import logging, shutil, traceback, os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image as PILImage
from PyQt6.QtCore import QThread, pyqtSignal
from constants import SEASON_CLASSES, SUPPORTED_EXT
from preprocessors import BasePreprocessor
from datasets import PersonalColorDataset


# ── Windows 멀티프로세스 DataLoader 안전 설정 ────────────────
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# ── cuDNN 최적화 (RTX 4080 Super Tensor Core 풀 활용) ────────
# benchmark=True  : 입력 크기 고정 시 가장 빠른 커널 자동 선택
# allow_tf32=True : Ampere 이상 GPU에서 Tensor Core 사용 → 최대 2배 빠름
torch.backends.cudnn.benchmark       = True
torch.backends.cudnn.allow_tf32      = True
torch.backends.cuda.matmul.allow_tf32 = True


class DatasetLoaderWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    def __init__(self, folder_path):
        super().__init__(); self.folder_path = folder_path; self._running = True
    def run(self):
        try:    self.finished.emit(PersonalColorDataset(self.folder_path).get_class_distribution())
        except Exception as e: self.error.emit(str(e))
    def stop(self):
        self._running = False; self.quit(); self.wait(2000)


class PreprocessWorker(QThread):
    progress    = pyqtSignal(int, str)
    image_ready = pyqtSignal(np.ndarray, dict)
    finished    = pyqtSignal(list)
    error       = pyqtSignal(str)

    def __init__(self, paths, mode: str = "standard", device: str = "cpu"):
        super().__init__()
        self.paths   = paths
        self._mode   = mode    # "standard" or "advanced"
        self._device = device
        self._running = True

    def run(self):
        from preprocessors import PersonalColorPreprocessor, AdvancedSkinPreprocessor
        # ★ 스레드 내부에서 FaceMesh 새로 생성
        if self._mode == "advanced":
            pre = AdvancedSkinPreprocessor(device=self._device)
        else:
            pre = PersonalColorPreprocessor(device=self._device)

        all_meta = []
        total    = len(self.paths)
        for i, path in enumerate(self.paths):
            if not self._running:
                break
            try:
                img, meta = pre.preprocess(path)
                all_meta.append(meta)
                if i == 0:
                    self.image_ready.emit(img.copy(), meta)
                self.progress.emit(int((i + 1) / total * 100), Path(path).name)
            except Exception as e:
                logging.warning(f"전처리 실패 {path}: {e}")
        self.finished.emit(all_meta)

    def stop(self):
        self._running = False
        self.quit()
        self.wait(3000)


# ─────────────────────────────────────────────────────────────
#  Val 전용 Dataset 래퍼
# ─────────────────────────────────────────────────────────────
class _ValWrapper(Dataset):
    def __init__(self, subset, transform):
        self._subset    = subset
        self._transform = transform

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        base_ds  = self._subset.dataset
        real_idx = self._subset.indices[idx]
        path, label = base_ds.samples[real_idx]
        try:
            img = PILImage.open(path).convert("RGB")
        except Exception:
            img = PILImage.new("RGB", (224, 224), (128, 128, 128))
        return self._transform(img), label


# ─────────────────────────────────────────────────────────────
#  경량 Dataset 래퍼 (경로 기반, 캐시 없음)
#  Windows QThread에서 num_workers=0 으로 사용
#  → 이미지 I/O를 최소화하기 위해 Pillow LOAD_TRUNCATED 활성화
# ─────────────────────────────────────────────────────────────
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # 손상 이미지 스킵 방지

class _FastDataset(Dataset):
    """경로 리스트 기반 Dataset. num_workers=0 환경 최적화."""
    def __init__(self, samples, transform):
        self._samples   = samples   # [(path, label), ...]
        self._transform = transform

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        path, label = self._samples[idx]
        try:
            img = PILImage.open(path).convert("RGB")
        except Exception:
            img = PILImage.new("RGB", (288, 288), (128, 128, 128))
        return self._transform(img), label


class TrainWorker(QThread):
    log_message    = pyqtSignal(str)
    epoch_complete = pyqtSignal(int, float, float, float, float)
    progress       = pyqtSignal(int)
    finished       = pyqtSignal(str)
    error          = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config    = config
        self._running  = True

    def run(self):
        try:
            self._train_loop()
        except Exception as e:
            self.error.emit(str(e))
            self.log_message.emit(traceback.format_exc())

    def _train_loop(self):
        # train_standalone의 CUDAPrefetchLoader 재사용
        from train_standalone import CUDAPrefetchLoader, OnTheFlyPreprocessDataset
        from datasets import PersonalColorDataset as _PCD, JSON_FEATURE_DIM, _extract_features

        # ── 디바이스 설정 ──────────────────────────────────────
        req = self.config.get("device", "cpu")
        if req == "cuda" and not torch.cuda.is_available():
            req = "cpu"; self.log_message.emit("[WARN] CUDA unavailable -> CPU")
        if req == "mps"  and not torch.backends.mps.is_available():
            req = "cpu"; self.log_message.emit("[WARN] MPS unavailable -> CPU")
        device   = torch.device(req)
        use_cuda = (device.type == "cuda")
        use_amp  = use_cuda
        self.log_message.emit(f"[INFO] Device: {device}  AMP: {use_amp}")

        if use_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.log_message.emit(f"[INFO] GPU: {gpu_name}  VRAM: {vram_gb:.1f}GB")
            self.log_message.emit("[INFO] cuDNN benchmark=ON  TF32=ON  AMP=ON")

        feat_mode    = self.config.get("feat_mode",    "jpg")
        preproc_mode = self.config.get("preproc_mode", "standard")
        preproc_dev  = self.config.get("preproc_dev",  "cpu")

        # ── Transform ─────────────────────────────────────────
        # 크롭 완료 이미지(전처리 폴더) 또는 온더플라이 크롭 결과:
        # 이미 얼굴 영역이므로 Resize(300,300)으로 크기만 통일
        _M = [0.485, 0.456, 0.406]
        _S = [0.229, 0.224, 0.225]
        trn_tf = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomCrop(260),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.06),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize(_M, _S),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop(260),
            transforms.ToTensor(),
            transforms.Normalize(_M, _S),
        ])

        # ── 데이터셋 스캔 ─────────────────────────────────────
        ds = _PCD(self.config["data_dir"], transform=None)
        if len(ds) == 0:
            self.error.emit("빈 데이터셋"); return

        nc = len(ds.classes)
        self.log_message.emit(f"[INFO] Classes: {ds.classes}")

        # Stratified 80/20 분할
        import random as _rng
        _rng.seed(42)
        class_indices = {i: [] for i in range(nc)}
        for idx, (_, label) in enumerate(ds.samples):
            class_indices[label].append(idx)
        tr_idx, vl_idx = [], []
        for label, idxs in class_indices.items():
            _rng.shuffle(idxs)
            n_val = max(1, int(len(idxs) * 0.2))
            vl_idx.extend(idxs[:n_val])
            tr_idx.extend(idxs[n_val:])

        tr_samples = [(ds.samples[i][0], ds.samples[i][1]) for i in tr_idx]
        vl_samples = [(ds.samples[i][0], ds.samples[i][1]) for i in vl_idx]
        ts, vs = len(tr_samples), len(vl_samples)

        # JSON 맵 구성
        use_json = feat_mode in ("json", "jpg_json")
        feat_dim = 0
        tr_json = vl_json = {}
        if use_json:
            feat_dim = JSON_FEATURE_DIM
            tr_json  = {ds.samples[i][0]: ds.json_map.get(ds.samples[i][0], {})
                        for i in tr_idx}
            vl_json  = {ds.samples[i][0]: ds.json_map.get(ds.samples[i][0], {})
                        for i in vl_idx}
            if not any(tr_json.values()):
                self.log_message.emit("[WARN] JSON 피처 없음 → jpg 모드 전환")
                use_json = False; feat_mode = "jpg"

        bs = self.config.get("batch_size", 32)
        self.log_message.emit(
            f"[INFO] Train:{ts}  Val:{vs}  배치:{bs}  "
            f"feat_mode:{feat_mode}  preproc:{preproc_mode}")

        # ── Dataset 생성 ──────────────────────────────────────
        # QThread: num_workers=0 고정 (Windows spawn 충돌 방지)
        # 원본 폴더(preprocess_meta.json 없음) → 온더플라이 크롭
        # 전처리 완료 폴더 → _FastDataset (크롭 저장된 이미지 직접 로드)
        is_raw = not ds.use_json and feat_mode != "json"

        if feat_mode == "json":
            from datasets import _extract_features as _ef

            class _JsonOnly(Dataset):
                def __init__(self, samples, jmap, augment=False):
                    self._s = samples; self._j = jmap; self._aug = augment
                def __len__(self): return len(self._s)
                def __getitem__(self, i):
                    p, lb = self._s[i]
                    feat = _ef(self._j.get(p, {}))
                    if self._aug:
                        noise = torch.randn_like(feat) * 0.02
                        feat  = (feat + noise).clamp(-1.0, 2.0)
                    return feat, lb

            tr_ds = _JsonOnly(tr_samples, tr_json, augment=True)
            vl_ds = _JsonOnly(vl_samples, vl_json, augment=False)

        elif is_raw:
            # 원본 폴더 → 온더플라이 크롭
            self.log_message.emit(
                f"[OnTheFly] 전처리 폴더 없음 → {preproc_mode} 크롭 학습 "
                f"(b*보정: {preproc_dev})")
            tr_ds = OnTheFlyPreprocessDataset(
                tr_samples, trn_tf,
                preproc_mode=preproc_mode, device=preproc_dev,
                json_map=tr_json if feat_mode == "jpg_json" else {})
            vl_ds = OnTheFlyPreprocessDataset(
                vl_samples, val_tf,
                preproc_mode=preproc_mode, device=preproc_dev,
                json_map=vl_json if feat_mode == "jpg_json" else {})
        else:
            # 전처리 완료 폴더 → 크롭 이미지 직접 로드
            tr_ds = _FastDataset(tr_samples, trn_tf)
            vl_ds = _FastDataset(vl_samples, val_tf)

        # ── DataLoader ────────────────────────────────────────
        tr_dl_raw = DataLoader(
            tr_ds, batch_size=bs, shuffle=True,
            num_workers=0, pin_memory=use_cuda, drop_last=True)
        vl_dl_raw = DataLoader(
            vl_ds, batch_size=bs * 2,
            num_workers=0, pin_memory=use_cuda)

        # VRAM Prefetch 래퍼: 디스크→핀메모리→CUDA 스트림 비동기 전송
        if use_cuda:
            self.log_message.emit(
                "[INFO] VRAM Prefetch 활성화: 디스크→핀메모리→VRAM 미리배치")
            tr_dl = CUDAPrefetchLoader(tr_dl_raw, device, feat_mode=feat_mode)
            vl_dl = CUDAPrefetchLoader(vl_dl_raw, device, feat_mode=feat_mode)
        else:
            tr_dl = tr_dl_raw
            vl_dl = vl_dl_raw

        self.log_message.emit(
            f"[INFO] Train배치수:{len(tr_dl)}  Val배치수:{len(vl_dl)}")

        # ── 모델 ─────────────────────────────────────────────
        backbone = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        cnn_dim  = backbone.classifier[1].in_features

        if feat_mode == "json":
            class _MLP(nn.Module):
                def __init__(self, fd, nc):
                    super().__init__()
                    self.bn_in = nn.BatchNorm1d(fd)
                    self.fc1   = nn.Linear(fd,  256)
                    self.bn1   = nn.BatchNorm1d(256)
                    self.fc2   = nn.Linear(256, 128)
                    self.bn2   = nn.BatchNorm1d(128)
                    self.fc3   = nn.Linear(128, 64)
                    self.bn3   = nn.BatchNorm1d(64)
                    self.out   = nn.Linear(64, nc)
                    self.drop1 = nn.Dropout(0.3)
                    self.drop2 = nn.Dropout(0.2)
                    self.drop3 = nn.Dropout(0.1)
                    self.res1  = nn.Linear(fd,  256)
                    self.res2  = nn.Linear(256, 128)
                def forward(self, x):
                    x  = self.bn_in(x)
                    r1 = self.res1(x)
                    x  = torch.relu(self.bn1(self.fc1(x)))
                    x  = self.drop1(x + r1)
                    r2 = self.res2(x)
                    x  = torch.relu(self.bn2(self.fc2(x)))
                    x  = self.drop2(x + r2)
                    x  = torch.relu(self.bn3(self.fc3(x)))
                    x  = self.drop3(x)
                    return self.out(x)
            model = _MLP(feat_dim, nc).to(device)
            self.log_message.emit(f"[INFO] 모델: ResidualMLP (JSON 전용) feat_dim={feat_dim}")
        else:
            backbone.classifier = nn.Sequential(
                nn.Dropout(0.3, True),
                nn.Linear(cnn_dim, 256), nn.ReLU(True),
                nn.Dropout(0.2),
                nn.Linear(256, nc),
            )
            model = backbone.to(device)
            self.log_message.emit(f"[INFO] 모델: EfficientNet-B2  nc={nc}")

        # ── 옵티마이저 & 스케줄러 ─────────────────────────────
        lr   = self.config.get("lr", 1e-3)
        ep_n = self.config.get("epochs", 10)

        if feat_mode == "json":
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            opt = optim.AdamW([
                {"params": model.features.parameters(),   "lr": lr * 0.05},
                {"params": model.classifier.parameters(), "lr": lr},
            ], weight_decay=1e-4)

        import math as _math

        warmup_ep = min(5, ep_n // 4)
        def _lr_lambda(ep):
            if ep < warmup_ep:
                return (ep + 1) / warmup_ep
            t = ep - warmup_ep
            T = max(ep_n - warmup_ep, 1)
            return 0.05 + 0.95 * (1 + _math.cos(_math.pi * t / T)) / 2

        sch    = optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        crit   = nn.CrossEntropyLoss(label_smoothing=0.1)
        best   = 0.0
        sp     = self.config.get("save_path", "best_model.pt")

        emit_every = max(1, len(tr_dl) // 20)

        for ep in range(1, ep_n + 1):
            if not self._running:
                self.log_message.emit("[INFO] 학습 중단"); break

            # ── Train ─────────────────────────────────────────
            model.train()
            tl = tc = tt = 0

            for bi, batch in enumerate(tr_dl):
                if not self._running: break

                # CUDAPrefetchLoader → 이미 CUDA 텐서
                # CPU fallback → .to()가 실제 이동
                if feat_mode == "json":
                    feat, y = batch
                    x = None
                    feat = feat.to(device, non_blocking=True)
                else:
                    x, y = batch
                    x    = x.to(device, non_blocking=True)
                    feat = None
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    o  = model(feat) if feat_mode == "json" else model(x)
                    lo = crit(o, y)

                scaler.scale(lo).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                n   = (feat if feat_mode == "json" else x).size(0)
                tl += lo.item() * n
                tc += o.argmax(1).eq(y).sum().item()
                tt += n

                if bi % emit_every == 0:
                    self.progress.emit(int((bi + 1) / len(tr_dl) * 100))

            # ── Validation ────────────────────────────────────
            model.eval()
            vl = vc = vt = 0
            with torch.no_grad():
                for batch in vl_dl:
                    if feat_mode == "json":
                        feat, y = batch
                        x = None
                        feat = feat.to(device, non_blocking=True)
                    else:
                        x, y = batch
                        x    = x.to(device, non_blocking=True)
                        feat = None
                    y = y.to(device, non_blocking=True)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        o  = model(feat) if feat_mode == "json" else model(x)
                        lo = crit(o, y)

                    n   = (feat if feat_mode == "json" else x).size(0)
                    vl += lo.item() * n
                    vc += o.argmax(1).eq(y).sum().item()
                    vt += n

            sch.step()

            atl    = tl / max(tt, 1)
            avl    = vl / max(vt, 1)
            ta     = tc / max(tt, 1) * 100
            va     = vc / max(vt, 1) * 100
            cur_lr = opt.param_groups[-1]["lr"]

            self.log_message.emit(
                f"[Epoch {ep:03d}/{ep_n}] "
                f"Train Loss={atl:.4f} Acc={ta:.2f}%  |  "
                f"Val Loss={avl:.4f} Acc={va:.2f}%  |  "
                f"LR={cur_lr:.2e}")
            self.epoch_complete.emit(ep, atl, ta, avl, va)
            self.progress.emit(100)

            if va > best:
                best = va
                torch.save({
                    "epoch":            ep,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          va,
                    "classes":          ds.classes,
                    "num_classes":      nc,
                    "model_arch":       "efficientnet_b2",
                    "feat_mode":        feat_mode,
                    "feat_dim":         feat_dim,
                    "preproc_mode":     preproc_mode,  # ★
                }, sp)
                self.log_message.emit(f"  [BEST] val_acc={va:.2f}% → {sp}")

        self.log_message.emit(
            f"\n{'='*60}\n[DONE] Best Val Acc: {best:.2f}%\n{'='*60}")
        self.finished.emit(sp)

    def stop(self):
        self._running = False


class InferenceWorker(QThread):
    finished = pyqtSignal(str, list, dict)   # pc, rk, season_probs
    error    = pyqtSignal(str)

    def __init__(self, model_path, image_path):
        super().__init__()
        self.model_path  = model_path
        self.image_path  = image_path

    def run(self):
        try:
            self.finished.emit(*self._infer())
        except Exception as e:
            self.error.emit(str(e))

    def _infer(self):
        dev = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu")
        ck       = torch.load(self.model_path, map_location=dev, weights_only=False)
        model_type = ck.get("model_type", "")

        # ── 모델 타입별 state_dict 키 정규화 ──────────────────
        # train_hierarchical.py 저장 포맷 자동 처리
        if model_type == "hierarchical_stage1":
            # Stage1 독립 저장: stage1_state 키 사용
            if "model_state_dict" not in ck and "stage1_state" in ck:
                ck["model_state_dict"] = ck["stage1_state"]
            ck.setdefault("num_classes", len(ck.get("stage1_classes", []) or [4]))
            ck.setdefault("classes",     ck.get("stage1_classes", ["Spring","Summer","Autumn","Winter"]))

        elif model_type == "hierarchical_stage2":
            # Stage2 독립 저장: 계절별 state 중 첫 번째 사용 (추론 불가 — 경고)
            raise RuntimeError(
                "Stage2 단독 모델은 직접 추론 불가합니다.\n"
                "'조합' 탭에서 Stage1 + Stage2를 합쳐 final.pt를 만든 후 사용하세요.")

        elif model_type == "hierarchical":
            # Stage1+Stage2 조합된 최종 모델 → 계층 추론 실행
            return self._infer_hierarchical(ck, dev)

        nc        = ck.get("num_classes", 4)
        cls       = ck.get("classes", ["Spring","Summer","Autumn","Winter"][:nc])
        feat_mode = ck.get("feat_mode", "jpg")

        from constants import parse_season_tone

        # ── JSON 전용 모드 (MLP) ──────────────────────────────
        if feat_mode == "json":
            feat_dim    = ck.get("feat_dim", 16)
            preproc_mode = ck.get("preproc_mode", "standard")  # ★ 학습 때 모드

            # ★ MLP 구조를 학습 시(_JsonOnlyModel)와 완전히 일치
            class _MLP(nn.Module):
                def __init__(self, fd, nc):
                    super().__init__()
                    self.bn_in = nn.BatchNorm1d(fd)
                    self.fc1   = nn.Linear(fd,  256)
                    self.bn1   = nn.BatchNorm1d(256)
                    self.fc2   = nn.Linear(256, 128)
                    self.bn2   = nn.BatchNorm1d(128)
                    self.fc3   = nn.Linear(128, 64)
                    self.bn3   = nn.BatchNorm1d(64)
                    self.out   = nn.Linear(64, nc)
                    self.drop1 = nn.Dropout(0.3)
                    self.drop2 = nn.Dropout(0.2)
                    self.drop3 = nn.Dropout(0.1)
                    self.res1  = nn.Linear(fd,  256)
                    self.res2  = nn.Linear(256, 128)
                def forward(self, x):
                    x  = self.bn_in(x)
                    r1 = self.res1(x)
                    x  = torch.relu(self.bn1(self.fc1(x)))
                    x  = self.drop1(x + r1)
                    r2 = self.res2(x)
                    x  = torch.relu(self.bn2(self.fc2(x)))
                    x  = self.drop2(x + r2)
                    x  = torch.relu(self.bn3(self.fc3(x)))
                    x  = self.drop3(x)
                    return self.out(x)

            m = _MLP(feat_dim, nc)
            m.load_state_dict(ck["model_state_dict"])
            m = m.to(dev).eval()

            # ★ 학습 때와 동일한 전처리기로 피처 추출
            try:
                from preprocessors import (PersonalColorPreprocessor,
                                            AdvancedSkinPreprocessor)
                from datasets import _extract_features
                if preproc_mode == "advanced":
                    pre = AdvancedSkinPreprocessor()
                else:
                    pre = PersonalColorPreprocessor()
                _, meta = pre.preprocess(self.image_path)
                feat = _extract_features(meta).unsqueeze(0).to(dev)
            except Exception as e:
                raise RuntimeError(f"피처 추출 실패: {e}")

            with torch.no_grad():
                m.eval()
                pr = torch.softmax(m(feat), 1).squeeze().cpu().numpy()

        # ── JPG+JSON 융합 모드 (FusionModel) ─────────────────
        elif feat_mode == "jpg_json":
            feat_dim = ck.get("feat_dim", 10)
            arch     = ck.get("model_arch", "efficientnet_b2")

            backbone = models.efficientnet_b2(weights=None) \
                if arch == "efficientnet_b2" else models.efficientnet_b0(weights=None)
            inp     = 260 if arch == "efficientnet_b2" else 224
            cnn_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()

            # FusionModel 재구성
            head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(cnn_dim + feat_dim, 512), nn.ReLU(True),
                nn.Dropout(0.2),
                nn.Linear(512, 256), nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Linear(256, nc),
            )

            class _Fusion(nn.Module):
                def __init__(self, bb, hd):
                    super().__init__()
                    self.backbone = bb
                    self.head     = hd
                def forward(self, x, feat=None):
                    out = self.backbone(x)
                    if feat is not None:
                        out = torch.cat([out, feat], dim=1)
                    return self.head(out)

            m = _Fusion(backbone, head)
            m.load_state_dict(ck["model_state_dict"])
            m = m.to(dev).eval()

            tf = transforms.Compose([
                transforms.Resize((inp, inp)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            img_t = tf(PILImage.open(self.image_path).convert("RGB")).unsqueeze(0).to(dev)

            try:
                from preprocessors import (PersonalColorPreprocessor,
                                            AdvancedSkinPreprocessor)
                from datasets import _extract_features
                preproc_mode = ck.get("preproc_mode", "standard")
                if preproc_mode == "advanced":
                    pre = AdvancedSkinPreprocessor()
                else:
                    pre = PersonalColorPreprocessor()
                _, meta = pre.preprocess(self.image_path)
                feat = _extract_features(meta).unsqueeze(0).to(dev)
            except Exception:
                feat = None

            with torch.no_grad():
                pr = torch.softmax(m(img_t, feat), 1).squeeze().cpu().numpy()

        # ── JPG 전용 모드 (CNN) ───────────────────────────────
        else:
            arch = ck.get("model_arch", "efficientnet_b0")
            if arch == "efficientnet_b2":
                m   = models.efficientnet_b2(weights=None)
                inp = 260
            else:
                m   = models.efficientnet_b0(weights=None)
                inp = 224

            in_f = m.classifier[1].in_features
            m.classifier = nn.Sequential(
                nn.Dropout(0.3, True), nn.Linear(in_f, 256),
                nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256, nc))
            m.load_state_dict(ck["model_state_dict"])
            m = m.to(dev).eval()

            tf = transforms.Compose([
                transforms.Resize((inp, inp)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            t = tf(PILImage.open(self.image_path).convert("RGB")).unsqueeze(0).to(dev)
            with torch.no_grad():
                pr = torch.softmax(m(t), 1).squeeze().cpu().numpy()

        # ── 공통: 결과 정리 ───────────────────────────────────
        pc = cls[int(pr.argmax())]
        rk = sorted(zip(cls, pr.tolist()), key=lambda x: x[1], reverse=True)

        season_probs: Dict[str, float] = {}
        for c, prob in zip(cls, pr.tolist()):
            season, _ = parse_season_tone(c)
            season_probs[season] = season_probs.get(season, 0.0) + prob

        return pc, rk, season_probs

    def _infer_hierarchical(self, ck: dict, dev):
        """Stage1(계절) → Stage2(톤) 2단계 계층 추론"""
        from constants import SEASON_CLASSES, TONE_CLASSES, parse_season_tone

        feat_mode    = ck.get("feat_mode", "jpg")
        feat_dim     = ck.get("feat_dim",  0)
        arch         = ck.get("model_arch", "efficientnet_b2")
        preproc_mode = ck.get("preproc_mode", "standard")
        inp          = 260 if arch == "efficientnet_b2" else 224
        season_cls   = ck.get("stage1_classes", SEASON_CLASSES)
        tone_cls     = ck.get("stage2_tone_classes", TONE_CLASSES)

        # ── 이미지 전처리 ──────────────────────────────────────
        tf = transforms.Compose([
            transforms.Resize((inp, inp)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        img = PILImage.open(self.image_path).convert("RGB")
        img_t = tf(img).unsqueeze(0).to(dev)

        # JSON 피처 추출 (jpg_json 모드)
        feat_t = None
        if feat_mode == "jpg_json" and feat_dim > 0:
            try:
                from preprocessors import (PersonalColorPreprocessor,
                                            AdvancedSkinPreprocessor)
                from datasets import _extract_features
                pre = AdvancedSkinPreprocessor() if preproc_mode == "advanced" \
                      else PersonalColorPreprocessor()
                _, meta = pre.preprocess(self.image_path)
                feat_t = _extract_features(meta).unsqueeze(0).to(dev)
            except Exception:
                feat_t = None

        def _build_and_load(state_dict, nc):
            """EfficientNet-B2 기반 모델 빌드 + state 로드"""
            backbone = models.efficientnet_b2(weights=None)
            cnn_dim  = backbone.classifier[1].in_features
            if feat_mode == "jpg_json" and feat_dim > 0:
                backbone.classifier = nn.Identity()
                head = nn.Sequential(
                    nn.Dropout(0.3), nn.Linear(cnn_dim + feat_dim, 512), nn.ReLU(True),
                    nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(True),
                    nn.Dropout(0.1), nn.Linear(256, nc),
                )
                class _Fusion(nn.Module):
                    def __init__(self, bb, hd):
                        super().__init__(); self.backbone=bb; self.head=hd
                    def forward(self, x, feat=None):
                        o = self.backbone(x)
                        if feat is not None: o = torch.cat([o, feat], dim=1)
                        return self.head(o)
                m = _Fusion(backbone, head)
            else:
                backbone.classifier = nn.Sequential(
                    nn.Dropout(0.3, True), nn.Linear(cnn_dim, 256),
                    nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256, nc))
                m = backbone
            m.load_state_dict(state_dict)
            return m.to(dev).eval()

        def _forward(m, img_t, feat_t):
            with torch.no_grad():
                if feat_mode == "jpg_json" and feat_t is not None:
                    out = m(img_t, feat_t)
                else:
                    out = m(img_t)
            return torch.softmax(out, 1).squeeze().cpu().numpy()

        # ── Stage1: 계절 분류 ──────────────────────────────────
        s1_state = ck.get("stage1_state")
        if s1_state is None:
            raise RuntimeError("stage1_state 키 없음 — 조합되지 않은 모델입니다.")
        m1 = _build_and_load(s1_state, len(season_cls))
        s1_pr = _forward(m1, img_t, feat_t)
        pred_season = season_cls[int(s1_pr.argmax())]
        del m1

        # ── Stage2: 톤 분류 (예측된 계절 전용 모델) ────────────
        s2_key   = f"stage2_{pred_season}_state"
        s2_state = ck.get(s2_key)
        if s2_state is None:
            # Stage2 없으면 Stage1 결과만 반환
            pc = pred_season
            rk = [(s, float(p)) for s, p in zip(season_cls, s1_pr)]
            rk.sort(key=lambda x: x[1], reverse=True)
            return pc, rk, {s: float(p) for s, p in zip(season_cls, s1_pr)}

        m2 = _build_and_load(s2_state, len(tone_cls))
        s2_pr = _forward(m2, img_t, feat_t)
        pred_tone = tone_cls[int(s2_pr.argmax())]
        del m2

        # ── 결과 조합 ──────────────────────────────────────────
        # 최종 예측: Spring_Warm 형태로
        pc = f"{pred_season}_{pred_tone}"

        # 랭킹: 해당 계절의 S1 확률 × S2 톤 확률로 12클래스 점수 생성
        rk_raw = []
        for si, season in enumerate(season_cls):
            for ti, tone in enumerate(tone_cls):
                score = float(s1_pr[si]) * float(s2_pr[ti])   # 결합 확률
                rk_raw.append((f"{season}_{tone}", score))
        rk_raw.sort(key=lambda x: x[1], reverse=True)
        # 정규화
        total = sum(s for _, s in rk_raw) or 1.0
        rk = [(c, s / total) for c, s in rk_raw]

        season_probs = {s: float(s1_pr[i]) for i, s in enumerate(season_cls)}

        return pc, rk, season_probs


class AutoLabelWorker(QThread):
    progress    = pyqtSignal(int, str, str, float)
    log_message = pyqtSignal(str)
    finished    = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, model_path, src_dir, out_dir, threshold=0.85):
        super().__init__()
        self.model_path = model_path
        self.src_dir    = src_dir
        self.out_dir    = out_dir
        self.threshold  = threshold
        self._running   = True

    def run(self):
        try:
            self._auto_label()
        except Exception as e:
            self.error.emit(str(e))
            self.log_message.emit(traceback.format_exc())

    def _auto_label(self):
        dev = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu")
        self.log_message.emit(f"[INFO] Device: {dev}")

        ck  = torch.load(self.model_path, map_location=dev)
        nc  = ck.get("num_classes", 4)
        cls = ck.get("classes", SEASON_CLASSES[:nc])

        arch = ck.get("model_arch", "efficientnet_b0")
        if arch == "efficientnet_b2":
            m   = models.efficientnet_b2(weights=None)
            inp = 260
        else:
            m   = models.efficientnet_b0(weights=None)
            inp = 224

        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3, True), nn.Linear(in_f, 256),
            nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256, nc))
        m.load_state_dict(ck["model_state_dict"])
        m = m.to(dev).eval()
        self.log_message.emit(f"[INFO] Model({arch}) loaded  classes={cls}")

        tf = transforms.Compose([
            transforms.Resize((inp, inp)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        imgs  = [p for p in Path(self.src_dir).rglob("*")
                 if p.suffix.lower() in SUPPORTED_EXT]
        total = len(imgs)
        self.log_message.emit(
            f"[INFO] Images:{total}  thr:{self.threshold:.0%}")
        if total == 0:
            self.finished.emit({"error": "이미지 없음"}); return

        out = Path(self.out_dir)
        for c in cls:
            (out / c).mkdir(parents=True, exist_ok=True)

        stats: Dict = {c: 0 for c in cls}
        stats["skipped_low_conf"] = 0
        stats["skipped_error"]    = 0

        for i, ip in enumerate(imgs):
            if not self._running:
                self.log_message.emit("[INFO] 중단"); break
            try:
                t = tf(PILImage.open(str(ip)).convert("RGB")).unsqueeze(0).to(dev)
                with torch.no_grad():
                    pr = torch.softmax(m(t), 1).squeeze().cpu().numpy()
                cf = float(pr.max())
                pc = cls[int(pr.argmax())]
                pct = int((i + 1) / total * 100)
                self.progress.emit(pct, ip.name, pc, cf)
                if cf >= self.threshold:
                    dst = out / pc / ip.name
                    if dst.exists():
                        dst = out / pc / f"{ip.stem}_{i}{ip.suffix}"
                    shutil.copy2(str(ip), str(dst))
                    stats[pc] += 1
                    self.log_message.emit(
                        f"  [OK]  [{pc:6s}] {ip.name}  conf={cf:.3f}")
                else:
                    stats["skipped_low_conf"] += 1
                    self.log_message.emit(
                        f"  [SKIP] {ip.name}  conf={cf:.3f}"
                        f"<{self.threshold:.2f}  (best={pc})")
            except Exception as e:
                stats["skipped_error"] += 1
                self.log_message.emit(f"  [ERR] {ip.name}: {e}")

        lb = sum(stats[c] for c in cls)
        self.log_message.emit(
            f"\n{'='*50}\n[DONE] 분류:{lb}  "
            f"건너뜀:{stats['skipped_low_conf']}  "
            f"오류:{stats['skipped_error']}\n{'='*50}")
        self.finished.emit(stats)

    def stop(self):
        self._running = False

# ─────────────────────────────────────────────────────────────
#  FullPreprocessWorker
#  데이터셋 전체를 전처리하여 폴더 구조 유지하며 저장
#  입력: src_root/Spring/Warm/img.jpg
#  출력: out_dir/Spring/Warm/img.jpg  (b* 보정 적용)
# ─────────────────────────────────────────────────────────────
class FullPreprocessWorker(QThread):
    progress    = pyqtSignal(int, str)       # (퍼센트, 파일명)
    log_message = pyqtSignal(str)
    finished    = pyqtSignal(str, int, int)  # (out_dir, 저장수, 실패수)
    error       = pyqtSignal(str)

    def __init__(self, image_paths: list, out_dir: str,
                 mode: str,
                 src_root: str,
                 device: str = "cpu",
                 apply_correction: bool = True,
                 output_format: str = "jpg",
                 num_workers: int = 1):
        super().__init__()
        self._paths            = image_paths
        self._out_dir          = out_dir
        self._mode             = mode
        self._src_root         = Path(src_root)
        self._device           = device
        self._apply_correction = apply_correction
        self._output_format    = output_format
        self._num_workers      = max(1, num_workers)
        self._running          = True

    def run(self):
        try:
            if self._num_workers == 1:
                self._process_sequential()
            else:
                self._process_parallel()
        except Exception as e:
            self.error.emit(str(e))

    def _process_sequential(self):
        import cv2 as _cv2
        import json
        from preprocessors import (
            PersonalColorPreprocessor, AdvancedSkinPreprocessor,
            GPULabCorrector)

        is_jpg  = self._output_format == "jpg"
        is_json = self._output_format == "json"

        if self._mode == "advanced":
            pre = AdvancedSkinPreprocessor(
                apply_correction=self._apply_correction,
                device=self._device)
        else:
            pre = PersonalColorPreprocessor(
                apply_correction=self._apply_correction,
                device=self._device)

        corrector = GPULabCorrector(self._device)
        out_root  = Path(self._out_dir)

        # ── 이전 실행 기록 로드 (이어하기) ───────────────────
        meta_path = out_root / "preprocess_meta.json"
        prev_records: list = []
        prev_done_paths: set = set()   # 이미 처리 완료된 원본 경로 set

        if meta_path.exists():
            try:
                prev_summary = json.loads(meta_path.read_text(encoding="utf-8"))
                # output_format이 다르면 이어하기 불가 → 처음부터
                if prev_summary.get("output_format") == self._output_format:
                    prev_records = prev_summary.get("records", [])
                    src_root_str = str(self._src_root)
                    for r in prev_records:
                        # image_path는 상대경로 → 원본 절대경로 복원
                        orig = str(self._src_root / r["image_path"])
                        prev_done_paths.add(orig)
                    self.log_message.emit(
                        f"[이어하기] 이전 기록 발견: {len(prev_done_paths)}개 완료됨 → 스킵합니다")
                else:
                    self.log_message.emit(
                        f"[이어하기] 출력 형식이 달라 처음부터 시작합니다 "
                        f"(이전: {prev_summary.get('output_format')}, "
                        f"현재: {self._output_format})")
            except Exception as e:
                self.log_message.emit(f"[이어하기] 기록 읽기 실패, 처음부터 시작: {e}")

        self.log_message.emit(
            f"[장치] MediaPipe: CPU  |  b* 보정: {corrector.info()}")
        self.log_message.emit(
            f"[출력] {'이미지(JPG)만 저장' if is_jpg else 'JSON 피처만 저장 (이미지 없음)'}")

        total           = len(self._paths)
        saved           = 0        # 이번 실행에서 새로 저장한 수
        skipped         = 0        # 이어하기로 스킵한 수
        failed          = 0
        corrected_count = 0
        index_records: list = list(prev_records)  # 이전 기록 유지하며 추가

        for i, img_path in enumerate(self._paths):
            if not self._running:
                self.log_message.emit("[STOP] 중단됨")
                break

            src = Path(img_path)

            # ── 이미 처리된 파일 스킵 ────────────────────────
            if str(src) in prev_done_paths:
                skipped += 1
                self.progress.emit(int((i + 1) / total * 100), src.name)
                continue

            try:
                try:
                    rel = src.relative_to(self._src_root)
                except ValueError:
                    rel = Path(src.name)

                dst_img  = out_root / rel
                dst_json = dst_img.with_suffix(".json")
                dst_img.parent.mkdir(parents=True, exist_ok=True)

                # ── 출력 파일이 이미 존재하는지도 확인 ────────
                # (meta 없이 파일만 남아있는 경우 대비)
                if is_jpg and dst_img.exists():
                    skipped += 1
                    self.progress.emit(int((i + 1) / total * 100), src.name)
                    continue
                if is_json and dst_json.exists():
                    skipped += 1
                    self.progress.emit(int((i + 1) / total * 100), src.name)
                    continue

                # MediaPipe 얼굴 감지 + 피처 추출
                _, meta = pre.preprocess(str(src))

                # ★ 디버그: 얼굴 감지 상태 확인
                _face_det = meta.get("face_detected", False)
                _has_crop = "face_crop_bgr" in meta
                _crop_box = meta.get("crop_box", None)
                logging.info(
                    f"[DEBUG] {src.name}: face_detected={_face_det}, "
                    f"has_crop_key={_has_crop}, crop_box={_crop_box}")

                # 클래스명 추출
                parts = rel.parts
                season_name = parts[0] if len(parts) >= 1 else "Unknown"
                tone_name   = parts[1] if len(parts) >= 2 else None
                from datasets import _norm_season, _norm_tone
                season     = _norm_season(season_name) or season_name
                tone       = _norm_tone(tone_name) if tone_name else None
                class_name = f"{season}_{tone}" if tone else season

                corrected = meta.get("corrected", False)
                b_delta   = meta.get("b_delta", 0.0)
                b_cv      = meta.get("b_cv", 128.0)

                # ── JPG 모드: 얼굴 크롭 영역만 저장 (JSON과 동일 기준) ──
                if is_jpg:
                    # meta에 face_crop_bgr 있으면 크롭 이미지, 없으면 전체
                    save_img = meta.get("face_crop_bgr")
                    if save_img is None:
                        logging.warning(
                            f"[DEBUG] {src.name}: face_crop_bgr=None → "
                            f"원본 폴백! face_detected={_face_det}")
                        bgr = _cv2.imread(str(src))
                        if bgr is None:
                            raise ValueError("이미지 읽기 실패")
                        save_img = bgr
                    else:
                        # ★ 크롭 vs 원본 크기 비교
                        _orig_bgr = _cv2.imread(str(src))
                        if _orig_bgr is not None:
                            _orig_shape = _orig_bgr.shape[:2]
                            _crop_shape = save_img.shape[:2]
                            if _orig_shape == _crop_shape:
                                logging.warning(
                                    f"[DEBUG] {src.name}: 크롭 크기 == 원본 크기! "
                                    f"({_crop_shape}) → 크롭이 적용되지 않았을 수 있음")
                            else:
                                logging.info(
                                    f"[DEBUG] {src.name}: 원본={_orig_shape} "
                                    f"→ 크롭={_crop_shape} ✓")

                    if corrected and save_img is not None:
                        corrected_count += 1
                        tag = f"crop+delta={b_delta:+.1f} [{corrector.device.upper()}]"
                    else:
                        tag = "crop" if meta.get("face_detected") else "no-face(full)"
                    _cv2.imwrite(str(dst_img), save_img)
                    index_records.append({
                        "image_path": str(rel).replace("\\", "/"),
                        "class"     : class_name,
                        "cropped"   : bool(meta.get("face_detected", False)),
                    })
                    self.log_message.emit(
                        f"  ✅ {src.name}  [{tag}]  → {rel}")

                # ── JSON 모드: 피처 JSON만 저장 ───────────────
                elif is_json:
                    json_data = {
                        "image_path"      : str(rel).replace("\\", "/"),
                        "class"           : class_name,
                        "season"          : season,
                        "tone"            : tone,
                        "face_detected"   : meta.get("face_detected", False),
                        "corrected"       : corrected,
                        "b_delta"         : round(b_delta, 4),
                        "b_cv"            : round(b_cv, 4),
                        "b_star"          : round(meta.get("b_star", 0.0), 4),
                        "lab_mean"        : [round(v, 4) for v in meta.get("lab_mean", [0, 128, 128])],
                        "lab_std"         : [round(v, 4) for v in meta.get("lab_std",  [0, 0, 0])],
                        "cheek_rgb"       : meta.get("cheek_rgb", [200, 150, 150]),
                        "hsv_s_mean"      : round(meta.get("hsv_s_mean", 0.0), 4),
                        "skin_pixel_ratio": round(meta.get("skin_pixel_ratio", 0.0), 4),
                        "mode"            : self._mode,
                        "device"          : corrector.device,
                    }
                    dst_json.write_text(
                        json.dumps(json_data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                    index_records.append({
                        "image_path": str(rel).replace("\\", "/"),
                        "json_path" : str(dst_json.relative_to(out_root)).replace("\\", "/"),
                        "class"     : class_name,
                    })
                    self.log_message.emit(
                        f"  ✅ {src.name}  → {dst_json.name}")

                saved += 1

            except Exception as e:
                failed += 1
                self.log_message.emit(f"  ❌ {src.name}: {e}")

            self.progress.emit(int((i + 1) / total * 100), src.name)

        # preprocess_meta.json 저장 — 누적 기록으로 덮어씀
        summary = {
            "total"          : len(index_records),   # 전체 누적 저장 수
            "saved_this_run" : saved,
            "skipped"        : skipped,
            "failed"         : failed,
            "corrected"      : corrected_count if is_jpg else 0,
            "output_format"  : self._output_format,
            "mode"           : self._mode,
            "device"         : self._device,
            "src_root"       : str(self._src_root),
            "records"        : index_records,
        }
        meta_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8")

        if is_jpg:
            self.log_message.emit(
                f"\n[완료] 새로저장:{saved}  스킵:{skipped}  "
                f"보정적용:{corrected_count}  실패:{failed}  "
                f"누적합계:{len(index_records)}")
        else:
            self.log_message.emit(
                f"\n[완료] JSON 새로저장:{saved}  스킵:{skipped}  "
                f"실패:{failed}  누적합계:{len(index_records)}")
        self.log_message.emit(f"[인덱스] {meta_path}")
        self.finished.emit(self._out_dir, len(index_records), failed)

    # ── 병렬 전처리 ──────────────────────────────────────────
    def _process_parallel(self):
        """
        ThreadPoolExecutor로 CPU 코어를 병렬 활용.
        각 워커가 독립 FaceMesh 인스턴스를 생성 (인스턴스 공유 불가).
        결과는 thread-safe queue로 수집 후 메인 스레드에서 일괄 저장.
        """
        import cv2 as _cv2
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from preprocessors import GPULabCorrector
        from pathlib import Path as _Path

        is_jpg  = self._output_format == "jpg"
        is_json = self._output_format == "json"
        out_root = _Path(self._out_dir)

        # ── 이전 실행 기록 로드 (이어하기) ───────────────────
        meta_path = out_root / "preprocess_meta.json"
        prev_records: list = []
        prev_done_paths: set = set()

        if meta_path.exists():
            try:
                prev_summary = json.loads(meta_path.read_text(encoding="utf-8"))
                if prev_summary.get("output_format") == self._output_format:
                    prev_records    = prev_summary.get("records", [])
                    prev_done_paths = {
                        str(self._src_root / r["image_path"])
                        for r in prev_records}
                    self.log_message.emit(
                        f"[이어하기] 이전 기록: {len(prev_done_paths)}개 스킵")
            except Exception:
                pass

        # 처리 대상 필터링
        todo = [p for p in self._paths if str(p) not in prev_done_paths]
        total   = len(self._paths)
        skipped = len(self._paths) - len(todo)

        self.log_message.emit(
            f"[병렬] workers={self._num_workers}  처리대상={len(todo)}장  "
            f"스킵={skipped}장")

        # 저장 디렉토리 사전 생성 (스레드 충돌 방지)
        dirs_created = set()
        for img_path in todo:
            src = _Path(img_path)
            try:
                rel = src.relative_to(self._src_root)
            except ValueError:
                rel = _Path(src.name)
            dst_dir = out_root / rel.parent
            if str(dst_dir) not in dirs_created:
                dst_dir.mkdir(parents=True, exist_ok=True)
                dirs_created.add(str(dst_dir))

        # ── 워커 함수 (각 스레드에서 독립 실행) ──────────────
        corrector  = GPULabCorrector(self._device)
        lock       = threading.Lock()
        done_count = [0]   # 리스트로 감싸야 스레드 내 수정 가능

        def _worker(img_path: str):
            """한 이미지를 처리하고 (record, log_msg, ok) 반환"""
            from preprocessors import (PersonalColorPreprocessor,
                                        AdvancedSkinPreprocessor)
            # ★ 스레드마다 독립 FaceMesh 인스턴스 생성
            if self._mode == "advanced":
                pre = AdvancedSkinPreprocessor(
                    apply_correction=True, device=self._device)
            else:
                pre = PersonalColorPreprocessor(
                    apply_correction=True, device=self._device)

            src = _Path(img_path)
            try:
                rel = src.relative_to(self._src_root)
            except ValueError:
                rel = _Path(src.name)

            dst_img  = out_root / rel
            dst_json = dst_img.with_suffix(".json")

            # 이미 존재하면 스킵
            if is_jpg  and dst_img.exists():
                return None, f"  ⏩ 스킵(존재): {src.name}", True
            if is_json and dst_json.exists():
                return None, f"  ⏩ 스킵(존재): {src.name}", True

            try:
                _, meta = pre.preprocess(str(src))

                parts = rel.parts
                season_name = parts[0] if len(parts) >= 1 else "Unknown"
                tone_name   = parts[1] if len(parts) >= 2 else None
                from datasets import _norm_season, _norm_tone
                season     = _norm_season(season_name) or season_name
                tone       = _norm_tone(tone_name) if tone_name else None
                class_name = f"{season}_{tone}" if tone else season

                b_cv      = meta.get("b_cv", 128.0)
                b_delta   = meta.get("b_delta", 0.0)
                corrected = meta.get("corrected", False)

                if is_jpg:
                    save_img = meta.get("face_crop_bgr")
                    if save_img is None:
                        save_img = _cv2.imread(str(src))
                        if save_img is None:
                            raise ValueError("이미지 읽기 실패")
                    tag = (f"crop+delta={b_delta:+.1f} [{corrector.device.upper()}]"
                           if corrected else
                           ("crop" if meta.get("face_detected") else "no-face(full)"))
                    _cv2.imwrite(str(dst_img), save_img)
                    record = {
                        "image_path": str(rel).replace("\\", "/"),
                        "class"     : class_name,
                        "cropped"   : bool(meta.get("face_detected", False)),
                    }
                    return record, f"  ✅ {src.name}  [{tag}]  → {rel}", True

                elif is_json:
                    json_data = {
                        "image_path"      : str(rel).replace("\\", "/"),
                        "class"           : class_name,
                        "season"          : season,
                        "tone"            : tone,
                        "face_detected"   : meta.get("face_detected", False),
                        "corrected"       : corrected,
                        "b_delta"         : round(b_delta, 4),
                        "b_cv"            : round(b_cv, 4),
                        "b_star"          : round(meta.get("b_star", 0.0), 4),
                        "lab_mean"        : [round(v, 4) for v in meta.get("lab_mean", [0, 128, 128])],
                        "lab_std"         : [round(v, 4) for v in meta.get("lab_std",  [0, 0, 0])],
                        "cheek_rgb"       : meta.get("cheek_rgb", [200, 150, 150]),
                        "hsv_s_mean"      : round(meta.get("hsv_s_mean", 0.0), 4),
                        "skin_pixel_ratio": round(meta.get("skin_pixel_ratio", 0.0), 4),
                        "mode"            : self._mode,
                        "device"          : corrector.device,
                    }
                    dst_json.write_text(
                        json.dumps(json_data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                    record = {
                        "image_path": str(rel).replace("\\", "/"),
                        "json_path" : str(dst_json.relative_to(out_root)).replace("\\", "/"),
                        "class"     : class_name,
                    }
                    return record, f"  ✅ {src.name}  → {dst_json.name}", True

            except Exception as e:
                return None, f"  ❌ {src.name}: {e}", False

        # ── 병렬 실행 ─────────────────────────────────────────
        index_records = list(prev_records)
        saved = failed = 0

        with ThreadPoolExecutor(max_workers=self._num_workers) as ex:
            futures = {ex.submit(_worker, p): p for p in todo}

            for fut in as_completed(futures):
                if not self._running:
                    ex.shutdown(wait=False, cancel_futures=True)
                    self.log_message.emit("[STOP] 중단됨")
                    break

                record, log_msg, ok = fut.result()
                self.log_message.emit(log_msg)

                with lock:
                    done_count[0] += 1
                    pct = int((done_count[0] + skipped) / total * 100)
                    fname = _Path(futures[fut]).name

                if ok and record is not None:
                    index_records.append(record)
                    saved += 1
                elif not ok:
                    failed += 1

                self.progress.emit(pct, fname)

        # ── preprocess_meta.json 저장 ─────────────────────────
        summary = {
            "total"          : len(index_records),
            "saved_this_run" : saved,
            "skipped"        : skipped,
            "failed"         : failed,
            "corrected"      : 0,
            "output_format"  : self._output_format,
            "mode"           : self._mode,
            "device"         : self._device,
            "src_root"       : str(self._src_root),
            "num_workers"    : self._num_workers,
            "records"        : index_records,
        }
        meta_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8")

        self.log_message.emit(
            f"\n[완료-병렬] 저장:{saved}  스킵:{skipped}  실패:{failed}  "
            f"누적:{len(index_records)}")
        self.log_message.emit(f"[인덱스] {meta_path}")
        self.finished.emit(self._out_dir, len(index_records), failed)

    def stop(self):
        self._running = False