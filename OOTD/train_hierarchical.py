"""
train_hierarchical.py — 2단계 계층 분류 학습
=============================================

구조:
  Stage 1 : 4계절 분류 (Spring / Summer / Autumn / Winter)
            → EfficientNet-B2 or FusionModel  →  4-class softmax

  Stage 2 : 계절별 톤 분류 (Warm / Bright / Light)
            → 계절마다 전용 모델 4개  →  3-class softmax
            → Spring 전용, Summer 전용, Autumn 전용, Winter 전용

저장 형식:
  save_path  (예: hierarchical.pt)
  ├── stage1_state  : Stage1 모델 가중치
  ├── stage2_Spring_state  : Spring 톤 모델 가중치
  ├── stage2_Summer_state  : Summer 톤 모델 가중치
  ├── stage2_Autumn_state  : Autumn 톤 모델 가중치
  ├── stage2_Winter_state  : Winter 톤 모델 가중치
  ├── feat_mode / feat_dim / model_arch / preproc_mode / img_mode
  └── val_acc_stage1, val_acc_stage2_{season}

사용법:
  python train_hierarchical.py \\
      --data_dir D:/data/img \\
      --json_dir D:/data/json \\
      --save_path D:/models/hier.pt \\
      --epochs_s1 40 --epochs_s2 40 \\
      --feat_mode jpg_json --img_mode crop \\
      --device cuda
"""

import argparse
import json
import logging
import math
import os
import signal
import subprocess
import sys
import io
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as PILImage, ImageFile
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms, models

sys.path.insert(0, str(Path(__file__).parent))
from datasets import PersonalColorDataset, JSON_FEATURE_DIM, _extract_features
from constants import SEASON_CLASSES, TONE_CLASSES

torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

_STOP = False
def _sig(sig, frame):
    global _STOP
    log.info("\n[STOP] Ctrl+C — 현재 에포크 후 종료")
    _STOP = True
signal.signal(signal.SIGINT, _sig)


# ─────────────────────────────────────────────────────────────
#  모델 빌더
# ─────────────────────────────────────────────────────────────
def _build_model(nc: int, feat_mode: str, feat_dim: int, device):
    """EfficientNet-B2 기반 모델 생성 (feat_mode 따라 분기)"""
    backbone = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    cnn_dim = backbone.classifier[1].in_features

    if feat_mode == "jpg_json":
        backbone.classifier = nn.Identity()
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
                self.backbone = bb; self.head = hd
            def forward(self, x, feat=None):
                o = self.backbone(x)
                if feat is not None:
                    o = torch.cat([o, feat], dim=1)
                return self.head(o)
        return _Fusion(backbone, head).to(device), cnn_dim

    else:  # jpg
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3, True),
            nn.Linear(cnn_dim, 256), nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, nc),
        )
        return backbone.to(device), cnn_dim


def _build_opt(model, feat_mode: str, lr: float):
    if feat_mode == "jpg_json":
        return optim.AdamW([
            {"params": model.backbone.features.parameters(), "lr": lr * 0.05},
            {"params": model.head.parameters(),              "lr": lr},
        ], weight_decay=1e-4)
    else:
        return optim.AdamW([
            {"params": model.features.parameters(),   "lr": lr * 0.05},
            {"params": model.classifier.parameters(), "lr": lr},
        ], weight_decay=1e-4)


# ─────────────────────────────────────────────────────────────
#  Dataset — 계층별 필터링 지원
# ─────────────────────────────────────────────────────────────
class _HierDataset(Dataset):
    """
    samples: [(img_path, orig_label_str), ...]  ← 문자열 레이블
    label_map: {label_str: int}
    json_map: {img_path: json_dict}
    transform: torchvision transform
    augment: bool
    """
    def __init__(self, samples, label_map: dict, transform,
                 json_map: dict = None, augment: bool = False):
        self._s      = samples
        self._lm     = label_map
        self._tf     = transform
        self._jm     = json_map or {}
        self._aug    = augment

    def __len__(self): return len(self._s)

    def __getitem__(self, idx):
        path, cls_str = self._s[idx]
        label = self._lm[cls_str]

        try:
            img = PILImage.open(path).convert("RGB")
        except Exception:
            img = PILImage.new("RGB", (300, 300), (128, 128, 128))
        img = self._tf(img)

        if self._jm:
            jd   = self._jm.get(path, {})
            feat = _extract_features(jd)
            if self._aug:
                feat = (feat + torch.randn_like(feat) * 0.015).clamp(-1., 2.)
            return img, label, feat
        return img, label


# ─────────────────────────────────────────────────────────────
#  GPU util
# ─────────────────────────────────────────────────────────────
def _gpu_util():
    try:
        o = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=3, encoding="utf-8", errors="replace").strip().split(",")
        u, mu, mt, t = [x.strip() for x in o]
        return f"GPU {u}%  VRAM {mu}/{mt}MiB  {t}C"
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
#  단일 스테이지 학습 루프
# ─────────────────────────────────────────────────────────────
def _train_one_stage(
    model, opt, sch, scaler, crit,
    tr_dl, vl_dl,
    ep_n: int, feat_mode: str,
    device, use_amp: bool, use_cuda: bool,
    stage_name: str, save_path: str,
    resume_ckpt: str = "",
    target_acc: float = 0.0,
    target_acc_path: str = "",
    patience: int = 0,
    test_dl=None,
) -> tuple:
    """
    단일 스테이지 학습.
    Returns: (best_val_acc, model_state_dict)
    """
    best      = 0.0
    start_ep  = 1
    best_state = None
    target_acc_saved = False
    es_counter = 0
    es_best    = 0.0
    if target_acc > 0 and not target_acc_path:
        target_acc_path = str(
            Path(save_path).parent /
            f"{Path(save_path).stem}_{stage_name}_acc{int(target_acc)}.pt")

    # ── Resume ──────────────────────────────────────────────
    ckpt_path = Path(save_path).parent / f"{Path(save_path).stem}_{stage_name}_ckpt.pt"

    load_path = resume_ckpt or (str(ckpt_path) if ckpt_path.exists() else "")
    if load_path and Path(load_path).exists():
        try:
            ck = torch.load(load_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            if "optimizer_state_dict" in ck:
                opt.load_state_dict(ck["optimizer_state_dict"])
            if "scheduler_state_dict" in ck:
                sch.load_state_dict(ck["scheduler_state_dict"])
            if "scaler_state_dict" in ck and use_amp:
                scaler.load_state_dict(ck["scaler_state_dict"])
            start_ep   = ck.get("epoch", 0) + 1
            best       = ck.get("val_acc", 0.0)
            best_state = ck.get("best_state")
            log.info(f"  [{stage_name}] Resume ep{start_ep}  best={best:.2f}%")
        except Exception as e:
            log.warning(f"  [{stage_name}] Resume 실패: {e} → 처음부터")

    log.info(f"\n{'='*60}")
    log.info(f"  [{stage_name}] 학습 시작  ep:{start_ep}~{ep_n}")
    log.info(f"{'='*60}")

    t0 = time.time()
    for ep in range(start_ep, ep_n + 1):
        if _STOP: log.info("[STOP]"); break

        t_ep = time.time()

        # ── Train ────────────────────────────────────────────
        model.train()
        tl = tc = tt = 0
        emit_every = max(1, len(tr_dl) // 10)

        for bi, batch in enumerate(tr_dl):
            if _STOP: break
            if feat_mode == "jpg_json":
                x, y, feat = batch
                x = x.to(device, non_blocking=True)
                feat = feat.to(device, non_blocking=True)
            else:
                x, y = batch; feat = None
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x, feat) if feat_mode == "jpg_json" else model(x)
                lo  = crit(out, y)
            scaler.scale(lo).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            n   = x.size(0)
            tl += lo.item() * n
            tc += out.argmax(1).eq(y).sum().item()
            tt += n
            if (bi + 1) % emit_every == 0:
                print(f"\r  [{stage_name}] Train {(bi+1)/len(tr_dl)*100:5.1f}%"
                      f"  loss={tl/max(tt,1):.4f}", end="", flush=True)
        print()

        # ── Val ──────────────────────────────────────────────
        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for batch in vl_dl:
                if feat_mode == "jpg_json":
                    x, y, feat = batch
                    x = x.to(device, non_blocking=True)
                    feat = feat.to(device, non_blocking=True)
                else:
                    x, y = batch; feat = None
                    x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(x, feat) if feat_mode == "jpg_json" else model(x)
                    lo  = crit(out, y)
                n  = x.size(0)
                vl += lo.item() * n
                vc += out.argmax(1).eq(y).sum().item()
                vt += n

        sch.step()

        atl   = tl / max(tt, 1)
        avl   = vl / max(vt, 1)
        ta    = tc / max(tt, 1) * 100
        va    = vc / max(vt, 1) * 100
        cur_lr = opt.param_groups[-1]["lr"]
        ep_sec = time.time() - t_ep
        gpu    = _gpu_util() if use_cuda else ""

        log.info(f"  [{stage_name}][Ep {ep:03d}/{ep_n}] "
                 f"Train Loss={atl:.4f} Acc={ta:.2f}%  |  "
                 f"Val Loss={avl:.4f} Acc={va:.2f}%  |  "
                 f"LR={cur_lr:.2e}  {ep_sec:.0f}s")
        if gpu: log.info(f"    [{gpu}]")

        # Best 모델 갱신
        if va > best:
            best       = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info(f"  [{stage_name}] ★ Best val_acc={va:.2f}%")

        # ★ 목표 정확도 도달 시 별도 저장 (최초 1회)
        if target_acc > 0 and not target_acc_saved and va >= target_acc:
            torch.save({
                "epoch":            ep,
                "model_state_dict": {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()},
                "val_acc":          va,
                "target_acc":       target_acc,
                "stage_name":       stage_name,
            }, target_acc_path)
            target_acc_saved = True
            log.info(f"  [{stage_name}] ✅ 목표 {target_acc:.1f}% 도달 "
                     f"(val={va:.2f}%)  ->  {target_acc_path}")

        # 에포크별 체크포인트 (이어학습용)
        torch.save({
            "epoch":               ep,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sch.state_dict(),
            "scaler_state_dict":    scaler.state_dict() if use_amp else {},
            "val_acc":             va,
            "best_state":          best_state,
        }, ckpt_path)

        # ── Early Stopping ─────────────────────────────────────
        if patience > 0:
            if va > es_best:
                es_best    = va
                es_counter = 0
            else:
                es_counter += 1
                log.info(f"  [{stage_name}] [ES] 개선 없음 {es_counter}/{patience}")
                if es_counter >= patience:
                    log.info(f"  [{stage_name}] [ES] ★ Early Stopping 발동! 학습 중단")
                    break

    elapsed = (time.time() - t0) / 60
    log.info(f"  [{stage_name}] 완료  Best={best:.2f}%  소요:{elapsed:.1f}분")

    # ── Test 셋 평가 ───────────────────────────────────────────
    if test_dl is not None and best_state is not None:
        log.info(f"  [{stage_name}] Test 셋 평가 중...")
        model.load_state_dict(best_state)
        model.eval()
        tc2 = tv2 = 0
        with torch.no_grad():
            for batch in test_dl:
                if feat_mode == "jpg_json":
                    x, y, feat = batch
                    x = x.to(device, non_blocking=True)
                    feat = feat.to(device, non_blocking=True)
                else:
                    x, y = batch; feat = None
                    x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(x, feat) if feat_mode == "jpg_json" else model(x)
                tc2 += out.argmax(1).eq(y).sum().item()
                tv2 += y.size(0)
        test_acc = tc2 / max(tv2, 1) * 100
        log.info(f"  [{stage_name}] ★ Test Acc: {test_acc:.2f}%  "
                 f"(Val: {best:.2f}%  Gap: {best - test_acc:+.2f}%)")
        if best - test_acc > 5.0:
            log.info(f"  [{stage_name}] ⚠️  Gap {best-test_acc:.1f}% 큼 → 과적합 가능성")

    return best, best_state


# ─────────────────────────────────────────────────────────────
#  DataLoader 생성 헬퍼
# ─────────────────────────────────────────────────────────────
def _make_dl(samples_with_str, label_map, trn_tf, val_tf,
             json_map, feat_mode, bs, nw, use_cuda,
             test_split=0.1):
    """(path, class_str) 샘플 → Train/Val/Test DataLoader 반환
    반환: tr_dl, vl_dl, te_dl(없으면 None), len(tr), len(vl), len(te)
    """
    import random as _rng
    _rng.seed(42)

    from collections import defaultdict
    by_cls = defaultdict(list)
    for item in samples_with_str:
        by_cls[item[1]].append(item)

    tr_s, vl_s, te_s = [], [], []
    for cls, items in by_cls.items():
        _rng.shuffle(items)
        n_te  = max(1, int(len(items) * test_split)) if test_split > 0 else 0
        n_val = max(1, int(len(items) * 0.1))   # Val: 10%
        te_s.extend(items[:n_te])
        vl_s.extend(items[n_te:n_te + n_val])
        tr_s.extend(items[n_te + n_val:])

    jm = json_map if feat_mode == "jpg_json" else {}

    tr_ds = _HierDataset(tr_s, label_map, trn_tf, jm, augment=True)
    vl_ds = _HierDataset(vl_s, label_map, val_tf, jm, augment=False)
    te_ds = _HierDataset(te_s, label_map, val_tf, jm, augment=False) if te_s else None

    pf = 4 if nw > 0 else None
    kw = dict(num_workers=nw, pin_memory=use_cuda,
              persistent_workers=(nw > 0), prefetch_factor=pf)
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                       drop_last=(len(tr_ds) > bs), **kw)
    vl_dl = DataLoader(vl_ds, batch_size=bs * 2, **kw)
    te_dl = DataLoader(te_ds, batch_size=bs * 2, **kw) if te_ds else None

    return tr_dl, vl_dl, te_dl, len(tr_s), len(vl_s), len(te_s)


# ─────────────────────────────────────────────────────────────
#  LR 스케줄러 lambda
# ─────────────────────────────────────────────────────────────
def _lr_lambda_factory(ep_n: int):
    warmup = min(5, ep_n // 4)
    def _fn(ep):
        if ep < warmup: return (ep + 1) / warmup
        t = ep - warmup; T = max(ep_n - warmup, 1)
        return 0.05 + 0.95 * (1 + math.cos(math.pi * t / T)) / 2
    return _fn


# ─────────────────────────────────────────────────────────────
#  메인 학습 함수
# ─────────────────────────────────────────────────────────────
def train_stage1(cfg: dict):
    """Stage1만 독립 학습 — 4계절 분류 모델 저장"""
    req = cfg["device"]
    if req == "cuda" and not torch.cuda.is_available():
        req = "cpu"
    device   = torch.device(req)
    use_cuda = device.type == "cuda"
    use_amp  = use_cuda
    if use_cuda:
        log.info(f"GPU: {torch.cuda.get_device_name(0)}"
                 f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    feat_mode   = cfg["feat_mode"]
    img_mode    = cfg["img_mode"]
    s1_jpg_dir  = cfg.get("s1_jpg_dir","") or cfg.get("jpg_dir","") or cfg["data_dir"]
    s1_json_dir = cfg.get("s1_json_dir","") or cfg.get("json_dir","") or s1_jpg_dir
    feat_dim    = JSON_FEATURE_DIM if feat_mode == "jpg_json" else 0
    bs          = cfg["batch_size"] or (128 if use_cuda else 32)
    nw          = cfg["workers"]
    lr          = cfg["lr"]
    ep_s1       = cfg["epochs_s1"]
    save_path   = cfg["save_path"]  # 예: stage1.pt

    log.info("=" * 65)
    log.info("STAGE 1 독립 학습 — 4계절 분류")
    log.info(f"  S1 JPG : {s1_jpg_dir}")
    log.info(f"  S1 JSON: {s1_json_dir}")
    log.info(f"  저장   : {save_path}")
    log.info("=" * 65)

    _M = [0.485,0.456,0.406]; _S = [0.229,0.224,0.225]
    trn_tf = transforms.Compose([
        transforms.Resize((300,300)), transforms.RandomCrop(260),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(12),
        transforms.ColorJitter(0.25,0.25,0.25,0.06),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(), transforms.Normalize(_M,_S)])
    val_tf = transforms.Compose([
        transforms.Resize((300,300)), transforms.CenterCrop(260),
        transforms.ToTensor(), transforms.Normalize(_M,_S)])

    ds_s1 = PersonalColorDataset(s1_jpg_dir, transform=None)
    if len(ds_s1) == 0:
        log.error("S1 데이터셋 비어있음"); return

    idx_to_cls_s1  = {i: c for c, i in ds_s1.class_to_idx.items()}
    s1_raw_samples = [(path, idx_to_cls_s1[label]) for path, label in ds_s1.samples]

    from collections import Counter
    dist = Counter(c for _, c in s1_raw_samples)
    log.info(f"S1 전체 샘플: {len(s1_raw_samples)}장  클래스: {len(dist)}개")
    for cls in sorted(dist): log.info(f"  {cls}: {dist[cls]}장")

    json_map_s1 = {}
    if feat_mode == "jpg_json":
        json_map_s1 = _load_json_for(s1_json_dir, s1_raw_samples, "S1")
        if not json_map_s1:
            log.warning("S1 JSON 없음 → jpg 모드"); feat_mode = "jpg"; feat_dim = 0

    # 계절 레이블 변환
    season_label_map = {s: i for i, s in enumerate(SEASON_CLASSES)}
    s1_samples = []
    for path, cls_str in s1_raw_samples:
        season = cls_str.split("_")[0] if "_" in cls_str else cls_str
        if season in season_label_map:
            s1_samples.append((path, season))
    log.info(f"Stage1 샘플: {len(s1_samples)}장")
    test_split = cfg.get("test_split", 0.1)
    patience   = cfg.get("patience",   0)

    tr_dl1, vl_dl1, te_dl1, ts1, vs1, tes1 = _make_dl(
        s1_samples, season_label_map, trn_tf, val_tf,
        json_map_s1, feat_mode, bs, nw, use_cuda, test_split)
    log.info(f"  Train: {ts1}  Val: {vs1}  Test: {tes1}")

    model_s1, _ = _build_model(4, feat_mode, feat_dim, device)
    opt_s1      = _build_opt(model_s1, feat_mode, lr)
    sch_s1      = optim.lr_scheduler.LambdaLR(opt_s1, _lr_lambda_factory(ep_s1))
    scaler_s1   = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit_s1     = nn.CrossEntropyLoss(label_smoothing=0.1)

    target_acc      = cfg.get("target_acc",      0.0)
    target_acc_path = cfg.get("target_acc_path", "")
    if target_acc > 0 and not target_acc_path:
        target_acc_path = str(
            Path(save_path).parent /
            f"{Path(save_path).stem}_acc{int(target_acc)}.pt")

    resume = cfg.get("resume","")
    best_s1, state_s1 = _train_one_stage(
        model_s1, opt_s1, sch_s1, scaler_s1, crit_s1,
        tr_dl1, vl_dl1, ep_s1, feat_mode, device, use_amp, use_cuda,
        "Stage1_Season", save_path, resume_ckpt=resume,
        target_acc=target_acc, target_acc_path=target_acc_path)

    # Stage1 단독 저장
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type":      "hierarchical_stage1",
        "model_arch":      "efficientnet_b2",
        "feat_mode":       feat_mode,
        "feat_dim":        feat_dim,
        "img_mode":        img_mode,
        "preproc_mode":    cfg.get("preproc_mode","standard"),
        "stage1_classes":  SEASON_CLASSES,
        "stage1_state":    state_s1,
        "val_acc_stage1":  best_s1,
    }, save_path)
    log.info(f"\n✅ Stage1 저장 완료: {save_path}  val_acc={best_s1:.2f}%")


def train_stage2(cfg: dict):
    """Stage2만 독립 학습 — 계절별 톤 분류 모델 저장"""
    req = cfg["device"]
    if req == "cuda" and not torch.cuda.is_available():
        req = "cpu"
    device   = torch.device(req)
    use_cuda = device.type == "cuda"
    use_amp  = use_cuda
    if use_cuda:
        log.info(f"GPU: {torch.cuda.get_device_name(0)}"
                 f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    feat_mode   = cfg["feat_mode"]
    img_mode    = cfg["img_mode"]
    s2_jpg_dir  = cfg.get("s2_jpg_dir","") or cfg.get("jpg_dir","") or cfg["data_dir"]
    s2_json_dir = cfg.get("s2_json_dir","") or cfg.get("json_dir","") or s2_jpg_dir
    feat_dim    = JSON_FEATURE_DIM if feat_mode == "jpg_json" else 0
    bs          = cfg["batch_size"] or (128 if use_cuda else 32)
    nw          = cfg["workers"]
    lr          = cfg["lr"]
    ep_s2       = cfg["epochs_s2"]
    save_path   = cfg["save_path"]  # 예: stage2.pt
    # 특정 계절만 학습 (비어있으면 전체)
    target_seasons = [s.strip() for s in cfg.get("seasons","").split(",") if s.strip()] \
                     or SEASON_CLASSES

    log.info("=" * 65)
    log.info("STAGE 2 독립 학습 — 톤 분류")
    log.info(f"  S2 JPG : {s2_jpg_dir}")
    log.info(f"  S2 JSON: {s2_json_dir}")
    log.info(f"  대상 계절: {target_seasons}")
    log.info(f"  저장   : {save_path}")
    log.info("=" * 65)

    _M = [0.485,0.456,0.406]; _S = [0.229,0.224,0.225]
    trn_tf = transforms.Compose([
        transforms.Resize((300,300)), transforms.RandomCrop(260),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(12),
        transforms.ColorJitter(0.25,0.25,0.25,0.06),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(), transforms.Normalize(_M,_S)])
    val_tf = transforms.Compose([
        transforms.Resize((300,300)), transforms.CenterCrop(260),
        transforms.ToTensor(), transforms.Normalize(_M,_S)])

    ds_s2 = PersonalColorDataset(s2_jpg_dir, transform=None)
    if len(ds_s2) == 0:
        log.error("S2 데이터셋 비어있음"); return

    idx_to_cls_s2 = {i: c for c, i in ds_s2.class_to_idx.items()}
    all_samples   = [(path, idx_to_cls_s2[label]) for path, label in ds_s2.samples]

    from collections import Counter
    dist = Counter(c for _, c in all_samples)
    log.info(f"S2 전체 샘플: {len(all_samples)}장  클래스: {len(dist)}개")

    json_map_s2 = {}
    if feat_mode == "jpg_json":
        json_map_s2 = _load_json_for(s2_json_dir, all_samples, "S2")
        if not json_map_s2:
            log.warning("S2 JSON 없음 → jpg 모드"); feat_mode = "jpg"; feat_dim = 0

    # 기존 Stage2 모델 로드 (이어학습)
    existing_state2 = {}
    existing_best2  = {}
    resume = cfg.get("resume","")
    if resume and Path(resume).exists():
        try:
            ck = torch.load(resume, map_location=device, weights_only=False)
            if ck.get("model_type") in ("hierarchical_stage2","hierarchical"):
                for s in SEASON_CLASSES:
                    k = f"stage2_{s}_state"
                    if k in ck: existing_state2[s] = ck[k]
                existing_best2 = ck.get("val_acc_stage2", {})
                log.info(f"[RESUME] 기존 Stage2 모델 로드: {resume}")
        except Exception as e:
            log.warning(f"[RESUME] 로드 실패: {e}")

    tone_label_map = {t: i for i, t in enumerate(TONE_CLASSES)}
    state_s2 = dict(existing_state2)
    best_s2  = dict(existing_best2)

    for season in target_seasons:
        if _STOP: break

        log.info(f"\n  ── {season} 전용 톤 모델 ──")
        s2_samples = []
        for path, cls_str in all_samples:
            if "_" not in cls_str: continue
            s, t = cls_str.split("_", 1)
            if s == season and t in tone_label_map:
                s2_samples.append((path, t))

        if not s2_samples:
            log.warning(f"  {season} 샘플 없음 — 스킵"); continue

        log.info(f"  {season} 샘플: {len(s2_samples)}장")
        test_split = cfg.get("test_split", 0.1)
        patience   = cfg.get("patience",   0)
        tr_dl2, vl_dl2, te_dl2, ts2, vs2, tes2 = _make_dl(
            s2_samples, tone_label_map, trn_tf, val_tf,
            json_map_s2, feat_mode, bs, nw, use_cuda, test_split)
        log.info(f"  Train: {ts2}  Val: {vs2}  Test: {tes2}")

        model_s2  = _build_model(3, feat_mode, feat_dim, device)[0]
        opt_s2    = _build_opt(model_s2, feat_mode, lr)
        sch_s2    = optim.lr_scheduler.LambdaLR(opt_s2, _lr_lambda_factory(ep_s2))
        scaler_s2 = torch.amp.GradScaler("cuda", enabled=use_amp)
        crit_s2   = nn.CrossEntropyLoss(label_smoothing=0.05)

        # 해당 계절의 기존 ckpt 경로
        season_ckpt = Path(save_path).parent / f"{Path(save_path).stem}_{season}_ckpt.pt"
        target_acc      = cfg.get("target_acc",      0.0)
        target_acc_path_s = cfg.get("target_acc_path", "")
        if target_acc > 0 and not target_acc_path_s:
            target_acc_path_s = str(
                Path(save_path).parent /
                f"{Path(save_path).stem}_{season}_acc{int(target_acc)}.pt")

        best_acc, state = _train_one_stage(
            model_s2, opt_s2, sch_s2, scaler_s2, crit_s2,
            tr_dl2, vl_dl2, ep_s2, feat_mode, device, use_amp, use_cuda,
            f"Stage2_{season}", save_path,
            resume_ckpt=str(season_ckpt) if season_ckpt.exists() else "",
            target_acc=target_acc, target_acc_path=target_acc_path_s)

        best_s2[season]  = best_acc
        state_s2[season] = state
        del model_s2, opt_s2, sch_s2, scaler_s2
        if use_cuda: torch.cuda.empty_cache()

    # Stage2 저장
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "model_type":          "hierarchical_stage2",
        "model_arch":          "efficientnet_b2",
        "feat_mode":           feat_mode,
        "feat_dim":            feat_dim,
        "img_mode":            img_mode,
        "preproc_mode":        cfg.get("preproc_mode","standard"),
        "stage2_tone_classes": TONE_CLASSES,
        "val_acc_stage2":      best_s2,
    }
    for season, state in state_s2.items():
        save_dict[f"stage2_{season}_state"] = state
    torch.save(save_dict, save_path)

    log.info("\n" + "="*65)
    log.info(f"✅ Stage2 저장 완료: {save_path}")
    for s, acc in best_s2.items():
        log.info(f"  {s:6s} 톤  val_acc={acc:.2f}%")
    log.info("="*65)


def combine_models(cfg: dict):
    """Stage1.pt + Stage2.pt → final hierarchical.pt 조합"""
    s1_path   = cfg["stage1_path"]
    s2_path   = cfg["stage2_path"]
    out_path  = cfg["save_path"]

    if not Path(s1_path).exists():
        log.error(f"Stage1 모델 없음: {s1_path}"); return
    if not Path(s2_path).exists():
        log.error(f"Stage2 모델 없음: {s2_path}"); return

    log.info("=" * 65)
    log.info("모델 조합 (Stage1 + Stage2 → Final)")
    log.info(f"  Stage1: {s1_path}")
    log.info(f"  Stage2: {s2_path}")
    log.info(f"  출력  : {out_path}")
    log.info("=" * 65)

    ck1 = torch.load(s1_path, map_location="cpu", weights_only=False)
    ck2 = torch.load(s2_path, map_location="cpu", weights_only=False)

    combined = {
        "model_type":          "hierarchical",
        "model_arch":          ck1.get("model_arch", "efficientnet_b2"),
        "feat_mode":           ck1.get("feat_mode",  "jpg"),
        "feat_dim":            ck1.get("feat_dim",   0),
        "img_mode":            ck1.get("img_mode",   "crop"),
        "preproc_mode":        ck1.get("preproc_mode","standard"),
        # Stage1
        "stage1_classes":      ck1.get("stage1_classes", SEASON_CLASSES),
        "stage1_state":        ck1.get("stage1_state"),
        "val_acc_stage1":      ck1.get("val_acc_stage1", 0.0),
        # Stage2
        "stage2_tone_classes": ck2.get("stage2_tone_classes", TONE_CLASSES),
        "val_acc_stage2":      ck2.get("val_acc_stage2", {}),
    }
    for season in SEASON_CLASSES:
        k = f"stage2_{season}_state"
        if k in ck2:
            combined[k] = ck2[k]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, out_path)

    log.info(f"✅ 조합 완료: {out_path}")
    log.info(f"  Stage1 val_acc: {combined['val_acc_stage1']:.2f}%")
    for s, acc in combined["val_acc_stage2"].items():
        log.info(f"  Stage2 {s:6s}: {acc:.2f}%")


def _load_json_for(json_src: str, samples, label: str) -> dict:
    """json_src 폴더에서 samples 경로 기준 JSON 매핑 반환"""
    if not json_src: return {}
    log.info(f"  {label} JSON 로드: {json_src}")
    meta_path = Path(json_src) / "preprocess_meta.json"
    jmap = {}
    if meta_path.exists():
        from datasets import PersonalColorDataset as _PCD
        jds = _PCD(json_src, transform=None)
        raw = jds.json_map
        stem_map = {Path(p).stem: d for p, d in raw.items()}
        for path, _ in samples:
            s = Path(path).stem
            if s in stem_map: jmap[path] = stem_map[s]
    else:
        stem_map = {}
        for jf in Path(json_src).rglob("*.json"):
            if jf.name == "preprocess_meta.json": continue
            try: stem_map[jf.stem] = json.loads(jf.read_text(encoding="utf-8"))
            except Exception: pass
        for path, _ in samples:
            s = Path(path).stem
            if s in stem_map: jmap[path] = stem_map[s]
    log.info(f"  {label} JSON 매칭: {len(jmap)}/{len(samples)}장")
    return jmap
    # ── 디바이스 ─────────────────────────────────────────────
    req = cfg["device"]
    if req == "cuda" and not torch.cuda.is_available():
        req = "cpu"; log.warning("CUDA 없음 → CPU")
    device   = torch.device(req)
    use_cuda = device.type == "cuda"
    use_amp  = use_cuda

    if use_cuda:
        log.info(f"GPU: {torch.cuda.get_device_name(0)}"
                 f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    feat_mode    = cfg["feat_mode"]
    img_mode     = cfg["img_mode"]
    # S1/S2 분리 폴더 지원
    s1_jpg_dir   = cfg.get("s1_jpg_dir", "") or cfg.get("jpg_dir", "")
    s1_json_dir  = cfg.get("s1_json_dir", "") or cfg.get("json_dir", "")
    s2_jpg_dir   = cfg.get("s2_jpg_dir", "") or cfg.get("jpg_dir", "")
    s2_json_dir  = cfg.get("s2_json_dir", "") or cfg.get("json_dir", "")
    # 폴백: 지정 없으면 data_dir 사용
    s1_jpg_dir   = s1_jpg_dir  or cfg["data_dir"]
    s2_jpg_dir   = s2_jpg_dir  or cfg["data_dir"]
    jpg_dir      = s2_jpg_dir  # 하위호환
    json_dir     = s2_json_dir
    bs           = cfg["batch_size"] or (128 if use_cuda else 32)
    nw           = cfg["workers"]
    lr           = cfg["lr"]
    ep_s1        = cfg["epochs_s1"]
    ep_s2        = cfg["epochs_s2"]
    save_path    = cfg["save_path"]
    feat_dim     = JSON_FEATURE_DIM if feat_mode == "jpg_json" else 0

    log.info("=" * 65)
    log.info("2단계 계층 분류 학습")
    log.info(f"  Stage1 에포크: {ep_s1}  Stage2 에포크: {ep_s2}")
    log.info(f"  feat_mode: {feat_mode}  img_mode: {img_mode}")
    log.info(f"  배치: {bs}  LR: {lr}  workers: {nw}")
    log.info("=" * 65)

    # ── Transform ─────────────────────────────────────────────
    _M = [0.485, 0.456, 0.406]; _S = [0.229, 0.224, 0.225]
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

    log.info("=" * 65)
    log.info(f"  Stage1 JPG  : {s1_jpg_dir}")
    log.info(f"  Stage1 JSON : {s1_json_dir or '(S1 JPG 사용)'}")
    log.info(f"  Stage2 JPG  : {s2_jpg_dir}")
    log.info(f"  Stage2 JSON : {s2_json_dir or '(S2 JPG 사용)'}")
    log.info("=" * 65)

    # ── S2 데이터셋 스캔 (12클래스 기준) ─────────────────────
    img_data_dir = s2_jpg_dir
    log.info(f"S2 데이터셋 로드: {img_data_dir}")
    ds = PersonalColorDataset(img_data_dir, transform=None)
    if len(ds) == 0:
        log.error("빈 데이터셋"); return

    idx_to_cls  = {i: c for c, i in ds.class_to_idx.items()}
    all_samples = [(path, idx_to_cls[label]) for path, label in ds.samples]

    from collections import Counter
    dist = Counter(c for _, c in all_samples)
    log.info(f"S2 전체 샘플: {len(all_samples)}장  클래스: {len(dist)}개")
    for cls in sorted(dist):
        log.info(f"  {cls}: {dist[cls]}장")

    # ── S1 데이터셋 스캔 (4계절 기준) ────────────────────────
    log.info(f"\nS1 데이터셋 로드: {s1_jpg_dir}")
    if s1_jpg_dir != s2_jpg_dir:
        ds_s1 = PersonalColorDataset(s1_jpg_dir, transform=None)
        if len(ds_s1) == 0:
            log.warning("S1 데이터셋 비어있음 → S2 데이터로 대체")
            ds_s1 = ds
        idx_to_cls_s1  = {i: c for c, i in ds_s1.class_to_idx.items()}
        s1_raw_samples = [(path, idx_to_cls_s1[label]) for path, label in ds_s1.samples]
    else:
        s1_raw_samples = all_samples  # 같은 폴더면 S2 그대로 사용
        ds_s1 = ds
    log.info(f"S1 샘플: {len(s1_raw_samples)}장")

    # ── JSON 피처 로드 (S1/S2 각각) ──────────────────────────
    json_map_s1: dict = {}
    json_map_s2: dict = {}

    def _load_json_map(json_src: str, img_samples, label: str) -> dict:
        """json_src 폴더에서 img_samples 경로 기준 JSON 매핑 반환"""
        if not json_src or feat_mode != "jpg_json":
            return {}
        log.info(f"  {label} JSON 로드: {json_src}")
        meta_path = Path(json_src) / "preprocess_meta.json"
        jmap = {}
        if meta_path.exists():
            jds    = PersonalColorDataset(json_src, transform=None)
            raw_jm = jds.json_map
            if Path(json_src).resolve() == Path(
                    img_samples[0][0] if img_samples else json_src).parent.parent.resolve():
                jmap = raw_jm
            else:
                stem_map = {Path(p).stem: d for p, d in raw_jm.items()}
                for path, _ in img_samples:
                    s = Path(path).stem
                    if s in stem_map: jmap[path] = stem_map[s]
        else:
            stem_map = {}
            for jf in Path(json_src).rglob("*.json"):
                if jf.name == "preprocess_meta.json": continue
                try:
                    stem_map[jf.stem] = json.loads(jf.read_text(encoding="utf-8"))
                except Exception: pass
            for path, _ in img_samples:
                s = Path(path).stem
                if s in stem_map: jmap[path] = stem_map[s]
        log.info(f"  {label} JSON 매칭: {len(jmap)}/{len(img_samples)}장")
        return jmap

    if feat_mode == "jpg_json":
        json_map_s1 = _load_json_map(s1_json_dir or s1_jpg_dir, s1_raw_samples, "S1")
        json_map_s2 = _load_json_map(s2_json_dir or s2_jpg_dir, all_samples,    "S2")
        if not json_map_s1 and not json_map_s2:
            log.warning("JSON 피처 없음 → jpg 모드로 전환")
            feat_mode = "jpg"

    # ─────────────────────────────────────────────────────────
    #  STAGE 1: 4계절 분류
    # ─────────────────────────────────────────────────────────
    log.info("\n" + "="*65)
    log.info("  STAGE 1: 4계절 분류 (Spring / Summer / Autumn / Winter)")
    log.info("="*65)

    season_label_map = {s: i for i, s in enumerate(SEASON_CLASSES)}
    s1_samples = []
    for path, cls_str in s1_raw_samples:
        season = cls_str.split("_")[0] if "_" in cls_str else cls_str
        if season in season_label_map:
            s1_samples.append((path, season))

    log.info(f"Stage1 샘플: {len(s1_samples)}장")
    test_split = cfg.get("test_split", 0.1)
    patience   = cfg.get("patience",   0)

    tr_dl1, vl_dl1, te_dl1, ts1, vs1, tes1 = _make_dl(
        s1_samples, season_label_map, trn_tf, val_tf,
        json_map_s1, feat_mode, bs, nw, use_cuda, test_split)
    log.info(f"  Train: {ts1}  Val: {vs1}  Test: {tes1}")

    model_s1, _ = _build_model(4, feat_mode, feat_dim, device)
    opt_s1      = _build_opt(model_s1, feat_mode, lr)
    sch_s1      = optim.lr_scheduler.LambdaLR(opt_s1, _lr_lambda_factory(ep_s1))
    scaler_s1   = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit_s1     = nn.CrossEntropyLoss(label_smoothing=0.1)

    target_acc      = cfg.get("target_acc",      0.0)
    target_acc_path = cfg.get("target_acc_path", "")

    best_s1, state_s1 = _train_one_stage(
        model_s1, opt_s1, sch_s1, scaler_s1, crit_s1,
        tr_dl1, vl_dl1, ep_s1, feat_mode, device, use_amp, use_cuda,
        "Stage1_Season", save_path,
        target_acc=target_acc,
        target_acc_path=str(Path(save_path).parent /
            f"{Path(save_path).stem}_Stage1_acc{int(target_acc)}.pt")
            if target_acc > 0 else "",
        patience=patience, test_dl=te_dl1)

    # ─────────────────────────────────────────────────────────
    #  STAGE 2: 계절별 톤 분류 (Warm / Bright / Light)
    # ─────────────────────────────────────────────────────────
    log.info("\n" + "="*65)
    log.info("  STAGE 2: 계절별 톤 분류 (Warm / Bright / Light)")
    log.info("="*65)

    tone_label_map = {t: i for i, t in enumerate(TONE_CLASSES)}
    state_s2   = {}
    best_s2    = {}

    for season in SEASON_CLASSES:
        if _STOP: break

        log.info(f"\n  ── {season} 전용 톤 모델 ──")

        # 해당 계절 샘플만 필터링 (path, tone_str)
        s2_samples = []
        for path, cls_str in all_samples:   # S2 데이터(12클래스)에서 필터링
            if "_" not in cls_str: continue
            s, t = cls_str.split("_", 1)
            if s == season and t in tone_label_map:
                s2_samples.append((path, t))

        if not s2_samples:
            log.warning(f"  {season} 샘플 없음 — 스킵")
            continue

        log.info(f"  {season} 샘플: {len(s2_samples)}장")

        tr_dl2, vl_dl2, te_dl2, ts2, vs2, tes2 = _make_dl(
            s2_samples, tone_label_map, trn_tf, val_tf,
            json_map_s2, feat_mode, bs, nw, use_cuda, test_split)
        log.info(f"  Train: {ts2}  Val: {vs2}  Test: {tes2}")

        model_s2, _ = _build_model(3, feat_mode, feat_dim, device)
        opt_s2      = _build_opt(model_s2, feat_mode, lr)
        sch_s2      = optim.lr_scheduler.LambdaLR(opt_s2, _lr_lambda_factory(ep_s2))
        scaler_s2   = torch.amp.GradScaler("cuda", enabled=use_amp)
        crit_s2     = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_acc, state = _train_one_stage(
            model_s2, opt_s2, sch_s2, scaler_s2, crit_s2,
            tr_dl2, vl_dl2, ep_s2, feat_mode, device, use_amp, use_cuda,
            f"Stage2_{season}", save_path,
            target_acc=target_acc,
            target_acc_path=str(Path(save_path).parent /
                f"{Path(save_path).stem}_{season}_acc{int(target_acc)}.pt")
                if target_acc > 0 else "",
            patience=patience, test_dl=te_dl2)

        best_s2[season]  = best_acc
        state_s2[season] = state

        # 메모리 해제
        del model_s2, opt_s2, sch_s2, scaler_s2
        if use_cuda: torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────
    #  최종 저장
    # ─────────────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_type":     "hierarchical",
        "model_arch":     "efficientnet_b2",
        "feat_mode":      feat_mode,
        "feat_dim":       feat_dim,
        "img_mode":       img_mode,
        "preproc_mode":   cfg.get("preproc_mode", "standard"),
        # Stage1
        "stage1_classes":     SEASON_CLASSES,
        "stage1_state":       state_s1,
        "val_acc_stage1":     best_s1,
        # Stage2
        "stage2_tone_classes": TONE_CLASSES,
        "val_acc_stage2":      best_s2,
    }
    for season, state in state_s2.items():
        save_dict[f"stage2_{season}_state"] = state

    torch.save(save_dict, save_path)

    log.info("\n" + "="*65)
    log.info("계층 학습 완료!")
    log.info(f"  Stage1 (계절) Best Val Acc: {best_s1:.2f}%")
    for s, acc in best_s2.items():
        log.info(f"  Stage2 ({s:6s} 톤) Best Val Acc: {acc:.2f}%")
    log.info(f"  저장: {save_path}")
    log.info("="*65)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="2단계 계층 분류 학습")
    p.add_argument("--mode",        default="combined",
                   choices=["combined","stage1_only","stage2_only","combine_models"],
                   help="combined=한번에 / stage1_only=Stage1만 / stage2_only=Stage2만 / combine_models=조합만")
    p.add_argument("--data_dir",    default="",  help="데이터셋 루트 (combined/stage2_only용)")
    p.add_argument("--save_path",   default="hierarchical.pt")
    # 분리 폴더
    p.add_argument("--jpg_dir",     default="",  help="JPG 폴더 (하위호환)")
    p.add_argument("--json_dir",    default="",  help="JSON 폴더 (하위호환)")
    p.add_argument("--s1_jpg_dir",  default="",  help="Stage1 JPG 폴더 (4계절)")
    p.add_argument("--s1_json_dir", default="",  help="Stage1 JSON 폴더 (4계절)")
    p.add_argument("--s2_jpg_dir",  default="",  help="Stage2 JPG 폴더 (12클래스)")
    p.add_argument("--s2_json_dir", default="",  help="Stage2 JSON 폴더 (12클래스)")
    # combine_models 전용
    p.add_argument("--stage1_path", default="",  help="조합할 Stage1 .pt 경로")
    p.add_argument("--stage2_path", default="",  help="조합할 Stage2 .pt 경로")
    # 학습 파라미터
    p.add_argument("--epochs_s1",   type=int, default=40)
    p.add_argument("--epochs_s2",   type=int, default=40)
    p.add_argument("--batch_size",  type=int, default=0, help="0=자동")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--workers",     type=int, default=4)
    p.add_argument("--device",      default="cuda", choices=["cuda","cpu"])
    p.add_argument("--feat_mode",   default="jpg", choices=["jpg","jpg_json"])
    p.add_argument("--img_mode",    default="crop", choices=["crop","original"])
    p.add_argument("--preproc_mode",default="standard", choices=["standard","advanced"])
    p.add_argument("--resume",      default="",  help="이어학습 체크포인트 경로")
    p.add_argument("--seasons",     default="",  help="stage2_only: 특정 계절만 (예: Spring,Summer)")
    p.add_argument("--target_acc",      type=float, default=0.0,
                   help="목표 Val Acc. 이 값 이상 최초 도달 시 별도 저장")
    p.add_argument("--target_acc_path", default="",
                   help="목표 정확도 도달 모델 저장 경로")
    p.add_argument("--patience",   type=int,   default=0,
                   help="Early Stopping patience (0=비활성, 예:10)")
    p.add_argument("--test_split", type=float, default=0.1,
                   help="Test 셋 비율 (기본 0.1=10%%, 0=Test 없음)")
    return vars(p.parse_args())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    cfg = parse_args()

    log.info("=" * 65)
    log.info(f"계층 분류 학습  모드: {cfg['mode']}")
    for k, v in cfg.items():
        if v: log.info(f"  {k:15s}: {v}")
    log.info("=" * 65)

    mode = cfg["mode"]
    if mode == "combined":
        # data_dir 필수 확인
        if not cfg["data_dir"] and not cfg.get("s2_jpg_dir"):
            log.error("--data_dir 또는 --s2_jpg_dir 필요"); sys.exit(1)
        train_hierarchical(cfg)
    elif mode == "stage1_only":
        if not cfg["data_dir"] and not cfg.get("s1_jpg_dir"):
            log.error("--data_dir 또는 --s1_jpg_dir 필요"); sys.exit(1)
        Path(cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)
        train_stage1(cfg)
    elif mode == "stage2_only":
        if not cfg["data_dir"] and not cfg.get("s2_jpg_dir"):
            log.error("--data_dir 또는 --s2_jpg_dir 필요"); sys.exit(1)
        Path(cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)
        train_stage2(cfg)
    elif mode == "combine_models":
        if not cfg["stage1_path"] or not cfg["stage2_path"]:
            log.error("--stage1_path, --stage2_path 필요"); sys.exit(1)
        combine_models(cfg)
    else:
        log.error(f"알 수 없는 모드: {mode}")
