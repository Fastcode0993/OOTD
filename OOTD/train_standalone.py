"""
train_standalone.py  (GPU 완전 활용 버전 - 디스크 I/O 병목 해결)
=================================================================

핵심 문제 진단:
  에포크당 20분 = GPU 34% = 디스크 I/O 병목
  원인: 7만장 원본 이미지를 에포크마다 디스크에서 전부 읽음
        PILImage.open() x 57,658회 x 30에포크 = 173만 번 디스크 접근

해결책:
  1. [권장] --cache_mode resize  (기본값)
     첫 실행 시 300x300 JPEG로 .cache_260/ 폴더에 저장
     2번째 에포크부터 작은 파일 읽기 -> I/O 5~10배 감소
     GPU 80%+ 달성 가능

  2. --cache_mode ram
     첫 에포크에 모든 이미지를 RAM에 올림 (72000장 ~15~20GB RAM 필요)
     이후 에포크 디스크 접근 0 -> GPU 90%+ 달성

  3. --cache_mode none
     기존 방식 (변경 없음)

사용법
------
# 권장 (디스크 캐시 - 처음 한 번만 느림)
python train_standalone.py --data_dir D:/dataset --save_path best.pt

# RAM 캐시 (RAM 32GB+ 권장)
python train_standalone.py --data_dir D:/dataset --save_path best.pt --cache_mode ram
"""

import argparse
import json
import logging
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import io
from pathlib import Path

# Windows 한글 깨짐 방지
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as PILImage, ImageFile
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

sys.path.insert(0, str(Path(__file__).parent))
from datasets import PersonalColorDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_STOP = False
def _sig_handler(sig, frame):
    global _STOP
    log.info("\n[STOP] Ctrl+C 감지 - 현재 에포크 완료 후 종료...")
    _STOP = True
signal.signal(signal.SIGINT, _sig_handler)


def _gpu_util() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=3, encoding="utf-8", errors="replace"
        ).strip().split(",")
        util, mu, mt, temp = [x.strip() for x in out]
        return f"GPU {util}%  VRAM {mu}/{mt}MiB  {temp}C"
    except Exception:
        return "nvidia-smi 조회 실패"


# ─────────────────────────────────────────────────────────────
#  FusionModel: EfficientNet-B2 CNN + JSON 피처 벡터 결합
#  CNN(1408) + 피처(10) -> Linear(256) -> nc 클래스
# ─────────────────────────────────────────────────────────────
class _FusionModel(nn.Module):
    def __init__(self, backbone, cnn_dim: int, feat_dim: int, nc: int):
        super().__init__()
        self.backbone = backbone          # EfficientNet (classifier=Identity)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(cnn_dim + feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, nc),
        )

    def forward(self, x, feat=None):
        cnn_out = self.backbone(x)        # (B, cnn_dim)
        if feat is not None:
            cnn_out = torch.cat([cnn_out, feat], dim=1)
        return self.head(cnn_out)


# ─────────────────────────────────────────────────────────────
#  JsonOnlyModel: JSON 피처만으로 분류하는 MLP
#  모듈 최상단에 정의해야 Windows multiprocessing pickle 가능
# ─────────────────────────────────────────────────────────────
class _JsonOnlyModel(nn.Module):
    """
    JSON 피처 전용 분류 MLP.
    피처 dim=16 → BatchNorm → 256 → 128 → 64 → nc
    잔차연결(Residual) + BatchNorm으로 수렴 안정화.
    """
    def __init__(self, feat_dim: int, nc: int):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(feat_dim)
        self.fc1   = nn.Linear(feat_dim, 256)
        self.bn1   = nn.BatchNorm1d(256)
        self.fc2   = nn.Linear(256, 128)
        self.bn2   = nn.BatchNorm1d(128)
        self.fc3   = nn.Linear(128, 64)
        self.bn3   = nn.BatchNorm1d(64)
        self.out   = nn.Linear(64, nc)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.1)
        # 잔차 연결용 프로젝션
        self.res1  = nn.Linear(feat_dim, 256)
        self.res2  = nn.Linear(256, 128)

    def forward(self, x):
        x  = self.bn_in(x)
        # Block 1: feat_dim → 256 (잔차)
        r1 = self.res1(x)
        x  = torch.relu(self.bn1(self.fc1(x)))
        x  = self.drop1(x + r1)
        # Block 2: 256 → 128 (잔차)
        r2 = self.res2(x)
        x  = torch.relu(self.bn2(self.fc2(x)))
        x  = self.drop2(x + r2)
        # Block 3: 128 → 64
        x  = torch.relu(self.bn3(self.fc3(x)))
        x  = self.drop3(x)
        return self.out(x)


# ─────────────────────────────────────────────────────────────
#  JsonOnlyDataset: JSON 피처만 반환 (이미지 로드 없음)
#  모듈 최상단에 정의해야 Windows multiprocessing pickle 가능
# ─────────────────────────────────────────────────────────────
class _JsonOnlyDataset(Dataset):
    """
    JSON 피처만 반환 (이미지 로드 없음).
    학습 시 가우시안 노이즈로 augmentation → 과적합 방지.
    """
    def __init__(self, samples, json_map: dict, augment: bool = False):
        self._samples  = samples
        self._json_map = json_map
        self._augment  = augment   # Train=True, Val=False

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        from datasets import _extract_features
        path, label = self._samples[idx]
        feat = _extract_features(self._json_map.get(path, {}))
        if self._augment:
            # 피처에 표준편차 0.02의 가우시안 노이즈 추가
            # clamp(-1, 2)로 정규화 범위 이탈 방지
            noise = torch.randn_like(feat) * 0.02
            feat  = (feat + noise).clamp(-1.0, 2.0)
        return feat, label


# ─────────────────────────────────────────────────────────────
#  온더플라이 전처리 Dataset
#  전처리 폴더 없이 원본 이미지 직접 사용 시
#  선택된 모드(standard / advanced)로 얼굴 크롭 후 학습
# ─────────────────────────────────────────────────────────────
class OnTheFlyPreprocessDataset(Dataset):
    """
    전처리 미완료 원본 이미지를 MediaPipe로 온더플라이 크롭하여 학습.

    Parameters
    ----------
    samples      : [(path, label), ...]
    transform    : torchvision transform (Crop/Flip/Normalize 등)
    preproc_mode : "standard" | "advanced"
    device       : "cuda" | "cpu"  (b* 보정 연산 장치)
    json_map     : {path: json_dict}  (jpg_json 모드 전용)
    """
    def __init__(self, samples, transform,
                 preproc_mode: str = "standard",
                 device: str = "cpu",
                 json_map: dict = None):
        self._samples      = samples
        self._transform    = transform
        self._preproc_mode = preproc_mode
        self._device       = device
        self._json_map     = json_map or {}
        self._preprocessor = None   # 지연 초기화 (워커 프로세스 safe)

    def _get_preprocessor(self):
        """처음 호출 시 초기화 (스레드/프로세스 안전)"""
        if self._preprocessor is None:
            from preprocessors import (PersonalColorPreprocessor,
                                        AdvancedSkinPreprocessor)
            if self._preproc_mode == "advanced":
                self._preprocessor = AdvancedSkinPreprocessor(
                    apply_correction=True, device=self._device)
            else:
                self._preprocessor = PersonalColorPreprocessor(
                    apply_correction=True, device=self._device)
        return self._preprocessor

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        from datasets import _extract_features
        path, label = self._samples[idx]
        try:
            pre = self._get_preprocessor()
            _, meta = pre.preprocess(path)
            # face_crop_bgr: 크롭된 BGR ndarray (preprocessors.py에서 설정)
            crop_bgr = meta.get("face_crop_bgr")
            if crop_bgr is not None and crop_bgr.size > 0:
                import cv2 as _cv2
                rgb = _cv2.cvtColor(crop_bgr, _cv2.COLOR_BGR2RGB)
                img = PILImage.fromarray(rgb)
            else:
                img = PILImage.open(path).convert("RGB")
        except Exception:
            img = PILImage.new("RGB", (300, 300), (128, 128, 128))

        if self._transform:
            img = self._transform(img)

        if self._json_map:
            feat = _extract_features(self._json_map.get(path, {}))
            return img, label, feat
        return img, label


# ─────────────────────────────────────────────────────────────
#  VRAM Prefetch 로더
#  DataLoader가 CPU로 준비한 배치를 미리 CUDA로 이동하여
#  "디스크읽기 → 미리배치 → GPU 학습" 파이프라인 구현.
#  VRAM에 임시 저장 후 학습 완료 시 해제 (del 로 VRAM 회수).
# ─────────────────────────────────────────────────────────────
class CUDAPrefetchLoader:
    """
    DataLoader를 감싸서 다음 배치를 CUDA 스트림으로 미리 전송.

    사용법:
        loader = CUDAPrefetchLoader(DataLoader(...), device)
        for batch in loader:
            x, y = batch   # 이미 CUDA 텐서
            ...             # 학습 완료 후 자동 VRAM 해제
    """
    def __init__(self, loader, device: torch.device,
                 feat_mode: str = "jpg"):
        self._loader    = loader
        self._device    = device
        self._feat_mode = feat_mode
        self._stream    = torch.cuda.Stream(device) if device.type == "cuda" else None

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        if self._device.type != "cuda":
            # CUDA 없으면 그냥 pass-through
            yield from self._loader
            return

        # ── VRAM 비동기 prefetch 로직 ────────────────────────
        it       = iter(self._loader)
        prefetch = None

        def _to_cuda(batch):
            """배치를 CUDA 스트림으로 비동기 이동"""
            with torch.cuda.stream(self._stream):
                if self._feat_mode == "json":
                    feat, y = batch
                    return (feat.to(self._device, non_blocking=True),
                            y.to(self._device, non_blocking=True))
                elif self._feat_mode == "jpg_json":
                    x, y, feat = batch
                    return (x.to(self._device, non_blocking=True),
                            y.to(self._device, non_blocking=True),
                            feat.to(self._device, non_blocking=True))
                else:
                    x, y = batch
                    return (x.to(self._device, non_blocking=True),
                            y.to(self._device, non_blocking=True))

        # 첫 배치 prefetch
        try:
            prefetch = _to_cuda(next(it))
        except StopIteration:
            return

        for batch in it:
            # 현재 배치 스트림 동기화 후 yield
            torch.cuda.current_stream(self._device).wait_stream(self._stream)
            cur = prefetch

            # 다음 배치 미리 올리기
            prefetch = _to_cuda(batch)

            yield cur
            # cur 참조 해제 → VRAM 즉시 반환 가능
            del cur

        # 마지막 배치
        if prefetch is not None:
            torch.cuda.current_stream(self._device).wait_stream(self._stream)
            yield prefetch
            del prefetch


# ─────────────────────────────────────────────────────────────
#  디스크 캐시 Dataset (json_map 지원)
# ─────────────────────────────────────────────────────────────
class DiskCachedDataset(Dataset):
    SAVE_SIZE = 300

    def __init__(self, samples, transform, cache_dir: Path, split: str,
                 json_map: dict = None):
        self._samples   = samples
        self._transform = transform
        self._json_map  = json_map or {}
        self._cached    = []
        cache_dir       = cache_dir / split
        cache_dir.mkdir(parents=True, exist_ok=True)

        for orig_path, label in samples:
            h = hash(orig_path) & 0xFFFF
            cp = cache_dir / f"{label}_{Path(orig_path).stem}_{h:04x}.jpg"
            self._cached.append((str(cp), label, orig_path))

        missing = [i for i, (cp, _, _) in enumerate(self._cached)
                   if not Path(cp).exists()]

        if not missing:
            log.info(f"  캐시 재사용 ({split}): {len(self._cached)}장")
            return

        log.info(f"  캐시 생성 ({split}): {len(missing)}장 -> {cache_dir}")
        log.info("  ** 첫 실행만 시간 소요. 이후 에포크는 빠릅니다 **")

        total = len(missing)
        emit_every = max(1, total // 20)
        for done, i in enumerate(missing, 1):
            orig_path, _, _ = self._samples[i] if len(self._samples[i]) == 3 \
                              else (*self._samples[i], None)
            orig_path = self._samples[i][0]
            cache_path = self._cached[i][0]
            try:
                img = PILImage.open(orig_path).convert("RGB")
                img = img.resize((self.SAVE_SIZE, self.SAVE_SIZE),
                                 PILImage.BILINEAR)
                img.save(cache_path, "JPEG", quality=92, optimize=True)
            except Exception:
                PILImage.new("RGB", (self.SAVE_SIZE, self.SAVE_SIZE),
                             (128, 128, 128)).save(cache_path, "JPEG")

            if done % emit_every == 0:
                print(f"\r  캐시 생성: {done/total*100:.0f}%  ({done}/{total})",
                      end="", flush=True)
        print()
        log.info(f"  캐시 완료: {cache_dir}")

    def __len__(self):
        return len(self._cached)

    def __getitem__(self, idx):
        from datasets import _extract_features
        cache_path, label, orig_path = self._cached[idx]
        try:
            img = PILImage.open(cache_path).convert("RGB")
        except Exception:
            img = PILImage.new("RGB", (self.SAVE_SIZE, self.SAVE_SIZE),
                               (128, 128, 128))
        if self._transform:
            img = self._transform(img)

        if self._json_map:
            feat = _extract_features(self._json_map.get(orig_path, {}))
            return img, label, feat
        return img, label


# ─────────────────────────────────────────────────────────────
#  RAM 캐시 Dataset (RAM 32GB+ 환경 권장)
# ─────────────────────────────────────────────────────────────
class RAMCachedDataset(Dataset):
    def __init__(self, samples, transform, json_map: dict = None):
        self._samples   = samples
        self._transform = transform
        self._json_map  = json_map or {}
        self._images    = {}  # idx -> PILImage

    def warm_up(self):
        from concurrent.futures import ThreadPoolExecutor
        total = len(self._samples)
        log.info(f"  RAM 캐시 로딩 중: {total}장...")

        def _load(idx):
            path, _ = self._samples[idx]
            try:
                return idx, PILImage.open(path).convert("RGB")
            except Exception:
                return idx, PILImage.new("RGB", (300, 300), (128, 128, 128))

        emit_every = max(1, total // 20)
        with ThreadPoolExecutor(max_workers=8) as ex:
            for done, (idx, img) in enumerate(
                    ex.map(_load, range(total)), 1):
                self._images[idx] = img
                if done % emit_every == 0:
                    print(f"\r  RAM 캐시: {done/total*100:.0f}%  ({done}/{total})",
                          end="", flush=True)
        print()
        log.info("  RAM 캐시 완료 - 이후 에포크 디스크 I/O = 0")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        from datasets import _extract_features
        path, label = self._samples[idx]
        img = self._images.get(idx)
        if img is None:
            try:
                img = PILImage.open(path).convert("RGB")
            except Exception:
                img = PILImage.new("RGB", (300, 300), (128, 128, 128))
        if self._transform:
            img = self._transform(img)
        if self._json_map:
            feat = _extract_features(self._json_map.get(path, {}))
            return img, label, feat
        return img, label


# ─────────────────────────────────────────────────────────────
#  일반 Dataset (cache_mode=none)
# ─────────────────────────────────────────────────────────────
class PlainDataset(Dataset):
    def __init__(self, samples, transform, json_map: dict = None):
        self._samples   = samples
        self._transform = transform
        self._json_map  = json_map or {}

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        from datasets import _extract_features
        path, label = self._samples[idx]
        try:
            img = PILImage.open(path).convert("RGB")
        except Exception:
            img = PILImage.new("RGB", (300, 300), (128, 128, 128))
        if self._transform:
            img = self._transform(img)
        if self._json_map:
            feat = _extract_features(self._json_map.get(path, {}))
            return img, label, feat
        return img, label


def _auto_batch(requested: int, use_cuda: bool) -> int:
    if requested > 0:
        return requested
    if not use_cuda:
        return 32
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 20: return 256
        if vram_gb >= 14: return 128
        if vram_gb >= 8:  return 64
        return 32
    except Exception:
        return 64


# ─────────────────────────────────────────────────────────────
#  학습 함수
# ─────────────────────────────────────────────────────────────
def train(cfg: dict):
    global _STOP

    req = cfg["device"]
    if req == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA 없음 -> CPU 사용"); req = "cpu"
    device   = torch.device(req)
    use_cuda = device.type == "cuda"
    use_amp  = use_cuda

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU : {gpu_name}  VRAM: {vram_gb:.1f}GB")
        log.info("cuDNN benchmark=ON  TF32=ON  AMP=ON")
        log.info(f"초기 GPU: {_gpu_util()}")

    bs   = _auto_batch(cfg["batch_size"], use_cuda)
    nw   = cfg["workers"]
    mode = cfg.get("cache_mode", "resize")
    feat_mode    = cfg.get("feat_mode",    "jpg")
    json_dir     = cfg.get("json_dir",     "")
    jpg_dir      = cfg.get("jpg_dir",      "")   # ★ JPG 폴더 (비면 data_dir 사용)
    preproc_mode = cfg.get("preproc_mode", "standard")
    preproc_dev  = cfg.get("preproc_dev",  "cpu")
    img_mode     = cfg.get("img_mode",     "crop")   # ★ "crop" / "original"
    resume_path  = cfg.get("resume",       "")       # ★ 이어학습 체크포인트 경로
    target_acc   = cfg.get("target_acc",   0.0)      # ★ 목표 정확도 (0=비활성)
    target_acc_path = cfg.get("target_acc_path", "")  # ★ 목표 정확도 저장 경로
    if target_acc > 0 and not target_acc_path:
        target_acc_path = str(
            Path(cfg["save_path"]).parent /
            f"{Path(cfg['save_path']).stem}_acc{int(target_acc)}.pt")
    target_acc_saved = False   # 목표 정확도 이미 저장했는지

    log.info(f"배치: {bs}  workers: {nw}  cache_mode: {mode}  feat_mode: {feat_mode}")
    log.info(f"preproc_mode: {preproc_mode}  preproc_dev: {preproc_dev}")

    # ── Transform 설정 ────────────────────────────────────────
    # 전처리 완료(크롭 저장) / 온더플라이 크롭 모두:
    #   이미지가 이미 얼굴 크롭 영역이므로 Resize 불필요.
    #   RandomCrop(260)으로 augmentation만 적용.
    # cache_mode=none + 전처리 미완료(원본) 폴더:
    #   원본 전체 이미지 → 온더플라이 크롭 → 아래 transform 적용
    _M = [0.485, 0.456, 0.406]
    _S = [0.229, 0.224, 0.225]

    # 크롭 완료 이미지 (전처리 폴더 또는 온더플라이 크롭) 용 transform
    trn_tf = transforms.Compose([
        transforms.Resize((300, 300)),   # 크롭 크기 통일 (얼굴 크기 불일치 대비)
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

    # ── 데이터셋 로드 ─────────────────────────────────────────
    # jpg_json 모드에서 JPG/JSON 폴더가 분리된 경우:
    #   ds      → jpg_dir (이미지 스캔용, 클래스 구조 기준)
    #   json_ds → json_dir (JSON 피처 로드용)
    # 나머지 모드: data_dir 단일 사용
    img_data_dir = jpg_dir if (feat_mode == "jpg_json" and jpg_dir) else cfg["data_dir"]

    log.info(f"데이터셋 로드: {img_data_dir}  (feat_mode={feat_mode})")
    if feat_mode == "jpg_json" and jpg_dir:
        log.info(f"  JPG 폴더 : {jpg_dir}")
        log.info(f"  JSON 폴더: {json_dir if json_dir else cfg['data_dir']}")

    ds = PersonalColorDataset(img_data_dir, transform=None)
    if len(ds) == 0:
        log.error("빈 데이터셋"); return

    nc = len(ds.classes)
    log.info(f"클래스: {ds.classes}")

    # feat_mode 에 따라 use_json, use_img 결정
    use_img  = feat_mode in ("jpg", "jpg_json")
    use_json = feat_mode in ("json", "jpg_json")

    # JSON 모드: json_map 로드
    # JPG/JSON 폴더 분리 구조 완전 지원:
    #   json_dir 폴더를 직접 탐색해 .json 파일을 stem 기준으로 매칭
    #   → jpg_path의 stem과 json_path의 stem이 같으면 연결
    json_map_all: dict = {}
    if use_json:
        json_src = json_dir if json_dir else (jpg_dir if jpg_dir else cfg["data_dir"])
        log.info(f"JSON 피처 로드: {json_src}")

        # ── 방법 A: preprocess_meta.json 있으면 그대로 사용
        meta_path = Path(json_src) / "preprocess_meta.json"
        if meta_path.exists():
            json_ds = PersonalColorDataset(json_src, transform=None)
            json_map_raw = json_ds.json_map  # key = json_src 기준 절대경로

            if json_map_raw:
                # JPG 폴더와 JSON 폴더가 같으면 키 그대로 사용
                if Path(json_src).resolve() == Path(img_data_dir).resolve():
                    json_map_all = json_map_raw
                else:
                    # 분리 폴더: stem 기준으로 jpg 경로에 재매핑
                    stem_to_jdata: dict = {}
                    for jpath, jdata in json_map_raw.items():
                        stem_to_jdata[Path(jpath).stem] = jdata
                    for img_path, _ in ds.samples:
                        stem = Path(img_path).stem
                        if stem in stem_to_jdata:
                            json_map_all[img_path] = stem_to_jdata[stem]
                    log.info(f"  stem 매칭 완료: {len(json_map_all)}/{len(ds.samples)}장")

        # ── 방법 B: preprocess_meta.json 없으면 .json 파일 직접 탐색
        if not json_map_all:
            log.info(f"  preprocess_meta.json 없음 → .json 파일 직접 탐색")
            # json_src 하위의 모든 .json 파일을 stem 기준으로 인덱싱
            stem_to_jfile: dict = {}
            for jf in Path(json_src).rglob("*.json"):
                if jf.name == "preprocess_meta.json":
                    continue
                stem_to_jfile[jf.stem] = jf

            matched = 0
            for img_path, _ in ds.samples:
                stem = Path(img_path).stem
                if stem in stem_to_jfile:
                    try:
                        jd = json.loads(stem_to_jfile[stem].read_text(encoding="utf-8"))
                        json_map_all[img_path] = jd
                        matched += 1
                    except Exception:
                        pass
            log.info(f"  직접 탐색 매칭: {matched}/{len(ds.samples)}장")

        if not json_map_all:
            log.warning(
                f"[WARN] JSON 피처를 찾지 못했습니다: {json_src}\n"
                f"       .json 파일이 있는지, 이미지 파일명과 stem이 일치하는지 확인하세요.\n"
                f"       feat_mode를 'jpg'로 전환합니다.")
            use_json = False
            feat_mode = "jpg"
        else:
            log.info(f"  JSON 피처 로드 완료: {len(json_map_all)}장")

    feat_dim = len(list(json_map_all.values())[0]) if json_map_all else 0
    # JSON 피처는 _extract_features()가 고정 10차원 반환
    from datasets import JSON_FEATURE_DIM
    if use_json:
        feat_dim = JSON_FEATURE_DIM

    log.info(f"학습 입력: {'이미지' if use_img else ''}"
             f"{'+' if use_img and use_json else ''}"
             f"{'JSON피처' if use_json else ''}  "
             f"feat_dim={feat_dim}")

    # ── Stratified Split — Train / Val / Test 3-way ──────────
    patience   = cfg.get("patience",   0)
    test_split = cfg.get("test_split", 0.1)

    import random as _rng
    _rng.seed(42)

    class_indices: dict = {i: [] for i in range(nc)}
    for idx, (_, label) in enumerate(ds.samples):
        class_indices[label].append(idx)

    tr_indices, vl_indices, ts_indices = [], [], []
    for label, idxs in class_indices.items():
        _rng.shuffle(idxs)
        n_test = max(1, int(len(idxs) * test_split)) if test_split > 0 else 0
        n_val  = max(1, int(len(idxs) * 0.1))        # Val: 10%
        ts_indices.extend(idxs[:n_test])
        vl_indices.extend(idxs[n_test:n_test + n_val])
        tr_indices.extend(idxs[n_test + n_val:])

    tr_samples = [(ds.samples[i][0], ds.samples[i][1]) for i in tr_indices]
    vl_samples = [(ds.samples[i][0], ds.samples[i][1]) for i in vl_indices]
    te_samples = [(ds.samples[i][0], ds.samples[i][1]) for i in ts_indices]

    log.info(f"Train: {len(tr_samples)}  Val: {len(vl_samples)}"
             f"  Test: {len(te_samples)}  [Stratified]")
    for label, cls in enumerate(ds.classes):
        tr_c = sum(1 for _, lb in tr_samples if lb == label)
        vl_c = sum(1 for _, lb in vl_samples if lb == label)
        te_c = sum(1 for _, lb in te_samples if lb == label)
        log.info(f"  {cls}: Train {tr_c}  Val {vl_c}  Test {te_c}")
    if patience > 0:
        log.info(f"Early Stopping: patience={patience} 에포크")

    # ── json_map 슬라이스 ─────────────────────────────────────
    if use_json and json_map_all:
        tr_json = {ds.samples[i][0]: json_map_all.get(ds.samples[i][0], {})
                   for i in tr_indices}
        vl_json = {ds.samples[i][0]: json_map_all.get(ds.samples[i][0], {})
                   for i in vl_indices}
        te_json = {ds.samples[i][0]: json_map_all.get(ds.samples[i][0], {})
                   for i in ts_indices}
    else:
        tr_json = vl_json = te_json = {}

    # ── Dataset 생성 — feat_mode 분기 ────────────────────────
    # JSON 전용 모드: 이미지 불필요 → _JsonOnlyDataset 사용
    # preprocess_meta.json 없음(원본 폴더) + 이미지 학습 모드
    #   → OnTheFlyPreprocessDataset (온더플라이 크롭)
    # preprocess_meta.json 있음(전처리 완료)
    #   → DiskCachedDataset or RAMCachedDataset (크롭 이미지 그대로)

    # img_mode="original" 이면 원본 폴더 → 온더플라이 크롭 강제
    # img_mode="crop"     이면 전처리 완료 폴더 사용 (ds.use_json 기준)
    is_raw_folder = (img_mode == "original") or (not ds.use_json and feat_mode != "json")

    if feat_mode == "json":
        tr_ds = _JsonOnlyDataset(tr_samples, tr_json, augment=True)
        vl_ds = _JsonOnlyDataset(vl_samples, vl_json, augment=False)
        te_ds = _JsonOnlyDataset(te_samples, te_json, augment=False) if te_samples else None
    elif is_raw_folder:
        log.info(f"[OnTheFly] 전처리 폴더 없음 → {preproc_mode} 모드 크롭 학습")
        log.info(f"           b* 보정 장치: {preproc_dev}")
        tr_ds = OnTheFlyPreprocessDataset(tr_samples, trn_tf, preproc_mode=preproc_mode,
                                          device=preproc_dev,
                                          json_map=tr_json if feat_mode=="jpg_json" else {})
        vl_ds = OnTheFlyPreprocessDataset(vl_samples, val_tf, preproc_mode=preproc_mode,
                                          device=preproc_dev,
                                          json_map=vl_json if feat_mode=="jpg_json" else {})
        te_ds = OnTheFlyPreprocessDataset(te_samples, val_tf, preproc_mode=preproc_mode,
                                          device=preproc_dev,
                                          json_map=te_json if feat_mode=="jpg_json" else {}) \
                if te_samples else None
        if nw > 0:
            log.info(f"  [주의] 온더플라이 크롭은 MediaPipe CPU 의존 → workers=0 강제")
            nw = 0
    elif mode == "resize":
        cache_dir = Path(cfg["data_dir"]) / ".cache_300"
        log.info(f"캐시 폴더: {cache_dir}")
        tr_ds = DiskCachedDataset(tr_samples, trn_tf, cache_dir, "train", json_map=tr_json)
        vl_ds = DiskCachedDataset(vl_samples, val_tf, cache_dir, "val",   json_map=vl_json)
        te_ds = DiskCachedDataset(te_samples, val_tf, cache_dir, "test",  json_map=te_json) \
                if te_samples else None
    elif mode == "ram":
        log.info("RAM 캐시 모드")
        tr_ds = RAMCachedDataset(tr_samples, trn_tf, json_map=tr_json)
        vl_ds = RAMCachedDataset(vl_samples, val_tf, json_map=vl_json)
        te_ds = RAMCachedDataset(te_samples, val_tf, json_map=te_json) if te_samples else None
        tr_ds.warm_up(); vl_ds.warm_up()
        if te_ds: te_ds.warm_up()
        nw = 0
    else:
        tr_ds = PlainDataset(tr_samples, trn_tf, json_map=tr_json)
        vl_ds = PlainDataset(vl_samples, val_tf, json_map=vl_json)
        te_ds = PlainDataset(te_samples, val_tf, json_map=te_json) if te_samples else None

    # DataLoader
    pf = 4 if nw > 0 else None
    _tr_dl_raw = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                            num_workers=nw, pin_memory=use_cuda,
                            persistent_workers=(nw > 0), prefetch_factor=pf, drop_last=True)
    _vl_dl_raw = DataLoader(vl_ds, batch_size=bs * 2,
                            num_workers=nw, pin_memory=use_cuda,
                            persistent_workers=(nw > 0), prefetch_factor=pf)
    _te_dl_raw = DataLoader(te_ds, batch_size=bs * 2,
                            num_workers=nw, pin_memory=use_cuda,
                            persistent_workers=(nw > 0), prefetch_factor=pf) \
                 if te_ds else None

    # ── VRAM Prefetch 래퍼 적용 ──────────────────────────────
    if use_cuda:
        log.info("VRAM Prefetch 활성화: 디스크→핀메모리→VRAM 미리배치 파이프라인")
        tr_dl = CUDAPrefetchLoader(_tr_dl_raw, device, feat_mode=feat_mode)
        vl_dl = CUDAPrefetchLoader(_vl_dl_raw, device, feat_mode=feat_mode)
        te_dl = CUDAPrefetchLoader(_te_dl_raw, device, feat_mode=feat_mode) \
                if _te_dl_raw else None
    else:
        tr_dl = _tr_dl_raw
        vl_dl = _vl_dl_raw
        te_dl = _te_dl_raw

    log.info(f"Train배치수: {len(tr_dl)}  Val배치수: {len(vl_dl)}"
             f"  Test배치수: {len(te_dl) if te_dl else 0}")

    # ── 모델 ─────────────────────────────────────────────────
    log.info("EfficientNet-B2 로드 중...")
    backbone = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    cnn_dim = backbone.classifier[1].in_features

    if feat_mode == "json":
        # JSON 전용: MLP만 사용 (_JsonOnlyModel은 모듈 최상단에 정의됨)
        model = _JsonOnlyModel(feat_dim, nc).to(device)
        log.info(f"모델: MLP (JSON 전용)  feat_dim={feat_dim}")

    elif feat_mode == "jpg_json":
        # JPG+JSON: FusionModel
        backbone.classifier = nn.Identity()
        model = _FusionModel(backbone, cnn_dim, feat_dim, nc).to(device)
        log.info(f"모델: FusionModel (CNN+MLP)  cnn_dim={cnn_dim}  feat_dim={feat_dim}")

    else:
        # JPG 전용: CNN only
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3, True),
            nn.Linear(cnn_dim, 256), nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, nc),
        )
        model = backbone.to(device)
        log.info(f"모델: EfficientNet-B2 CNN only  nc={nc}")
    # 옵티마이저 — feat_mode 별 파라미터 그룹
    lr   = cfg["lr"]
    ep_n = cfg["epochs"]

    if feat_mode == "json":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif feat_mode == "jpg_json":
        opt = optim.AdamW([
            {"params": model.backbone.features.parameters(), "lr": lr * 0.05},
            {"params": model.head.parameters(),              "lr": lr},
        ], weight_decay=1e-4)
    else:   # jpg
        opt = optim.AdamW([
            {"params": model.features.parameters(),   "lr": lr * 0.05},
            {"params": model.classifier.parameters(), "lr": lr},
        ], weight_decay=1e-4)

    warmup_ep = min(5, ep_n // 4)
    def _lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep
        t = ep - warmup_ep
        T = max(ep_n - warmup_ep, 1)
        return 0.05 + 0.95 * (1 + math.cos(math.pi * t / T)) / 2

    sch    = optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit   = nn.CrossEntropyLoss(label_smoothing=0.1)

    sp       = cfg["save_path"]
    log_path = Path(sp).parent / "train_log.txt"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(fh)

    best       = 0.0
    start_ep   = 1
    # ── Early Stopping 변수 ──────────────────────────────────
    es_counter = 0   # patience 카운터
    es_best    = 0.0 # early stopping 기준 best val acc
    t_total    = time.time()

    # ── Resume (이어학습) ────────────────────────────────────────
    resume_ckpt_path = resume_path or ""
    # resume 미지정이면 save_path 자동 탐색 (중단 전 저장된 best 모델)
    if not resume_ckpt_path:
        checkpoint_path = Path(sp).with_suffix("") .parent / (Path(sp).stem + "_ckpt.pt")
        if checkpoint_path.exists():
            resume_ckpt_path = str(checkpoint_path)
            log.info(f"[RESUME] 자동 감지된 체크포인트: {checkpoint_path}")

    if resume_ckpt_path and Path(resume_ckpt_path).exists():
        try:
            ck = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            if "optimizer_state_dict" in ck:
                opt.load_state_dict(ck["optimizer_state_dict"])
            if "scheduler_state_dict" in ck:
                sch.load_state_dict(ck["scheduler_state_dict"])
            if "scaler_state_dict" in ck and use_amp:
                scaler.load_state_dict(ck["scaler_state_dict"])
            resumed_ep = ck.get("epoch", 0)
            best       = ck.get("val_acc", 0.0)
            start_ep   = resumed_ep + 1
            log.info(f"[RESUME] ✅ 이어학습 시작: epoch {resumed_ep} → {start_ep}  best_acc={best:.2f}%")
            log.info(f"[RESUME] 체크포인트: {resume_ckpt_path}")
            if start_ep > ep_n:
                log.info(f"[RESUME] 이미 {ep_n} 에포크 완료 → 추가 에포크 필요 시 --epochs 증가")
        except Exception as e:
            log.warning(f"[RESUME] 체크포인트 로드 실패 ({e}) → 처음부터 학습")
            start_ep = 1
            best     = 0.0
    else:
        if resume_ckpt_path:
            log.warning(f"[RESUME] 체크포인트 없음: {resume_ckpt_path} → 처음부터 학습")

    # 에포크별 체크포인트 저장 경로 (이어학습용 — best와 별도)
    ckpt_path = Path(sp).parent / (Path(sp).stem + "_ckpt.pt")

    log.info("=" * 65)
    log.info(f"학습 시작  에포크:{start_ep}~{ep_n}  배치:{bs}  LR:{lr}  warmup:{warmup_ep}ep")
    log.info("=" * 65)

    for ep in range(start_ep, ep_n + 1):
        if _STOP:
            log.info("[STOP] 학습 중단"); break

        t_ep = time.time()

        # Train
        model.train()
        tl = tc = tt = 0
        emit_every = max(1, len(tr_dl) // 10)

        for bi, batch in enumerate(tr_dl):
            if _STOP: break

            # CUDAPrefetchLoader 사용 시 배치는 이미 CUDA 텐서.
            # CPU fallback(no CUDA) 시에는 아래 .to()가 실제 이동 수행.
            if feat_mode == "json":
                feat, y = batch
                x    = None
                feat = feat.to(device, non_blocking=True)
            elif feat_mode == "jpg_json":
                x, y, feat = batch
                x    = x.to(device, non_blocking=True)
                feat = feat.to(device, non_blocking=True)
            else:   # jpg
                x, y = batch
                x    = x.to(device, non_blocking=True)
                feat = None

            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if feat_mode == "json":
                    out = model(feat)        # _JsonOnlyModel: feat만 입력
                elif feat_mode == "jpg_json":
                    out = model(x, feat)     # _FusionModel
                else:
                    out = model(x)           # CNN only
                lo  = crit(out, y)
            scaler.scale(lo).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            tl += lo.item() * (x.size(0) if x is not None else feat.size(0))
            tc += out.argmax(1).eq(y).sum().item()
            tt += (x.size(0) if x is not None else feat.size(0))

            if (bi + 1) % emit_every == 0:
                pct = (bi + 1) / len(tr_dl) * 100
                print(f"\r  Train {pct:5.1f}%  loss={tl/max(tt,1):.4f}",
                      end="", flush=True)
        print()

        model.eval()
        vl_loss = vc = vt = 0
        with torch.no_grad():
            for batch in vl_dl:
                # CUDAPrefetchLoader 사용 시 이미 CUDA 텐서
                if feat_mode == "json":
                    feat, y = batch
                    x    = None
                    feat = feat.to(device, non_blocking=True)
                elif feat_mode == "jpg_json":
                    x, y, feat = batch
                    x    = x.to(device, non_blocking=True)
                    feat = feat.to(device, non_blocking=True)
                else:
                    x, y = batch
                    x    = x.to(device, non_blocking=True)
                    feat = None

                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if feat_mode == "json":
                        out = model(feat)
                    elif feat_mode == "jpg_json":
                        out = model(x, feat)
                    else:
                        out = model(x)
                    lo  = crit(out, y)
                n = (x.size(0) if x is not None else feat.size(0))
                vl_loss += lo.item() * n
                vc      += out.argmax(1).eq(y).sum().item()
                vt      += n

        sch.step()

        atl    = tl      / max(tt, 1)
        avl    = vl_loss / max(vt, 1)
        ta     = tc / max(tt, 1) * 100
        va     = vc / max(vt, 1) * 100
        cur_lr = opt.param_groups[-1]["lr"]   # json=그룹1개, 나머지=마지막그룹
        ep_sec = time.time() - t_ep

        gpu_stat = _gpu_util() if use_cuda else ""
        log.info(
            f"[Epoch {ep:03d}/{ep_n}] "
            f"Train Loss={atl:.4f} Acc={ta:.2f}%  |  "
            f"Val Loss={avl:.4f} Acc={va:.2f}%  |  "
            f"LR={cur_lr:.2e}  |  {ep_sec:.0f}s")
        if gpu_stat:
            log.info(f"  [{gpu_stat}]")

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
                "preproc_mode":     preproc_mode,
                "img_mode":         img_mode,
            }, sp)
            log.info(f"  [BEST] val_acc={va:.2f}%  ->  {sp}")

        # ★ 목표 정확도 도달 시 별도 저장 (최초 1회)
        if target_acc > 0 and not target_acc_saved and va >= target_acc:
            torch.save({
                "epoch":            ep,
                "model_state_dict": model.state_dict(),
                "val_acc":          va,
                "classes":          ds.classes,
                "num_classes":      nc,
                "model_arch":       "efficientnet_b2",
                "feat_mode":        feat_mode,
                "feat_dim":         feat_dim,
                "preproc_mode":     preproc_mode,
                "img_mode":         img_mode,
                "target_acc":       target_acc,
            }, target_acc_path)
            target_acc_saved = True
            log.info(f"  [TARGET] ✅ 목표 {target_acc:.1f}% 도달 (val={va:.2f}%)  ->  {target_acc_path}")

        # ★ 에포크마다 이어학습용 체크포인트 저장 (옵티마이저/스케줄러 포함)
        torch.save({
            "epoch":               ep,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sch.state_dict(),
            "scaler_state_dict":    scaler.state_dict() if use_amp else {},
            "val_acc":             va,
            "best_val_acc":        best,
            "classes":             ds.classes,
            "num_classes":         nc,
            "model_arch":          "efficientnet_b2",
            "feat_mode":           feat_mode,
            "feat_dim":            feat_dim,
            "preproc_mode":        preproc_mode,
            "img_mode":            img_mode,
            "total_epochs":        ep_n,
        }, ckpt_path)
        log.info(f"  [CKPT] 체크포인트 저장 → {ckpt_path}")

        # ── Early Stopping ─────────────────────────────────────
        if patience > 0:
            if va > es_best:
                es_best    = va
                es_counter = 0
            else:
                es_counter += 1
                log.info(f"  [ES] Val Acc 개선 없음 {es_counter}/{patience}"
                         f"  (best={es_best:.2f}%)")
                if es_counter >= patience:
                    log.info(f"  [ES] ★ Early Stopping 발동! "
                             f"{patience}에포크 동안 개선 없음 → 학습 중단")
                    break

        elapsed = time.time() - t_total
        per_ep  = elapsed / ep
        remain  = per_ep * (ep_n - ep)
        log.info(f"  경과: {elapsed/60:.1f}분  에포크당: {per_ep:.0f}s  "
                 f"남은 예상: {remain/60:.1f}분")

    log.info("=" * 65)
    log.info(f"[DONE] Best Val Acc: {best:.2f}%")
    log.info(f"[DONE] 모델 저장: {sp}")
    log.info(f"[DONE] 총 소요: {(time.time()-t_total)/60:.1f}분")
    if use_cuda:
        log.info(f"[DONE] 최종 GPU: {_gpu_util()}")

    # ── Test 셋 최종 평가 (Best 모델 로드 후) ────────────────
    if te_dl is not None:
        log.info("\n[TEST] Best 모델로 Test 셋 평가 중...")
        best_ck = torch.load(sp, map_location=device, weights_only=False)
        model.load_state_dict(best_ck["model_state_dict"])
        model.eval()
        tc2 = tv2 = 0
        with torch.no_grad():
            for batch in te_dl:
                if feat_mode == "json":
                    feat, y = batch; x = None
                    feat = feat.to(device, non_blocking=True)
                elif feat_mode == "jpg_json":
                    x, y, feat = batch
                    x    = x.to(device, non_blocking=True)
                    feat = feat.to(device, non_blocking=True)
                else:
                    x, y = batch; feat = None
                    x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if feat_mode == "json":   out = model(feat)
                    elif feat_mode == "jpg_json": out = model(x, feat)
                    else:                     out = model(x)
                tc2 += out.argmax(1).eq(y).sum().item()
                tv2 += y.size(0)
        test_acc = tc2 / max(tv2, 1) * 100
        log.info(f"[TEST] ★ Test Acc: {test_acc:.2f}%  "
                 f"(Val Best: {best:.2f}%  Gap: {best - test_acc:+.2f}%)")
        if best - test_acc > 5.0:
            log.info(f"[TEST] ⚠️  Val-Test Gap이 {best - test_acc:.1f}%로 큼 "
                     f"→ 과적합 가능성 있음")
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────
#  진입점
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="퍼스널컬러 EfficientNet-B2 학습 (GPU 완전 활용)")
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--save_path",  default="best_model.pt")
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch_size", type=int,   default=0,
                   help="0=VRAM 자동결정 (RTX4080S->128)")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--device",     default="cuda", choices=["cuda","cpu"])
    p.add_argument("--cache_mode", default="resize",
                   choices=["resize","ram","none"],
                   help="resize=디스크캐시(권장)/ram=RAM캐시/none=기존")
    p.add_argument("--feat_mode",    default="jpg",
                   choices=["jpg", "json", "jpg_json"],
                   help="jpg=이미지만 / json=피처만 / jpg_json=결합(FusionModel)")
    p.add_argument("--json_dir",     default="",
                   help="jpg_json 모드 전용: JSON 전처리 폴더 경로")
    p.add_argument("--jpg_dir",      default="",
                   help="jpg_json 모드 전용: JPG 전처리 폴더 경로 (비우면 data_dir 사용)")
    p.add_argument("--preproc_mode", default="standard",
                   choices=["standard", "advanced"],
                   help="온더플라이 크롭 모드 (전처리 폴더 없을 때만 적용)\n"
                        "standard=볼 영역 크롭 / advanced=Face Oval 정밀 크롭")
    p.add_argument("--preproc_dev",  default="cpu",
                   choices=["cuda", "cpu"],
                   help="온더플라이 크롭 시 b* 보정 연산 장치 (cuda 권장)")
    p.add_argument("--resume",       default="",
                   help="이어학습할 체크포인트 .pt 파일 경로 (비어있으면 처음부터)")
    p.add_argument("--img_mode",     default="crop",
                   choices=["crop", "original"],
                   help="jpg 학습 시 이미지 소스: crop=크롭전처리폴더 / original=원본폴더+온더플라이")
    p.add_argument("--target_acc",      type=float, default=0.0,
                   help="목표 Val Acc (예: 90.0). 이 값 이상 최초 도달 시 별도 저장")
    p.add_argument("--target_acc_path", default="",
                   help="목표 정확도 도달 모델 저장 경로 (비우면 자동 생성)")
    p.add_argument("--patience",   type=int, default=0,
                   help="Early Stopping patience (0=비활성, 예: 10 → Val Acc 10에포크 동안 개선 없으면 중단)")
    p.add_argument("--test_split", type=float, default=0.1,
                   help="Test 셋 비율 (기본 0.1=10%%, 0=Test 셋 없음)")
    return vars(p.parse_args())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    # Windows spawn 방식: 워커 프로세스가 이 파일을 재임포트할 때
    # __main__ 블록이 다시 실행되지 않도록 보호됨.
    # set_start_method는 이미 spawn이 기본이므로 생략.

    cfg = parse_args()
    Path(cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("퍼스널컬러 학습 스크립트 (GPU 완전 활용 버전)")
    log.info(f"  data_dir   : {cfg['data_dir']}")
    log.info(f"  save_path  : {cfg['save_path']}")
    log.info(f"  epochs     : {cfg['epochs']}")
    log.info(f"  batch_size : {cfg['batch_size']} (0=자동)")
    log.info(f"  lr         : {cfg['lr']}")
    log.info(f"  workers    : {cfg['workers']}")
    log.info(f"  device     : {cfg['device']}")
    log.info(f"  cache_mode : {cfg['cache_mode']}")
    log.info(f"  feat_mode  : {cfg['feat_mode']}  "
             f"({'이미지만' if cfg['feat_mode']=='jpg' else 'JSON피처만' if cfg['feat_mode']=='json' else '이미지+JSON결합'})")
    log.info(f"  preproc_mode: {cfg['preproc_mode']}  preproc_dev: {cfg['preproc_dev']}")
    log.info(f"  img_mode   : {cfg['img_mode']}  ({'크롭 전처리 이미지' if cfg['img_mode']=='crop' else '원본 이미지 + 온더플라이 크롭'})")
    if cfg['feat_mode'] == 'jpg_json':
        log.info(f"  jpg_dir    : {cfg.get('jpg_dir','') or '(data_dir 사용)'}")
        log.info(f"  json_dir   : {cfg.get('json_dir','') or '(data_dir 사용)'}")
    if cfg['resume']:
        log.info(f"  resume     : {cfg['resume']}")
    if cfg.get('target_acc', 0) > 0:
        log.info(f"  target_acc : {cfg['target_acc']:.1f}%")
    pt = cfg.get('patience', 0)
    ts = cfg.get('test_split', 0.1)
    log.info(f"  patience   : {pt}  ({'비활성' if pt == 0 else str(pt) + '에포크 개선없으면 중단'})")
    log.info(f"  test_split : {ts:.0%}  ({'Test 셋 없음' if ts == 0 else '전체의 ' + str(int(ts*100)) + '%를 Test로 분리'})")
    log.info("=" * 65)

    train(cfg)