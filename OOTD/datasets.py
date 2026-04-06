"""
datasets.py — 전처리 JSON 메타데이터 지원 버전

지원 모드:
  1. JSON 모드 (전처리 완료된 폴더)
     preprocess_meta.json 이 있으면 자동 감지
     → 이미지 + JSON 피처(lab_mean, b_star 등) 함께 로드

  2. 폴더 모드 (기존 — 원본 또는 전처리 미완료)
     폴더 구조로 클래스 판별 (기존 동작 유지)

학습 코드에서는 아무 변경 없이 PersonalColorDataset(path) 만 호출하면 됨.
JSON이 있으면 자동으로 피처를 포함한 샘플을 반환.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage

from constants import SEASON_CLASSES, TONE_CLASSES, SEASON_TONE_CLASSES, SUPPORTED_EXT


# ─────────────────────────────────────────────────────────────
#  폴더명 정규화
# ─────────────────────────────────────────────────────────────
_SEASON_ALIAS: Dict[str, str] = {}
for _cls in SEASON_CLASSES:
    _SEASON_ALIAS[_cls.lower()] = _cls
_SEASON_ALIAS.update({
    "fall": "Autumn", "autumn": "Autumn",
    "winter": "Winter", "spring": "Spring", "summer": "Summer",
})

_TONE_ALIAS: Dict[str, str] = {}
for _t in TONE_CLASSES:
    _TONE_ALIAS[_t.lower()] = _t


def _norm_season(name: str) -> Optional[str]:
    return _SEASON_ALIAS.get(name.lower())

def _norm_tone(name: str) -> Optional[str]:
    return _TONE_ALIAS.get(name.lower())


# ─────────────────────────────────────────────────────────────
#  JSON 피처 키 목록 (학습에 사용할 수치형 피처)
#  순서 고정: 모델 입력 차원이 여기에 의존
# ─────────────────────────────────────────────────────────────
JSON_FEATURE_KEYS = [
    # Lab 평균 (L, a, b) — 정규화 기준 별도
    "lab_mean_L",       # 0~100  → /100
    "lab_mean_a",       # -128~127 → /128
    "lab_mean_b",       # -128~127 → /128
    # Lab 표준편차
    "lab_std_L",        # /50
    "lab_std_a",        # /64
    "lab_std_b",        # /64
    # 논문 b* (−127~127)
    "b_star",           # /127
    # HSV 채도 (0~100)
    "hsv_s_mean",       # /100
    # 피부 면적 비율 (0~1)
    "skin_pixel_ratio",
    # 보정 여부
    "corrected",
    # ── 추가 피처 ────────────────────────────────────────────
    # ITA (Individual Typology Angle) — 피부톤 밝기 지표
    # ITA = atan((L-50)/b) * 180/pi  → -90~90 → /90
    "ita_angle",
    # a채널 절댓값 — 붉은기/초록기 강도
    "a_abs",            # 0~128 → /128
    # Lab b채널 분산 (표준편차의 제곱, 분포 넓이)
    "b_variance",       # /1000
    # 채도 대비 (a_std / a_mean_abs) — 피부 균일도
    "chroma_contrast",  # 0~5 → /5
    # 밝기 대비 (L_std / L_mean) — 조명 균일도
    "luminance_contrast", # 0~1
    # b* 보정 전후 delta 절댓값 — 보정 강도
    "b_delta_abs",      # 0~50 → /50
]
JSON_FEATURE_DIM = len(JSON_FEATURE_KEYS)  # 16


def _extract_features(json_data: dict) -> "torch.Tensor":
    """JSON 레코드 → 정규화된 float32 텐서 (dim=16)"""
    import math as _math

    lm = json_data.get("lab_mean", [50.0, 128.0, 128.0])
    ls = json_data.get("lab_std",  [0.0,  0.0,   0.0])

    # Lab 스케일: OpenCV 기준
    # L: 0~255 (실제 Lab L*=0~100 → OpenCV L=0~255)
    # a,b: 0~255 (중심 128 = 0)
    L_cv  = float(lm[0])
    a_cv  = float(lm[1])
    b_cv  = float(lm[2])

    # OpenCV → 실제 Lab 스케일 변환
    L_real = L_cv / 255.0 * 100.0          # 0~100
    a_real = (a_cv - 128.0)                # -128~127
    b_real = (b_cv - 128.0)                # -128~127

    # ITA 각도 계산
    try:
        ita = _math.degrees(_math.atan2(L_real - 50.0, b_real)) \
              if b_real != 0.0 else 0.0
    except Exception:
        ita = 0.0

    b_star   = float(json_data.get("b_star",          0.0))
    b_delta  = float(json_data.get("b_delta",         0.0))
    a_abs    = abs(a_real)
    b_std    = float(ls[2]) if len(ls) > 2 else 0.0
    b_var    = b_std ** 2   # OpenCV b std² → 최대 64²=4096

    a_std    = float(ls[1]) if len(ls) > 1 else 0.0
    chroma_c = (a_std / (a_abs + 1e-6)) if a_abs > 0.5 else 0.0

    L_std    = float(ls[0]) if len(ls) > 0 else 0.0
    lum_c    = (L_std / (L_real + 1e-6)) if L_real > 1.0 else 0.0

    raw = [
        L_real  / 100.0,                    # L 정규화 (0~1)
        a_real  / 128.0,                    # a 중심화+정규화
        b_real  / 128.0,                    # b 중심화+정규화
        L_std   / 50.0,                     # L std
        a_std   / 64.0,                     # a std
        b_std   / 64.0,                     # b std
        b_star  / 127.0,                    # b* 논문 스케일
        float(json_data.get("hsv_s_mean", 0.0)) / 100.0,
        float(json_data.get("skin_pixel_ratio", 0.0)),
        float(json_data.get("corrected", False)),
        # ── 추가 피처 ──────────────────────────────────────
        max(-1.0, min(1.0, ita / 90.0)),    # ITA 정규화 (-1~1)
        min(1.0, a_abs / 128.0),            # a 절댓값 정규화
        min(1.0, b_var / 4096.0),           # b 분산 정규화 (64²=4096)
        min(1.0, chroma_c / 5.0),           # 채도 대비
        min(1.0, lum_c),                    # 밝기 대비
        min(1.0, abs(b_delta) / 50.0),      # b* 보정 강도
    ]
    return torch.tensor(raw, dtype=torch.float32)


class PersonalColorDataset(Dataset):
    """
    JSON 모드와 폴더 모드를 자동 감지합니다.

    JSON 모드 (전처리 결과 폴더):
        root/preprocess_meta.json  ← 존재하면 JSON 모드
        root/Spring/Warm/img.jpg   ← 보정된 이미지
        root/Spring/Warm/img.json  ← 피처 JSON

        __getitem__ 반환: (image_tensor, label, feature_tensor)

    폴더 모드 (원본 데이터셋):
        root/Spring/Warm/img.jpg
        __getitem__ 반환: (image_tensor, label)
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir   = Path(root_dir)
        self.transform  = transform
        self.samples:      List[Tuple[str, int]] = []  # (img_path, label)
        self.json_map:     Dict[str, dict]       = {}  # img_path → json_data
        self.classes:      List[str]             = []
        self.class_to_idx: Dict[str, int]        = {}
        self.use_json      = False   # JSON 모드 여부

        self._scan()

    # ── 스캔 ─────────────────────────────────────────────────
    def _scan(self):
        meta_path = self.root_dir / "preprocess_meta.json"
        if meta_path.exists():
            self._scan_json(meta_path)
        else:
            self._scan_folder()

    def _scan_json(self, meta_path: Path):
        """preprocess_meta.json 기반 로드"""
        self.use_json = True
        try:
            summary = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"preprocess_meta.json 읽기 실패: {e}")

        records = summary.get("records", [])
        if not records:
            raise RuntimeError("preprocess_meta.json에 records가 없습니다.")

        # 클래스 수집
        found_classes = set()
        for r in records:
            found_classes.add(r["class"])

        priority = {c: i for i, c in enumerate(SEASON_TONE_CLASSES)}
        for i, c in enumerate(SEASON_CLASSES):
            priority.setdefault(c, len(SEASON_TONE_CLASSES) + i)

        self.classes      = sorted(found_classes,
                                   key=lambda x: priority.get(x, 999))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 샘플 + JSON 데이터 로드
        self.samples  = []
        self.json_map = {}

        for r in records:
            img_rel  = r["image_path"]                    # "Spring/Warm/img.jpg"
            json_rel = r.get("json_path",
                             str(Path(img_rel).with_suffix(".json")))

            img_path  = str(self.root_dir / img_rel)
            json_path = self.root_dir / json_rel

            cls   = r["class"]
            label = self.class_to_idx.get(cls, 0)

            self.samples.append((img_path, label))

            # JSON 개별 파일 로드
            if json_path.exists():
                try:
                    jd = json.loads(json_path.read_text(encoding="utf-8"))
                    self.json_map[img_path] = jd
                except Exception:
                    pass  # JSON 읽기 실패 시 피처 없이 진행

    def _scan_folder(self):
        """기존 폴더 구조 스캔 (JSON 없음)"""
        self.use_json = False
        collected: Dict[str, List[str]] = {}

        for season_dir in self.root_dir.iterdir():
            if not season_dir.is_dir() or season_dir.name.startswith("."):
                continue
            season = _norm_season(season_dir.name)
            if season is None:
                continue

            direct = [str(p) for p in season_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
            if direct:
                collected.setdefault(season, []).extend(direct)

            for sub in season_dir.iterdir():
                if not sub.is_dir() or sub.name.startswith("."):
                    continue
                tone = _norm_tone(sub.name)
                if tone is not None:
                    cls  = f"{season}_{tone}"
                    imgs = [str(p) for p in sub.rglob("*")
                            if p.suffix.lower() in SUPPORTED_EXT]
                    if imgs:
                        collected.setdefault(cls, []).extend(imgs)
                else:
                    imgs = [str(p) for p in sub.rglob("*")
                            if p.suffix.lower() in SUPPORTED_EXT]
                    if imgs:
                        collected.setdefault(season, []).extend(imgs)

        if not collected:
            self.classes = []
            return

        priority = {c: i for i, c in enumerate(SEASON_TONE_CLASSES)}
        for i, c in enumerate(SEASON_CLASSES):
            priority.setdefault(c, len(SEASON_TONE_CLASSES) + i)

        self.classes      = sorted(collected.keys(),
                                   key=lambda x: priority.get(x, 999))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls, paths in collected.items():
            label = self.class_to_idx[cls]
            for p in paths:
                self.samples.append((p, label))

    # ── Dataset 인터페이스 ────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            img = PILImage.open(path).convert("RGB")
        except Exception:
            img = PILImage.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        if self.use_json:
            jd      = self.json_map.get(path, {})
            feat    = _extract_features(jd)
            return img, label, feat
        else:
            return img, label

    # ── 유틸 ─────────────────────────────────────────────────
    def get_class_distribution(self) -> Dict[str, int]:
        dist = {c: 0 for c in self.classes}
        for _, lbl in self.samples:
            dist[self.classes[lbl]] += 1
        return dist

    def get_detailed_distribution(self) -> Dict[str, Dict[str, int]]:
        detail: Dict[str, Dict[str, int]] = {c: {} for c in SEASON_CLASSES}
        for cls, cnt in self.get_class_distribution().items():
            if "_" in cls:
                season, tone = cls.split("_", 1)
                if season in detail:
                    detail[season][tone] = detail[season].get(tone, 0) + cnt
            elif cls in detail:
                detail[cls]["_root"] = detail[cls].get("_root", 0) + cnt
        return detail

    def feature_dim(self) -> int:
        """JSON 피처 차원 (use_json=False면 0)"""
        return JSON_FEATURE_DIM if self.use_json else 0

    def mode_info(self) -> str:
        return f"JSON모드 (피처{JSON_FEATURE_DIM}차원)" if self.use_json \
               else "폴더모드 (피처없음)"
