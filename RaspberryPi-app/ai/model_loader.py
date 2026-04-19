"""
ai/model_loader.py
──────────────────
계층적 퍼스널 컬러 분류 모델 (Stage1: 4-class 계절, Stage2: 3-class 톤).
라즈베리파이5 CPU 추론에 최적화된 경량 MobileNetV3-Small 기반 아키텍처.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)

# ── 계절 레이블 (Stage 1, 4-class) ─────────────────────────────────────────
SEASONS = ["spring", "summer", "autumn", "winter"]

# ── 계절별 톤 매핑 (Stage 2, 3-class per season) → 퍼스널 컬러 코드 ────────
TONE_LABELS: dict[int, list[str]] = {
    0: ["spring_warm_light",  "spring_warm_bright", "spring_warm_deep"],
    1: ["summer_cool_light",  "summer_cool_muted",  "summer_cool_bright"],
    2: ["autumn_warm_deep",   "autumn_warm_muted",  "autumn_warm_light"],
    3: ["winter_cool_deep",   "winter_cool_bright", "winter_cool_muted"],
}

# 전체 12종 (인덱스 고정)
PERSONAL_COLOR_CLASSES: list[str] = [c for tones in TONE_LABELS.values() for c in tones]
NUM_CLASSES = len(PERSONAL_COLOR_CLASSES)  # 12

_META_DIM         = 6    # LAB 색상 메타: mean_L, mean_a, mean_b, std_L, std_a, std_b
_BACKBONE_FEAT    = 576  # MobileNetV3-Small avgpool 출력 채널 수


# ── 계층적 모델 아키텍처 ────────────────────────────────────────────────────
class HierarchicalPersonalColorModel(nn.Module):
    """
    2-stage 계층적 퍼스널 컬러 분류기.

    Stage 1: 이미지(224×224) + LAB 색상 메타(6-dim) → 4 계절
    Stage 2: 이미지 + 색상 메타 → 3 톤 (계절별 독립 헤드 × 4)

    최종 결과: season_idx + tone_idx → TONE_LABELS[season_idx][tone_idx]
    """

    def __init__(self) -> None:
        super().__init__()

        # 공유 백본 (MobileNetV3-Small feature extractor)
        backbone        = models.mobilenet_v3_small(weights=None)
        self.features   = backbone.features   # (B,3,224,224) → (B,576,7,7)
        self.avgpool    = backbone.avgpool     # (B,576,7,7)   → (B,576,1,1)

        combined = _BACKBONE_FEAT + _META_DIM  # 582

        # Stage 1 헤드: 4 계절 분류
        self.stage1_head = nn.Sequential(
            nn.Linear(combined, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )

        # Stage 2 헤드: 계절당 3 톤 분류 (4개 독립 헤드)
        self.stage2_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined, 64),
                nn.Hardswish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )
            for _ in range(4)
        ])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """이미지 텐서 → 576-dim 특징 벡터."""
        feat = self.features(x)               # (B,576,7,7)
        feat = self.avgpool(feat)             # (B,576,1,1)
        return feat.flatten(1)               # (B,576)

    def forward(
        self,
        x: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x    : (B, 3, 224, 224)
        meta : (B, 6)  LAB 색상 메타 — None 이면 0으로 패딩

        Returns
        -------
        stage1_logits : (B, 4)
        stage2_logits : (B, 4, 3)
        """
        feat = self.extract_features(x)       # (B, 576)

        if meta is not None:
            combined = torch.cat([feat, meta], dim=1)         # (B, 582)
        else:
            pad      = torch.zeros(feat.size(0), _META_DIM,
                                   device=feat.device, dtype=feat.dtype)
            combined = torch.cat([feat, pad], dim=1)

        s1 = self.stage1_head(combined)                       # (B, 4)
        s2 = torch.stack(
            [head(combined) for head in self.stage2_heads],
            dim=1,
        )                                                     # (B, 4, 3)
        return s1, s2


# ── 싱글톤 모델 로더 ────────────────────────────────────────────────────────
class ModelLoader:
    """Thread-safe 싱글톤 — 계층적 모델을 앱 전체에서 한 번만 로드."""

    _instance: Optional["ModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelLoader":
        with cls._lock:
            if cls._instance is None:
                inst              = super().__new__(cls)
                inst._model       = None
                inst._loaded      = False
                cls._instance     = inst
        return cls._instance

    # ------------------------------------------------------------------ #
    def load(self, model_path: str = "ai/models/hierarchical.pt") -> None:
        """
        hierarchical.pt 에서 가중치를 로드한다.
        파일이 없으면 랜덤 초기화 더미 모델로 대체 (개발 환경용).
        """
        if self._loaded:
            logger.info("Model already loaded — skipping.")
            return

        model = HierarchicalPersonalColorModel()
        pt    = Path(model_path)

        if pt.exists():
            try:
                # weights_only=True: CVE-2025-32434 패치 (torch.load pickle RCE 차단)
                # torch 2.6.0+ 에서 weights_only=True가 완전히 안전함
                state = torch.load(str(pt), map_location="cpu", weights_only=True)

                # state_dict 만 저장된 경우
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]

                if isinstance(state, dict):
                    model.load_state_dict(state, strict=False)
                    logger.info(f"Loaded state_dict from {pt}")
                else:
                    logger.warning("Unknown checkpoint format — using random-init.")

            except Exception as e:
                logger.warning(f"Weight load failed ({e}) — using random-init model.")
        else:
            logger.warning(
                f"'{pt}' not found — using random-initialized model for development."
            )

        model.eval()
        self._model  = model
        self._loaded = True
        logger.info("HierarchicalPersonalColorModel ready for inference.")

    # ------------------------------------------------------------------ #
    @property
    def model(self) -> HierarchicalPersonalColorModel:
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call ModelLoader().load() first.")
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._loaded
