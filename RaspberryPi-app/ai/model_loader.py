"""
ai/model_loader.py
──────────────────
final_hierarchical.pt 로딩 및 EfficientNet-B2 FusionModel 관리.

저장 포맷 (OOTD/train_hierarchical.py 출력):
  stage1_state        : EfficientNet-B2 + 16-dim head → 4계절
  stage2_{Season}_state : EfficientNet-B2 + 16-dim head → 3톤 (계절별 독립)
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

# ── 클래스 정의 ──────────────────────────────────────────────────────────────
SEASON_CLASSES = ["Spring", "Summer", "Autumn", "Winter"]
TONE_CLASSES   = ["Warm", "Bright", "Light"]

# 계절별 웜/쿨 구분 (봄·가을=웜톤, 여름·겨울=쿨톤)
UNDERTONE_MAP: Dict[str, str] = {
    "Spring": "웜톤",
    "Summer": "쿨톤",
    "Autumn": "웜톤",
    "Winter": "쿨톤",
}

SEASON_KO: Dict[str, str] = {
    "Spring": "봄",
    "Summer": "여름",
    "Autumn": "가을",
    "Winter": "겨울",
}

TONE_KO: Dict[str, str] = {
    "Warm":   "웜",
    "Bright": "브라이트",
    "Light":  "라이트",
}

# EfficientNet-B2 특징 차원 / 피처 차원
_EFF_B2_CNN_DIM = 1408
_FEAT_DIM       = 16   # 학습 feat_dim=16

# 12클래스 한국어/영어 레이블
_LABEL_MAP: Dict[str, Dict[str, str]] = {
    "Spring_Warm":   {"ko": "봄 웜",          "en": "Spring Warm"},
    "Spring_Bright": {"ko": "봄 브라이트",     "en": "Spring Bright"},
    "Spring_Light":  {"ko": "봄 라이트",       "en": "Spring Light"},
    "Summer_Warm":   {"ko": "여름 웜",         "en": "Summer Warm"},
    "Summer_Bright": {"ko": "여름 브라이트",   "en": "Summer Bright"},
    "Summer_Light":  {"ko": "여름 라이트",     "en": "Summer Light"},
    "Autumn_Warm":   {"ko": "가을 웜",         "en": "Autumn Warm"},
    "Autumn_Bright": {"ko": "가을 브라이트",   "en": "Autumn Bright"},
    "Autumn_Light":  {"ko": "가을 라이트",     "en": "Autumn Light"},
    "Winter_Warm":   {"ko": "겨울 웜",         "en": "Winter Warm"},
    "Winter_Bright": {"ko": "겨울 브라이트",   "en": "Winter Bright"},
    "Winter_Light":  {"ko": "겨울 라이트",     "en": "Winter Light"},
}

# 12클래스 설명 (한국어)
_DESCRIPTION_KO: Dict[str, str] = {
    "Spring_Warm":   "선명하고 따뜻한 봄 웜톤. 밝고 생기있는 살구·코랄·황금빛이 잘 어울립니다.",
    "Spring_Bright": "화사하고 투명한 봄 브라이트. 선명한 원색과 형광계열 컬러가 생기를 더합니다.",
    "Spring_Light":  "부드럽고 밝은 봄 라이트. 파스텔·아이보리·민트 등 연한 색조가 잘 어울립니다.",
    "Summer_Warm":   "차분하고 우아한 여름 웜톤. 로즈·모브·라벤더 계열의 은은한 색감이 잘 어울립니다.",
    "Summer_Bright": "또렷하고 시원한 여름 브라이트. 로열블루·퍼플·핫핑크 등 채도높은 쿨톤이 잘 맞습니다.",
    "Summer_Light":  "연하고 서늘한 여름 라이트. 파우더블루·라일락·실버그레이 같은 안개빛 색조가 돋보입니다.",
    "Autumn_Warm":   "깊고 풍성한 가을 웜톤. 테라코타·카멜·올리브·머스터드 등 어스톤이 자연스럽게 어울립니다.",
    "Autumn_Bright": "강렬하고 진한 가을 브라이트. 버건디·카키·딥오렌지 등 채도 높은 어스톤이 인상적입니다.",
    "Autumn_Light":  "따뜻하고 부드러운 가을 라이트. 베이지·카멜·크림 등 밝은 어스톤이 피부를 환하게 합니다.",
    "Winter_Warm":   "날카롭고 강렬한 겨울 웜톤. 와인·버건디·다크초콜릿 등 깊은 쿨-웜 경계 색이 잘 맞습니다.",
    "Winter_Bright": "대비가 선명한 겨울 브라이트. 블랙·화이트·원색의 강한 콘트라스트가 인상을 극대화합니다.",
    "Winter_Light":  "차갑고 투명한 겨울 라이트. 아이시화이트·실버·페일블루 등 냉색 계열 파스텔이 어울립니다.",
}

# 12클래스 색상 팔레트
_PALETTE: Dict[str, list] = {
    "Spring_Warm":   ["#FF9B7B", "#FFA07A", "#FFD700", "#FF8C69", "#FFDAB9"],
    "Spring_Bright": ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#FF922B"],
    "Spring_Light":  ["#FFB6C1", "#FFFACD", "#98FB98", "#E0FFFF", "#FFF0F5"],
    "Summer_Warm":   ["#D4A5A5", "#C9A0A0", "#B0A0C0", "#9E9EC0", "#A8C0C8"],
    "Summer_Bright": ["#B0C4DE", "#DDA0DD", "#E6A8D7", "#9370DB", "#6495ED"],
    "Summer_Light":  ["#B8B8D1", "#8FB8D7", "#C0C0C0", "#D8BFD8", "#E8D5E8"],
    "Autumn_Warm":   ["#B7410E", "#8B4513", "#D2691E", "#CD853F", "#A0522D"],
    "Autumn_Bright": ["#FF6600", "#CC5500", "#808000", "#556B2F", "#8B0000"],
    "Autumn_Light":  ["#DEB887", "#F5DEB3", "#D2B48C", "#C19A6B", "#BC8F5F"],
    "Winter_Warm":   ["#8B0000", "#C0392B", "#4B0082", "#2C3E50", "#800020"],
    "Winter_Bright": ["#000080", "#4B0082", "#FF0000", "#FFFFFF", "#000000"],
    "Winter_Light":  ["#E8E8E8", "#C0D8E8", "#F0F0FF", "#D8D8F0", "#C0C0C0"],
}


# ── 모델 아키텍처 빌더 ────────────────────────────────────────────────────────
def _build_fusion_model(nc: int) -> nn.Module:
    """
    학습 코드(OOTD/train_hierarchical.py _build_model)와 동일한 구조.
    EfficientNet-B2 backbone + 16-dim 피처 fusion head.
    """
    backbone = models.efficientnet_b2(weights=None)
    backbone.classifier = nn.Identity()   # 분류기 제거, 특징 벡터만 출력

    head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(_EFF_B2_CNN_DIM + _FEAT_DIM, 512), nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(512, 256), nn.ReLU(True),
        nn.Dropout(0.1),
        nn.Linear(256, nc),
    )

    class _FusionModel(nn.Module):
        def __init__(self, bb: nn.Module, hd: nn.Module) -> None:
            super().__init__()
            self.backbone = bb
            self.head     = hd

        def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None) -> torch.Tensor:
            out = self.backbone(x)                          # (B, 1408)
            if feat is not None:
                out = torch.cat([out, feat], dim=1)         # (B, 1424)
            return self.head(out)

    return _FusionModel(backbone, head)


# ── 싱글톤 모델 로더 ──────────────────────────────────────────────────────────
class ModelLoader:
    """
    Thread-safe 싱글톤.
    Stage1 모델 1개 + Stage2 계절별 모델 4개를 보관.
    """

    _instance: Optional["ModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelLoader":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._stage1: Optional[nn.Module]       = None
                inst._stage2: Dict[str, nn.Module]      = {}
                inst._loaded = False
                cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------ #
    def load(self, model_path: str = "ai/models/final_hierarchical.pt") -> None:
        if self._loaded:
            logger.info("Model already loaded — skipping.")
            return

        pt = Path(model_path)
        if not pt.exists():
            logger.warning(f"'{pt}' not found — random-init fallback (dev only).")
            self._stage1 = _build_fusion_model(4).eval()
            for s in SEASON_CLASSES:
                self._stage2[s] = _build_fusion_model(3).eval()
            self._loaded = True
            return

        ck = torch.load(str(pt), map_location="cpu", weights_only=False)

        # ── Stage1 로드 ──────────────────────────────────────────────────
        s1_state = ck.get("stage1_state")
        if s1_state is None:
            raise RuntimeError("stage1_state 키 없음. 올바른 hierarchical.pt 인지 확인.")
        m1 = _build_fusion_model(4)
        m1.load_state_dict(s1_state)
        self._stage1 = m1.eval()
        logger.info(f"Stage1 loaded — val_acc={ck.get('val_acc_stage1', 0):.2f}%")

        # ── Stage2 계절별 로드 ───────────────────────────────────────────
        val2 = ck.get("val_acc_stage2", {})
        for season in SEASON_CLASSES:
            key      = f"stage2_{season}_state"
            s2_state = ck.get(key)
            if s2_state is None:
                logger.warning(f"'{key}' 없음 — {season} Stage2 랜덤 초기화")
                self._stage2[season] = _build_fusion_model(3).eval()
            else:
                m2 = _build_fusion_model(3)
                m2.load_state_dict(s2_state)
                self._stage2[season] = m2.eval()
                logger.info(f"Stage2[{season}] loaded — val_acc={val2.get(season, 0):.2f}%")

        self._loaded = True
        logger.info("All models ready for hierarchical inference.")

    # ------------------------------------------------------------------ #
    @property
    def stage1(self) -> nn.Module:
        if not self._loaded:
            raise RuntimeError("Call ModelLoader().load() first.")
        return self._stage1

    def stage2(self, season: str) -> nn.Module:
        if not self._loaded:
            raise RuntimeError("Call ModelLoader().load() first.")
        if season not in self._stage2:
            raise KeyError(f"Unknown season: {season}")
        return self._stage2[season]

    @property
    def is_loaded(self) -> bool:
        return self._loaded
