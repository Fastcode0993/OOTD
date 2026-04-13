"""
ai/model_loader.py
──────────────────
PyTorch .pt 모델을 메모리에 한 번만 로드하고 싱글톤으로 관리.
라즈베리파이5(CPU 전용)에서 메모리 낭비 없이 재사용.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)

# 지원하는 퍼스널 컬러 클래스 (12종)
PERSONAL_COLOR_CLASSES = [
    "spring_warm_light",
    "spring_warm_bright",
    "spring_warm_deep",
    "summer_cool_light",
    "summer_cool_muted",
    "summer_cool_bright",
    "autumn_warm_deep",
    "autumn_warm_muted",
    "autumn_warm_light",
    "winter_cool_deep",
    "winter_cool_bright",
    "winter_cool_muted",
]

NUM_CLASSES = len(PERSONAL_COLOR_CLASSES)


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    MobileNetV3-Small 백본 기반 분류 모델 빌드.
    라즈베리파이5 CPU에서 경량 추론에 최적화.
    실제 학습된 .pt 가중치가 없을 경우 이 구조로 더미 추론 가능.
    """
    backbone = models.mobilenet_v3_small(weights=None)
    # 마지막 분류 헤드를 num_classes 로 교체
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    return backbone


class ModelLoader:
    """Thread-safe 싱글톤 모델 로더."""

    _instance: Optional["ModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelLoader":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._model = None
                cls._instance._loaded = False
        return cls._instance

    # ------------------------------------------------------------------ #
    def load(self, model_path: str = "ai/models/personal_color.pt") -> None:
        """
        .pt 파일에서 가중치를 로드한다.
        파일이 없으면 더미(랜덤 초기화) 모델로 대체해 개발 환경에서도 동작.
        """
        if self._loaded:
            logger.info("Model already loaded — skipping.")
            return

        device = torch.device("cpu")  # 라즈베리파이5 는 CPU 전용
        model = build_model()

        pt_path = Path(model_path)
        if pt_path.exists():
            try:
                state = torch.load(str(pt_path), map_location=device)
                # state_dict 만 저장된 경우와 전체 모델이 저장된 경우 모두 처리
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if isinstance(state, dict):
                    model.load_state_dict(state, strict=True)
                    logger.info(f"Loaded state_dict from {pt_path}")
                else:
                    model = state  # 전체 모델 객체로 저장된 경우
                    logger.info(f"Loaded full model object from {pt_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}. Using random-init model.")
        else:
            logger.warning(
                f"Model file not found at '{pt_path}'. "
                "Using random-initialized model for development."
            )

        model.eval()
        # TorchScript로 변환 → 추론 속도 개선 (라즈베리파이5 최적화)
        try:
            model = torch.jit.script(model)
            logger.info("Model compiled with TorchScript.")
        except Exception as e:
            logger.warning(f"TorchScript compilation failed ({e}), using eager mode.")

        self._model = model
        self._loaded = True
        logger.info("Model ready for inference.")

    # ------------------------------------------------------------------ #
    @property
    def model(self) -> nn.Module:
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call ModelLoader().load() first.")
        return self._model

    @property
    def classes(self) -> list[str]:
        return PERSONAL_COLOR_CLASSES

    @property
    def is_loaded(self) -> bool:
        return self._loaded
