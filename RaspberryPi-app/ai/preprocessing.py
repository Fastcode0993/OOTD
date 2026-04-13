"""
ai/preprocessing.py
────────────────────
MediaPipe Face Mesh를 사용해 얼굴을 검출하고,
랜드마크 기반으로 얼굴 영역을 Crop → Resize → 정규화하여
PyTorch 텐서로 반환한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── 모델 입력 사이즈 ───────────────────────────────────────────────────────
TARGET_SIZE = (224, 224)

# ImageNet 정규화 (MobileNetV3 사전학습 기준)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


@dataclass
class FaceDetectionResult:
    success: bool
    face_crop: Optional[np.ndarray] = None   # BGR uint8, shape (H, W, 3)
    tensor: Optional[torch.Tensor] = None    # shape (1, 3, 224, 224)
    bbox: Optional[Tuple[int,int,int,int]] = None  # x, y, w, h (원본 좌표)
    message: str = ""


class FacePreprocessor:
    """
    싱글톤 패턴으로 MediaPipe 세션을 재사용하여
    라즈베리파이5 CPU 부하를 최소화.
    """

    _instance: Optional["FacePreprocessor"] = None

    def __new__(cls) -> "FacePreprocessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_mediapipe()
        return cls._instance

    # ------------------------------------------------------------------ #
    def _init_mediapipe(self) -> None:
        mp_face_mesh = mp.solutions.face_mesh
        # 정적 이미지 모드: 단일 프레임 처리에 최적
        self._face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,   # False → 속도 우선
            min_detection_confidence=0.6,
        )
        logger.info("MediaPipe FaceMesh initialized.")

    # ------------------------------------------------------------------ #
    def process(self, bgr_frame: np.ndarray) -> FaceDetectionResult:
        """
        BGR 이미지(OpenCV)를 받아 전처리된 텐서를 반환.

        Parameters
        ----------
        bgr_frame : np.ndarray   shape (H, W, 3)  dtype uint8

        Returns
        -------
        FaceDetectionResult
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return FaceDetectionResult(success=False, message="빈 프레임")

        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return FaceDetectionResult(success=False, message="얼굴을 찾을 수 없습니다.")

        landmarks = results.multi_face_landmarks[0].landmark

        # ── 랜드마크 → 픽셀 좌표 ─────────────────────────────────────────
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        # 얼굴 영역에 여백(20%) 추가 → 이마·턱 포함
        pad_x = int((x_max - x_min) * 0.20)
        pad_y = int((y_max - y_min) * 0.25)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        face_crop = bgr_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return FaceDetectionResult(success=False, message="얼굴 크롭 실패")

        # ── 피부 영역 강조: LAB 색공간에서 A 채널 피부 마스크 ──────────
        face_crop = _enhance_skin_region(face_crop)

        # ── BGR → RGB → Tensor ──────────────────────────────────────────
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = _transform(face_rgb).unsqueeze(0)  # (1, 3, 224, 224)

        return FaceDetectionResult(
            success=True,
            face_crop=face_crop,
            tensor=tensor,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            message="OK",
        )

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._face_mesh.close()
        logger.info("MediaPipe FaceMesh closed.")


# ── 헬퍼 ──────────────────────────────────────────────────────────────────
def _enhance_skin_region(bgr: np.ndarray) -> np.ndarray:
    """
    LAB 색공간에서 피부 영역을 살짝 밝기 보정하여
    조명 편차를 줄인다 (CLAHE 적용).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
