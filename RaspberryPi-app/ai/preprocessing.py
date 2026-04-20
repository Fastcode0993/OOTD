"""
ai/preprocessing.py
────────────────────
OpenCV Haarcascade로 얼굴 검출 → 크롭 →
EfficientNet-B2 입력 텐서(260×260) + 16차원 피처 반환.

16차원 피처는 OOTD/datasets.py 의 _extract_features 와
완전히 동일한 정규화 기준으로 계산한다.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# EfficientNet-B2 입력 크기 (학습: Resize(300)→CenterCrop(260), 추론: Resize(260))
TARGET_SIZE    = (260, 260)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

FEAT_DIM = 16  # 학습 시 feat_dim=16 과 일치


@dataclass
class FaceDetectionResult:
    success:     bool
    face_crop:   Optional[np.ndarray]   = None   # BGR uint8
    tensor:      Optional[torch.Tensor] = None   # (1,3,260,260)
    meta_tensor: Optional[torch.Tensor] = None   # (1,16)
    bbox:        Optional[Tuple[int,int,int,int]] = None
    message:     str = ""


class FacePreprocessor:
    """싱글톤 OpenCV 얼굴 검출기. RPi5 CPU 부하 최소화."""

    _instance: Optional["FacePreprocessor"] = None

    def __new__(cls) -> "FacePreprocessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_detector()
        return cls._instance

    def _init_detector(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(f"Haarcascade 로드 실패: {cascade_path}")
        logger.info("OpenCV Haarcascade face detector initialized.")

    def process(self, bgr_frame: np.ndarray) -> FaceDetectionResult:
        """BGR 이미지 → 얼굴 텐서(260×260) + 16차원 피처 텐서 반환."""
        if bgr_frame is None or bgr_frame.size == 0:
            return FaceDetectionResult(success=False, message="빈 프레임")

        h, w = bgr_frame.shape[:2]
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60),
        )

        if len(faces) == 0:
            return FaceDetectionResult(success=False, message="얼굴을 찾을 수 없습니다.")

        # 가장 큰 얼굴 선택
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)

        face_crop = bgr_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return FaceDetectionResult(success=False, message="얼굴 크롭 실패")

        # 16-dim 피처 추출
        meta_tensor = extract_16dim_features(face_crop)   # (1,16)

        # BGR → RGB → 260×260 정규화 텐서
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor   = _transform(face_rgb).unsqueeze(0)      # (1,3,260,260)

        return FaceDetectionResult(
            success=True,
            face_crop=face_crop,
            tensor=tensor,
            meta_tensor=meta_tensor,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            message="OK",
        )

    def close(self) -> None:
        logger.info("FacePreprocessor closed.")


def extract_16dim_features(bgr_crop: np.ndarray) -> torch.Tensor:
    """
    얼굴 BGR 크롭 → 16차원 피처 텐서 (1, 16).

    OOTD/datasets.py _extract_features 와 동일한 정규화:
      OpenCV LAB 스케일 L: 0~255, a/b: 0~255 (중심 128=0)
      → 실제 스케일: L_real=L/255*100, a_real=a-128, b_real=b-128
    """
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    # OpenCV 스케일 평균/표준편차
    L_cv  = float(lab[:, :, 0].mean())
    a_cv  = float(lab[:, :, 1].mean())
    b_cv  = float(lab[:, :, 2].mean())
    L_std = float(lab[:, :, 0].std())
    a_std = float(lab[:, :, 1].std())
    b_std = float(lab[:, :, 2].std())

    # 실제 Lab 스케일 변환
    L_real = L_cv / 255.0 * 100.0   # 0~100
    a_real = a_cv - 128.0            # -128~127
    b_real = b_cv - 128.0            # -128~127

    # b* (논문 스케일 = 실제 b*)
    b_star = b_real

    # ITA 각도 (Individual Typology Angle)
    try:
        ita = math.degrees(math.atan2(L_real - 50.0, b_real)) if b_real != 0.0 else 0.0
    except Exception:
        ita = 0.0

    # HSV 채도 평균 (0~100 스케일로 환산)
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_s_mean = float(hsv[:, :, 1].mean()) / 255.0 * 100.0

    # 파생 피처
    a_abs    = abs(a_real)
    b_var    = b_std ** 2
    chroma_c = (a_std / (a_abs + 1e-6)) if a_abs > 0.5 else 0.0
    lum_c    = (L_std / (L_real + 1e-6)) if L_real > 1.0 else 0.0

    raw = [
        L_real  / 100.0,                          # lab_mean_L      (0~1)
        a_real  / 128.0,                          # lab_mean_a      (-1~1)
        b_real  / 128.0,                          # lab_mean_b      (-1~1)
        L_std   / 50.0,                           # lab_std_L
        a_std   / 64.0,                           # lab_std_a
        b_std   / 64.0,                           # lab_std_b
        max(-1.0, min(1.0, b_star / 127.0)),      # b_star
        min(1.0, hsv_s_mean / 100.0),             # hsv_s_mean
        1.0,                                       # skin_pixel_ratio (crop = face)
        0.0,                                       # corrected        (no correction)
        max(-1.0, min(1.0, ita / 90.0)),          # ita_angle
        min(1.0, a_abs / 128.0),                  # a_abs
        min(1.0, b_var / 4096.0),                 # b_variance       (64²=4096)
        min(1.0, chroma_c / 5.0),                 # chroma_contrast
        min(1.0, lum_c),                          # luminance_contrast
        0.0,                                       # b_delta_abs      (no correction)
    ]

    return torch.tensor([raw], dtype=torch.float32)  # (1,16)
