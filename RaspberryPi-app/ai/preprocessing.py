"""
ai/preprocessing.py
────────────────────
MediaPipe FaceMesh로 얼굴을 검출하고,
랜드마크 기반 Crop → CLAHE 보정 → 정규화 → 텐서 변환.
LAB 색상 메타데이터(6-dim)를 함께 추출하여 계층적 모델에 제공.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

TARGET_SIZE     = (224, 224)
_IMAGENET_MEAN  = [0.485, 0.456, 0.406]
_IMAGENET_STD   = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


@dataclass
class FaceDetectionResult:
    success:     bool
    face_crop:   Optional[np.ndarray]   = None   # BGR uint8 (H,W,3)
    tensor:      Optional[torch.Tensor] = None   # (1,3,224,224)
    meta_tensor: Optional[torch.Tensor] = None   # (1,6) LAB 색상 메타
    bbox:        Optional[Tuple[int,int,int,int]] = None  # x,y,w,h
    message:     str = ""


class FacePreprocessor:
    """
    싱글톤 패턴으로 MediaPipe 세션을 재사용.
    라즈베리파이5 CPU 부하 최소화.
    """

    _instance: Optional["FacePreprocessor"] = None

    def __new__(cls) -> "FacePreprocessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_mediapipe()
        return cls._instance

    def _init_mediapipe(self) -> None:
        mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
        )
        logger.info("MediaPipe FaceMesh initialized.")

    # ------------------------------------------------------------------ #
    def process(self, bgr_frame: np.ndarray) -> FaceDetectionResult:
        """
        BGR 이미지 → 얼굴 크롭 텐서 + LAB 색상 메타 텐서 반환.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return FaceDetectionResult(success=False, message="빈 프레임")

        h, w     = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results   = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return FaceDetectionResult(success=False, message="얼굴을 찾을 수 없습니다.")

        landmarks = results.multi_face_landmarks[0].landmark
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        pad_x = int((x_max - x_min) * 0.20)
        pad_y = int((y_max - y_min) * 0.25)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        face_crop = bgr_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return FaceDetectionResult(success=False, message="얼굴 크롭 실패")

        # CLAHE 조명 보정
        face_crop = _enhance_skin_region(face_crop)

        # LAB 색상 메타 추출 (6-dim)
        meta_tensor = _extract_skin_meta(face_crop)  # (1,6)

        # BGR → RGB → 정규화 텐서
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor   = _transform(face_rgb).unsqueeze(0)  # (1,3,224,224)

        return FaceDetectionResult(
            success=True,
            face_crop=face_crop,
            tensor=tensor,
            meta_tensor=meta_tensor,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            message="OK",
        )

    def close(self) -> None:
        self._face_mesh.close()
        logger.info("MediaPipe FaceMesh closed.")


# ── 헬퍼 ──────────────────────────────────────────────────────────────────
def _enhance_skin_region(bgr: np.ndarray) -> np.ndarray:
    """CLAHE로 조명 편차를 줄인다 (LAB L채널 적용)."""
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_ch  = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _extract_skin_meta(bgr_crop: np.ndarray) -> torch.Tensor:
    """
    얼굴 크롭에서 LAB 채널의 mean/std 6개 값을 추출.
    피부 톤 계절/웜쿨 분류에 직접적으로 유용한 특징:
      - mean_L : 밝기 평균 (라이트/딥 구분)
      - mean_a : 적-녹 축 평균 (웜/쿨 구분 핵심)
      - mean_b : 황-청 축 평균 (웜/쿨 보조)
      - std_L/a/b : 각 채널 표준편차 (대비 정보)

    Returns
    -------
    torch.Tensor  shape (1, 6)  float32, 0~1 정규화
    """
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    mean_L = float(lab[:, :, 0].mean()) / 255.0
    mean_a = float(lab[:, :, 1].mean()) / 255.0
    mean_b = float(lab[:, :, 2].mean()) / 255.0
    std_L  = float(lab[:, :, 0].std())  / 255.0
    std_a  = float(lab[:, :, 1].std())  / 255.0
    std_b  = float(lab[:, :, 2].std())  / 255.0

    meta = torch.tensor(
        [[mean_L, mean_a, mean_b, std_L, std_a, std_b]],
        dtype=torch.float32,
    )
    return meta
