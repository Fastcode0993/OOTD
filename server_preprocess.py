"""
서버 전처리 모듈 (MediaPipe Face Mesh)
HMI 가 보낸 이미지를 받아 얼굴 468 개 좌표와 피부색 데이터를 추출
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FaceAnalysisResult:
    """얼굴 분석 결과"""
    face_detected: bool
    landmarks: List[Tuple[int, int]]  # 468 개 좌표
    cheek_rgb: List[int]  # 양볼 평균 RGB
    cheek_lab: List[float]  # 양볼 평균 Lab
    forehead_rgb: List[int]  # 이마 평균 RGB
    forehead_lab: List[float]  # 이마 평균 Lab
    face_confidence: float


class FaceMeshProcessor:
    """
    MediaPipe Face Mesh 를 이용한 얼굴 분석
    - HMI 가 보낸 이미지를 받아 468 개 랜드마크 추출
    - 양볼과 이마의 평균 RGB/Lab 값 계산
    """
    
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._initialized = True
        except ImportError:
            self._initialized = False
    
    def analyze(self, image_bgr: np.ndarray) -> Optional[FaceAnalysisResult]:
        """
        이미지에서 얼굴 분석 수행
        
        Args:
            image_bgr: BGR 형식의 이미지 (4K 또는 원본)
            
        Returns:
            FaceAnalysisResult 또는 None
        """
        if not self._initialized:
            return None
        
        # RGB 로 변환
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb_image = np.expand_dims(rgb_image, axis=0)
        
        try:
            res = self.face_mesh.process(rgb_image)
            if res and res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                return self._extract_features(lm, image_bgr)
        except Exception:
            pass
        
        return None
    
    def _extract_features(self, lm, image_bgr: np.ndarray) -> FaceAnalysisResult:
        """랜드마크에서 특징 추출"""
        # 468 개 좌표 추출 (이미지 좌표로 변환)
        landmarks = []
        for idx in range(468):
            x = int(lm.landmark[idx].x * image_bgr.shape[1])
            y = int(lm.landmark[idx].y * image_bgr.shape[0])
            landmarks.append((x, y))
        
        # 양볼 영역 색상 계산
        cheek_rgb, cheek_lab = self._calculate_region_color(
            landmarks, 
            LEFT_CHEEK_LANDMARKS, 
            RIGHT_CHEEK_LANDMARKS,
            image_bgr
        )
        
        # 이마 영역 색상 계산
        forehead_rgb, forehead_lab = self._calculate_region_color(
            landmarks,
            FOREHEAD_LANDMARKS,
            image_bgr
        )
        
        confidence = float(lm.detection_confidence) if hasattr(lm, 'detection_confidence') else 0.95
        
        return FaceAnalysisResult(
            face_detected=True,
            landmarks=landmarks,
            cheek_rgb=cheek_rgb,
            cheek_lab=cheek_lab,
            forehead_rgb=forehead_rgb,
            forehead_lab=forehead_lab,
            face_confidence=confidence
        )
    
    def _calculate_region_color(self, landmarks: List[Tuple[int, int]], 
                                 indices: List[int], frame_bgr: np.ndarray) -> Tuple[List[int], List[float]]:
        """지정된 랜드마크 영역의 평균 RGB/Lab 값 계산"""
        if not landmarks or not indices:
            return [0, 0, 0], [0.0, 0.0, 0.0]
        
        # 마스크 생성
        pts = np.array([landmarks[i] for i in indices], dtype=np.int32)
        mask = np.zeros((landmarks[0][1], landmarks[0][0]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        
        # 이미지가 BGR 형식
        bgr_pixels = frame_bgr[mask == 255]
        
        if len(bgr_pixels) == 0:
            return [0, 0, 0], [0.0, 0.0, 0.0]
        
        # BGR -> RGB 변환
        rgb_pixels = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        
        # 평균 RGB 계산
        avg_rgb = rgb_pixels.mean(axis=0).astype(np.uint8).tolist()
        
        # Lab 색상 공간 변환
        pixels_reshaped = rgb_pixels.reshape(-1, 1, 3)
        lab = cv2.cvtColor(pixels_reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3)
        avg_lab = lab.mean(axis=0).tolist()
        
        return avg_rgb, avg_lab


# 상수 정의 (MediaPipe Face Mesh 랜드마크 인덱스)
LEFT_CHEEK_LANDMARKS: List[int] = [50, 101, 118, 117, 123, 187, 207, 206, 205, 36, 93]
RIGHT_CHEEK_LANDMARKS: List[int] = [280, 330, 347, 346, 352, 411, 427, 426, 425, 266, 323]
FACE_OVAL_LANDMARKS: List[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYE_LANDMARKS: List[int] = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_LANDMARKS: List[int] = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW_LANDMARKS: List[int] = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_LANDMARKS: List[int] = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_LANDMARKS: List[int] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
    310, 311, 312, 13, 82, 81, 42, 183, 78,
]

# 이마 영역 랜드마크 (forehead:眉心 ~발际)
FOREHEAD_LANDMARKS: List[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
]