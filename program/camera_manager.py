"""
HMI 전용 카메라 관리 클래스 (ED-HMI3010-101C 기반)
4K IMX415 카메라 최적화 및 전처리 파이프라인
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import queue

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class CameraConfig:
    """카메라 구성 설정"""
    resolution_4k: Tuple[int, int] = (3840, 2160)
    resolution_inference: Tuple[int, int] = (224, 224)
    resolution_display: Tuple[int, int] = (640, 480)  # 10.1" 화면용 Low-res
    fps: int = 30
    exposure_auto: bool = True
    white_balance_auto: bool = True
    gain_auto: bool = True


class CameraManager:
    """
    4K 카메라 최적화 관리 클래스
    - 추론용 (224x224) 과 화면 출력용 (Low-res) 으로 분리 캡처
    - HMI 성능 하락 방지
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # MediaPipe 초기화
        self._face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # HMI 디스플레이 해상도 (10.1" 터치스크린)
        self.HMI_WIDTH = 1280
        self.HMI_HEIGHT = 800
        
    def start(self) -> bool:
        """카메라 시작"""
        with self._lock:
            if self.cap is not None:
                return True
            
            self.cap = cv2.VideoCapture(0)
            
            # 4K 해상도 설정 (IMX415)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution_4k[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution_4k[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # HMI 최적화 설정
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 0 if not self.config.exposure_auto else -1)
            self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 0 if not self.config.white_balance_auto else -1)
            self.cap.set(cv2.CAP_PROP_GAIN, 0 if not self.config.gain_auto else -1)
            
            if not self.cap.isOpened():
                self.cap = None
                return False
            
            self._running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            return True
    
    def stop(self):
        """카메라 중지"""
        with self._lock:
            self._running = False
            if self._capture_thread:
                self._capture_thread.join(timeout=2.0)
                self._capture_thread = None
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def _capture_loop(self):
        """비동기 캡처 루프"""
        while self._running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # 큐가 꽉 차면 버림
    
    def get_frame_4k(self) -> Optional[np.ndarray]:
        """4K 원본 프레임 반환 (추론용)"""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_frame_inference(self) -> Optional[np.ndarray]:
        """추론용 프레임 (224x224) 반환"""
        frame = self.get_frame_4k()
        if frame is None:
            return None
        return cv2.resize(frame, self.config.resolution_inference, interpolation=cv2.INTER_AREA)
    
    def get_frame_display(self) -> Optional[np.ndarray]:
        """HMI 화면 출력용 프레임 (Low-res) 반환"""
        frame = self.get_frame_4k()
        if frame is None:
            return None
        # 10.1" 화면에 최적화된 해상도
        return cv2.resize(frame, (400, 225), interpolation=cv2.INTER_AREA)
    
    def get_frame_hmi(self) -> Optional[np.ndarray]:
        """HMI 전체 화면용 프레임 (1280x800) 반환"""
        frame = self.get_frame_4k()
        if frame is None:
            return None
        return cv2.resize(frame, (self.HMI_WIDTH, self.HMI_HEIGHT), interpolation=cv2.INTER_AREA)
    
    def analyze_face(self, frame: np.ndarray) -> Dict:
        """
        MediaPipe Face Mesh 를 이용한 얼굴 분석
        - 양볼과 이마 좌표 추출
        - 해당 구역의 평균 RGB/Lab 값 계산
        """
        result = {
            "face_detected": False,
            "landmarks": None,
            "cheek_rgb": None,
            "cheek_lab": None,
            "forehead_rgb": None,
            "forehead_lab": None,
            "face_confidence": 0.0
        }
        
        if not MEDIAPIPE_AVAILABLE or self._face_mesh is None:
            return result
        
        rgb_image = frame.copy()
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        try:
            res = self._face_mesh.process(rgb_image)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                result["face_detected"] = True
                result["face_confidence"] = float(lm.detection_confidence) if hasattr(lm, 'detection_confidence') else 0.95
                
                # 랜드마크 좌표 추출
                landmarks = []
                for idx in range(468):
                    x = int(lm.landmark[idx].x * frame.shape[1])
                    y = int(lm.landmark[idx].y * frame.shape[0])
                    landmarks.append((x, y))
                result["landmarks"] = landmarks
                
                # 양볼 영역 색상 계산
                cheek_rgb, cheek_lab = self._calculate_region_color(
                    landmarks, 
                    LEFT_CHEEK_LANDMARKS, 
                    RIGHT_CHEEK_LANDMARKS,
                    frame
                )
                result["cheek_rgb"] = cheek_rgb
                result["cheek_lab"] = cheek_lab
                
                # 이마 영역 색상 계산
                forehead_rgb, forehead_lab = self._calculate_region_color(
                    landmarks,
                    FOREHEAD_LANDMARKS,
                    frame
                )
                result["forehead_rgb"] = forehead_rgb
                result["forehead_lab"] = forehead_lab
                
        except Exception as e:
            pass
        
        return result
    
    def _calculate_region_color(self, landmarks: List[Tuple[int, int]], 
                                 indices: List[int], frame_bgr: np.ndarray) -> Tuple[List[int], List[float]]:
        """지정된 랜드마크 영역의 평균 RGB/Lab 값 계산"""
        if not landmarks or not indices:
            return [0, 0, 0], [0.0, 0.0, 0.0]
        
        # 마스크 생성
        pts = np.array([landmarks[i] for i in indices], dtype=np.int32)
        mask = np.zeros((landmarks[0][1], landmarks[0][0]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        
        # 이미지가 RGB 형식인지 확인하고 변환
        if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
            # 이미지가 RGB 형식
            rgb_pixels = frame_bgr[mask == 255]
        else:
            # BGR 형식으로 가정하고 RGB 로 변환
            rgb_pixels = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)[mask == 255]
        
        if len(rgb_pixels) == 0:
            return [0, 0, 0], [0.0, 0.0, 0.0]
        
        # 평균 RGB 계산
        avg_rgb = rgb_pixels.mean(axis=0).astype(np.uint8).tolist()
        
        # Lab 색상 공간 변환
        pixels_reshaped = rgb_pixels.reshape(-1, 1, 3)
        lab = cv2.cvtColor(pixels_reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3)
        avg_lab = lab.mean(axis=0).tolist()
        
        return avg_rgb, avg_lab
    
    def get_face_landmarks_indices(self) -> Dict[str, List[int]]:
        """얼굴 주요 부위 랜드마크 인덱스 반환"""
        return {
            "left_cheek": LEFT_CHEEK_LANDMARKS,
            "right_cheek": RIGHT_CHEEK_LANDMARKS,
            "forehead": FOREHEAD_LANDMARKS,
            "face_ovall": FACE_OVAL_LANDMARKS,
            "left_eye": LEFT_EYE_LANDMARKS,
            "right_eye": RIGHT_EYE_LANDMARKS,
            "left_eyebrow": LEFT_EYEBROW_LANDMARKS,
            "right_eyebrow": RIGHT_EYEBROW_LANDMARKS,
            "lips": LIPS_LANDMARKS,
        }
    
    def resize_for_hmi(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """HMI 화면에 맞게 프레임 리사이즈"""
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    
    def __del__(self):
        self.stop()


# 상수 정의 (constants.py 와 일관성 유지)
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