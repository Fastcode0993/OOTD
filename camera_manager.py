"""
HMI 전용 카메라 관리 클래스 (ED-HMI3010-101C 기반)
4K IMX415 카메라 캡처 전용 (MediaPipe 로직은 서버로 이동)
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import threading
import queue


@dataclass
class CameraConfig:
    """카메라 구성 설정"""
    resolution_4k: Tuple[int, int] = (3840, 2160)
    resolution_inference: Tuple[int, int] = (224, 224)
    fps: int = 30
    exposure_auto: bool = True
    white_balance_auto: bool = True
    gain_auto: bool = True


class CameraManager:
    """
    4K 카메라 최적화 관리 클래스
    - HMI 전용: 4K 프레임 캡처만 담당
    - MediaPipe 로직은 서버로 이동
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
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
        while self._running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # 큐가 꽉 차면 버림
    
    def get_frame_4k(self) -> Optional[np.ndarray]:
        """4K 원본 프레임 반환 (서버 전송용)"""
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
    
    def resize_for_hmi(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """HMI 화면에 맞게 프레임 리사이즈"""
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    
    def __del__(self):
        self.stop()