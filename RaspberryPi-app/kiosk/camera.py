"""
kiosk/camera.py
────────────────
OpenCV로 USB UVC 카메라(U30CAM 4K IMX415)를 제어하고
QImage 시그널을 발행하는 백그라운드 스레드.

• 캡처 해상도: 1920×1080 (4K 원본에서 다운샘플 → CPU 부하 절감)
• 미리보기 해상도: UI 위젯 크기에 맞춰 스케일
• 스냅샷: 촬영 시 고해상도 프레임 1장 캡처
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt6.QtGui import QImage

logger = logging.getLogger(__name__)

CAPTURE_W = 1920
CAPTURE_H = 1080
FPS_TARGET = 30


class CameraThread(QThread):
    """
    백그라운드에서 카메라 프레임을 읽어 QImage 시그널로 전달.
    메인 스레드를 블록하지 않음.
    """

    frame_ready   = pyqtSignal(QImage)    # 미리보기 프레임
    error_occurred = pyqtSignal(str)      # 오류 메시지

    def __init__(self, camera_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._mutex   = QMutex()
        self._snapshot_requested = False
        self._snapshot_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            # V4L2 실패 시 기본 백엔드로 재시도
            self._cap = cv2.VideoCapture(self._camera_index)

        if not self._cap.isOpened():
            self.error_occurred.emit(f"카메라({self._camera_index})를 열 수 없습니다.")
            return

        # 카메라 설정
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        self._cap.set(cv2.CAP_PROP_FPS,          FPS_TARGET)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)  # 버퍼 최소화 → 지연 감소

        self._running = True
        logger.info(
            f"Camera opened: {int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
            f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"@ {int(self._cap.get(cv2.CAP_PROP_FPS))}fps"
        )

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # ── 스냅샷 요청 처리 ────────────────────────────────────────
            with QMutexLocker(self._mutex):
                if self._snapshot_requested:
                    self._snapshot_frame = frame.copy()
                    self._snapshot_requested = False

            # ── 미리보기 프레임 변환 ────────────────────────────────────
            # 미리보기는 절반 해상도(960×540)로 다운샘플 → CPU 절감
            preview = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

            h, w, ch = preview_rgb.shape
            qimg = QImage(
                preview_rgb.data, w, h,
                w * ch,
                QImage.Format.Format_RGB888,
            )
            self.frame_ready.emit(qimg.copy())

        if self._cap:
            self._cap.release()
            logger.info("Camera released.")

    # ------------------------------------------------------------------ #
    def request_snapshot(self) -> None:
        """촬영 버튼 클릭 시 호출 — 다음 프레임을 스냅샷으로 저장."""
        with QMutexLocker(self._mutex):
            self._snapshot_requested = True

    def take_snapshot(self, timeout_ms: int = 2000) -> Optional[np.ndarray]:
        """
        스냅샷이 준비될 때까지 대기 후 반환.
        BGR numpy 배열 (원본 해상도).
        """
        self.request_snapshot()
        t0 = time.time()
        while time.time() - t0 < timeout_ms / 1000:
            with QMutexLocker(self._mutex):
                if self._snapshot_frame is not None:
                    frame = self._snapshot_frame
                    self._snapshot_frame = None
                    return frame
            time.sleep(0.033)
        logger.warning("Snapshot timeout.")
        return None

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        self._running = False
        self.wait(3000)
