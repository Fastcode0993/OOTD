"""
kiosk/ui.py
────────────
PyQt6 메인 윈도우 및 화면 전환 관리자.
QStackedWidget으로 6개 화면을 전환.
1280×800 해상도에 최적화.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional

import cv2
import httpx
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QStackedWidget

from kiosk.camera import CameraThread
from kiosk.screens.idle_screen           import IdleScreen
from kiosk.screens.guide_screen          import GuideScreen
from kiosk.screens.analysis_screen       import AnalysisScreen
from kiosk.screens.result_screen         import ResultScreen
from kiosk.screens.qr_screen             import QRScreen

logger = logging.getLogger(__name__)

API_BASE = os.getenv("KIOSK_LOCAL_API_BASE", "http://127.0.0.1:8000/api/v1")
API_KEY  = os.getenv("KIOSK_API_KEY", "kiosk-dev-key-2024")


# ── 백그라운드 API 워커 ───────────────────────────────────────────────────
class _AnalyzeWorker(QThread):
    """FastAPI /analyze 를 별도 스레드에서 호출."""

    finished = pyqtSignal(dict)
    failed   = pyqtSignal(str)

    def __init__(self, bgr_frame: np.ndarray, session_id: str) -> None:
        super().__init__()
        self._frame      = bgr_frame
        self._session_id = session_id

    def run(self) -> None:
        try:
            ok, buf = cv2.imencode(".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                self.failed.emit("이미지 인코딩 실패")
                return

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{API_BASE}/analyze",
                    headers={"x-api-key": API_KEY},
                    files={"image_file": ("face.jpg", buf.tobytes(), "image/jpeg")},
                    data={"session_id": self._session_id},
                )
            if resp.status_code == 200:
                self.finished.emit(resp.json())
            else:
                try:
                    body = resp.json()
                    detail = body.get("message") or body.get("detail") or resp.text
                except Exception:
                    detail = resp.text
                self.failed.emit(f"서버 오류: {resp.status_code} - {detail}")
        except Exception as e:
            self.failed.emit(str(e))


# ── 메인 윈도우 ───────────────────────────────────────────────────────────
class KioskWindow(QMainWindow):
    """전체화면 키오스크 메인 윈도우 (스크린 해상도 자동 감지)."""

    _IDX_IDLE     = 0
    _IDX_GUIDE    = 1
    _IDX_ANALYSIS = 2
    _IDX_RESULT   = 3
    _IDX_QR       = 4

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Personal Color Kiosk")
        # 실제 스크린 해상도 기반 최소 크기 설정 (setFixedSize 제거)
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            self.setMinimumSize(geom.width(), geom.height())
        self.setCursor(Qt.CursorShape.BlankCursor)

        self._session_id:     str              = ""
        self._result_data:    Dict[str, Any]   = {}
        self._snapshot:       Optional[np.ndarray] = None
        self._analyze_worker: Optional[_AnalyzeWorker] = None

        self._build_stack()
        self._init_camera()
        self._goto_idle()

    # ------------------------------------------------------------------ #
    def _build_stack(self) -> None:
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._idle     = IdleScreen()
        self._guide    = GuideScreen()
        self._analysis = AnalysisScreen()
        self._result   = ResultScreen()
        self._qr       = QRScreen()

        self._stack.addWidget(self._idle)       # 0
        self._stack.addWidget(self._guide)      # 1
        self._stack.addWidget(self._analysis)   # 2
        self._stack.addWidget(self._result)     # 3
        self._stack.addWidget(self._qr)         # 4

        self._idle.start_requested.connect(self._goto_guide)

        self._guide.capture_requested.connect(self._on_capture)
        self._guide.auto_capture_requested.connect(self._on_capture)
        self._guide.back_requested.connect(self._goto_idle)

        self._result.next_requested.connect(self._goto_qr)
        self._result.retry_requested.connect(self._goto_guide)

        self._qr.home_requested.connect(self._goto_idle)

    def _init_camera(self) -> None:
        self._cam = CameraThread(camera_index=0)
        self._cam.frame_ready.connect(self._guide.update_frame)
        self._cam.face_detected.connect(self._guide.set_face_detected)
        self._cam.error_occurred.connect(self._on_camera_error)
        self._cam.start()

    # ── 화면 전환 ─────────────────────────────────────────────────────────
    def _goto_idle(self) -> None:
        self._session_id = str(uuid.uuid4())
        self._stack.setCurrentIndex(self._IDX_IDLE)

    def _goto_guide(self) -> None:
        self._guide.reset()
        self._stack.setCurrentIndex(self._IDX_GUIDE)

    def _goto_analysis(self) -> None:
        self._analysis.start()
        self._stack.setCurrentIndex(self._IDX_ANALYSIS)

    def _goto_result(self) -> None:
        self._analysis.stop()
        if self._result_data:
            self._result.set_result(self._result_data)
        self._stack.setCurrentIndex(self._IDX_RESULT)

    def _goto_qr(self) -> None:
        # 클라우드 연동 QR URL 또는 로컬 세션 URL 사용
        qr_url = self._result_data.get("qr_url") or None
        self._qr.set_session(
            session_id=self._session_id,
            label_ko=self._result_data.get("label_ko", ""),
            qr_url=qr_url,
        )
        self._stack.setCurrentIndex(self._IDX_QR)

    def _goto_result_direct(self) -> None:
        self._stack.setCurrentIndex(self._IDX_RESULT)

    # ── 이벤트 핸들러 ────────────────────────────────────────────────────
    def _on_capture(self) -> None:
        snapshot = self._cam.take_snapshot()
        if snapshot is None:
            self._guide.reset()
            return

        self._snapshot = snapshot

        h, w = snapshot.shape[:2]
        rgb  = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._analysis.set_face_pixmap(QPixmap.fromImage(qimg))

        self._goto_analysis()
        self._start_analyze()

    def _start_analyze(self) -> None:
        self._analyze_worker = _AnalyzeWorker(self._snapshot, self._session_id)
        self._analyze_worker.finished.connect(self._on_analyze_finished)
        self._analyze_worker.failed.connect(self._on_analyze_failed)
        self._analyze_worker.start()

    def _on_analyze_finished(self, data: Dict[str, Any]) -> None:
        self._result_data = data
        self._goto_result()

    def _on_analyze_failed(self, msg: str) -> None:
        logger.error(f"Analyze failed: {msg}")
        self._analysis.stop()
        self._goto_guide()

    def _on_camera_error(self, msg: str) -> None:
        logger.error(f"Camera error: {msg}")

    def closeEvent(self, event) -> None:
        self._cam.stop()
        if self._analyze_worker and self._analyze_worker.isRunning():
            self._analyze_worker.quit()
            self._analyze_worker.wait(2000)
        super().closeEvent(event)
