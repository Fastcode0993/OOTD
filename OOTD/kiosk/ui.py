from __future__ import annotations

import logging
import os
import socket
import uuid
from typing import Any, Dict, Optional

import cv2
import httpx
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QStackedWidget

from kiosk.camera import CameraThread
from kiosk.screens.idle_screen import IdleScreen
from kiosk.screens.guide_screen import GuideScreen
from kiosk.screens.analysis_screen import AnalysisScreen
from kiosk.screens.result_screen import ResultScreen
from kiosk.screens.recommendation_screen import RecommendationScreen
from kiosk.screens.qr_screen import QRScreen

logger = logging.getLogger(__name__)
DISPLAY_W = 1280
DISPLAY_H = 800
API_KEY = "kiosk-dev-key-2024"


class _AnalyzeWorker(QThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, bgr_frame: np.ndarray, session_id: str, api_base: str) -> None:
        super().__init__()
        self._frame = bgr_frame
        self._session_id = session_id
        self._api_base = api_base

    def run(self) -> None:
        try:
            ok, buf = cv2.imencode(".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                self.failed.emit("이미지 인코딩 실패")
                return
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self._api_base}/analyze",
                    headers={"x-api-key": API_KEY},
                    files={"image_file": ("face.jpg", buf.tobytes(), "image/jpeg")},
                    data={"session_id": self._session_id},
                )
            if resp.status_code == 200:
                self.finished.emit(resp.json())
            else:
                self.failed.emit(f"서버 오류: {resp.status_code}")
        except httpx.ConnectError:
            self.failed.emit("분석 서버 연결에 실패했습니다. 다시 시도해주세요.")
        except Exception as e:
            self.failed.emit(str(e))


class KioskWindow(QMainWindow):
    _IDX_IDLE = 0
    _IDX_GUIDE = 1
    _IDX_ANALYSIS = 2
    _IDX_RESULT = 3
    _IDX_RECO = 4
    _IDX_QR = 5

    def __init__(self, api_host: str = "127.0.0.1", api_port: int = 8000, hide_cursor: bool = True, preview_mode: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Personal Color Kiosk")
        self.setFixedSize(DISPLAY_W, DISPLAY_H)
        if hide_cursor:
            self.setCursor(Qt.CursorShape.BlankCursor)

        self._preview_mode = preview_mode
        self._api_base = f"http://{api_host}:{api_port}/api/v1"
        self._result_base_url = os.getenv("KIOSK_RESULT_BASE_URL") or f"http://{self._detect_local_ip()}:{api_port}"

        self._session_id = ""
        self._result_data: Dict[str, Any] = {}
        self._snapshot: Optional[np.ndarray] = None
        self._analyze_worker: Optional[_AnalyzeWorker] = None

        self._build_stack()
        if not self._preview_mode:
            self._init_camera()
        else:
            self._load_preview_data()
        self._goto_idle()

    def _detect_local_ip(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def _build_stack(self) -> None:
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self._idle = IdleScreen()
        self._guide = GuideScreen()
        self._analysis = AnalysisScreen()
        self._result = ResultScreen()
        self._reco = RecommendationScreen()
        self._qr = QRScreen()
        for w in [self._idle, self._guide, self._analysis, self._result, self._reco, self._qr]:
            self._stack.addWidget(w)

        self._idle.start_requested.connect(self._goto_guide)
        self._guide.capture_requested.connect(self._on_capture)
        self._guide.back_requested.connect(self._goto_idle)
        self._result.next_requested.connect(self._goto_reco)
        self._result.retry_requested.connect(self._goto_guide)
        self._reco.qr_requested.connect(self._goto_qr)
        self._reco.back_requested.connect(self._goto_result)
        self._qr.home_requested.connect(self._goto_idle)

    def _init_camera(self) -> None:
        raw_camera_index = os.getenv("KIOSK_CAMERA_INDEX", "1")
        try:
            camera_index = int(raw_camera_index)
        except ValueError:
            logger.warning(
                "Invalid KIOSK_CAMERA_INDEX=%r; falling back to camera index 1.",
                raw_camera_index,
            )
            camera_index = 1
            self._guide.set_error_message("카메라 설정값이 잘못되어 기본 카메라로 시작합니다.")

        self._cam = CameraThread(camera_index=camera_index)
        self._cam.frame_ready.connect(self._guide.update_frame)
        self._cam.error_occurred.connect(self._on_camera_error)
        self._cam.start()

    def _load_preview_data(self) -> None:
        self._result_data = {
            "label_ko": "가을 웜톤",
            "recommendations": {
                "fashion": [{"item_name_ko": "베이지 자켓", "color_name_ko": "웜 베이지", "color_hex": "#C7A27C", "tip_ko": "부드러운 톤 매치"}],
                "makeup": [], "hair": [], "interior": []
            }
        }

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

    def _goto_reco(self) -> None:
        self._reco.set_data(self._result_data.get("label_ko", ""), self._result_data.get("recommendations", {}))
        self._stack.setCurrentIndex(self._IDX_RECO)

    def _goto_qr(self) -> None:
        qr_url = self._result_data.get("qr_url") or f"{self._result_base_url}/result/{self._session_id}"
        self._qr.set_session(session_id=self._session_id, label_ko=self._result_data.get("label_ko", ""), qr_url=qr_url)
        self._stack.setCurrentIndex(self._IDX_QR)

    def _on_capture(self) -> None:
        if self._preview_mode:
            self._goto_analysis(); self._on_analyze_finished(self._result_data); return
        snapshot = self._cam.take_snapshot()
        if snapshot is None:
            self._guide.set_error_message("카메라를 열 수 없습니다. 직원에게 문의해주세요.")
            self._guide.reset(clear_status=False)
            return
        self._snapshot = snapshot
        h, w = snapshot.shape[:2]
        rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._analysis.set_face_pixmap(QPixmap.fromImage(qimg))
        self._goto_analysis(); self._start_analyze()

    def _start_analyze(self) -> None:
        self._analyze_worker = _AnalyzeWorker(self._snapshot, self._session_id, self._api_base)
        self._analyze_worker.finished.connect(self._on_analyze_finished)
        self._analyze_worker.failed.connect(self._on_analyze_failed)
        self._analyze_worker.start()

    def _on_analyze_finished(self, data: Dict[str, Any]) -> None:
        self._result_data = data
        self._goto_result()

    def _on_analyze_failed(self, msg: str) -> None:
        logger.error("Analyze failed: %s", msg)
        error_message = msg if "서버" in msg else "분석에 실패했습니다. 다시 촬영해주세요."
        self._analysis.stop()
        self._analysis.set_error_message("분석에 실패했습니다. 다시 촬영해주세요.")
        self._guide.set_error_message(error_message)
        self._guide.reset(clear_status=False)
        self._stack.setCurrentIndex(self._IDX_GUIDE)

    def _on_camera_error(self, msg: str) -> None:
        logger.error("Camera error: %s", msg)
        self._guide.set_error_message("카메라를 열 수 없습니다. 직원에게 문의해주세요.")

    def closeEvent(self, event) -> None:
        if not self._preview_mode:
            self._cam.stop()
        if self._analyze_worker and self._analyze_worker.isRunning():
            self._analyze_worker.quit(); self._analyze_worker.wait(2000)
        super().closeEvent(event)
