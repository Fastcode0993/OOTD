"""
testfolder/dev_kiosk.py
────────────────────────
개발 환경용 키오스크 런처.

실제 kiosk/main.py 와 동일한 GUI + FastAPI 서버를 실행하되,
카메라 입력 대신 testfolder/img/ 폴더의 이미지를 순환하여 사용한다.

사용법:
  python testfolder/dev_kiosk.py

동작:
  1. FastAPI 서버 백그라운드 시작 (포트 8000)
  2. PyQt6 GUI 윈도우 모드로 실행 (1280×800, 풀스크린 아님)
  3. 가이드 화면에서 testfolder/img/ 이미지가 0.8초마다 순환 표시
  4. '촬영하기' 버튼 클릭 → 현재 표시 중인 이미지로 AI 추론 실행
  5. 결과 → 추천 → QR 코드 생성까지 실제 동작
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── 프로젝트 루트를 sys.path 에 추가 ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── .env 파일 로드 ───────────────────────────────────────────────────────────
_env_file = _ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from kiosk.ui import KioskWindow

IMG_DIR   = Path(__file__).parent / "img"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── 개발용 가짜 카메라 스레드 ────────────────────────────────────────────────
class DevCameraThread(QThread):
    """
    testfolder/img/ 이미지를 0.8초마다 순환하며 frame_ready 시그널로 emit.
    CameraThread 와 동일한 퍼블릭 인터페이스를 제공하므로
    KioskWindow 가 차이를 알 필요 없음.
    """

    frame_ready    = pyqtSignal(QImage)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._running  = False
        self._mutex    = QMutex()
        self._snapshot_requested = False
        self._snapshot_frame: Optional[np.ndarray] = None
        self._current_frame:  Optional[np.ndarray] = None
        self._images: List[Path] = self._scan_images()

    # ── 이미지 목록 수집 ───────────────────────────────────────────────────
    def _scan_images(self) -> List[Path]:
        if not IMG_DIR.exists():
            IMG_DIR.mkdir(parents=True, exist_ok=True)
            return []
        files: List[Path] = []
        for ext in SUPPORTED:
            files.extend(IMG_DIR.glob(f"*{ext}"))
            files.extend(IMG_DIR.glob(f"*{ext.upper()}"))
        result = []
        for path in sorted(set(files)):
            if not path.exists():
                continue
            if cv2.imread(str(path)) is None:
                logger.warning(f"DevCamera: unreadable image skipped: {path}")
                continue
            result.append(path)
        logger.info(f"DevCamera: {len(result)}개 이미지 발견 ({IMG_DIR})")
        return result

    # ── 스레드 메인 루프 ───────────────────────────────────────────────────
    def run(self) -> None:
        if not self._images:
            self.error_occurred.emit(
                f"testfolder/img/ 에 이미지가 없습니다. "
                f"({', '.join(sorted(SUPPORTED))} 형식 지원)"
            )
            return

        self._running = True
        idx = 0

        while self._running:
            path = self._images[idx % len(self._images)]
            bgr  = cv2.imread(str(path))
            if bgr is None:
                logger.warning(f"DevCamera: image disappeared or cannot be read, removing: {path}")
                self._images.pop(idx % len(self._images))
                if not self._images:
                    self.error_occurred.emit("testfolder/img/ 에 읽을 수 있는 이미지가 없습니다.")
                    break
                idx += 1
                continue

            with QMutexLocker(self._mutex):
                self._current_frame = bgr.copy()
                if self._snapshot_requested:
                    self._snapshot_frame = bgr.copy()
                    self._snapshot_requested = False

            # 미리보기용 960×540 다운샘플
            preview = cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA)
            rgb     = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg    = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qimg.copy())

            idx += 1
            self.msleep(800)  # 0.8초마다 이미지 전환

    # ── 스냅샷 인터페이스 (CameraThread 호환) ─────────────────────────────
    def request_snapshot(self) -> None:
        with QMutexLocker(self._mutex):
            self._snapshot_requested = True

    def take_snapshot(self, timeout_ms: int = 2000) -> Optional[np.ndarray]:
        """현재 프레임을 즉시 반환 (이미 메모리에 있음)."""
        with QMutexLocker(self._mutex):
            if self._current_frame is not None:
                logger.info(
                    f"DevCamera snapshot: {self._images[(self._images.index(self._images[0]) if self._images else 0)]}"
                    if self._images else "DevCamera snapshot"
                )
                return self._current_frame.copy()

        # 아직 첫 프레임이 없는 극히 드문 경우만 대기
        self.request_snapshot()
        t0 = time.time()
        while time.time() - t0 < timeout_ms / 1000:
            with QMutexLocker(self._mutex):
                if self._snapshot_frame is not None:
                    frame = self._snapshot_frame
                    self._snapshot_frame = None
                    return frame
            time.sleep(0.033)

        logger.warning("DevCamera: 스냅샷 타임아웃")
        return None

    def stop(self) -> None:
        self._running = False
        self.wait(3000)


# ── 개발용 KioskWindow ────────────────────────────────────────────────────
class DevKioskWindow(KioskWindow):
    """
    KioskWindow 를 상속하여 카메라만 DevCameraThread 로 교체.
    _init_camera() 가 KioskWindow.__init__ 에서 호출되므로
    MRO 에 의해 자동으로 이 버전이 사용됨.
    """

    def _init_camera(self) -> None:
        self._cam = DevCameraThread()
        self._cam.frame_ready.connect(self._guide.update_frame)
        self._cam.error_occurred.connect(self._on_camera_error)
        self._cam.start()
        logger.info("DevCameraThread started (img 폴더 모드)")

    def _on_camera_error(self, msg: str) -> None:
        logger.error(f"DevCamera error: {msg}")
        # 에러 메시지를 가이드 화면 상태 라벨에 표시
        self._guide._status_label.setText(f"⚠ {msg}")
        self._guide._status_label.setStyleSheet("color: #E87070; font-size: 16px;")


# ── 서버 유틸리티 ─────────────────────────────────────────────────────────
def _start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT)
    # 클라우드 URL은 analyze.py 기본값(personalootd.kro.kr) 그대로 사용
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "server.app:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--workers", "1",
            "--log-level", "warning",
        ],
        cwd=str(_ROOT),
        env=env,
    )
    logger.info(f"FastAPI server started (PID={proc.pid})")
    return proc


def _wait_for_server(timeout: float = 20.0) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
            logger.info("FastAPI server ready.")
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("FastAPI server did not stop in time; killing it.")
        proc.kill()
        proc.wait(timeout=5)


# ── 메인 ─────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=== Personal Color Kiosk [DEV MODE] ===")
    logger.info(f"이미지 폴더: {IMG_DIR}")
    logger.info("QR URL → https://personalootd.kro.kr/result/<qr_code>")

    # FastAPI 서버 시작
    server_proc = _start_server()
    if not _wait_for_server():
        logger.warning("서버가 제때 시작되지 않았습니다 — GUI는 계속 진행합니다.")

    exit_code = 0
    try:
        # PyQt6 앱
        app = QApplication(sys.argv)
        app.setApplicationName("Personal Color Kiosk [DEV]")
        app.setOrganizationName("ColorLab")
        app.setStyleSheet("""
            * {
                font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
            }
            QScrollBar:vertical {
                background: #1A1512; width: 6px; border: none;
            }
            QScrollBar::handle:vertical {
                background: #4A4030; border-radius: 3px; min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QToolTip {
                background: #231E1A; color: #C9A96E;
                border: 1px solid #4A4030; font-size: 13px;
            }
        """)

        window = DevKioskWindow()
        window.setWindowTitle("Personal Color Kiosk  ·  DEV — img 폴더 모드")
        # 개발 환경: 마우스 커서 표시, 윈도우 모드
        window.setCursor(Qt.CursorShape.ArrowCursor)
        window.show()

        exit_code = app.exec()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; shutting down.")
        exit_code = 130
    finally:
        _stop_server(server_proc)
        logger.info("서버 종료 완료.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
