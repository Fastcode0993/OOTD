"""
kiosk/main.py
──────────────
프로그램 시작점.
FastAPI 서버를 서브프로세스로 실행 후 PyQt6 키오스크 UI를 시작한다.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

# ── 프로젝트 루트를 sys.path 에 추가 ────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kiosk.ui import KioskWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _start_server() -> subprocess.Popen:
    """FastAPI 서버를 백그라운드 프로세스로 시작."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "server.app:app",
         "--host", "127.0.0.1",
         "--port", "8000",
         "--workers", "1",
         "--log-level", "warning"],
        cwd=str(_ROOT),
        env=env,
    )
    logger.info(f"FastAPI server started (PID={proc.pid})")
    return proc


def _wait_for_server(timeout: float = 15.0) -> bool:
    """서버가 준비될 때까지 대기."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
            logger.info("Server is ready.")
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> None:
    # ── FastAPI 서버 시작 ─────────────────────────────────────────────
    server_proc = _start_server()
    if not _wait_for_server():
        logger.warning("Server did not start in time — continuing anyway.")

    # ── PyQt6 앱 초기화 ───────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("Personal Color Kiosk")
    app.setOrganizationName("ColorLab")

    # 전역 스타일시트 — Noto Sans KR 폰트 로드 시도 (한글 렌더링)
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

    # ── 메인 윈도우 ───────────────────────────────────────────────────
    window = KioskWindow()

    # 전체화면 모드 (터치 키오스크)
    if "--windowed" not in sys.argv:
        window.showFullScreen()
    else:
        window.show()

    exit_code = app.exec()

    # ── 서버 종료 ─────────────────────────────────────────────────────
    server_proc.terminate()
    try:
        server_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_proc.kill()
    logger.info("Server stopped.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
