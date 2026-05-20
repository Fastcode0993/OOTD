"""
kiosk/main.py
──────────────
프로그램 시작점.
FastAPI 서버를 서브프로세스로 실행 후 PyQt6 키오스크 UI를 시작한다.
"""

from __future__ import annotations

import argparse
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


def _env_host_port() -> tuple[str, int]:
    host = os.getenv("KIOSK_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("KIOSK_PORT", "8000"))
    except ValueError:
        logger.warning("Invalid KIOSK_PORT, fallback to 8000")
        port = 8000
    return host, port


def _start_server(host: str, port: int) -> subprocess.Popen:
    """FastAPI 서버를 백그라운드 프로세스로 시작."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "server.app:app",
         "--host", host,
         "--port", str(port),
         "--workers", "1",
         "--log-level", "warning"],
        cwd=str(_ROOT),
        env=env,
    )
    logger.info("FastAPI server started (PID=%s, %s:%s)", proc.pid, host, port)
    return proc


def _wait_for_server(host: str, port: int, timeout: float = 15.0) -> bool:
    """서버가 준비될 때까지 대기."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1)
            logger.info("Server is ready.")
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--show-cursor", action="store_true")
    parser.add_argument("--ui-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    host, port = _env_host_port()

    server_proc: subprocess.Popen | None = None
    if not args.ui_preview:
        server_proc = _start_server(host, port)
        if not _wait_for_server(host, port):
            logger.warning("Server did not start in time — continuing anyway.")

    app = QApplication(sys.argv)
    app.setApplicationName("Personal Color Kiosk")
    app.setOrganizationName("ColorLab")

    app.setStyleSheet("""
        * {
            font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
        }
    """)

    window = KioskWindow(
        api_host=host,
        api_port=port,
        hide_cursor=not args.show_cursor,
        preview_mode=args.ui_preview,
    )

    if not args.windowed:
        window.showFullScreen()
    else:
        window.show()

    exit_code = app.exec()

    if server_proc is not None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        logger.info("Server stopped.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
