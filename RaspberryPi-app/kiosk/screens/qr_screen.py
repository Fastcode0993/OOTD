"""
kiosk/screens/qr_screen.py
───────────────────────────
QR 코드 저장/공유 화면.
세션 ID 기반 결과 URL을 QR로 생성하여 표시.
"""

from __future__ import annotations

import io
import logging

import qrcode
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
)

logger = logging.getLogger(__name__)

GOLD  = "#C9A96E"
CREAM = "#F5EFE6"
DARK  = "#1A1512"
CARD  = "#231E1A"

_RESULT_BASE_URL = "http://localhost:8000/result/"  # 실제 배포 URL로 변경


class QRScreen(QWidget):
    """QR 코드 화면."""

    home_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {DARK};")
        self._auto_home_timer = QTimer(self)
        self._auto_home_timer.setSingleShot(True)
        self._auto_home_timer.timeout.connect(self.home_requested.emit)
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._tick_countdown)
        self._remaining = 30
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(20)

        header = QLabel("✦  결과 저장")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            f"color: {GOLD}; font-size: 14px; letter-spacing: 4px;"
        )
        layout.addWidget(header)

        title = QLabel("QR 코드를 스캔하세요")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {CREAM}; font-family: Georgia; font-size: 34px;"
        )
        layout.addWidget(title)

        sub = QLabel("스마트폰으로 QR 코드를 스캔하면\n나의 퍼스널 컬러 결과를 확인할 수 있어요.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(f"color: {GOLD}; font-size: 16px;")
        layout.addWidget(sub)

        # QR 이미지
        self._qr_label = QLabel()
        self._qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qr_label.setFixedSize(260, 260)
        self._qr_label.setStyleSheet(
            f"background: white; border-radius: 16px; border: 2px solid {GOLD};"
        )
        layout.addWidget(self._qr_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 세션 ID 표시
        self._session_label = QLabel("")
        self._session_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._session_label.setStyleSheet("color: #555; font-size: 13px;")
        layout.addWidget(self._session_label)

        # 카운트다운
        self._countdown_label = QLabel("")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._countdown_label.setStyleSheet(f"color: #666; font-size: 15px;")
        layout.addWidget(self._countdown_label)

        # 홈 버튼
        home_btn = QPushButton("처음으로 돌아가기")
        home_btn.setFixedSize(260, 60)
        home_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        home_btn.clicked.connect(self._go_home)
        home_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {GOLD};
                font-size: 17px; border: 1px solid {GOLD};
                border-radius: 30px;
            }}
            QPushButton:hover {{ background: rgba(201,169,110,0.1); }}
        """)
        layout.addWidget(home_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    # ------------------------------------------------------------------ #
    def set_session(self, session_id: str, label_ko: str) -> None:
        url = f"{_RESULT_BASE_URL}{session_id}"
        self._generate_qr(url)
        self._session_label.setText(f"세션 ID: {session_id[:8]}...  |  {label_ko}")
        self._remaining = 30
        self._countdown_label.setText(f"{self._remaining}초 후 자동으로 처음 화면으로 돌아갑니다")
        self._auto_home_timer.start(30_000)
        self._countdown_timer.start()

    def _generate_qr(self, url: str) -> None:
        try:
            qr = qrcode.QRCode(
                version=2,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=8,
                border=2,
            )
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            qpixmap = QPixmap()
            qpixmap.loadFromData(buf.read())
            scaled = qpixmap.scaled(
                240, 240,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._qr_label.setPixmap(scaled)
        except Exception as e:
            logger.error(f"QR generation failed: {e}")
            self._qr_label.setText("QR 생성 실패")

    def _tick_countdown(self) -> None:
        self._remaining -= 1
        self._countdown_label.setText(
            f"{self._remaining}초 후 자동으로 처음 화면으로 돌아갑니다"
        )
        if self._remaining <= 0:
            self._go_home()

    def _go_home(self) -> None:
        self._auto_home_timer.stop()
        self._countdown_timer.stop()
        self.home_requested.emit()
