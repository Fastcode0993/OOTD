"""
kiosk/screens/qr_screen.py
───────────────────────────
QR 코드 표시 화면.
클라우드 서버에서 받은 qr_url 또는 로컬 세션 URL을 QR로 인코딩해 표시.
30초 자동 귀환 타이머 포함.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import qrcode
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
)

logger = logging.getLogger(__name__)

GOLD  = "#C9A96E"
CREAM = "#F5EFE6"
DARK  = "#1A1512"
CARD  = "#231E1A"


class QRScreen(QWidget):
    """QR 코드 저장/공유 화면."""

    home_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {DARK};")
        self._remaining       = 30
        self._auto_home_timer = QTimer(self)
        self._auto_home_timer.setSingleShot(True)
        self._auto_home_timer.timeout.connect(self.home_requested.emit)
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._tick_countdown)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(18)

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

        self._qr_label = QLabel()
        self._qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qr_label.setFixedSize(260, 260)
        self._qr_label.setStyleSheet(
            f"background: white; border-radius: 16px; border: 2px solid {GOLD};"
        )
        layout.addWidget(self._qr_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._info_label = QLabel("")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #888; font-size: 13px;")
        layout.addWidget(self._info_label)

        self._countdown_label = QLabel("")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._countdown_label.setStyleSheet(f"color: #666; font-size: 15px;")
        layout.addWidget(self._countdown_label)

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
    def set_session(
        self,
        session_id: str,
        label_ko: str,
        qr_url: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        session_id : 로컬 UUID
        label_ko   : 퍼스널 컬러 한국어 이름
        qr_url     : QR에 인코딩할 URL
                     (클라우드 sync 성공 시 클라우드 URL, 아니면 로컬 URL)
        """
        url = qr_url or f"http://localhost:3000/result/{session_id}"
        self._generate_qr(url)
        self._info_label.setText(f"{label_ko}  |  {session_id[:8]}...")
        self._remaining = 30
        self._countdown_label.setText(
            f"{self._remaining}초 후 자동으로 처음 화면으로 돌아갑니다"
        )
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

            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            scaled = pixmap.scaled(
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

