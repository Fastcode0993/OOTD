from __future__ import annotations

import io
import logging
from typing import Optional

import qrcode
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

GOLD = "#C8942E"
INK = "#1F1B18"
MUTED = "#62564A"
PAGE = "#F7F4EF"
BORDER = "rgba(200,148,46,.34)"


class QRScreen(QWidget):
    home_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background:{PAGE}; color:{INK}; font-family: Pretendard, Arial;")
        self._remaining = 30
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

        header = QLabel("나의 컬러 QR")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"color:{GOLD}; font-size:18px; font-weight:900; letter-spacing:4px;")
        layout.addWidget(header)

        title = QLabel("QR 코드를 스캔하세요")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color:{INK}; font-size:42px; font-weight:900;")
        layout.addWidget(title)

        sub = QLabel("모바일에서 자세한 팔레트와 추천 아이템을 바로 확인할 수 있습니다.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(f"color:{MUTED}; font-size:18px;")
        layout.addWidget(sub)

        self._qr_label = QLabel()
        self._qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qr_label.setFixedSize(300, 300)
        self._qr_label.setStyleSheet(
            f"background:white; border-radius:22px; border:2px solid {BORDER}; padding:18px;"
        )
        layout.addWidget(self._qr_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._info_label = QLabel("")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet(f"color:{MUTED}; font-size:14px;")
        layout.addWidget(self._info_label)

        self._countdown_label = QLabel("")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._countdown_label.setStyleSheet(f"color:{MUTED}; font-size:15px;")
        layout.addWidget(self._countdown_label)

        home_btn = QPushButton("처음으로 돌아가기")
        home_btn.setFixedSize(260, 60)
        home_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        home_btn.clicked.connect(self._go_home)
        home_btn.setStyleSheet(f"""
            QPushButton {{
                background: #FFFFFF;
                color: {INK};
                font-size: 17px;
                font-weight: 800;
                border: 1px solid {BORDER};
                border-radius: 30px;
            }}
            QPushButton:hover {{ background: #FFF8E9; }}
        """)
        layout.addWidget(home_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def set_session(
        self,
        session_id: str,
        label_ko: str,
        qr_url: Optional[str] = None,
    ) -> None:
        url = qr_url or f"http://127.0.0.1:8000/result/{session_id}"
        self._generate_qr(url)
        self._info_label.setText(f"{label_ko} | {session_id[:8]}...")
        self._remaining = 30
        self._countdown_label.setText(f"{self._remaining}초 후 처음 화면으로 돌아갑니다.")
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
                260,
                260,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._qr_label.setPixmap(scaled)
        except Exception as e:
            logger.error(f"QR generation failed: {e}")
            self._qr_label.setText("QR 생성 실패")

    def _tick_countdown(self) -> None:
        self._remaining -= 1
        self._countdown_label.setText(f"{self._remaining}초 후 처음 화면으로 돌아갑니다.")
        if self._remaining <= 0:
            self._go_home()

    def _go_home(self) -> None:
        self._auto_home_timer.stop()
        self._countdown_timer.stop()
        self.home_requested.emit()
