"""
kiosk/screens/guide_screen.py
──────────────────────────────
촬영 가이드 화면:
  • 카메라 미리보기 + 얼굴 정렬용 타원 오버레이
  • 사용자 안내 메시지
  • 촬영 / 취소 버튼

디자인: 스펙트럼 컬러 스튜디오 분위기.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRect
from PyQt6.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QBrush,
    QPixmap, QRadialGradient, QLinearGradient,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QSizePolicy,
)

logger = logging.getLogger(__name__)

GOLD      = "#FFE08A"
ROSE_GOLD = "#FF8FB3"
CREAM     = "#F5EFE6"
DARK_BG   = "#101426"
SUCCESS   = "#7EC8A4"
_BASE_FONT = "Georgia"


class GuideScreen(QWidget):
    """촬영 가이드 + 카메라 미리보기 화면."""

    capture_requested = pyqtSignal()
    back_requested    = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            background:
                qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #101426,
                    stop:0.35 #142345,
                    stop:0.72 #1B1F3F,
                    stop:1 #211832);
        """)
        self._face_detected = False
        self._countdown     = 0
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(12)

        # ── 헤더 ──────────────────────────────────────────────────────
        header = QLabel("얼굴을 화면 중앙에 맞춰주세요")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            f"color: {CREAM}; font-family: {_BASE_FONT}; font-size: 24px;"
        )
        layout.addWidget(header)

        # ── 카메라 + 오버레이 컨테이너 ───────────────────────────────
        self._cam_overlay = _CameraOverlayWidget()
        self._cam_overlay.setFixedSize(760, 520)
        layout.addWidget(
            self._cam_overlay, alignment=Qt.AlignmentFlag.AlignCenter
        )

        # ── 상태 메시지 ───────────────────────────────────────────────
        self._status_label = QLabel("카메라를 초기화 중입니다...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            f"color: {GOLD}; font-size: 17px; letter-spacing: 1px;"
        )
        layout.addWidget(self._status_label)

        # ── 버튼 영역 ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(24)

        self._back_btn = _SmallButton("← 돌아가기", accent="#888")
        self._back_btn.clicked.connect(self.back_requested.emit)

        self._capture_btn = _CaptureButton()
        self._capture_btn.clicked.connect(self._on_capture_clicked)

        btn_row.addWidget(self._back_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._capture_btn)

        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    def update_frame(self, qimage: QImage) -> None:
        """CameraThread 로부터 프레임 수신."""
        self._cam_overlay.set_frame(qimage)
        self._status_label.setText("얼굴 타원 안에 얼굴을 맞춰주세요")

    def set_face_detected(self, detected: bool) -> None:
        self._face_detected = detected
        self._cam_overlay.set_face_detected(detected)
        if detected:
            self._status_label.setText("✓ 얼굴이 인식되었습니다 — 촬영 버튼을 눌러주세요")
            self._status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 17px;")
        else:
            self._status_label.setText("얼굴 타원 안에 얼굴을 맞춰주세요")
            self._status_label.setStyleSheet(f"color: {GOLD}; font-size: 17px;")

    # ------------------------------------------------------------------ #
    def _on_capture_clicked(self) -> None:
        self._capture_btn.setEnabled(False)
        self.capture_requested.emit()

    def reset(self) -> None:
        self._capture_btn.setEnabled(True)
        self._status_label.setText("카메라를 초기화 중입니다...")
        self._status_label.setStyleSheet(f"color: {GOLD}; font-size: 17px;")


# ── 카메라 + 타원 오버레이 위젯 ──────────────────────────────────────────
class _CameraOverlayWidget(QWidget):
    """카메라 프레임 위에 얼굴 가이드 타원을 그리는 위젯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._face_detected = False
        self.setStyleSheet("border: none; background: #0B1020;")

    def set_frame(self, qimage: QImage) -> None:
        self._pixmap = QPixmap.fromImage(qimage).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.update()

    def set_face_detected(self, detected: bool) -> None:
        self._face_detected = detected
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 카메라 프레임
        if self._pixmap:
            painter.drawPixmap(self.rect(), self._pixmap)
        else:
            painter.fillRect(self.rect(), QColor("#0B1020"))

        w, h = self.width(), self.height()

        # 어두운 비네팅 마스크
        grad = QRadialGradient(w / 2, h / 2, max(w, h) * 0.52)
        grad.setColorAt(0.55, QColor(0, 0, 0, 0))
        grad.setColorAt(1.0,  QColor(0, 0, 0, 200))
        painter.fillRect(self.rect(), grad)

        # 은은한 컬러 렌즈 느낌의 프레임.
        band = QLinearGradient(0, 0, w, 0)
        band.setColorAt(0.00, QColor(255, 122, 144, 120))
        band.setColorAt(0.22, QColor(255, 184, 107, 110))
        band.setColorAt(0.42, QColor(255, 230, 109, 110))
        band.setColorAt(0.62, QColor(110, 231, 168, 110))
        band.setColorAt(0.82, QColor(104, 216, 255, 120))
        band.setColorAt(1.00, QColor(167, 139, 250, 120))
        painter.setPen(QPen(QBrush(band), 5))
        painter.drawRoundedRect(self.rect().adjusted(3, 3, -3, -3), 22, 22)

        # 얼굴 가이드 타원
        oval_w, oval_h = int(w * 0.38), int(h * 0.75)
        oval_rect = QRect(
            (w - oval_w) // 2, (h - oval_h) // 2, oval_w, oval_h
        )
        color = QColor(SUCCESS) if self._face_detected else QColor(GOLD)
        color.setAlpha(220)
        pen = QPen(color, 3, Qt.PenStyle.DashLine if not self._face_detected else Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(oval_rect)

        # 코너 마커
        _draw_corner_marks(painter, oval_rect, color)


def _draw_corner_marks(painter: QPainter, rect: QRect, color: QColor) -> None:
    """타원 상하좌우에 짧은 십자 마커 그리기."""
    pen = QPen(color, 3)
    painter.setPen(pen)
    cx, cy = rect.center().x(), rect.center().y()
    hw, hh = rect.width() // 2, rect.height() // 2
    L = 14  # 마커 길이

    # Top
    painter.drawLine(cx - L, cy - hh, cx + L, cy - hh)
    painter.drawLine(cx, cy - hh - L, cx, cy - hh + L)
    # Bottom
    painter.drawLine(cx - L, cy + hh, cx + L, cy + hh)
    painter.drawLine(cx, cy + hh - L, cx, cy + hh + L)
    # Left
    painter.drawLine(cx - hw - L, cy, cx - hw + L, cy)
    # Right
    painter.drawLine(cx + hw - L, cy, cx + hw + L, cy)


# ── 버튼 헬퍼 ─────────────────────────────────────────────────────────────
class _CaptureButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("📷  촬영하기", parent)
        self.setFixedSize(220, 66)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8FB3, stop:0.32 #FFE08A, stop:0.68 #6EE7A8, stop:1 #68D8FF);
                color: #101426;
                font-family: {_BASE_FONT};
                font-size: 20px;
                font-weight: bold;
                border-radius: 33px;
                border: none;
            }}
            QPushButton:hover {{ color: #101426; }}
            QPushButton:pressed {{ color: #101426; }}
            QPushButton:disabled {{ background-color: #555; color: #888; }}
        """)


class _SmallButton(QPushButton):
    def __init__(self, text: str, accent: str = GOLD, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {accent};
                font-size: 16px;
                border: 1px solid {accent};
                border-radius: 24px;
                padding: 0 24px;
            }}
            QPushButton:hover {{ background-color: rgba(200,180,140,0.1); }}
        """)
