"""
kiosk/screens/guide_screen.py
──────────────────────────────
촬영 가이드 화면.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRect, QRectF
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

PAGE = "#F8F4EF"
INK = "#201915"
MUTED = "#6E6258"
BORDER = "#201915"
READY = "#6DB7E8"
WAITING = "#D6B979"
SOFT = "#FFFDF9"
METAL = "#D6B979"
_AUTO_CAPTURE_SEC = 3


class GuideScreen(QWidget):
    """촬영 가이드 + 카메라 미리보기 화면."""

    capture_requested = pyqtSignal()
    auto_capture_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background:{PAGE}; color:{INK}; font-family: Pretendard, Arial;")
        self._face_detected = False
        self._countdown_remaining = 0
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(1000)
        self._auto_timer.timeout.connect(self._on_auto_tick)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(58, 34, 58, 34)
        layout.setSpacing(14)

        title = QLabel("얼굴을 원 안에 맞춰주세요")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color:{INK}; font-size:38px; font-weight:900;")
        layout.addWidget(title)

        subtitle = QLabel("정면을 바라보고 얼굴이 원 안에 들어오면 자동으로 촬영됩니다.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color:{MUTED}; font-size:18px; font-weight:700;")
        layout.addWidget(subtitle)

        self._cam_overlay = _CameraOverlayWidget()
        self._cam_overlay.setFixedSize(640, 640)
        layout.addWidget(self._cam_overlay, alignment=Qt.AlignmentFlag.AlignCenter)

        self._status_label = QLabel("카메라를 준비 중입니다")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setFixedHeight(54)
        layout.addWidget(self._status_label)

        self._countdown_label = QLabel("")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._countdown_label.setFixedHeight(54)
        self._countdown_label.setStyleSheet(f"color:{READY}; font-size:44px; font-weight:900;")
        layout.addWidget(self._countdown_label)

        hint = QLabel("밝은 조명에서 얼굴에 그림자가 생기지 않게 서 주세요")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(f"color:{MUTED}; font-size:16px;")
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)
        self._back_btn = _LineButton("돌아가기")
        self._back_btn.clicked.connect(self.back_requested.emit)
        self._capture_btn = _CaptureButton()
        self._capture_btn.clicked.connect(self._on_capture_clicked)
        btn_row.addWidget(self._back_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._capture_btn)
        layout.addLayout(btn_row)

        self._set_status(False, "얼굴을 원 안에 맞춰주세요")

    def update_frame(self, qimage: QImage) -> None:
        self._cam_overlay.set_frame(qimage)
        if not self._face_detected:
            self._set_status(False, "얼굴을 원 안에 맞춰주세요")

    def set_face_detected(self, detected: bool) -> None:
        self._face_detected = detected
        self._cam_overlay.set_face_detected(detected)
        if detected:
            if not self._auto_timer.isActive():
                self._countdown_remaining = _AUTO_CAPTURE_SEC
                self._countdown_label.setText(f"{self._countdown_remaining}")
                self._auto_timer.start()
            self._set_status(True, "얼굴 인식 완료, 잠시만 기다려주세요")
        else:
            self._auto_timer.stop()
            self._countdown_label.setText("")
            self._set_status(False, "얼굴을 원 안에 맞춰주세요")

    def _set_status(self, ready: bool, text: str) -> None:
        color = READY if ready else WAITING
        bg = "#E7F4FB" if ready else "#FFF8E7"
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"""
            background: {bg};
            color: {INK};
            border: 2px solid {color};
            border-radius: 8px;
            font-size: 19px;
            font-weight: 900;
            padding: 8px 18px;
        """)

    def _on_auto_tick(self) -> None:
        self._countdown_remaining -= 1
        if self._countdown_remaining > 0:
            self._countdown_label.setText(f"{self._countdown_remaining}")
        else:
            self._auto_timer.stop()
            self._countdown_label.setText("")
            if self._capture_btn.isEnabled():
                self._capture_btn.setEnabled(False)
                self.auto_capture_requested.emit()

    def _on_capture_clicked(self) -> None:
        self._auto_timer.stop()
        self._countdown_label.setText("")
        self._capture_btn.setEnabled(False)
        self.capture_requested.emit()

    def reset(self) -> None:
        self._auto_timer.stop()
        self._countdown_label.setText("")
        self._capture_btn.setEnabled(True)
        self._face_detected = False
        self._cam_overlay.set_face_detected(False)
        self._set_status(False, "카메라를 준비 중입니다")


class _CameraOverlayWidget(QWidget):
    """카메라 프레임 위에 원형 얼굴 가이드를 그리는 위젯."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._face_detected = False
        self.setStyleSheet(f"background:{SOFT}; border:none;")

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

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._pixmap:
            painter.drawPixmap(self.rect(), self._pixmap)
        else:
            painter.fillRect(self.rect(), QColor(SOFT))

        w, h = self.width(), self.height()
        diameter = int(min(w, h) * 0.68)
        circle_rect = QRect((w - diameter) // 2, (h - diameter) // 2, diameter, diameter)
        circle_rect_f = QRectF(circle_rect)

        mask = QPainterPath()
        mask.addRect(0, 0, w, h)
        hole = QPainterPath()
        hole.addEllipse(circle_rect_f)
        mask = mask.subtracted(hole)
        painter.fillPath(mask, QColor(255, 253, 248, 178))

        border_color = QColor(READY if self._face_detected else WAITING)
        painter.setPen(QPen(border_color, 5, Qt.PenStyle.SolidLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(circle_rect)

        self._draw_cross_marks(painter, circle_rect, border_color)

        painter.setPen(QPen(QColor(METAL), 4))
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 12, 12)

        self._draw_palette_ticks(painter)

    def _draw_cross_marks(self, painter: QPainter, rect: QRect, color: QColor) -> None:
        painter.setPen(QPen(color, 4))
        cx, cy = rect.center().x(), rect.center().y()
        radius = rect.width() // 2
        marker = 18
        points = [
            (cx, cy - radius),
            (cx, cy + radius),
            (cx - radius, cy),
            (cx + radius, cy),
        ]
        for x, y in points:
            painter.drawLine(x - marker, y, x + marker, y)
            painter.drawLine(x, y - marker, x, y + marker)

    def _draw_palette_ticks(self, painter: QPainter) -> None:
        colors = ["#FF8FB8", "#FF3348", "#C96F2D", "#071A44"]
        step = self.width() // (len(colors) + 1)
        y = self.height() - 18
        for i, color in enumerate(colors, 1):
            painter.setBrush(QColor(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(i * step - 18, y, 36, 8, 4, 4)


class _CaptureButton(QPushButton):
    def __init__(self, parent=None) -> None:
        super().__init__("촬영하기", parent)
        self.setFixedSize(220, 64)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: {INK};
                color: {PAGE};
                border: 2px solid {METAL};
                border-radius: 8px;
                font-size: 22px;
                font-weight: 900;
            }}
            QPushButton:pressed {{
                background: {MUTED};
            }}
            QPushButton:disabled {{
                background: #CFC7BD;
                color: #8A8178;
                border-color: #CFC7BD;
            }}
        """)


class _LineButton(QPushButton):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setFixedSize(160, 58)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: {SOFT};
                color: {INK};
                border: 2px solid {METAL};
                border-radius: 8px;
                font-size: 18px;
                font-weight: 800;
            }}
            QPushButton:pressed {{
                background: {SOFT};
            }}
        """)
