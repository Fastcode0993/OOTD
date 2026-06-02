"""
kiosk/screens/analysis_screen.py
──────────────────────────────────
AI 분석 중 로딩 화면.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsOpacityEffect

PAGE = "#F8F4EF"
SURFACE = "#FFFDF9"
INK = "#251D19"
MUTED = "#71665D"
METAL = "#D6B979"
SPRING = "#FF8FB8"
SUMMER = "#FF3348"
AUTUMN = "#C96F2D"
WINTER = "#071A44"

_STEPS = [
    "얼굴 영역을 정리하고 있습니다",
    "피부톤과 명도를 추출하고 있습니다",
    "계절 팔레트를 비교하고 있습니다",
    "결과 화면을 준비하고 있습니다",
]


class AnalysisScreen(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background:{PAGE}; color:{INK}; font-family: Pretendard, Arial;")
        self._angle = 0
        self._step_idx = 0
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(22)

        self._face_preview = QLabel()
        self._face_preview.setFixedSize(190, 190)
        self._face_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._face_preview.setStyleSheet(f"""
            background: {SURFACE};
            border: 2px solid {METAL};
            border-radius: 95px;
        """)
        layout.addWidget(self._face_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        self._spinner = _SpinnerWidget()
        self._spinner.setFixedSize(118, 118)
        layout.addWidget(self._spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        title = QLabel("퍼스널 컬러 분석 중")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color:{INK}; font-size:38px; font-weight:900;")
        layout.addWidget(title)

        self._step_label = QLabel(_STEPS[0])
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_label.setStyleSheet(f"""
            background: {SURFACE};
            color: {MUTED};
            border: 2px solid {METAL};
            border-radius: 8px;
            padding: 12px 34px;
            font-size: 19px;
            font-weight: 800;
        """)
        self._step_effect = QGraphicsOpacityEffect(self._step_label)
        self._step_label.setGraphicsEffect(self._step_effect)
        layout.addWidget(self._step_label)

        note = QLabel("봄 핑크 · 여름 레드 · 가을 브라운 · 겨울 블루를 비교합니다")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet(f"color:{MUTED}; font-size:16px;")
        layout.addWidget(note)

    def start(self) -> None:
        self._angle = 0
        self._step_idx = 0
        self._step_label.setText(_STEPS[0])
        self._timer.start(30)

    def stop(self) -> None:
        self._timer.stop()

    def set_face_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap:
            scaled = pixmap.scaled(
                190,
                190,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._face_preview.setPixmap(scaled)

    def _tick(self) -> None:
        self._angle = (self._angle + 4) % 360
        self._spinner.set_angle(self._angle)
        idx = min((self._angle // 90), len(_STEPS) - 1)
        if idx != self._step_idx:
            self._step_idx = idx
            self._step_label.setText(_STEPS[idx])


class _SpinnerWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._angle = 0

    def set_angle(self, angle: int) -> None:
        self._angle = angle
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - 8

        painter.setPen(QPen(QColor("#E7D8C2"), 6))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        colors = [SPRING, SUMMER, AUTUMN, WINTER]
        for i, color in enumerate(colors):
            pen = QPen(QColor(color), 6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(
                QRectF(cx - r, cy - r, r * 2, r * 2),
                int((-self._angle + i * 90) * 16),
                46 * 16,
            )
