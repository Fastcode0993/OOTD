"""
kiosk/screens/analysis_screen.py
──────────────────────────────────
AI 분석 중 로딩 화면.
스피너 + 단계별 진행 메시지를 표시.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsOpacityEffect

GOLD   = "#C9A96E"
CREAM  = "#F5EFE6"
DARK   = "#1A1512"

_STEPS = [
    "얼굴 특징점 분석 중...",
    "피부톤 데이터 추출 중...",
    "AI 퍼스널 컬러 추론 중...",
    "결과 준비 중...",
]


class AnalysisScreen(QWidget):
    """AI 분석 로딩 화면."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {DARK};")
        self._angle    = 0
        self._step_idx = 0
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(28)

        # 캡처된 얼굴 미리보기 (선택)
        self._face_preview = QLabel()
        self._face_preview.setFixedSize(180, 180)
        self._face_preview.setStyleSheet(
            f"border-radius: 90px; border: 2px solid {GOLD}; background: #231E1A;"
        )
        self._face_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._face_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        # 커스텀 스피너
        self._spinner = _SpinnerWidget()
        self._spinner.setFixedSize(100, 100)
        layout.addWidget(self._spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        # 메인 메시지
        self._title = QLabel("분석 중입니다")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            f"color: {CREAM}; font-family: Georgia; font-size: 30px;"
        )
        layout.addWidget(self._title)

        # 단계 메시지
        self._step_label = QLabel(_STEPS[0])
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_label.setStyleSheet(
            f"color: {GOLD}; font-size: 17px; letter-spacing: 1px;"
        )
        self._step_effect = QGraphicsOpacityEffect(self._step_label)
        self._step_label.setGraphicsEffect(self._step_effect)
        layout.addWidget(self._step_label)

        self._error_label = QLabel("")
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_label.setStyleSheet("color: #FF6B6B; font-size: 15px;")
        layout.addWidget(self._error_label)

        note = QLabel("잠시만 기다려 주세요 — 약 5~10초 소요됩니다")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(note)

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._angle    = 0
        self._step_idx = 0
        self._step_label.setText(_STEPS[0])
        self._error_label.clear()
        self._timer.start(30)   # 33 fps

    def stop(self) -> None:
        self._timer.stop()

    def set_error_message(self, msg: str) -> None:
        self._error_label.setText(msg)

    def set_face_pixmap(self, pixmap) -> None:
        if pixmap:
            scaled = pixmap.scaled(
                180, 180,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._face_preview.setPixmap(scaled)

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        self._angle = (self._angle + 4) % 360
        self._spinner.set_angle(self._angle)

        # 단계 메시지 순환 (1.5초마다)
        elapsed_ticks = self._angle // (4 * 45)  # ~1.5초
        idx = min(elapsed_ticks, len(_STEPS) - 1)
        if idx != self._step_idx:
            self._step_idx = idx
            self._step_label.setText(_STEPS[idx])


# ── 골드 스피너 위젯 ──────────────────────────────────────────────────────
class _SpinnerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0

    def set_angle(self, angle: int) -> None:
        self._angle = angle
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - 6

        # 배경 원
        pen = QPen(QColor(50, 42, 36), 5)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # 골드 아크
        from PyQt6.QtCore import QRectF
        pen = QPen(QColor(GOLD), 5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(
            QRectF(cx - r, cy - r, r * 2, r * 2),
            (-self._angle) * 16,
            270 * 16,
        )
