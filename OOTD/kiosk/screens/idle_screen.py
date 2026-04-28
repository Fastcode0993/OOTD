"""
kiosk/screens/idle_screen.py
─────────────────────────────
대기 화면 — 키오스크가 사용되지 않을 때 표시.
터치/클릭 시 촬영 가이드 화면으로 전환.

디자인 컨셉: 퍼스널 컬러 스튜디오 + 스펙트럼 무드
  • 배경: 잉크 블루 + 무지개 색조 리본 + 물방울 컬러 포인트
  • 폰트: 세리프 디스플레이 (우아함) + 산세리프 UI
  • 컬러 팔레트: 워밍 골드 / 로즈 골드 / 크림
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QRect, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QBrush, QPixmap, QPainterPath
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsOpacityEffect,
)

# ── 공통 스타일 상수 ──────────────────────────────────────────────────────
GOLD      = "#FFE08A"
ROSE_GOLD = "#FF8FB3"
CREAM     = "#F5EFE6"
DARK_BG   = "#101426"
DARK_CARD = "#182039"
RAINBOW   = ["#FF7A90", "#FFB86B", "#FFE66D", "#6EE7A8", "#68D8FF", "#A78BFA"]

_BASE_FONT = "Georgia"


class IdleScreen(QWidget):
    """대기 화면."""

    start_requested = pyqtSignal()   # 시작 버튼 → 메인 흐름 진입

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {DARK_BG};")
        self._build_ui()
        self._start_pulse_animation()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 60, 60, 60)
        root.setSpacing(0)

        # ── 상단 로고 영역 ─────────────────────────────────────────────
        logo_label = QLabel("✦  PERSONAL COLOR")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet(
            f"color: {GOLD}; font-family: {_BASE_FONT}; font-size: 14px;"
            "letter-spacing: 6px; font-weight: normal;"
        )
        root.addWidget(logo_label)

        # ── 골드 구분선 ───────────────────────────────────────────────
        root.addSpacing(20)
        line = _GoldLine()
        root.addWidget(line)
        root.addSpacing(40)

        # ── 메인 타이틀 ───────────────────────────────────────────────
        title = QLabel("나만의\n퍼스널 컬러를\n찾아보세요")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {CREAM}; font-family: {_BASE_FONT}; font-size: 56px;"
            "font-weight: normal; line-height: 1.4;"
        )
        root.addWidget(title)

        root.addSpacing(16)

        subtitle = QLabel("AI가 당신의 피부톤을 분석하여\n봄·여름·가을·겨울 중 어울리는 색을 알려드립니다.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            f"color: {GOLD}; font-family: 'Noto Sans KR', sans-serif; font-size: 18px;"
            "font-weight: normal; opacity: 0.85;"
        )
        root.addWidget(subtitle)

        root.addSpacing(60)

        # ── 시작 버튼 ─────────────────────────────────────────────────
        self._start_btn = _TouchButton("진단 시작하기", accent=GOLD)
        self._start_btn.clicked.connect(self.start_requested.emit)
        root.addWidget(self._start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addSpacing(20)

        hint = QLabel("화면을 터치하여 시작")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            f"color: {ROSE_GOLD}; font-size: 15px; letter-spacing: 2px;"
        )
        self._hint_label = hint
        root.addWidget(hint)

        root.addStretch()

        # ── 하단 안내 ─────────────────────────────────────────────────
        bottom = QLabel("소요시간 약 30초  |  개인정보 미수집  |  무료 체험")
        bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom.setStyleSheet(f"color: {GOLD}; font-size: 13px; opacity: 0.6;")
        root.addWidget(bottom)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(DARK_BG))

        # Soft spectrum ribbons.
        for i, color in enumerate(RAINBOW):
            grad = QLinearGradient(0, 0, self.width(), self.height())
            grad.setColorAt(0.0, QColor(color + "00"))
            c = QColor(color)
            c.setAlpha(58)
            grad.setColorAt(0.45 + i * 0.035, c)
            grad.setColorAt(1.0, QColor(color + "00"))
            pen = QPen(QBrush(grad), 42)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            y = 130 + i * 62
            painter.drawArc(-180 + i * 65, y, self.width() + 360, 520, 18 * 16, 118 * 16)

        # Water-drop color accents.
        drops = [
            (118, 136, 42, "#FF7A90"), (1040, 112, 56, "#68D8FF"),
            (948, 566, 44, "#FFE66D"), (220, 610, 54, "#6EE7A8"),
            (1110, 430, 34, "#A78BFA"),
        ]
        for x, y, size, color in drops:
            path = QPainterPath()
            path.moveTo(x, y - size)
            path.cubicTo(x + size, y - size * 0.15, x + size * 0.55, y + size, x, y + size)
            path.cubicTo(x - size * 0.55, y + size, x - size, y - size * 0.15, x, y - size)
            fill = QColor(color)
            fill.setAlpha(92)
            painter.setBrush(fill)
            painter.setPen(QPen(QColor(255, 255, 255, 46), 1))
            painter.drawPath(path)

    # ------------------------------------------------------------------ #
    def _start_pulse_animation(self) -> None:
        """힌트 레이블 페이드 인/아웃 반복."""
        self._opacity_effect = QGraphicsOpacityEffect(self._hint_label)
        self._hint_label.setGraphicsEffect(self._opacity_effect)

        self._anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._anim.setDuration(1500)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.3)
        self._anim.setLoopCount(-1)  # 무한 반복
        self._anim.start()

    # ── 마우스/터치 클릭으로도 시작 ──────────────────────────────────
    def mousePressEvent(self, event) -> None:
        self.start_requested.emit()


# ── 재사용 위젯 ───────────────────────────────────────────────────────────
class _GoldLine(QWidget):
    """얇은 골드 그라디언트 수평선."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(2)

    def paintEvent(self, event):
        painter = QPainter(self)
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, QColor(0, 0, 0, 0))
        grad.setColorAt(0.4, QColor(201, 169, 110, 220))
        grad.setColorAt(0.6, QColor(201, 169, 110, 220))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), grad)


class _TouchButton(QPushButton):
    """키오스크용 대형 터치 버튼."""

    def __init__(self, text: str, accent: str = GOLD, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(340, 72)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 2px solid {accent};
                color: {accent};
                font-family: {_BASE_FONT};
                font-size: 20px;
                letter-spacing: 3px;
                border-radius: 36px;
            }}
            QPushButton:hover {{
                background-color: {accent};
                color: {DARK_BG};
            }}
            QPushButton:pressed {{
                background-color: {accent};
                color: {DARK_BG};
                opacity: 0.85;
            }}
        """)
