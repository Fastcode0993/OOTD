"""
kiosk/screens/idle_screen.py
─────────────────────────────
대기 화면.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QPropertyAnimation, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget, QGraphicsOpacityEffect

PAGE = "#F8F4EF"
SURFACE = "#FFFDF9"
INK = "#251D19"
MUTED = "#71665D"
METAL = "#D6B979"
SPRING = ["#FF8FB8", "#FFC4D8", "#FFE7EF"]
SUMMER = ["#FFB3B8", "#FF6B78", "#FF3348"]
AUTUMN = ["#E7A765", "#C96F2D", "#8E4D24"]
WINTER = ["#8BB6FF", "#2558C7", "#071A44"]


class IdleScreen(QWidget):
    start_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background:{PAGE}; color:{INK}; font-family: Pretendard, Arial;")
        self._build_ui()
        self._start_pulse_animation()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(76, 46, 76, 38)
        root.setSpacing(18)

        brand = QLabel("PERSONAL COLOR STUDIO")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setStyleSheet(f"color:{MUTED}; font-size:15px; font-weight:900; letter-spacing:5px;")
        root.addWidget(brand)

        root.addStretch(1)

        title = QLabel("나에게 가장 선명한\n계절 컬러를 찾아보세요")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color:{INK}; font-size:58px; font-weight:900; line-height:1.12;")
        root.addWidget(title)

        desc = QLabel("봄 핑크, 여름 레드, 가을 브라운, 겨울 블루 중 어울리는 색을 안내합니다.")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet(f"color:{MUTED}; font-size:21px; font-weight:600;")
        root.addWidget(desc)

        root.addSpacing(12)

        self._season_cards = _SeasonCards()
        root.addWidget(self._season_cards)

        root.addSpacing(18)

        self._start_btn = _StartButton("진단 시작")
        self._start_btn.clicked.connect(self.start_requested.emit)
        root.addWidget(self._start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._hint_label = QLabel("화면을 터치해도 시작됩니다")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setStyleSheet(f"color:{MUTED}; font-size:16px;")
        root.addWidget(self._hint_label)

        root.addStretch(1)

        bottom = QLabel("약 30초 소요 | 사진은 분석에만 사용")
        bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom.setStyleSheet(f"color:{MUTED}; font-size:15px;")
        root.addWidget(bottom)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(PAGE))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(METAL), 2))
        painter.drawRoundedRect(self.rect().adjusted(38, 28, -38, -28), 18, 18)

    def _start_pulse_animation(self) -> None:
        self._opacity_effect = QGraphicsOpacityEffect(self._hint_label)
        self._hint_label.setGraphicsEffect(self._opacity_effect)
        self._anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._anim.setDuration(1400)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.35)
        self._anim.setLoopCount(-1)
        self._anim.start()

    def mousePressEvent(self, event) -> None:
        self.start_requested.emit()


class _SeasonCards(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(150)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        seasons = [
            ("봄", "Pink", SPRING),
            ("여름", "Red", SUMMER),
            ("가을", "Clear Brown", AUTUMN),
            ("겨울", "Deep Blue", WINTER),
        ]
        gap = 18
        width = (self.width() - gap * 3) // 4
        for idx, (ko, en, colors) in enumerate(seasons):
            x = idx * (width + gap)
            rect = self.rect().adjusted(x, 6, -(self.width() - x - width), -8)
            painter.setBrush(QColor(SURFACE))
            painter.setPen(QPen(QColor(METAL), 2))
            painter.drawRoundedRect(rect, 10, 10)
            swatch_w = (rect.width() - 34) // 3
            for i, color in enumerate(colors):
                painter.setBrush(QColor(color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(rect.left() + 14 + i * swatch_w, rect.top() + 18, swatch_w - 4, 34, 6, 6)
            painter.setPen(QColor(INK))
            painter.setFont(QFont("Pretendard", 22, QFont.Weight.Black))
            painter.drawText(rect.adjusted(0, 64, 0, 0), Qt.AlignmentFlag.AlignHCenter, ko)
            painter.setPen(QColor(MUTED))
            painter.setFont(QFont("Pretendard", 10, QFont.Weight.Bold))
            painter.drawText(rect.adjusted(10, 102, -10, 0), Qt.AlignmentFlag.AlignHCenter, en)


class _StartButton(QPushButton):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(320, 74)
        self.setStyleSheet(f"""
            QPushButton {{
                background: {INK};
                color: {SURFACE};
                border: 2px solid {METAL};
                border-radius: 10px;
                font-size: 24px;
                font-weight: 900;
            }}
            QPushButton:pressed {{
                background: #44352E;
            }}
        """)
