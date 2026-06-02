from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from kiosk.personal_color_content import get_profile, normalize_code

INK = "#050505"
PAGE = "#FFFFFF"


@dataclass(frozen=True)
class _Theme:
    page: str
    ink: str
    muted: str
    line: str
    accent: str
    accent_soft: str
    button: str
    button_text: str
    confidence: str
    panels: Tuple[str, str, str, str]
    name: str


_THEMES = {
    "spring": _Theme(
        page="#FFF3F8",
        ink="#3A1626",
        muted="#8D526B",
        line="#D8A6B8",
        accent="#FF7EAD",
        accent_soft="#FFD7E6",
        button="#FF7EAD",
        button_text="#3A1626",
        confidence="#FFD7E6",
        panels=("#FFE1EC", "#FFD4E4", "#FFC2D8", "#FFEAF2"),
        name="Spring",
    ),
    "summer": _Theme(
        page="#FFF3F3",
        ink="#3D1215",
        muted="#8C454B",
        line="#E19AA1",
        accent="#FF5B68",
        accent_soft="#FFD7DB",
        button="#FF5B68",
        button_text="#FFFFFF",
        confidence="#FFDDE0",
        panels=("#FFE1E4", "#FFD1D6", "#FFC1C8", "#FFF0F1"),
        name="Summer",
    ),
    "autumn": _Theme(
        page="#FFF4E7",
        ink="#3B2418",
        muted="#875336",
        line="#D6A46D",
        accent="#C96F2D",
        accent_soft="#FFD9B8",
        button="#C96F2D",
        button_text="#FFFFFF",
        confidence="#FFD9B8",
        panels=("#FFE0BF", "#FFD3A6", "#FFC58D", "#FFEAD8"),
        name="Autumn",
    ),
    "winter": _Theme(
        page="#EEF4FF",
        ink="#071A44",
        muted="#435985",
        line="#6989D8",
        accent="#0B3C9D",
        accent_soft="#CADBFF",
        button="#071A44",
        button_text="#FFFFFF",
        confidence="#CADBFF",
        panels=("#DDE8FF", "#C9DCFF", "#B8CEFA", "#E8F0FF"),
        name="Winter",
    ),
}

_TYPE_COLORS = {
    "spring_warm": "#FF7EAD",
    "spring_bright": "#FF5F9E",
    "spring_light": "#FFB6D2",
    "summer_warm": "#FF7A7A",
    "summer_bright": "#FF3348",
    "summer_light": "#FF8B94",
    "autumn_warm": "#C96F2D",
    "autumn_bright": "#D98731",
    "autumn_light": "#E7A765",
    "winter_warm": "#0B2F80",
    "winter_bright": "#003BBA",
    "winter_light": "#4169E1",
    "spring_warm_light": "#FFB6D2",
    "spring_warm_bright": "#FF5F9E",
    "summer_cool_light": "#FF8B94",
    "summer_cool_mute": "#FF6B78",
    "autumn_warm_mute": "#E7A765",
    "autumn_warm_deep": "#C96F2D",
    "winter_cool_bright": "#003BBA",
    "winter_cool_deep": "#071A44",
}


class ResultScreen(QWidget):
    next_requested = pyqtSignal()
    retry_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._theme = _THEMES["summer"]
        self._retry_button: _BoxButton | None = None
        self._qr_button: _BoxButton | None = None
        self._season_badge: QLabel | None = None
        self._apply_theme()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(70, 34, 70, 34)
        root.setSpacing(18)

        top = QHBoxLayout()
        top.setSpacing(28)

        title_block = QVBoxLayout()
        title_block.setSpacing(8)

        self._season_badge = QLabel("Summer")
        self._season_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._season_badge.setFixedSize(138, 36)
        title_block.addWidget(self._season_badge)

        self._title = QLabel("퍼스널 컬러: -")
        self._title.setStyleSheet("font-size: 46px; font-weight: 900;")
        self._title.setMinimumHeight(58)
        title_block.addWidget(self._title)

        self._confidence = QLabel("정확도: -%")
        self._confidence.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._confidence.setStyleSheet("""
            background: white;
            border: none;
            border-radius: 8px;
            padding: 8px 18px;
            font-size: 42px;
            font-weight: 900;
        """)
        self._confidence.setFixedHeight(82)
        title_block.addWidget(self._confidence)
        top.addLayout(title_block, stretch=1)

        self._color_circle = _ColorCircle()
        top.addWidget(self._color_circle, alignment=Qt.AlignmentFlag.AlignTop)
        root.addLayout(top)

        panels = QGridLayout()
        panels.setHorizontalSpacing(34)
        panels.setVerticalSpacing(24)
        panels.setColumnStretch(0, 1)
        panels.setColumnStretch(1, 1)

        self._about = _Panel("퍼스널 컬러란?")
        self._meaning = _Panel("이 컬러는 무슨 의미?")
        self._recommend = _Panel("추천색상")
        self._avoid = _Panel("피해야 할 색상")
        for index, panel in enumerate((self._about, self._meaning, self._recommend, self._avoid)):
            panel.set_panel_index(index)
        panels.addWidget(self._about, 0, 0)
        panels.addWidget(self._meaning, 0, 1)
        panels.addWidget(self._recommend, 1, 0)
        panels.addWidget(self._avoid, 1, 1)
        root.addLayout(panels, stretch=1)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(20)
        self._retry_button = _BoxButton("재촬영", 220, 82)
        self._retry_button.clicked.connect(self.retry_requested.emit)
        self._qr_button = _BoxButton("QR", 160, 82)
        self._qr_button.clicked.connect(self.next_requested.emit)
        bottom.addWidget(self._retry_button)
        bottom.addStretch()
        bottom.addWidget(self._qr_button)
        root.addLayout(bottom)
        self._apply_theme()

    def set_result(self, data: Dict[str, Any]) -> None:
        personal_color = data.get("personal_color") or data.get("code")
        code = normalize_code(personal_color)
        season = code.split("_", 1)[0]
        self._theme = _THEMES.get(season, _THEMES["summer"])
        self._apply_theme()

        profile = get_profile(personal_color)
        label = data.get("label_ko") or profile["label_ko"]
        confidence = float(data.get("final_confidence", data.get("confidence", 0.0)) or 0.0)
        palette = data.get("color_palette") or {}
        best_colors: Sequence[Tuple[str, str]] = profile["best"]
        avoid_colors: Sequence[Tuple[str, str]] = profile["avoid"]
        colors = (
            palette.get("primary")
            or data.get("recommended_colors")
            or [hex_value for hex_value, _ in best_colors]
        )

        self._title.setText(f"퍼스널 컬러: {label}")
        self._confidence.setText(f"정확도: {confidence * 100:.0f}%")
        self._color_circle.set_color(_result_color(personal_color, code, colors))
        self._color_circle.set_theme(self._theme)
        if self._season_badge:
            self._season_badge.setText(self._theme.name)

        summary = data.get("description_ko") or palette.get("description") or profile["summary"]
        meaning = profile["meaning"]
        best_text = " · ".join(name for _, name in best_colors[:5])
        avoid_text = " · ".join(name for _, name in avoid_colors[:4])

        self._about.set_content(
            "피부톤, 눈동자, 머리색과 조화가 이루어 얼굴빛을 더 맑고 생기 있게 보여주는 색상 그룹입니다.",
            colors[:4],
        )
        self._meaning.set_content(summary or meaning, colors[:5])
        self._recommend.set_content(best_text, [hex_value for hex_value, _ in best_colors[:5]])
        self._avoid.set_content(avoid_text, [hex_value for hex_value, _ in avoid_colors[:4]])
        self._apply_theme()

    def _apply_theme(self) -> None:
        theme = self._theme
        self.setStyleSheet(f"""
            ResultScreen {{
                background: {theme.page};
                color: {theme.ink};
                font-family: Pretendard, Arial;
            }}
            QLabel {{
                color: {theme.ink};
            }}
        """)
        if hasattr(self, "_title"):
            self._title.setStyleSheet(f"color:{theme.ink}; font-size:46px; font-weight:900;")
            self._confidence.setStyleSheet(f"""
                background: {theme.confidence};
                color: {theme.ink};
                border: 2px solid {theme.line};
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 42px;
                font-weight: 900;
            """)
            self._color_circle.set_theme(theme)
            for panel in (self._about, self._meaning, self._recommend, self._avoid):
                panel.set_theme(theme)
        if self._season_badge:
            self._season_badge.setStyleSheet(f"""
                background: {theme.accent};
                color: {theme.button_text};
                border: none;
                border-radius: 17px;
                font-size: 16px;
                font-weight: 900;
            """)
        if self._retry_button:
            self._retry_button.set_theme(theme, filled=False)
        if self._qr_button:
            self._qr_button.set_theme(theme, filled=True)


class _Panel(QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._theme = _THEMES["summer"]
        self._panel_index = 0
        self.setMinimumHeight(210)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(0)

        self._title = QLabel(title)
        self._title.setStyleSheet("border:none;")
        self._title.setFixedHeight(44)
        self._title.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._title, alignment=Qt.AlignmentFlag.AlignLeft)

        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body.setStyleSheet("border:none;")
        layout.addWidget(self._body, stretch=1)

        self._swatches = _Swatches()
        layout.addWidget(self._swatches, alignment=Qt.AlignmentFlag.AlignRight)
        self.set_theme(self._theme)

    def set_content(self, text: str, colors: Sequence[str]) -> None:
        self._body.setText(text)
        self._swatches.set_colors(list(colors)[:5])

    def set_panel_index(self, index: int) -> None:
        self._panel_index = index
        self.set_theme(self._theme)

    def set_theme(self, theme: _Theme) -> None:
        self._theme = theme
        panel_bg = theme.panels[self._panel_index % len(theme.panels)]
        self.setStyleSheet(f"""
            QFrame {{
                background: {panel_bg};
                border: 2px solid {theme.line};
                border-radius: 8px;
            }}
        """)
        self._title.setStyleSheet(f"""
            background: rgba(255, 255, 255, 150);
            color: {theme.ink};
            border: 1px solid {theme.line};
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 24px;
            font-weight: 900;
        """)
        self._body.setStyleSheet(f"""
            border: none;
            color: {theme.ink};
            padding: 20px 6px 8px;
            font-size: 22px;
            line-height: 1.35;
            font-weight: 700;
        """)


class _ColorCircle(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._color = "#FFFFFF"
        self._theme = _THEMES["summer"]
        self.setFixedSize(188, 188)

    def set_color(self, color: str) -> None:
        self._color = color
        self.update()

    def set_theme(self, theme: _Theme) -> None:
        self._theme = theme
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill = QColor(self._color)
        painter.setBrush(fill if fill.isValid() else QColor("#FFFFFF"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(6, 6, self.width() - 12, self.height() - 12)


class _Swatches(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._colors: List[str] = []
        self.setFixedSize(210, 36)

    def set_colors(self, colors: List[str]) -> None:
        self._colors = colors
        self.update()

    def paintEvent(self, event) -> None:
        if not self._colors:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        x = self.width() - (len(self._colors[:5]) * 32)
        for hex_color in self._colors[:5]:
            color = QColor(hex_color)
            painter.setBrush(color if color.isValid() else QColor("#FFFFFF"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(x, 3, 24, 24)
            x += 32


class _BoxButton(QPushButton):
    def __init__(self, text: str, width: int, height: int, parent=None) -> None:
        super().__init__(text, parent)
        self.setFixedSize(width, height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_theme(_THEMES["summer"], filled=False)

    def set_theme(self, theme: _Theme, filled: bool) -> None:
        background = theme.button if filled else theme.accent_soft
        text = theme.button_text if filled else theme.ink
        pressed = theme.accent if filled else theme.accent_soft
        self.setStyleSheet(f"""
            QPushButton {{
                background: {background};
                color: {text};
                border: 2px solid {theme.line};
                border-radius: 8px;
                font-size: 28px;
                font-weight: 900;
            }}
            QPushButton:pressed {{
                background: {pressed};
            }}
        """)


def _result_color(personal_color: Any, code: str, colors: Sequence[str]) -> str:
    raw = str(personal_color or "").strip().lower().replace("-", "_")
    if raw in _TYPE_COLORS:
        return _TYPE_COLORS[raw]
    if code in _TYPE_COLORS:
        return _TYPE_COLORS[code]
    return str(colors[0]) if colors else "#FFFFFF"
