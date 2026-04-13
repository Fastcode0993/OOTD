"""
kiosk/screens/result_screen.py
────────────────────────────────
결과 화면: 퍼스널 컬러 유형 + 색상 팔레트 표시
"""

from __future__ import annotations

from typing import Any, Dict, List

from PyQt6.QtCore import Qt, QPropertyAnimation, QRect, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QLinearGradient, QPen
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGraphicsOpacityEffect,
)

GOLD   = "#C9A96E"
CREAM  = "#F5EFE6"
DARK   = "#1A1512"
CARD   = "#231E1A"


class ResultScreen(QWidget):
    """진단 결과 화면."""

    next_requested = pyqtSignal()
    retry_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {DARK};")
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(0)

        # 소제목
        sub = QLabel("✦  진단 결과")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(
            f"color: {GOLD}; font-size: 14px; letter-spacing: 5px;"
        )
        layout.addWidget(sub)
        layout.addSpacing(16)

        # 퍼스널 컬러 라벨 (크게)
        self._color_label = QLabel("—")
        self._color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._color_label.setStyleSheet(
            f"color: {CREAM}; font-family: Georgia; font-size: 48px;"
        )
        layout.addWidget(self._color_label)

        # 신뢰도
        self._confidence_label = QLabel("")
        self._confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._confidence_label.setStyleSheet(
            f"color: {GOLD}; font-size: 18px;"
        )
        layout.addWidget(self._confidence_label)

        layout.addSpacing(24)

        # 설명
        self._desc_label = QLabel("")
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet(
            f"color: #C8BFB0; font-size: 17px; line-height: 1.6;"
        )
        layout.addWidget(self._desc_label)

        layout.addSpacing(32)

        # 팔레트 영역
        palette_title = QLabel("어울리는 색상 팔레트")
        palette_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        palette_title.setStyleSheet(f"color: {GOLD}; font-size: 14px; letter-spacing: 3px;")
        layout.addWidget(palette_title)
        layout.addSpacing(12)

        self._palette_widget = _PaletteWidget()
        layout.addWidget(self._palette_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addSpacing(12)

        # TOP 3 표시
        self._top3_widget = _Top3Widget()
        layout.addWidget(self._top3_widget)

        layout.addStretch()

        # 버튼
        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)

        retry_btn = _OutlineButton("다시 촬영", accent="#888")
        retry_btn.clicked.connect(self.retry_requested.emit)

        next_btn = _FillButton("추천 아이템 보기 →")
        next_btn.clicked.connect(self.next_requested.emit)

        btn_row.addWidget(retry_btn)
        btn_row.addStretch()
        btn_row.addWidget(next_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    def set_result(self, data: Dict[str, Any]) -> None:
        """FastAPI 응답 데이터를 받아 UI를 업데이트."""
        label_ko   = data.get("label_ko", "알 수 없음")
        confidence = data.get("confidence", 0.0)
        description = data.get("description_ko", "")
        colors     = data.get("recommended_colors", [])
        top3       = data.get("top3", [])

        self._color_label.setText(label_ko)
        self._confidence_label.setText(f"적합도  {confidence * 100:.1f}%")
        self._desc_label.setText(description)
        self._palette_widget.set_colors(colors)
        self._top3_widget.set_top3(top3)


# ── 팔레트 ────────────────────────────────────────────────────────────────
class _PaletteWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._colors: List[str] = []
        self.setFixedHeight(70)
        self.setMinimumWidth(400)

    def set_colors(self, colors: List[str]) -> None:
        self._colors = colors
        self.update()

    def paintEvent(self, event):
        if not self._colors:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self._colors)
        swatch_w = min(80, self.width() // n - 8)
        total_w  = swatch_w * n + 8 * (n - 1)
        x_start  = (self.width() - total_w) // 2
        y        = (self.height() - 60) // 2

        for i, hex_color in enumerate(self._colors):
            x = x_start + i * (swatch_w + 8)
            painter.setBrush(QColor(hex_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(x, y, swatch_w, 60, 10, 10)


# ── TOP 3 ─────────────────────────────────────────────────────────────────
class _Top3Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.setSpacing(16)

    def set_top3(self, top3: list) -> None:
        # 기존 위젯 제거
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, item in enumerate(top3[:3]):
            card = _Top3Card(
                rank=i + 1,
                label=item.get("label_ko", ""),
                conf=item.get("confidence", 0.0),
                is_best=(i == 0),
            )
            self._layout.addWidget(card)


class _Top3Card(QWidget):
    def __init__(self, rank: int, label: str, conf: float, is_best: bool, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 72)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        accent = GOLD if is_best else "#555"

        top = QLabel(f"{'👑 ' if is_best else ''}{rank}위  {label}")
        top.setStyleSheet(f"color: {accent}; font-size: 14px;")
        conf_lbl = QLabel(f"{conf * 100:.1f}%")
        conf_lbl.setStyleSheet(f"color: #888; font-size: 13px;")
        layout.addWidget(top)
        layout.addWidget(conf_lbl)

        self.setStyleSheet(
            f"background: {CARD}; border: 1px solid {accent}; border-radius: 10px;"
        )


# ── 버튼 ─────────────────────────────────────────────────────────────────
class _FillButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(260, 60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: {GOLD}; color: {DARK};
                font-family: Georgia; font-size: 18px;
                border-radius: 30px; border: none; font-weight: bold;
            }}
            QPushButton:hover {{ background: #D4B47E; }}
            QPushButton:pressed {{ background: #B8935E; }}
        """)


class _OutlineButton(QPushButton):
    def __init__(self, text: str, accent: str = GOLD, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(160, 56)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {accent};
                font-size: 16px; border: 1px solid {accent};
                border-radius: 28px;
            }}
            QPushButton:hover {{ background: rgba(200,180,140,0.1); }}
        """)
