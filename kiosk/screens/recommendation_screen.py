"""
kiosk/screens/recommendation_screen.py
────────────────────────────────────────
추천 아이템 화면: 패션/메이크업/헤어/인테리어 탭으로 구성.
"""

from __future__ import annotations

from typing import Any, Dict, List

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QFrame, QGridLayout,
)

GOLD  = "#FFE08A"
CREAM = "#F5EFE6"
DARK  = "#101426"
CARD  = "#182039"
CARD2 = "#202A4F"

_CATEGORIES = [
    ("fashion",  "👗 패션"),
    ("makeup",   "💄 메이크업"),
    ("hair",     "💇 헤어"),
    ("interior", "🪴 인테리어"),
]


class RecommendationScreen(QWidget):
    qr_requested   = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            background:
                qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #101426,
                    stop:0.34 #142345,
                    stop:0.7 #242B5A,
                    stop:1 #301C39);
        """)
        self._reco_data: Dict[str, List] = {}
        self._current_tab = "fashion"
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 28, 40, 28)
        layout.setSpacing(16)

        # 헤더
        header = QLabel("✦  나에게 어울리는 아이템")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            f"color: {GOLD}; font-size: 14px; letter-spacing: 4px;"
        )
        layout.addWidget(header)

        self._title = QLabel("")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            f"color: {CREAM}; font-family: Georgia; font-size: 28px;"
        )
        layout.addWidget(self._title)

        # 탭 버튼
        tab_row = QHBoxLayout()
        tab_row.setSpacing(8)
        self._tab_buttons: Dict[str, QPushButton] = {}
        for key, label in _CATEGORIES:
            btn = _TabButton(label, key == self._current_tab)
            btn.clicked.connect(lambda _, k=key: self._switch_tab(k))
            self._tab_buttons[key] = btn
            tab_row.addWidget(btn)
        layout.addLayout(tab_row)

        # 스크롤 아이템 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._items_container = QWidget()
        self._items_container.setStyleSheet("background: transparent;")
        self._items_grid = QGridLayout(self._items_container)
        self._items_grid.setSpacing(14)
        scroll.setWidget(self._items_container)
        layout.addWidget(scroll)

        # 버튼 행
        btn_row = QHBoxLayout()
        back_btn = _SmallBtn("← 결과로")
        back_btn.clicked.connect(self.back_requested.emit)
        qr_btn = _GoldBtn("QR로 저장하기")
        qr_btn.clicked.connect(self.qr_requested.emit)
        btn_row.addWidget(back_btn)
        btn_row.addStretch()
        btn_row.addWidget(qr_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    def set_data(self, label_ko: str, recommendations: Dict[str, List]) -> None:
        self._title.setText(label_ko)
        self._reco_data = recommendations
        self._switch_tab(self._current_tab)

    def _switch_tab(self, key: str) -> None:
        self._current_tab = key
        for k, btn in self._tab_buttons.items():
            btn.set_active(k == key)
        self._render_items(self._reco_data.get(key, []))

    def _render_items(self, items: List[Dict]) -> None:
        # 기존 아이템 제거
        while self._items_grid.count():
            item = self._items_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not items:
            empty = QLabel("추천 아이템이 없습니다.")
            empty.setStyleSheet(f"color: #666; font-size: 16px;")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._items_grid.addWidget(empty, 0, 0)
            return

        cols = 3
        for i, item in enumerate(items):
            card = _ItemCard(item)
            self._items_grid.addWidget(card, i // cols, i % cols)


# ── 아이템 카드 ───────────────────────────────────────────────────────────
class _ItemCard(QWidget):
    def __init__(self, item: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 140)
        self.setStyleSheet(
            f"background: rgba(24,32,57,0.92); border-radius: 12px; border: 1px solid #38507D;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # 색상 스와치
        swatch = _ColorSwatch(item.get("color_hex", "#888"))
        layout.addWidget(swatch)

        # 텍스트
        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        name = QLabel(item.get("item_name_ko", ""))
        name.setStyleSheet(f"color: {CREAM}; font-size: 16px; font-weight: bold;")
        color_name = QLabel(item.get("color_name_ko", ""))
        color_name.setStyleSheet(f"color: {GOLD}; font-size: 13px;")
        live_meta = ""
        if item.get("product_price"):
            live_meta = f"{item.get('product_mall') or item.get('brand', '')} · {int(item['product_price']):,}원"
        elif item.get("product_mall"):
            live_meta = item.get("product_mall", "")
        product_meta = QLabel(live_meta)
        product_meta.setStyleSheet("color: #6EE7A8; font-size: 11px;")
        tip = QLabel(item.get("tip_ko", ""))
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #888; font-size: 12px;")
        text_col.addWidget(name)
        text_col.addWidget(color_name)
        if live_meta:
            text_col.addWidget(product_meta)
        text_col.addWidget(tip)
        layout.addLayout(text_col)


class _ColorSwatch(QWidget):
    def __init__(self, hex_color: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(48, 48)
        self._color = QColor(hex_color)
        self.setStyleSheet(f"border-radius: 24px;")

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self._color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self.rect())


# ── 버튼 ─────────────────────────────────────────────────────────────────
class _TabButton(QPushButton):
    def __init__(self, text: str, active: bool = False, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(46)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._active = active
        self._refresh_style()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._refresh_style()

    def _refresh_style(self) -> None:
        if self._active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #FF8FB3, stop:0.35 #FFE08A, stop:0.7 #6EE7A8, stop:1 #68D8FF);
                    color: {DARK};
                    border-radius: 23px; font-size: 15px;
                    font-weight: bold; padding: 0 20px; border: none;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(24,32,57,0.55); color: {GOLD};
                    border-radius: 23px; font-size: 15px;
                    border: 1px solid #38507D; padding: 0 20px;
                }}
                QPushButton:hover {{ background: rgba(201,169,110,0.1); }}
            """)


class _GoldBtn(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(220, 58)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8FB3, stop:0.35 #FFE08A, stop:0.7 #6EE7A8, stop:1 #68D8FF);
                color: {DARK};
                font-family: Georgia; font-size: 17px;
                font-weight: bold; border-radius: 29px; border: none;
            }}
            QPushButton:hover {{ color: {DARK}; }}
        """)


class _SmallBtn(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(140, 52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: #888;
                font-size: 15px; border: 1px solid #555; border-radius: 26px;
            }}
            QPushButton:hover {{ background: rgba(200,200,200,0.05); }}
        """)
