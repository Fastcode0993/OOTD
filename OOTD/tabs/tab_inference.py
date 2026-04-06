import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Optional, Tuple, Dict
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QSplitter, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont
from constants import SEASON_PALETTES, SEASON_TONE_PALETTES, parse_season_tone
from workers import InferenceWorker

# AUTO-INJECTED: Korean font setup for matplotlib
import os as _os
import matplotlib.font_manager as _fm
import matplotlib.pyplot as _plt
if not any('NanumGothic' in f.name for f in _fm.fontManager.ttflist):
    for _font in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                  '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf']:
        if _os.path.exists(_font):
            _fm.fontManager.addfont(_font)
_plt.rcParams.update({'font.family': 'NanumGothic', 'axes.unicode_minus': False})
del _os, _fm, _plt
# END AUTO-INJECTED Korean font setup

# 계절별 색상/이모지
_SEASON_COLOR = {"Spring":"#FF7043","Summer":"#1E88E5","Autumn":"#8D6E63","Winter":"#5C6BC0"}
_SEASON_EMOJI = {"Spring":"🌸","Summer":"☀️","Autumn":"🍂","Winter":"❄️"}
_TONE_EMOJI   = {"Warm":"🔥","Bright":"✨","Light":"🌤️"}
# 12클래스용 바 색상
_CLASS_COLOR = {
    "Spring_Warm":"#FF8C69","Spring_Bright":"#FF6B35","Spring_Light":"#FFB6C1",
    "Summer_Warm":"#C9A0A0","Summer_Bright":"#6495ED","Summer_Light":"#B8C8D8",
    "Autumn_Warm":"#CD853F","Autumn_Bright":"#B7410E","Autumn_Light":"#DEB887",
    "Winter_Warm":"#800020","Winter_Bright":"#000080","Winter_Light":"#A8C0D0",
}


class InferenceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mp = ""
        self._ip = ""
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        sel = QGroupBox("📂 파일 선택")
        sr  = QHBoxLayout(sel)
        sr.addWidget(QLabel("모델(.pt):"))
        self._me = QLineEdit()
        self._me.setReadOnly(True)
        self._me.setPlaceholderText("학습 완료된 .pt 파일...")
        bm = QPushButton("찾기")
        bm.setFixedWidth(60)
        bm.clicked.connect(self._pm)
        sr.addWidget(self._me)
        sr.addWidget(bm)
        sr.addSpacing(15)

        sr.addWidget(QLabel("이미지:"))
        self._ie = QLineEdit()
        self._ie.setReadOnly(True)
        self._ie.setPlaceholderText("테스트 이미지...")
        bi = QPushButton("찾기")
        bi.setFixedWidth(60)
        bi.clicked.connect(self._pi)
        sr.addWidget(self._ie)
        sr.addWidget(bi)

        self._btn = QPushButton("🔍 분석 시작")
        self._btn.setEnabled(False)
        self._btn.setMinimumWidth(110)
        self._btn.clicked.connect(self._run)
        sr.addWidget(self._btn)
        root.addWidget(sel)

        sp = QSplitter(Qt.Orientation.Horizontal)

        # 입력 이미지
        ig = QGroupBox("🖼️ 입력 이미지")
        il = QVBoxLayout(ig)
        self._il = QLabel("이미지 없음")
        self._il.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._il.setMinimumSize(240, 300)
        self._il.setStyleSheet("border:1px solid #ddd; background:#fafafa;")
        il.addWidget(self._il)
        sp.addWidget(ig)

        # 결과 패널
        cg = QGroupBox("📊 분류 결과")
        cl = QVBoxLayout(cg)

        # 메인 결과 레이블 (계절_톤)
        self._rl = QLabel("결과 대기 중...")
        self._rl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rl.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        cl.addWidget(self._rl)

        # 계절 집계 확률 바 차트
        self._cf = Figure(figsize=(3.5, 2.2), tight_layout=True)
        self._cc = FigureCanvas(self._cf)
        cl.addWidget(self._cc)

        # 12클래스 상세 바 차트
        self._df = Figure(figsize=(3.5, 3.5), tight_layout=True)
        self._dc = FigureCanvas(self._df)
        cl.addWidget(self._dc)
        sp.addWidget(cg)

        # 팔레트 패널
        pg = QGroupBox("👗 추천 패션 팔레트")
        pl = QVBoxLayout(pg)
        self._pt = QLabel("시즌 분류 후 팔레트가 표시됩니다")
        self._pt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pt.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        pl.addWidget(self._pt)

        self._sw: List[QLabel] = []
        self._sn: List[QLabel] = []
        for _ in range(6):
            rw = QWidget()
            rl = QHBoxLayout(rw)
            rl.setContentsMargins(4, 2, 4, 2)
            s = QLabel()
            s.setFixedSize(55, 38)
            s.setStyleSheet("border:1px solid #888; border-radius:5px; background:#eee;")
            n = QLabel("—")
            n.setFont(QFont("Arial", 9))
            n.setWordWrap(True)
            self._sw.append(s)
            self._sn.append(n)
            rl.addWidget(s)
            rl.addWidget(n)
            rl.addStretch()
            pl.addWidget(rw)
        pl.addStretch()
        sp.addWidget(pg)
        sp.setSizes([240, 360, 200])
        root.addWidget(sp, 1)

        self._st = QLabel("모델과 이미지를 선택하세요.")
        self._st.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._st)

    def _pm(self):
        p, _ = QFileDialog.getOpenFileName(self, "모델 파일", "", "PyTorch (*.pt *.pth)")
        if p:
            self._mp = p; self._me.setText(p); self._chk()

    def _pi(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "이미지", "", "이미지 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if p:
            self._ip = p; self._ie.setText(p)
            px = QPixmap(p)
            if not px.isNull():
                self._il.setPixmap(px.scaled(
                    240, 300,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            self._chk()

    def _chk(self):
        self._btn.setEnabled(bool(self._mp and self._ip))

    def _run(self):
        self._btn.setEnabled(False)
        self._rl.setText("분석 중...")
        self._st.setText("추론 중...")
        self._worker = InferenceWorker(self._mp, self._ip)
        self._worker.finished.connect(self._done)
        self._worker.error.connect(self._err)
        self._worker.start()

    def _done(self, pc: str, rk: list, season_probs: dict):
        self._btn.setEnabled(True)

        # pc 파싱: "Spring_Warm" or "Spring"
        season, tone = parse_season_tone(pc)
        s_color = _SEASON_COLOR.get(season, "#607D8B")
        s_emoji = _SEASON_EMOJI.get(season, "🎨")
        t_emoji = _TONE_EMOJI.get(tone, "") if tone else ""

        if tone:
            display = f"{s_emoji} {season}  {t_emoji} {tone}"
        else:
            display = f"{s_emoji} {season}"

        self._rl.setText(display)
        self._rl.setStyleSheet(
            f"color:{s_color}; font-size:18px; font-weight:bold;")

        self._draw_season(season_probs)
        self._draw_detail(rk)
        self._pal(pc)

        top_conf = rk[0][1] * 100 if rk else 0
        self._st.setText(
            f"✅ 예측: {pc}  (Confidence: {top_conf:.1f}%)")

    def _draw_season(self, season_probs: dict):
        """계절 집계 확률 바 차트"""
        self._cf.clear()
        ax = self._cf.add_subplot(111)
        seasons = [s for s in ["Spring","Summer","Autumn","Winter"]
                   if s in season_probs]
        vals  = [season_probs[s] * 100 for s in seasons]
        colors= [_SEASON_COLOR.get(s,"#95A5A6") for s in seasons]
        bars  = ax.bar(seasons, vals, color=colors, edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("확률 (%)", fontsize=8)
        ax.set_title("계절 확률 (집계)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        self._cf.tight_layout()
        self._cc.draw()

    def _draw_detail(self, rk: list):
        """12클래스 상세 수평 바 차트"""
        self._df.clear()
        ax = self._df.add_subplot(111)
        # 상위 12개 (또는 전체)
        top = rk[:12]
        labels = [r[0] for r in reversed(top)]
        vals   = [r[1] * 100 for r in reversed(top)]
        colors = [_CLASS_COLOR.get(l, "#95A5A6") for l in labels]
        bars = ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(min(v + 0.5, 98), b.get_y() + b.get_height()/2,
                    f"{v:.1f}%", va="center", fontsize=7)
        ax.set_xlim(0, 115)
        ax.set_xlabel("Confidence (%)", fontsize=8)
        ax.set_title("클래스별 상세 확률", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=7)
        self._df.tight_layout()
        self._dc.draw()

    def _pal(self, pc: str):
        """팔레트 표시 — 12클래스 팔레트 우선, 없으면 4계절 팔레트"""
        # 12클래스 팔레트
        pl = SEASON_TONE_PALETTES.get(pc)
        if pl is None:
            # 4클래스 팔레트 (하위 호환)
            season, _ = parse_season_tone(pc)
            pl = SEASON_PALETTES.get(season, [])

        season, tone = parse_season_tone(pc)
        t_label = f"  {_TONE_EMOJI.get(tone,'')} {tone}" if tone else ""
        self._pt.setText(
            f"🎨  {_SEASON_EMOJI.get(season,'')} {season}{t_label} 추천 팔레트")

        for i, (sw, nm) in enumerate(zip(self._sw, self._sn)):
            if i < len(pl):
                hx, cn = pl[i]
                sw.setStyleSheet(
                    f"background:{hx}; border:1px solid #888; border-radius:5px;")
                nm.setText(f"{cn}\n{hx}")
            else:
                sw.setStyleSheet(
                    "background:#f0f0f0; border:1px solid #ccc; border-radius:5px;")
                nm.setText("")

    def _err(self, m):
        self._btn.setEnabled(True)
        self._rl.setText("오류 발생")
        self._st.setText(f"❌ {m}")



