import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Optional, Tuple
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QSplitter, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont
from constants import SEASON_PALETTES
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

        ig = QGroupBox("🖼️ 입력 이미지")
        il = QVBoxLayout(ig)
        self._il = QLabel("이미지 없음")
        self._il.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._il.setMinimumSize(260, 320)
        self._il.setStyleSheet("border:1px solid #ddd; background:#fafafa;")
        il.addWidget(self._il)
        sp.addWidget(ig)

        cg = QGroupBox("📊 분류 결과")
        cl = QVBoxLayout(cg)
        self._rl = QLabel("결과 대기 중...")
        self._rl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rl.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        cl.addWidget(self._rl)
        self._cf = Figure(figsize=(3.5, 3.5), tight_layout=True)
        self._cc = FigureCanvas(self._cf)
        cl.addWidget(self._cc)
        sp.addWidget(cg)

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
        sp.setSizes([285, 285, 210])
        root.addWidget(sp, 1)

        self._st = QLabel("모델과 이미지를 선택하세요.")
        self._st.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._st)

    def _pm(self):
        p, _ = QFileDialog.getOpenFileName(self, "모델 파일", "", "PyTorch (*.pt *.pth)")
        if p:
            self._mp = p
            self._me.setText(p)
            self._chk()

    def _pi(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "이미지", "", "이미지 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if p:
            self._ip = p
            self._ie.setText(p)
            px = QPixmap(p)
            if not px.isNull():
                self._il.setPixmap(px.scaled(
                    260, 320,
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

    def _done(self, pc, rk):
        self._btn.setEnabled(True)
        A = {"Spring":"#FF7043","Summer":"#1E88E5","Autumn":"#8D6E63","Winter":"#5C6BC0"}
        E = {"Spring":"🌸","Summer":"☀️","Autumn":"🍂","Winter":"❄️"}
        c = A.get(pc, "#607D8B")
        e = E.get(pc, "🎨")
        self._rl.setText(f"{e}  {pc}")
        self._rl.setStyleSheet(f"color:{c}; font-size:22px; font-weight:bold;")
        self._draw(rk)
        self._pal(pc)
        tp = rk[0][1] * 100 if rk else 0
        self._st.setText(f"✅ 예측: {pc}  (Confidence: {tp:.1f}%)")

    def _draw(self, rk):
        self._cf.clear()
        ax = self._cf.add_subplot(111)
        C  = {"Spring":"#FFB347","Summer":"#87CEEB","Autumn":"#CD853F","Winter":"#6495ED"}
        cn = [r[0] for r in rk]
        pt = [r[1] * 100 for r in rk]
        co = [C.get(c, "#95A5A6") for c in cn]
        bars = ax.barh(cn, pt, color=co, edgecolor="white", linewidth=0.7)
        for b, p in zip(bars, pt):
            ax.text(min(p + 1.5, 98), b.get_y() + b.get_height() / 2,
                    f"{p:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_xlabel("Confidence (%)")
        ax.set_title("분류 확률", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
        self._cf.tight_layout()
        self._cc.draw()

    def _pal(self, s):
        pl = SEASON_PALETTES.get(s, [])
        self._pt.setText(f"🎨  {s} 타입 추천 팔레트")
        for i, (sw, nm) in enumerate(zip(self._sw, self._sn)):
            if i < len(pl):
                hx, cn = pl[i]
                sw.setStyleSheet(f"background:{hx}; border:1px solid #888; border-radius:5px;")
                nm.setText(f"{cn}\n{hx}")
            else:
                sw.setStyleSheet("background:#f0f0f0; border:1px solid #ccc; border-radius:5px;")
                nm.setText("")

    def _err(self, m):
        self._btn.setEnabled(True)
        self._rl.setText("오류 발생")
        self._st.setText(f"❌ {m}")
