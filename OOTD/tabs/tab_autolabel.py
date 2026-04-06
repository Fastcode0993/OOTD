import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, List, Optional
import matplotlib; matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QDoubleSpinBox, QSlider,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QSplitter, QHeaderView, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from workers import AutoLabelWorker
from constants import SEASON_CLASSES

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


class AutoLabelTab(QWidget):
    _SC={"Spring":"#FFB347","Summer":"#87CEEB","Autumn":"#CD853F","Winter":"#6495ED"}
    def __init__(self, parent=None):
        super().__init__(parent); self._mp=""; self._sd=""; self._od=""; self._worker=None; self._build_ui()
    def _make_row(self, lay, lbl, ph, slot):
        row=QHBoxLayout(); l=QLabel(lbl); l.setFixedWidth(120)
        e=QLineEdit(); e.setReadOnly(True); e.setPlaceholderText(ph)
        b=QPushButton("찾기"); b.setFixedWidth(60); b.clicked.connect(slot)
        row.addWidget(l); row.addWidget(e); row.addWidget(b); lay.addLayout(row); return e
    def _build_ui(self):
        root=QVBoxLayout(self); root.setSpacing(10)
        fg=QGroupBox("📂 파일 및 폴더 설정"); fl=QVBoxLayout(fg)
        self._me=self._make_row(fl,"모델 파일 (.pt):","학습 완료된 .pt 파일...",self._pm)
        self._se=self._make_row(fl,"소스 폴더:","미분류 이미지 폴더...",self._ps)
        self._oe=self._make_row(fl,"출력 폴더:","분류 결과 저장 위치...",self._po)
        root.addWidget(fg)
        tg=QGroupBox("🎚️ 분류 임계값 (Confidence Threshold)"); tr=QHBoxLayout(tg)
        tr.addWidget(QLabel("임계값:"))
        self._tspin=QDoubleSpinBox(); self._tspin.setRange(0.50,0.99); self._tspin.setDecimals(2)
        self._tspin.setValue(0.85); self._tspin.setSingleStep(0.05); self._tspin.setFixedWidth(80)
        self._tspin.valueChanged.connect(self._sv2sl); tr.addWidget(self._tspin)
        self._tsl=QSlider(Qt.Orientation.Horizontal); self._tsl.setRange(50,99); self._tsl.setValue(85)
        self._tsl.setTickInterval(5); self._tsl.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._tsl.valueChanged.connect(self._sl2sv); tr.addWidget(self._tsl,1)
        td=QLabel("이 값 이상의 신뢰도를 가진 이미지만 분류됩니다."); td.setWordWrap(True); td.setStyleSheet("color:#555; font-size:11px;"); tr.addWidget(td)
        root.addWidget(tg)
        br=QHBoxLayout()
        self._bst=QPushButton("🚀 자동 라벨링 시작"); self._bst.setEnabled(False)
        self._bst.setFont(QFont("Arial",11,QFont.Weight.Bold)); self._bst.setMinimumHeight(38); self._bst.clicked.connect(self._start)
        self._bsp=QPushButton("■ 중지"); self._bsp.setEnabled(False); self._bsp.setMinimumWidth(80); self._bsp.setMinimumHeight(38); self._bsp.clicked.connect(self._stop)
        br.addWidget(self._bst,1); br.addWidget(self._bsp); root.addLayout(br)
        self._bar=QProgressBar(); self._bar.setRange(0,100); self._bar.setFormat("%p%"); root.addWidget(self._bar)
        self._cl=QLabel("대기 중..."); self._cl.setAlignment(Qt.AlignmentFlag.AlignCenter); self._cl.setStyleSheet("color:#2C3E50; font-size:11px;"); root.addWidget(self._cl)
        sp=QSplitter(Qt.Orientation.Horizontal)
        lg=QGroupBox("📋 실시간 로그"); ll=QVBoxLayout(lg)
        self._log=QTextEdit(); self._log.setReadOnly(True); self._log.setFont(QFont("Courier New",9)); self._log.setStyleSheet("background:#1E1E1E; color:#E0E0E0;")
        ll.addWidget(self._log); sp.addWidget(lg)
        sg=QGroupBox("📊 분류 통계"); sl=QVBoxLayout(sg)
        self._tbl=QTableWidget(0,3); self._tbl.setHorizontalHeaderLabels(["클래스","분류 수","비율"])
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self._tbl.setAlternatingRowColors(True); sl.addWidget(self._tbl)
        self._sf=Figure(figsize=(3,3),tight_layout=True); self._sc=FigureCanvas(self._sf); sl.addWidget(self._sc)
        self._sum=QLabel(""); self._sum.setAlignment(Qt.AlignmentFlag.AlignCenter); self._sum.setWordWrap(True); sl.addWidget(self._sum)
        sp.addWidget(sg); sp.setSizes([550,330]); root.addWidget(sp,1)
    def _sv2sl(self, v): self._tsl.blockSignals(True); self._tsl.setValue(int(v*100)); self._tsl.blockSignals(False)
    def _sl2sv(self, v): self._tspin.blockSignals(True); self._tspin.setValue(v/100.); self._tspin.blockSignals(False)
    def _pm(self):
        p,_=QFileDialog.getOpenFileName(self,"모델 파일","","PyTorch (*.pt *.pth)")
        if p: self._mp=p; self._me.setText(p); self._chk()
    def _ps(self):
        d=QFileDialog.getExistingDirectory(self,"소스 폴더","",QFileDialog.Option.ShowDirsOnly)
        if d: self._sd=d; self._se.setText(d); self._chk()
    def _po(self):
        d=QFileDialog.getExistingDirectory(self,"출력 폴더","",QFileDialog.Option.ShowDirsOnly)
        if d: self._od=d; self._oe.setText(d); self._chk()
    def _chk(self): self._bst.setEnabled(bool(self._mp and self._sd and self._od))
    def _start(self):
        if self._worker and self._worker.isRunning(): self._worker.stop(); self._worker.wait(2000)
        self._log.clear(); self._tbl.setRowCount(0); self._sf.clear(); self._sc.draw(); self._sum.setText(""); self._bar.setValue(0)
        thr=self._tspin.value(); self._bst.setEnabled(False); self._bsp.setEnabled(True)
        self._app(f"[START] 임계값={thr:.0%}  소스:{self._sd}  출력:{self._od}")
        self._worker=AutoLabelWorker(self._mp,self._sd,self._od,thr)
        self._worker.log_message.connect(self._app); self._worker.progress.connect(self._prog)
        self._worker.finished.connect(self._fin); self._worker.error.connect(self._err); self._worker.start()
    def _stop(self):
        if self._worker and self._worker.isRunning(): self._worker.stop()
        self._bst.setEnabled(True); self._bsp.setEnabled(False); self._cl.setText("중단됨")
    def _app(self, m):
        self._log.append(m); sb=self._log.verticalScrollBar(); sb.setValue(sb.maximum())
    def _prog(self, pct, fn, pc, cf):
        self._bar.setValue(pct)
        ic="✅" if cf*100>=self._tspin.value()*100 else "⏭️"
        self._cl.setText(f"{ic}  {fn}  ->  {pc}  ({cf*100:.1f}%)")
    def _fin(self, stats):
        self._bst.setEnabled(True); self._bsp.setEnabled(False)
        cls=[c for c in SEASON_CLASSES if c in stats]; lb={c:stats.get(c,0) for c in cls}
        tot=sum(lb.values()); sk=stats.get("skipped_low_conf",0); er=stats.get("skipped_error",0)
        self._tbl.setRowCount(0)
        for c in cls:
            cnt=lb[c]; r=self._tbl.rowCount(); self._tbl.insertRow(r)
            it=QTableWidgetItem(c); it.setBackground(QColor(self._SC.get(c,"#ccc"))); self._tbl.setItem(r,0,it)
            self._tbl.setItem(r,1,QTableWidgetItem(str(cnt)))
            self._tbl.setItem(r,2,QTableWidgetItem(f"{cnt/max(tot,1)*100:.1f}%" if tot else "-"))
        self._draw_chart(lb); self._sum.setText(f"✅ 분류:{tot}  ⏭️ 건너뜀:{sk}  ❌ 오류:{er}"); self._cl.setText("🎉 자동 라벨링 완료!")
    def _draw_chart(self, lb):
        self._sf.clear(); ax=self._sf.add_subplot(111)
        cls=[c for c in SEASON_CLASSES if c in lb and lb[c]>0]; vs=[lb[c] for c in cls]; co=[self._SC.get(c,"#95A5A6") for c in cls]
        if not vs or not sum(vs): ax.text(0.5,0.5,"분류된 이미지 없음",ha="center",va="center",transform=ax.transAxes,fontsize=10)
        else:
            bars=ax.bar(cls,vs,color=co,edgecolor="white",linewidth=0.8)
            for b,v in zip(bars,vs): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,str(v),ha="center",va="bottom",fontsize=9)
            ax.set_ylabel("분류 수"); ax.set_title("시즌별 분류 결과",fontsize=10); ax.grid(True,axis="y",alpha=0.3)
        self._sf.tight_layout(); self._sc.draw()
    def _err(self, m): self._bst.setEnabled(True); self._bsp.setEnabled(False); self._app(f"❌ {m}"); self._cl.setText("❌ 오류 발생")
