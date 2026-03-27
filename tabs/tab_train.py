import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Optional
import matplotlib; matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QProgressBar,
    QSplitter, QLineEdit, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from workers import TrainWorker

class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._dir=""; self._worker=None
        self._tl: List[float]=[]; self._vl: List[float]=[]
        self._ta: List[float]=[]; self._va: List[float]=[]
        self._build_ui()
    def _build_ui(self):
        root=QVBoxLayout(self); root.setSpacing(10)
        hp=QGroupBox("🔧 학습 하이퍼파라미터"); hr=QHBoxLayout(hp)
        hr.addWidget(QLabel("에포크:"))
        self._ep=QSpinBox(); self._ep.setRange(1,500); self._ep.setValue(10); hr.addWidget(self._ep)
        hr.addWidget(QLabel("배치:"))
        self._bs=QSpinBox(); self._bs.setRange(4,256); self._bs.setValue(16); hr.addWidget(self._bs)
        hr.addWidget(QLabel("LR:"))
        self._lr=QDoubleSpinBox(); self._lr.setRange(1e-6,0.1); self._lr.setDecimals(6); self._lr.setValue(1e-3); self._lr.setSingleStep(1e-4); hr.addWidget(self._lr)
        hr.addWidget(QLabel("디바이스:"))
        self._dev=QComboBox(); self._dev.addItems(["cpu","cuda","mps"]); hr.addWidget(self._dev)
        hr.addStretch()
        hr.addWidget(QLabel("저장:"))
        self._sv=QLineEdit("best_model.pt"); self._sv.setMinimumWidth(160); hr.addWidget(self._sv)
        bs2=QPushButton("📂"); bs2.setFixedWidth(28); bs2.clicked.connect(self._pick); hr.addWidget(bs2)
        root.addWidget(hp)
        sp=QSplitter(Qt.Orientation.Horizontal)
        lg=QGroupBox("📋 학습 로그"); ll=QVBoxLayout(lg)
        self._log=QTextEdit(); self._log.setReadOnly(True); self._log.setFont(QFont("Courier New",9))
        self._log.setStyleSheet("background:#1E1E1E; color:#00FF7F;"); ll.addWidget(self._log); sp.addWidget(lg)
        cg=QGroupBox("📉 학습 커브 (실시간)"); cl=QVBoxLayout(cg)
        self._fig=Figure(figsize=(5,4),tight_layout=True); self._canvas=FigureCanvas(self._fig)
        self._init_chart(); cl.addWidget(self._canvas); sp.addWidget(cg); sp.setSizes([430,430]); root.addWidget(sp,1)
        bot=QHBoxLayout()
        self._bar=QProgressBar(); self._bar.setRange(0,100); bot.addWidget(self._bar,1)
        self._btn_tr=QPushButton("▶ 학습 시작"); self._btn_tr.setEnabled(False); self._btn_tr.setMinimumWidth(110); self._btn_tr.clicked.connect(self._start)
        self._btn_st=QPushButton("■ 중지"); self._btn_st.setEnabled(False); self._btn_st.setMinimumWidth(70); self._btn_st.clicked.connect(self._stop)
        bot.addWidget(self._btn_tr); bot.addWidget(self._btn_st); root.addLayout(bot)
    def _init_chart(self):
        self._fig.clear(); self._axL=self._fig.add_subplot(211); self._axA=self._fig.add_subplot(212)
        for ax,t,y in [(self._axL,"Loss","Loss"),(self._axA,"Accuracy (%)","Acc (%)")]:
            ax.set_title(t,fontsize=10); ax.set_xlabel("Epoch"); ax.set_ylabel(y); ax.grid(True,alpha=0.3)
        self._canvas.draw()
    def set_dataset_dir(self, path):
        self._dir=path; self._btn_tr.setEnabled(True)
    def _pick(self):
        p,_=QFileDialog.getSaveFileName(self,"모델 저장","best_model.pt","PyTorch 모델 (*.pt *.pth)")
        if p: self._sv.setText(p)
    def _start(self):
        if not self._dir: self._app("[ERROR] 데이터셋 경로 없음"); return
        self._tl.clear(); self._vl.clear(); self._ta.clear(); self._va.clear()
        self._init_chart(); self._log.clear()
        cfg={"data_dir":self._dir,"save_path":self._sv.text() or "best_model.pt",
             "epochs":self._ep.value(),"batch_size":self._bs.value(),"lr":self._lr.value(),"device":self._dev.currentText()}
        self._btn_tr.setEnabled(False); self._btn_st.setEnabled(True)
        self._worker=TrainWorker(cfg)
        self._worker.log_message.connect(self._app); self._worker.epoch_complete.connect(self._upd)
        self._worker.progress.connect(self._bar.setValue); self._worker.finished.connect(self._done)
        self._worker.error.connect(self._err); self._worker.start()
    def _stop(self):
        if self._worker and self._worker.isRunning(): self._worker.stop()
        self._btn_tr.setEnabled(True); self._btn_st.setEnabled(False)
    def _app(self, msg):
        self._log.append(msg); sb=self._log.verticalScrollBar(); sb.setValue(sb.maximum())
    def _upd(self, ep, tl, ta, vl, va):
        self._tl.append(tl); self._vl.append(vl); self._ta.append(ta); self._va.append(va)
        xs=list(range(1,len(self._tl)+1))
        for ax,td,vd,t,y,yl in [(self._axL,self._tl,self._vl,"Loss","Loss",None),(self._axA,self._ta,self._va,"Accuracy (%)","Acc (%)",(0,105))]:
            ax.clear(); ax.plot(xs,td,"b-o",label="Train",markersize=4); ax.plot(xs,vd,"r-s",label="Val",markersize=4)
            ax.set_title(t,fontsize=10); ax.set_xlabel("Epoch"); ax.set_ylabel(y); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
            if yl: ax.set_ylim(*yl)
        self._fig.tight_layout(); self._canvas.draw()
    def _done(self, p): self._btn_tr.setEnabled(True); self._btn_st.setEnabled(False); self._app(f"\n✅ 완료 -> {p}")
    def _err(self, m): self._btn_tr.setEnabled(True); self._btn_st.setEnabled(False); self._app(f"❌ {m}")
