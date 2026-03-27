import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
try: import mediapipe; _MP=True
except: _MP=False
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from tabs.tab_dataset    import DatasetTab
from tabs.tab_preprocess import PreprocessTab
from tabs.tab_train      import TrainTab
from tabs.tab_inference  import InferenceTab
from tabs.tab_autolabel  import AutoLabelTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("AI Personal Color & Fashion Platform  v2.0"); self.setMinimumSize(1280,820)
        self._build_ui(); self._wire(); self._style()
    def _build_ui(self):
        c=QWidget(); self.setCentralWidget(c); ml=QVBoxLayout(c); ml.setContentsMargins(8,6,8,6); ml.setSpacing(6)
        h=QLabel("AI Personal Color & Fashion Integrated Learning Platform  v2.0")
        h.setAlignment(Qt.AlignmentFlag.AlignCenter); h.setFont(QFont("Arial",14,QFont.Weight.Bold)); h.setStyleSheet("color:#2C3E50; padding:4px 0;"); ml.addWidget(h)
        self._tabs=QTabWidget(); self._tabs.setFont(QFont("Arial",10))
        self._d=DatasetTab(); self._p=PreprocessTab(); self._t=TrainTab(); self._i=InferenceTab(); self._a=AutoLabelTab()
        self._tabs.addTab(self._d,"📁  1. 데이터셋"); self._tabs.addTab(self._p,"🔬  2. 전처리")
        self._tabs.addTab(self._t,"🚀  3. 학습");     self._tabs.addTab(self._i,"🎯  4. 추론 & 추천")
        self._tabs.addTab(self._a,"🏷️  5. 자동 라벨링"); ml.addWidget(self._tabs)
        cu="✅" if torch.cuda.is_available() else "❌"; mp="✅" if _MP else "❌"
        self.statusBar().showMessage(f"준비 완료  |  PyTorch {torch.__version__}  |  CUDA: {cu}  |  MediaPipe: {mp}")
    def _wire(self):
        self._d.dataset_loaded.connect(self._p.set_dataset_dir)
        self._d.dataset_loaded.connect(self._t.set_dataset_dir)
        self._d.dataset_loaded.connect(lambda p: self.statusBar().showMessage(f"데이터셋 로드됨: {p}"))
    def _style(self):
        self.setStyleSheet("""
            QMainWindow{background:#F4F6F9}
            QGroupBox{font-weight:bold;border:1px solid #D0D7DE;border-radius:7px;margin-top:8px;padding-top:10px;background:white}
            QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 6px;color:#3D5A73}
            QPushButton{background:#3A7BD5;color:white;border:none;border-radius:5px;padding:6px 14px;font-weight:bold}
            QPushButton:hover{background:#2E63B8} QPushButton:pressed{background:#254F99} QPushButton:disabled{background:#B0BAC8;color:#7A8490}
            QTabWidget::pane{border:1px solid #D0D7DE;border-radius:5px;background:#F4F6F9}
            QTabBar::tab{background:#E4EAF0;border:1px solid #D0D7DE;padding:8px 16px;margin-right:3px;border-radius:5px 5px 0 0;color:#5A6A7A}
            QTabBar::tab:selected{background:white;color:#3A7BD5;font-weight:bold;border-bottom-color:white}
            QProgressBar{border:1px solid #D0D7DE;border-radius:5px;text-align:center;background:#EEF1F5;height:18px}
            QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #3A7BD5,stop:1 #00D2FF);border-radius:4px}
            QTableWidget{gridline-color:#E0E5EC;alternate-background-color:#F7F9FC;selection-background-color:#D6E4F7}
            QHeaderView::section{background:#E4EAF0;border:1px solid #D0D7DE;padding:5px;font-weight:bold;color:#3D5A73}
            QLineEdit{border:1px solid #D0D7DE;border-radius:4px;padding:4px 8px;background:white}
            QTextEdit{border:1px solid #D0D7DE;border-radius:4px}
            QCheckBox{font-weight:bold;color:#2C3E50;spacing:6px}
            QCheckBox::indicator{width:16px;height:16px;border-radius:3px;border:1px solid #3A7BD5}
            QCheckBox::indicator:checked{background:#3A7BD5}
            QSlider::groove:horizontal{height:6px;background:#D0D7DE;border-radius:3px}
            QSlider::handle:horizontal{width:16px;height:16px;border-radius:8px;background:#3A7BD5;margin:-5px 0}
            QSlider::sub-page:horizontal{background:#3A7BD5;border-radius:3px}
        """)
    def closeEvent(self, e):
        for w in [getattr(self._d,"_worker",None), getattr(self._p,"_worker",None),
                  getattr(self._t,"_worker",None), getattr(self._i,"_worker",None), getattr(self._a,"_worker",None)]:
            if w and hasattr(w,"stop") and w.isRunning(): w.stop(); w.wait(3000)
        e.accept()
