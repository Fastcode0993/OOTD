import sys, os, re, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import List, Optional
import matplotlib; matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QProgressBar,
    QSplitter, QLineEdit, QFileDialog, QCheckBox, QFrame, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont


# ─────────────────────────────────────────────────────────────
#  Subprocess 학습 워커
# ─────────────────────────────────────────────────────────────
class SubprocessTrainWorker(QThread):
    line_ready     = pyqtSignal(str)
    epoch_complete = pyqtSignal(int, float, float, float, float)
    progress       = pyqtSignal(int)
    finished       = pyqtSignal(str)
    error          = pyqtSignal(str)

    _RE_EPOCH = re.compile(
        r"\[(?:Stage\S*\s*)?\[?Ep(?:och)?\s+(\d+)/\d+\].*"
        r"Train Loss=([0-9.]+)\s+Acc=([0-9.]+)%.*"
        r"Val Loss=([0-9.]+)\s+Acc=([0-9.]+)%")
    _RE_PROG  = re.compile(r"Train\s+([0-9.]+)%")

    def __init__(self, script_path: str, args: List[str]):
        super().__init__()
        self._script    = script_path
        self._args      = args
        self._proc: Optional[subprocess.Popen] = None
        self._save_path = ""

    def run(self):
        cmd = [sys.executable, self._script] + self._args
        self.line_ready.emit(f"[CMD] {' '.join(cmd)}\n")
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"]       = "1"
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, env=env)
            for raw in self._proc.stdout:
                line = raw.rstrip()
                if not line: continue
                self.line_ready.emit(line)
                m = self._RE_EPOCH.search(line)
                if m:
                    ep = int(m.group(1))
                    tl, ta, vl, va = (float(m.group(i)) for i in range(2, 6))
                    self.epoch_complete.emit(ep, tl, ta, vl, va)
                    self.progress.emit(100)
                mp = self._RE_PROG.search(line)
                if mp:
                    self.progress.emit(int(float(mp.group(1))))
                if "[BEST]" in line and "->" in line:
                    self._save_path = line.split("->")[-1].strip()
            self._proc.wait()
            rc = self._proc.returncode
            if rc == 0: self.finished.emit(self._save_path)
            else:       self.error.emit(f"프로세스 종료 코드: {rc}")
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try: self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired: self._proc.kill()


# ─────────────────────────────────────────────────────────────
#  UI 헬퍼
# ─────────────────────────────────────────────────────────────
_GRP = (
    "QGroupBox{font-weight:600;font-size:12px;color:#1E293B;"
    "border:1px solid #CBD5E1;border-radius:8px;"
    "margin-top:10px;padding-top:6px;background:#FAFBFC;}"
    "QGroupBox::title{subcontrol-origin:margin;left:10px;"
    "padding:0 5px;color:#334155;background:#FAFBFC;border-radius:3px;}"
)
_SB = (
    "QSpinBox,QDoubleSpinBox{border:1px solid #CBD5E1;border-radius:6px;"
    "padding:4px 6px;font-size:13px;min-height:28px;background:white;color:#1E293B;}"
    "QSpinBox:focus,QDoubleSpinBox:focus{border-color:#2563EB;}"
)
_CB = (
    "QComboBox{border:1px solid #CBD5E1;border-radius:6px;"
    "padding:4px 8px;font-size:13px;min-height:28px;background:white;color:#1E293B;}"
    "QComboBox:focus{border-color:#2563EB;}"
    "QComboBox::drop-down{border:none;width:20px;}"
)

def _grp(t):
    g = QGroupBox(t); g.setStyleSheet(_GRP); return g

def _ibtn(icon, slot):
    b = QPushButton(icon); b.setFixedSize(28, 28)
    b.setStyleSheet(
        "QPushButton{border:1px solid #CBD5E1;border-radius:5px;"
        "background:#F1F5F9;font-size:13px;}"
        "QPushButton:hover{background:#E2E8F0;border-color:#94A3B8;}")
    b.clicked.connect(slot); return b

def _slbl(t):
    l = QLabel(t)
    l.setStyleSheet(
        "font-size:10px;font-weight:600;color:#64748B;"
        "letter-spacing:0.5px;padding:3px 0 1px 0;")
    return l

def _plbl(t, w=52):
    l = QLabel(t)
    l.setStyleSheet(f"font-size:12px;color:#374151;font-weight:600;min-width:{w}px;")
    return l

def _hline():
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color:#E2E8F0;margin:2px 0;"); return f

def _folder_edit(placeholder="", color="#334155", bg="white", border="#CBD5E1"):
    e = QLineEdit(); e.setReadOnly(True); e.setPlaceholderText(placeholder)
    e.setStyleSheet(
        f"QLineEdit{{border:1px solid {border};border-radius:5px;"
        f"padding:3px 7px;font-size:11px;background:{bg};color:{color};}}")
    return e


# ─────────────────────────────────────────────────────────────
#  데이터 행 위젯 빌더
# ─────────────────────────────────────────────────────────────
def _data_row(label: str, edit: QLineEdit, btn: QPushButton,
              lbl_color="#1E293B", lbl_bg="#F1F5F9") -> QHBoxLayout:
    row = QHBoxLayout(); row.setSpacing(4)
    lbl = QLabel(label); lbl.setFixedWidth(110)
    lbl.setStyleSheet(
        f"font-size:11px;font-weight:600;color:{lbl_color};"
        f"background:{lbl_bg};border-radius:4px;padding:2px 5px;")
    row.addWidget(lbl)
    row.addWidget(edit, 1)
    row.addWidget(btn)
    return row


# ─────────────────────────────────────────────────────────────
#  TrainTab
# ─────────────────────────────────────────────────────────────
class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 경로 저장소
        self._paths = {
            # 공통
            "raw":        "",   # 원본 데이터셋
            "pre":        "",   # 전처리 데이터셋
            # jpg / jpg_json
            "jpg":        "",   # JPG 폴더
            "json":       "",   # JSON 폴더
            # hierarchical
            "s1_jpg":     "",   # Stage1 JPG (4계절)
            "s1_json":    "",   # Stage1 JSON (4계절)
            "s2_jpg":     "",   # Stage2 JPG (12클래스)
            "s2_json":    "",   # Stage2 JSON (12클래스)
        }
        self._pre_mode_name = ""
        self._worker: Optional[SubprocessTrainWorker] = None
        self._tl: List[float] = []
        self._vl: List[float] = []
        self._ta: List[float] = []
        self._va: List[float] = []
        self._build_ui()

    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(5); root.setContentsMargins(8, 8, 8, 8)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(5)

        # ══ 왼쪽 설정 패널 ═══════════════════════════════════
        left = QWidget(); left.setMaximumWidth(440); left.setMinimumWidth(320)
        ll = QVBoxLayout(left); ll.setSpacing(5); ll.setContentsMargins(0,0,4,0)

        # ── 1. 학습 입력 모드 (최상단) ───────────────────────
        g1 = _grp("🧬  학습 입력 모드")
        g1l = QVBoxLayout(g1); g1l.setSpacing(4)

        self._feat_combo = QComboBox()
        self._feat_combo.setStyleSheet(_CB)
        self._feat_combo.addItem("🖼️  JPG  — 이미지만 학습",                "jpg")
        self._feat_combo.addItem("📄  JSON  — 피처 수치만 학습",             "json")
        self._feat_combo.addItem("🖼️＋📄  JPG + JSON  — 이미지＋피처 결합",  "jpg_json")
        self._feat_combo.addItem("🏗️  계층 분류  — 계절→톤 2단계 학습",      "hierarchical")
        self._feat_combo.currentIndexChanged.connect(self._on_mode_changed)
        g1l.addWidget(self._feat_combo)

        # 이미지 소스 (크롭/원본) — JPG 계열에만 표시
        self._img_src_row = QWidget()
        isr = QVBoxLayout(self._img_src_row)
        isr.setContentsMargins(0,2,0,0); isr.setSpacing(2)
        isr.addWidget(_slbl("이미지 소스"))
        self._img_crop_radio = QCheckBox("🔪  크롭 전처리 이미지  (권장)")
        self._img_crop_radio.setChecked(True)
        self._img_crop_radio.setStyleSheet("font-size:11px;color:#166534;")
        self._img_orig_radio = QCheckBox("🖼️  원본 이미지  (온더플라이 크롭)")
        self._img_orig_radio.setStyleSheet("font-size:11px;color:#92400E;")
        self._img_crop_radio.toggled.connect(
            lambda v: self._img_orig_radio.setChecked(not v) if v else None)
        self._img_orig_radio.toggled.connect(
            lambda v: (self._img_crop_radio.setChecked(not v) if v else None,
                       self._otf_row.setVisible(v)))
        isr.addWidget(self._img_crop_radio)
        isr.addWidget(self._img_orig_radio)
        g1l.addWidget(self._img_src_row)

        # 온더플라이 크롭 옵션
        self._otf_row = QWidget()
        otf = QHBoxLayout(self._otf_row)
        otf.setContentsMargins(14,0,0,0); otf.setSpacing(6)
        otf.addWidget(_plbl("크롭:", 36))
        self._otf_mode_combo = QComboBox()
        self._otf_mode_combo.addItem("표준 (Standard)", "standard")
        self._otf_mode_combo.addItem("정밀 (Advanced)", "advanced")
        self._otf_mode_combo.setStyleSheet(_CB)
        otf.addWidget(self._otf_mode_combo)
        otf.addSpacing(6); otf.addWidget(_plbl("보정:", 36))
        self._otf_dev_combo = QComboBox()
        self._otf_dev_combo.setStyleSheet(_CB)
        try:
            import torch as _t
            if _t.cuda.is_available():
                self._otf_dev_combo.addItem("GPU (CUDA)", "cuda")
            self._otf_dev_combo.addItem("CPU", "cpu")
        except Exception:
            self._otf_dev_combo.addItem("CPU", "cpu")
        otf.addWidget(self._otf_dev_combo); otf.addStretch()
        self._otf_row.setVisible(False)
        g1l.addWidget(self._otf_row)
        ll.addWidget(g1)

        # ── 2. 데이터셋 (모드에 따라 동적 변경) ───────────────
        self._ds_grp = _grp("📂  데이터셋")
        self._ds_lay = QVBoxLayout(self._ds_grp)
        self._ds_lay.setSpacing(5)
        ll.addWidget(self._ds_grp)

        # 모든 입력창을 미리 생성 (show/hide로 관리)
        self._make_all_edits()

        # 모드 상태 힌트
        self._ds_hint = QLabel("")
        self._ds_hint.setStyleSheet("color:#7C8FA6;font-size:10px;padding:1px 2px;")
        ll.addWidget(self._ds_hint)

        # ── 3. 학습 파라미터 ──────────────────────────────────
        g3 = _grp("⚙️  학습 파라미터")
        g3l = QVBoxLayout(g3); g3l.setSpacing(8)

        r1 = QHBoxLayout(); r1.setSpacing(10)

        # 일반 에포크
        self._ep_wrap = QWidget()
        vep = QVBoxLayout(self._ep_wrap); vep.setContentsMargins(0,0,0,0); vep.setSpacing(3)
        vep.addWidget(_plbl("에포크"))
        self._ep = QSpinBox(); self._ep.setRange(1,500); self._ep.setValue(40)
        self._ep.setMinimumWidth(80); self._ep.setStyleSheet(_SB)
        vep.addWidget(self._ep)
        r1.addWidget(self._ep_wrap)

        # 계층 S1/S2 에포크
        self._hier_ep_wrap = QWidget()
        hew = QHBoxLayout(self._hier_ep_wrap)
        hew.setContentsMargins(0,0,0,0); hew.setSpacing(10)
        vs1 = QVBoxLayout(); vs1.setSpacing(3)
        vs1.addWidget(_plbl("S1 에포크 (계절)"))
        self._ep_s1 = QSpinBox(); self._ep_s1.setRange(1,500); self._ep_s1.setValue(40)
        self._ep_s1.setMinimumWidth(80); self._ep_s1.setStyleSheet(_SB)
        vs1.addWidget(self._ep_s1); hew.addLayout(vs1)
        vs2 = QVBoxLayout(); vs2.setSpacing(3)
        vs2.addWidget(_plbl("S2 에포크 (톤)"))
        self._ep_s2 = QSpinBox(); self._ep_s2.setRange(1,500); self._ep_s2.setValue(40)
        self._ep_s2.setMinimumWidth(80); self._ep_s2.setStyleSheet(_SB)
        vs2.addWidget(self._ep_s2); hew.addLayout(vs2)
        hew.addStretch()
        self._hier_ep_wrap.setVisible(False)
        r1.addWidget(self._hier_ep_wrap)

        vbs = QVBoxLayout(); vbs.setSpacing(3)
        vbs.addWidget(_plbl("배치"))
        self._bs = QSpinBox(); self._bs.setRange(4,256); self._bs.setValue(64)
        self._bs.setMinimumWidth(70); self._bs.setStyleSheet(_SB)
        vbs.addWidget(self._bs); r1.addLayout(vbs)

        vnw = QVBoxLayout(); vnw.setSpacing(3)
        vnw.addWidget(_plbl("Workers"))
        self._nw = QSpinBox(); self._nw.setRange(0,16); self._nw.setValue(4)
        self._nw.setMinimumWidth(65); self._nw.setStyleSheet(_SB)
        vnw.addWidget(self._nw); r1.addLayout(vnw)
        r1.addStretch()
        g3l.addLayout(r1)

        r2 = QHBoxLayout(); r2.setSpacing(10)
        vlr = QVBoxLayout(); vlr.setSpacing(3)
        vlr.addWidget(_plbl("학습률 (LR)"))
        self._lr = QDoubleSpinBox()
        self._lr.setRange(1e-6,0.1); self._lr.setDecimals(6)
        self._lr.setValue(1e-3); self._lr.setSingleStep(1e-4)
        self._lr.setMinimumWidth(120); self._lr.setStyleSheet(_SB)
        vlr.addWidget(self._lr); r2.addLayout(vlr)

        vdv = QVBoxLayout(); vdv.setSpacing(3)
        vdv.addWidget(_plbl("디바이스"))
        self._dev = QComboBox(); self._dev.addItems(["cuda","cpu"])
        self._dev.setMinimumWidth(90); self._dev.setStyleSheet(_CB)
        vdv.addWidget(self._dev); r2.addLayout(vdv)

        # Early Stopping patience
        vpa = QVBoxLayout(); vpa.setSpacing(3)
        vpa.addWidget(_plbl("Patience"))
        self._patience = QSpinBox()
        self._patience.setRange(0, 100); self._patience.setValue(0)
        self._patience.setMinimumWidth(70); self._patience.setStyleSheet(_SB)
        self._patience.setToolTip("Early Stopping patience\n"
                                  "0 = 비활성\n"
                                  "N = Val Acc가 N에포크 동안 개선 없으면 자동 중단")
        vpa.addWidget(self._patience); r2.addLayout(vpa)

        # Test Split
        vts = QVBoxLayout(); vts.setSpacing(3)
        vts.addWidget(_plbl("Test 비율"))
        self._test_split = QDoubleSpinBox()
        self._test_split.setRange(0.0, 0.3); self._test_split.setDecimals(2)
        self._test_split.setValue(0.1); self._test_split.setSingleStep(0.05)
        self._test_split.setSuffix("  (10%)")
        self._test_split.setMinimumWidth(100); self._test_split.setStyleSheet(_SB)
        self._test_split.setToolTip("Test 셋 분리 비율\n"
                                    "0.0 = Test 셋 없음\n"
                                    "0.1 = 전체의 10%를 Test로 분리")
        self._test_split.valueChanged.connect(
            lambda v: self._test_split.setSuffix(f"  ({int(v*100)}%)"))
        vts.addWidget(self._test_split); r2.addLayout(vts)

        r2.addStretch()
        g3l.addLayout(r2)
        ll.addWidget(g3)

        # ── 4. 저장 & 이어학습 ────────────────────────────────
        g4 = _grp("💾  저장 & 이어학습")
        g4l = QVBoxLayout(g4); g4l.setSpacing(5)

        g4l.addWidget(_slbl("모델 저장 경로"))
        sv_row = QHBoxLayout(); sv_row.setSpacing(4)
        self._sv = QLineEdit("best_model.pt")
        self._sv.setStyleSheet(
            "QLineEdit{border:1px solid #CBD5E1;border-radius:5px;"
            "padding:3px 7px;font-size:11px;}")
        self._sv.textChanged.connect(self._update_ckpt_status)
        sv_row.addWidget(self._sv, 1)
        sv_row.addWidget(_ibtn("📂", self._pick_save))
        g4l.addLayout(sv_row)

        # ── 목표 정확도 저장 ──────────────────────────────────
        g4l.addWidget(_slbl("목표 정확도 도달 시 추가 저장  (선택)"))
        acc_row = QHBoxLayout(); acc_row.setSpacing(6)
        self._acc_chk = QCheckBox("활성화")
        self._acc_chk.setStyleSheet("font-size:11px;font-weight:600;color:#166534;")
        self._acc_chk.toggled.connect(self._on_acc_chk_toggled)
        acc_row.addWidget(self._acc_chk)
        acc_row.addWidget(QLabel("목표 정확도:"))
        self._acc_spin = QDoubleSpinBox()
        self._acc_spin.setRange(50.0, 100.0)
        self._acc_spin.setDecimals(1)
        self._acc_spin.setValue(90.0)
        self._acc_spin.setSuffix(" %")
        self._acc_spin.setFixedWidth(90)
        self._acc_spin.setEnabled(False)
        self._acc_spin.setStyleSheet(_SB)
        acc_row.addWidget(self._acc_spin)
        self._acc_status = QLabel("목표 미설정")
        self._acc_status.setStyleSheet("color:#7C8FA6;font-size:10px;")
        acc_row.addWidget(self._acc_status)
        acc_row.addStretch()
        g4l.addLayout(acc_row)

        # 목표 저장 경로 (기본: best_model_acc90.pt)
        self._acc_path_row = QWidget()
        apr = QHBoxLayout(self._acc_path_row)
        apr.setContentsMargins(16,0,0,0); apr.setSpacing(4)
        self._acc_path_edit = QLineEdit()
        self._acc_path_edit.setPlaceholderText("비워두면 자동 생성: best_model_acc90.pt")
        self._acc_path_edit.setStyleSheet(
            "QLineEdit{border:1px solid #BBF7D0;border-radius:5px;"
            "padding:3px 7px;font-size:11px;color:#166534;background:#F0FDF4;}")
        apr.addWidget(self._acc_path_edit, 1)
        apr.addWidget(_ibtn("📂", self._pick_acc_path))
        self._acc_path_row.setVisible(False)
        g4l.addWidget(self._acc_path_row)

        g4l.addWidget(_hline())

        # ── 이어학습 ──────────────────────────────────────────
        g4l.addWidget(_slbl("이어학습 (Resume)"))
        self._resume_chk = QCheckBox("중단된 학습 이어하기")
        self._resume_chk.setStyleSheet("font-size:11px;font-weight:600;color:#1D4ED8;")
        self._resume_chk.toggled.connect(self._on_resume_toggled)
        g4l.addWidget(self._resume_chk)

        self._resume_row = QWidget()
        rr = QHBoxLayout(self._resume_row)
        rr.setContentsMargins(16,0,0,0); rr.setSpacing(4)
        self._resume_edit = QLineEdit()
        self._resume_edit.setPlaceholderText("비워두면 _ckpt.pt 자동 탐색...")
        self._resume_edit.setStyleSheet(
            "QLineEdit{border:1px solid #BFDBFE;border-radius:5px;"
            "padding:3px 7px;font-size:11px;color:#1D4ED8;background:#EFF6FF;}")
        rr.addWidget(self._resume_edit, 1)
        rr.addWidget(_ibtn("📂", self._pick_resume))
        self._resume_row.setVisible(False)
        g4l.addWidget(self._resume_row)

        self._ckpt_status = QLabel("")
        self._ckpt_status.setStyleSheet("font-size:10px;padding:1px 2px;")
        g4l.addWidget(self._ckpt_status)
        ll.addWidget(g4)

        # ── 5. 실행 모드 ──────────────────────────────────────
        g5 = _grp("🚀  실행 모드")
        g5l = QHBoxLayout(g5)
        self._standalone_chk = QCheckBox(
            "Standalone  (subprocess — num_workers 정상, GPU 풀 활용)")
        self._standalone_chk.setChecked(True)
        self._standalone_chk.setStyleSheet("font-size:11px;")
        g5l.addWidget(self._standalone_chk)
        ll.addWidget(g5)

        ll.addStretch()

        # ── 버튼 & 진행바 ─────────────────────────────────────
        self._btn_tr = QPushButton("▶  학습 시작")
        self._btn_tr.setEnabled(False)
        self._btn_tr.setMinimumHeight(36)
        self._btn_tr.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self._btn_tr.setStyleSheet(
            "QPushButton{background:#2563EB;color:white;border:none;"
            "border-radius:7px;padding:7px 20px;}"
            "QPushButton:hover{background:#1D4ED8;}"
            "QPushButton:disabled{background:#94A3B8;color:#E2E8F0;}")
        self._btn_tr.clicked.connect(self._start)

        self._btn_st = QPushButton("■  중지")
        self._btn_st.setEnabled(False)
        self._btn_st.setMinimumHeight(36)
        self._btn_st.setStyleSheet(
            "QPushButton{background:#DC2626;color:white;border:none;"
            "border-radius:7px;padding:7px 14px;}"
            "QPushButton:hover{background:#B91C1C;}"
            "QPushButton:disabled{background:#94A3B8;color:#E2E8F0;}")
        self._btn_st.clicked.connect(self._stop)

        btn_r = QHBoxLayout(); btn_r.setSpacing(6)
        btn_r.addWidget(self._btn_tr, 1); btn_r.addWidget(self._btn_st)
        ll.addLayout(btn_r)

        self._bar = QProgressBar()
        self._bar.setRange(0,100); self._bar.setFixedHeight(7)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            "QProgressBar{border:none;border-radius:4px;background:#E2E8F0;}"
            "QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #2563EB,stop:1 #06B6D4);border-radius:4px;}")
        ll.addWidget(self._bar)

        split.addWidget(left)

        # ══ 오른쪽 로그+차트 ═════════════════════════════════
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(4,0,0,0); rl.setSpacing(5)
        vsplit = QSplitter(Qt.Orientation.Vertical); vsplit.setHandleWidth(5)

        log_grp = _grp("📋  학습 로그")
        log_lay = QVBoxLayout(log_grp)
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet(
            "QTextEdit{background:#0D1117;color:#3FB950;"
            "border:none;border-radius:6px;padding:8px;}")
        log_lay.addWidget(self._log)
        vsplit.addWidget(log_grp)

        chart_grp = _grp("📈  학습 커브")
        chart_lay = QVBoxLayout(chart_grp)
        self._fig = Figure(figsize=(5,3.5), facecolor="#0D1117", tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setStyleSheet("background:#0D1117;border-radius:6px;")
        self._init_chart()
        chart_lay.addWidget(self._canvas)
        vsplit.addWidget(chart_grp)

        vsplit.setSizes([300, 260])
        rl.addWidget(vsplit, 1)
        split.addWidget(right)
        split.setSizes([400, 590])
        root.addWidget(split, 1)

        # 초기 모드 적용
        self._rebuild_dataset_ui("jpg")
        self._update_ckpt_status()

    # ─────────────────────────────────────────────────────────
    #  모든 입력창 미리 생성
    # ─────────────────────────────────────────────────────────
    def _make_all_edits(self):
        """모든 폴더 입력창을 dict로 관리"""
        def _make(ph, color="#334155", bg="#F8FAFC", border="#CBD5E1"):
            e = _folder_edit(ph, color, bg, border); return e

        self._edits = {
            "raw":     _make("원본 데이터셋 폴더..."),
            "pre":     _make("전처리 완료 폴더...", "#1A5276", "#EFF6FF", "#93C5FD"),
            "jpg":     _make("JPG 이미지 폴더...",  "#166534", "#F0FDF4", "#86EFAC"),
            "json":    _make("JSON 피처 폴더...",   "#1D4ED8", "#EFF6FF", "#93C5FD"),
            "s1_jpg":  _make("Stage1 JPG (4계절)...",  "#166534", "#F0FDF4", "#86EFAC"),
            "s1_json": _make("Stage1 JSON (4계절)...", "#1D4ED8", "#EFF6FF", "#93C5FD"),
            "s2_jpg":  _make("Stage2 JPG (12클래스)...","#7C3AED", "#FAF5FF", "#C4B5FD"),
            "s2_json": _make("Stage2 JSON (12클래스)...","#B45309","#FFFBEB","#FCD34D"),
        }

    # ─────────────────────────────────────────────────────────
    #  데이터셋 UI 재구성 (모드별 동적 변경)
    # ─────────────────────────────────────────────────────────
    def _rebuild_dataset_ui(self, mode: str):
        """데이터셋 섹션을 모드에 맞게 완전히 재구성"""
        # 기존 위젯 제거
        while self._ds_lay.count():
            item = self._ds_lay.takeAt(0)
            if item.widget(): item.widget().setParent(None)
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget(): sub.widget().setParent(None)

        e = self._edits  # 단축

        if mode == "jpg":
            # ── JPG: 이미지 폴더 1칸
            self._ds_grp.setTitle("📂  데이터셋")
            r = _data_row("🖼️  이미지 폴더", e["jpg"], _ibtn("📂", lambda: self._pick("jpg")),
                          "#166534", "#DCFCE7")
            self._ds_lay.addLayout(r)
            self._ds_hint.setText("ℹ️  크롭 전처리 이미지 폴더 또는 원본 폴더를 선택하세요.")

        elif mode == "json":
            # ── JSON: JSON 폴더 1칸
            self._ds_grp.setTitle("📂  데이터셋")
            r = _data_row("📄  JSON 폴더", e["json"], _ibtn("📂", lambda: self._pick("json")),
                          "#1D4ED8", "#DBEAFE")
            self._ds_lay.addLayout(r)
            self._ds_hint.setText("ℹ️  JSON 전처리 완료 폴더를 선택하세요.")

        elif mode == "jpg_json":
            # ── JPG+JSON: 이미지 2칸
            self._ds_grp.setTitle("📂  데이터셋  (JPG + JSON 각각 지정)")
            r1 = _data_row("🖼️  JPG 폴더", e["jpg"], _ibtn("📂", lambda: self._pick("jpg", auto_pair="json")),
                           "#166534", "#DCFCE7")
            r2 = _data_row("📄  JSON 폴더", e["json"], _ibtn("📂", lambda: self._pick("json")),
                           "#1D4ED8", "#DBEAFE")
            self._ds_lay.addLayout(r1)
            self._ds_lay.addLayout(r2)
            hint = QLabel("💡  JPG·JSON 폴더가 같으면 JPG 폴더만 지정해도 됩니다.")
            hint.setStyleSheet("color:#6B7280;font-size:10px;padding:2px;")
            self._ds_lay.addWidget(hint)
            self._ds_hint.setText("ℹ️  크롭 이미지 폴더와 JSON 피처 폴더를 각각 지정하세요.")

        elif mode == "hierarchical":
            # ── 계층: 서브탭 4개
            self._ds_grp.setTitle("📂  데이터셋  &  계층 학습 방식 선택")

            sub_tabs = QTabWidget()
            sub_tabs.setStyleSheet(
                "QTabBar::tab{padding:5px 12px;font-size:11px;font-weight:600;}"
                "QTabBar::tab:selected{color:#2563EB;border-bottom:2px solid #2563EB;}")

            # ── 탭1: 한번에 (combined) ─────────────────────────
            t_combined = QWidget()
            tc = QVBoxLayout(t_combined); tc.setSpacing(4)
            tc.addWidget(_slbl("🌿 Stage1 — 4계절 데이터"))
            tc.addLayout(_data_row("🖼️  S1 JPG", e["s1_jpg"],
                _ibtn("📂", lambda: self._pick("s1_jpg","s1_json")), "#166534","#DCFCE7"))
            tc.addLayout(_data_row("📄  S1 JSON", e["s1_json"],
                _ibtn("📂", lambda: self._pick("s1_json")), "#1D4ED8","#DBEAFE"))
            tc.addWidget(_hline())
            tc.addWidget(_slbl("🌈 Stage2 — 12클래스 데이터"))
            tc.addLayout(_data_row("🖼️  S2 JPG", e["s2_jpg"],
                _ibtn("📂", lambda: self._pick("s2_jpg","s2_json")), "#7C3AED","#F3E8FF"))
            tc.addLayout(_data_row("📄  S2 JSON", e["s2_json"],
                _ibtn("📂", lambda: self._pick("s2_json")), "#B45309","#FFFBEB"))
            hint_c = QLabel("⚡ S1→S2 순서로 자동 학습 → 하나의 .pt로 저장")
            hint_c.setStyleSheet("color:#2563EB;font-size:10px;padding:3px;")
            tc.addWidget(hint_c)
            sub_tabs.addTab(t_combined, "⚡ 한번에")

            # ── 탭2: Stage1만 ──────────────────────────────────
            t_s1 = QWidget()
            ts1 = QVBoxLayout(t_s1); ts1.setSpacing(4)
            ts1.addWidget(_slbl("🌿 Stage1 전용 — 4계절 분류 모델 학습"))
            ts1.addLayout(_data_row("🖼️  S1 JPG", e["s1_jpg"],
                _ibtn("📂", lambda: self._pick("s1_jpg","s1_json")), "#166534","#DCFCE7"))
            ts1.addLayout(_data_row("📄  S1 JSON", e["s1_json"],
                _ibtn("📂", lambda: self._pick("s1_json")), "#1D4ED8","#DBEAFE"))
            # Stage1 이어학습
            ts1_resume_row = QHBoxLayout()
            self._s1_resume_chk = QCheckBox("기존 Stage1.pt 이어학습")
            self._s1_resume_chk.setStyleSheet("font-size:11px;color:#1D4ED8;")
            self._s1_resume_chk.toggled.connect(
                lambda v: (self._s1_resume_edit.setVisible(v),
                           self._update_start_btn()))
            ts1_resume_row.addWidget(self._s1_resume_chk)
            ts1.addLayout(ts1_resume_row)
            self._s1_resume_edit = QLineEdit()
            self._s1_resume_edit.setPlaceholderText("기존 stage1.pt 경로...")
            self._s1_resume_edit.setVisible(False)
            self._s1_resume_edit.setStyleSheet(
                "QLineEdit{border:1px solid #93C5FD;border-radius:5px;"
                "padding:3px 7px;font-size:11px;color:#1D4ED8;}")
            s1r = QHBoxLayout()
            s1r.addWidget(self._s1_resume_edit,1)
            s1r.addWidget(_ibtn("📂", lambda: self._pick_file("s1_resume")))
            ts1.addLayout(s1r)
            hint_s1 = QLabel("→ 결과: stage1.pt  (나중에 '조합' 탭에서 Stage2와 합치기)")
            hint_s1.setStyleSheet("color:#166534;font-size:10px;padding:3px;")
            ts1.addWidget(hint_s1)
            sub_tabs.addTab(t_s1, "🌿 Stage1만")

            # ── 탭3: Stage2만 ──────────────────────────────────
            t_s2 = QWidget()
            ts2 = QVBoxLayout(t_s2); ts2.setSpacing(4)
            ts2.addWidget(_slbl("🌈 Stage2 전용 — 톤 분류 모델 학습"))
            ts2.addLayout(_data_row("🖼️  S2 JPG", e["s2_jpg"],
                _ibtn("📂", lambda: self._pick("s2_jpg","s2_json")), "#7C3AED","#F3E8FF"))
            ts2.addLayout(_data_row("📄  S2 JSON", e["s2_json"],
                _ibtn("📂", lambda: self._pick("s2_json")), "#B45309","#FFFBEB"))
            # 계절 선택
            ts2.addWidget(_slbl("학습할 계절 선택"))
            self._season_checks = {}
            sc_row = QHBoxLayout()
            for s, emoji in [("Spring","🌸"),("Summer","☀️"),("Autumn","🍂"),("Winter","❄️")]:
                cb = QCheckBox(f"{emoji}{s}")
                cb.setChecked(True)
                cb.setStyleSheet("font-size:11px;")
                self._season_checks[s] = cb
                sc_row.addWidget(cb)
            sc_row.addStretch()
            ts2.addLayout(sc_row)
            # Stage2 이어학습
            self._s2_resume_chk = QCheckBox("기존 Stage2.pt 이어학습 (개선할 계절만 재학습)")
            self._s2_resume_chk.setStyleSheet("font-size:11px;color:#7C3AED;")
            self._s2_resume_chk.toggled.connect(
                lambda v: (self._s2_resume_edit.setVisible(v),
                           self._update_start_btn()))
            ts2.addWidget(self._s2_resume_chk)
            self._s2_resume_edit = QLineEdit()
            self._s2_resume_edit.setPlaceholderText("기존 stage2.pt 경로...")
            self._s2_resume_edit.setVisible(False)
            self._s2_resume_edit.setStyleSheet(
                "QLineEdit{border:1px solid #C4B5FD;border-radius:5px;"
                "padding:3px 7px;font-size:11px;color:#7C3AED;}")
            s2r = QHBoxLayout()
            s2r.addWidget(self._s2_resume_edit,1)
            s2r.addWidget(_ibtn("📂", lambda: self._pick_file("s2_resume")))
            ts2.addLayout(s2r)
            hint_s2 = QLabel("→ 결과: stage2.pt  (나중에 '조합' 탭에서 Stage1과 합치기)")
            hint_s2.setStyleSheet("color:#7C3AED;font-size:10px;padding:3px;")
            ts2.addWidget(hint_s2)
            sub_tabs.addTab(t_s2, "🌈 Stage2만")

            # ── 탭4: 조합 ─────────────────────────────────────
            t_combine = QWidget()
            tcmb = QVBoxLayout(t_combine); tcmb.setSpacing(6)
            tcmb.addWidget(_slbl("✅ 학습 완료된 Stage1 + Stage2 → 최종 모델 조합"))

            # Stage1 모델 선택 (정확도 표시)
            s1_row = QHBoxLayout(); s1_row.setSpacing(4)
            s1_lbl2 = QLabel("🌿  Stage1 모델")
            s1_lbl2.setFixedWidth(110)
            s1_lbl2.setStyleSheet(
                "font-size:11px;font-weight:600;color:#166534;"
                "background:#DCFCE7;border-radius:4px;padding:2px 5px;")
            self._combine_s1_edit = _folder_edit("stage1.pt ...", "#166534","#F0FDF4","#86EFAC")
            self._combine_s1_acc  = QLabel("?%")
            self._combine_s1_acc.setStyleSheet("font-size:11px;color:#166534;min-width:40px;")
            s1_row.addWidget(s1_lbl2)
            s1_row.addWidget(self._combine_s1_edit,1)
            s1_row.addWidget(self._combine_s1_acc)
            s1_row.addWidget(_ibtn("📂", self._pick_combine_s1))
            tcmb.addLayout(s1_row)

            # Stage2 모델 선택 (정확도 표시)
            s2_row2 = QHBoxLayout(); s2_row2.setSpacing(4)
            s2_lbl2 = QLabel("🌈  Stage2 모델")
            s2_lbl2.setFixedWidth(110)
            s2_lbl2.setStyleSheet(
                "font-size:11px;font-weight:600;color:#7C3AED;"
                "background:#F3E8FF;border-radius:4px;padding:2px 5px;")
            self._combine_s2_edit = _folder_edit("stage2.pt ...", "#7C3AED","#FAF5FF","#C4B5FD")
            self._combine_s2_acc  = QLabel("?%")
            self._combine_s2_acc.setStyleSheet("font-size:11px;color:#7C3AED;min-width:40px;")
            s2_row2.addWidget(s2_lbl2)
            s2_row2.addWidget(self._combine_s2_edit,1)
            s2_row2.addWidget(self._combine_s2_acc)
            s2_row2.addWidget(_ibtn("📂", self._pick_combine_s2))
            tcmb.addLayout(s2_row2)

            # 최종 저장 경로
            tcmb.addWidget(_slbl("최종 모델 저장 경로"))
            final_row = QHBoxLayout(); final_row.setSpacing(4)
            self._combine_out_edit = _folder_edit("final_hierarchical.pt ...")
            self._combine_out_edit.setReadOnly(False)
            self._combine_out_edit.setText("final_hierarchical.pt")
            final_row.addWidget(self._combine_out_edit,1)
            final_row.addWidget(_ibtn("📂", self._pick_combine_out))
            tcmb.addLayout(final_row)

            # 조합 버튼
            self._combine_btn = QPushButton("🔗  조합하기")
            self._combine_btn.setMinimumHeight(32)
            self._combine_btn.setStyleSheet(
                "QPushButton{background:#7C3AED;color:white;border:none;"
                "border-radius:6px;font-size:12px;font-weight:700;padding:6px;}"
                "QPushButton:hover{background:#6D28D9;}"
                "QPushButton:disabled{background:#94A3B8;}")
            self._combine_btn.clicked.connect(self._do_combine)
            tcmb.addWidget(self._combine_btn)
            tcmb.addStretch()
            sub_tabs.addTab(t_combine, "🔗 조합")

            self._hier_sub_tabs = sub_tabs
            self._ds_lay.addWidget(sub_tabs)
            self._ds_hint.setText(
                "⚡ 한번에: Stage1→Stage2 순서로 자동  |  "
                "🌿/🌈 개별: 각각 독립 학습 후 🔗 조합")

        self._update_start_btn()

    # ─────────────────────────────────────────────────────────
    #  이벤트
    # ─────────────────────────────────────────────────────────
    def _on_mode_changed(self, _=0):
        mode = self._feat_combo.currentData()
        is_hier = (mode == "hierarchical")
        is_img  = mode in ("jpg", "jpg_json", "hierarchical")

        # 이미지 소스 / 에포크 전환
        self._img_src_row.setVisible(is_img)
        self._ep_wrap.setVisible(not is_hier)
        self._hier_ep_wrap.setVisible(is_hier)

        if not is_img:
            self._otf_row.setVisible(False)
        else:
            self._otf_row.setVisible(self._img_orig_radio.isChecked())

        self._rebuild_dataset_ui(mode)

    def _pick(self, key: str, auto_pair: str = ""):
        """폴더 선택 → _paths 업데이트 → 자동 페어링"""
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if not d: return
        self._paths[key] = d
        self._edits[key].setText(d)

        # 페어링: JPG 선택 시 JSON이 비어있으면 같은 경로로 자동 채움
        if auto_pair and not self._paths.get(auto_pair):
            self._paths[auto_pair] = d
            self._edits[auto_pair].setText(d)

        self._update_start_btn()

    def _on_resume_toggled(self, checked: bool):
        self._resume_row.setVisible(checked)
        # 이어학습 체크 시 버튼 재평가 (폴더 없어도 OK)
        self._update_start_btn()

    def _on_acc_chk_toggled(self, checked: bool):
        self._acc_spin.setEnabled(checked)
        self._acc_path_row.setVisible(checked)
        if checked:
            self._update_acc_status()
        else:
            self._acc_status.setText("목표 미설정")
            self._acc_status.setStyleSheet("color:#7C8FA6;font-size:10px;")

    def _update_acc_status(self):
        acc = self._acc_spin.value()
        self._acc_status.setText(f"Val Acc ≥ {acc:.1f}% 도달 시 별도 저장")
        self._acc_status.setStyleSheet("color:#166534;font-size:10px;")

    def _pick_acc_path(self):
        acc = self._acc_spin.value()
        sp  = self._sv.text().strip() or "best_model.pt"
        default = str(Path(sp).parent / f"{Path(sp).stem}_acc{int(acc)}.pt")
        p, _ = QFileDialog.getSaveFileName(
            self, "목표 정확도 모델 저장", default, "PyTorch (*.pt *.pth)")
        if p: self._acc_path_edit.setText(p)

    def _update_start_btn(self):
        mode = self._feat_combo.currentData()
        # 이어학습 체크되어 있으면 폴더 없어도 학습 가능
        resume_ok = self._resume_chk.isChecked()
        ok = False
        if mode == "jpg":
            ok = bool(self._paths["jpg"]) or resume_ok
        elif mode == "json":
            ok = bool(self._paths["json"]) or resume_ok
        elif mode == "jpg_json":
            ok = bool(self._paths["jpg"]) or resume_ok
        elif mode == "hierarchical":
            sub = getattr(self, "_hier_sub_tabs", None)
            if sub is None: ok = False
            else:
                tab_idx = sub.currentIndex()
                if tab_idx == 0:   # 한번에
                    ok = bool(self._paths["s1_jpg"] and self._paths["s2_jpg"])
                elif tab_idx == 1: # Stage1만
                    s1_resume = getattr(self, "_s1_resume_chk", None)
                    s1_resume_ok = s1_resume is not None and s1_resume.isChecked()
                    ok = bool(self._paths["s1_jpg"]) or s1_resume_ok
                elif tab_idx == 2: # Stage2만
                    s2_resume = getattr(self, "_s2_resume_chk", None)
                    s2_resume_ok = s2_resume is not None and s2_resume.isChecked()
                    ok = bool(self._paths["s2_jpg"]) or s2_resume_ok
                elif tab_idx == 3: # 조합
                    ok = bool(getattr(self,"_combine_s1_edit",None) and
                               self._combine_s1_edit.text().strip() and
                               self._combine_s2_edit.text().strip())
        self._btn_tr.setEnabled(ok)

    def _update_ckpt_status(self):
        sp = self._sv.text().strip()
        if not sp: self._ckpt_status.setText(""); return
        ckpt = Path(sp).parent / (Path(sp).stem + "_ckpt.pt")
        if ckpt.exists():
            sz = os.path.getsize(ckpt) // (1024*1024)
            self._ckpt_status.setText(
                f"✅ 체크포인트 발견: {ckpt.name}  ({sz} MB)  → 이어학습 체크 시 자동 로드")
            self._ckpt_status.setStyleSheet(
                "color:#166534;font-size:10px;padding:1px 2px;")
        else:
            self._ckpt_status.setText("ℹ️  체크포인트 없음 — 에포크마다 자동 생성")
            self._ckpt_status.setStyleSheet(
                "color:#7C8FA6;font-size:10px;padding:1px 2px;")

    # ─────────────────────────────────────────────────────────
    #  파일 선택
    # ─────────────────────────────────────────────────────────
    def _pick_save(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "모델 저장", "best_model.pt", "PyTorch (*.pt *.pth)")
        if p: self._sv.setText(p); self._update_ckpt_status()

    def _pick_resume(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "체크포인트 선택", "", "PyTorch (*.pt *.pth)")
        if p: self._resume_edit.setText(p)

    def _pick_file(self, key: str):
        p, _ = QFileDialog.getOpenFileName(
            self, "모델 파일 선택", "", "PyTorch (*.pt *.pth)")
        if not p: return
        if key == "s1_resume" and hasattr(self, "_s1_resume_edit"):
            self._s1_resume_edit.setText(p)
        elif key == "s2_resume" and hasattr(self, "_s2_resume_edit"):
            self._s2_resume_edit.setText(p)

    def _pick_combine_s1(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Stage1 모델 선택", "", "PyTorch (*.pt *.pth)")
        if not p: return
        self._combine_s1_edit.setText(p)
        self._read_model_acc(p, self._combine_s1_acc, "stage1")
        self._update_start_btn()

    def _pick_combine_s2(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Stage2 모델 선택", "", "PyTorch (*.pt *.pth)")
        if not p: return
        self._combine_s2_edit.setText(p)
        self._read_model_acc(p, self._combine_s2_acc, "stage2")
        self._update_start_btn()

    def _pick_combine_out(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "최종 모델 저장", "final_hierarchical.pt", "PyTorch (*.pt *.pth)")
        if p: self._combine_out_edit.setText(p)

    def _read_model_acc(self, path: str, label: QLabel, stage: str):
        try:
            import torch as _t
            ck = _t.load(path, map_location="cpu", weights_only=False)
            if stage == "stage1":
                acc = ck.get("val_acc_stage1", None)
                label.setText(f"{acc:.1f}%" if acc else "?%")
            else:
                accs = ck.get("val_acc_stage2", {})
                if accs:
                    avg = sum(accs.values()) / len(accs)
                    label.setText(f"avg {avg:.1f}%")
                else: label.setText("?%")
        except Exception: label.setText("?%")

    def _do_combine(self):
        s1  = self._combine_s1_edit.text().strip()
        s2  = self._combine_s2_edit.text().strip()
        out = self._combine_out_edit.text().strip() or "final_hierarchical.pt"
        if not s1 or not s2:
            self._app("[ERROR] Stage1, Stage2 모델을 모두 선택하세요."); return
        script = Path(__file__).parent / "train_hierarchical.py"
        if not script.exists():
            script = Path(__file__).parent.parent / "train_hierarchical.py"
        if not script.exists():
            self._app("❌ train_hierarchical.py 없음"); return
        self._cleanup_worker()
        args = [
            "--mode","combine_models",
            "--stage1_path", s1,
            "--stage2_path", s2,
            "--save_path",   out,
            "--data_dir",    ".",
        ]
        self._app(f"[COMBINE] {Path(s1).name} + {Path(s2).name} → {out}")
        self._tl.clear(); self._vl.clear(); self._ta.clear(); self._va.clear()
        self._init_chart(); self._bar.setValue(0)
        self._btn_tr.setEnabled(False); self._btn_st.setEnabled(True)
        self._worker = SubprocessTrainWorker(str(script), args)
        self._worker.line_ready.connect(self._app)
        self._worker.epoch_complete.connect(self._upd)
        self._worker.progress.connect(self._bar.setValue)
        self._worker.finished.connect(self._done)
        self._worker.error.connect(self._err)
        self._worker.start()

    # ─────────────────────────────────────────────────────────
    #  외부 연결 (다른 탭에서 호출)
    # ─────────────────────────────────────────────────────────
    def set_dataset_dir(self, path: str):
        self._paths["jpg"] = path
        self._edits["jpg"].setText(path)
        self._update_start_btn()

    def set_preprocessed_dir(self, path: str, mode_name: str, out_fmt: str = "jpg"):
        self._pre_mode_name = mode_name
        key = "json" if out_fmt == "json" else "jpg"
        self._paths[key] = path
        self._edits[key].setText(path)
        # JSON 폴더도 자동 세팅
        if out_fmt == "jpg_json":
            self._paths["jpg"]  = path
            self._paths["json"] = path
            self._edits["jpg"].setText(path)
            self._edits["json"].setText(path)
        self._update_start_btn()
        self._app(
            f"[전처리 연결] {mode_name}  출력:{out_fmt}\n"
            f"  경로: {path}\n  ▶ 학습 시작 버튼을 누르세요.")
        # 탭 이동
        parent = self.parent()
        while parent:
            if hasattr(parent, '_tabs'):
                for i in range(parent._tabs.count()):
                    if parent._tabs.widget(i) is self:
                        parent._tabs.setCurrentIndex(i); break
                break
            parent = parent.parent()

    # ─────────────────────────────────────────────────────────
    #  차트
    # ─────────────────────────────────────────────────────────
    def _init_chart(self):
        self._fig.clear(); self._fig.patch.set_facecolor("#0D1117")
        self._axL = self._fig.add_subplot(211)
        self._axA = self._fig.add_subplot(212)
        for ax, t, y in [
            (self._axL, "Loss", "Loss"),
            (self._axA, "Accuracy (%)", "Acc (%)")
        ]:
            ax.set_facecolor("#161B22"); ax.set_title(t, fontsize=9, color="#8B949E")
            ax.set_xlabel("Epoch", fontsize=8, color="#6E7681")
            ax.set_ylabel(y, fontsize=8, color="#6E7681")
            ax.tick_params(colors="#6E7681", labelsize=7)
            ax.grid(True, alpha=0.12, color="#30363D")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
        self._fig.tight_layout(pad=1.0); self._canvas.draw()

    # ─────────────────────────────────────────────────────────
    #  학습 시작
    # ─────────────────────────────────────────────────────────
    def _start(self):
        mode     = self._feat_combo.currentData()
        img_mode = "original" if self._img_orig_radio.isChecked() else "crop"
        p        = self._paths

        # 경로 수집 및 검증
        if mode == "jpg":
            data_dir = p["jpg"]; jpg_dir = ""; json_dir = ""
            if not data_dir: self._app("[ERROR] JPG 폴더를 지정하세요."); return

        elif mode == "json":
            data_dir = p["json"]; jpg_dir = ""; json_dir = p["json"]
            if not data_dir: self._app("[ERROR] JSON 폴더를 지정하세요."); return

        elif mode == "jpg_json":
            jpg_dir  = p["jpg"] or ""; json_dir = p["json"] or jpg_dir
            data_dir = jpg_dir
            if not data_dir: self._app("[ERROR] JPG 폴더를 지정하세요."); return

        elif mode == "hierarchical":
            sub   = getattr(self, "_hier_sub_tabs", None)
            tab_i = sub.currentIndex() if sub else 0

            if tab_i == 3:   # 조합 탭 → _do_combine
                self._do_combine(); return

            s1_jpg  = p["s1_jpg"]; s1_json = p["s1_json"] or s1_jpg
            s2_jpg  = p["s2_jpg"]; s2_json = p["s2_json"] or s2_jpg
            jpg_dir = ""; json_dir = ""

            if tab_i == 0:   # 한번에
                hier_mode = "combined"
                data_dir  = s2_jpg or s1_jpg
                if not s1_jpg: self._app("[ERROR] S1 JPG 폴더를 지정하세요."); return
                if not s2_jpg: self._app("[ERROR] S2 JPG 폴더를 지정하세요."); return
            elif tab_i == 1: # Stage1만
                hier_mode = "stage1_only"
                data_dir  = s1_jpg
                if not s1_jpg: self._app("[ERROR] S1 JPG 폴더를 지정하세요."); return
            else:            # Stage2만
                hier_mode = "stage2_only"
                data_dir  = s2_jpg
                if not s2_jpg: self._app("[ERROR] S2 JPG 폴더를 지정하세요."); return
        else:
            data_dir = ""; jpg_dir = ""; json_dir = ""

        self._tl.clear(); self._vl.clear()
        self._ta.clear(); self._va.clear()
        self._init_chart(); self._log.clear(); self._bar.setValue(0)
        self._btn_tr.setEnabled(False); self._btn_st.setEnabled(True)

        # 이어학습
        resume = ""
        if self._resume_chk.isChecked():
            resume = self._resume_edit.text().strip()
            if not resume:
                sp = self._sv.text().strip()
                if sp:
                    ckpt = Path(sp).parent / (Path(sp).stem + "_ckpt.pt")
                    if ckpt.exists():
                        resume = str(ckpt)
                        self._app(f"[RESUME] 체크포인트 자동 로드: {ckpt.name}")
                    else:
                        self._app("[RESUME] ⚠️ 체크포인트 없음 — 처음부터 학습합니다.")

        save_path    = self._sv.text() or "best_model.pt"
        preproc_mode = self._otf_mode_combo.currentData()
        preproc_dev  = self._otf_dev_combo.currentData()

        # 목표 정확도
        target_acc      = 0.0
        target_acc_path = ""
        if self._acc_chk.isChecked():
            target_acc = self._acc_spin.value()
            target_acc_path = self._acc_path_edit.text().strip()
            if not target_acc_path:
                sp = save_path
                target_acc_path = str(
                    Path(sp).parent / f"{Path(sp).stem}_acc{int(target_acc)}.pt")
            self._app(f"[ACC]    목표 정확도 {target_acc:.1f}% → {target_acc_path}")

        # 로그
        self._app(f"[MODE]   {mode}  이미지소스: {img_mode}")
        if mode == "hierarchical":
            self._app(f"[S1 JPG] {p['s1_jpg']}")
            self._app(f"[S1 JSON]{p['s1_json'] or '(S1 JPG 사용)'}")
            self._app(f"[S2 JPG] {p['s2_jpg']}")
            self._app(f"[S2 JSON]{p['s2_json'] or '(S2 JPG 사용)'}")
        elif mode == "jpg_json":
            self._app(f"[JPG ]   {jpg_dir}")
            self._app(f"[JSON]   {json_dir}")
        else:
            self._app(f"[DATA]   {data_dir}")
        if resume: self._app(f"[RESUME] {resume}")

        if self._standalone_chk.isChecked():
            self._start_standalone(
                save_path, mode, img_mode,
                data_dir, jpg_dir, json_dir,
                resume, preproc_mode, preproc_dev,
                s1_jpg  = p.get("s1_jpg",""),
                s1_json = p.get("s1_json",""),
                s2_jpg  = p.get("s2_jpg",""),
                s2_json = p.get("s2_json",""),
                hier_mode    = locals().get("hier_mode","combined"),
                target_acc   = target_acc,
                target_acc_path = target_acc_path,
            )

    def _cleanup_worker(self):
        if self._worker:
            if self._worker.isRunning():
                self._worker.stop(); self._worker.wait(5000)
            self._worker.deleteLater(); self._worker = None

    def _start_standalone(self, save_path, mode, img_mode,
                          data_dir, jpg_dir, json_dir,
                          resume, preproc_mode, preproc_dev,
                          s1_jpg="", s1_json="",
                          s2_jpg="", s2_json="",
                          hier_mode="combined",
                          target_acc=0.0, target_acc_path=""):
        is_hier = (mode == "hierarchical")

        if is_hier:
            script = Path(__file__).parent / "train_hierarchical.py"
            if not script.exists():
                script = Path(__file__).parent.parent / "train_hierarchical.py"
        else:
            script = Path(__file__).parent / "train_standalone.py"
            if not script.exists():
                script = Path(__file__).parent.parent / "train_standalone.py"

        if not script.exists():
            self._app(f"❌ 스크립트 없음: {script}")
            self._btn_tr.setEnabled(True); self._btn_st.setEnabled(False); return

        self._cleanup_worker()

        if is_hier:
            hier_feat = "jpg_json" if (s1_json or s2_json) else "jpg"
            # Stage1/Stage2 이어학습 경로
            s1_resume = ""
            s2_resume = ""
            if hier_mode == "stage1_only" and hasattr(self,"_s1_resume_chk"):
                if self._s1_resume_chk.isChecked():
                    s1_resume = self._s1_resume_edit.text().strip()
            if hier_mode == "stage2_only" and hasattr(self,"_s2_resume_chk"):
                if self._s2_resume_chk.isChecked():
                    s2_resume = self._s2_resume_edit.text().strip()

            # 선택된 계절 (stage2_only)
            seasons_str = ""
            if hier_mode == "stage2_only" and hasattr(self,"_season_checks"):
                sel = [s for s, cb in self._season_checks.items() if cb.isChecked()]
                seasons_str = ",".join(sel)

            args = [
                "--mode",         hier_mode,
                "--data_dir",     data_dir or s2_jpg or s1_jpg or ".",
                "--save_path",    save_path,
                "--epochs_s1",    str(self._ep_s1.value()),
                "--epochs_s2",    str(self._ep_s2.value()),
                "--batch_size",   str(self._bs.value()),
                "--lr",           str(self._lr.value()),
                "--workers",      str(self._nw.value()),
                "--device",       self._dev.currentText(),
                "--feat_mode",    hier_feat,
                "--img_mode",     img_mode,
                "--preproc_mode", preproc_mode,
            ]
            if s1_jpg:  args += ["--s1_jpg_dir",  s1_jpg]
            if s1_json: args += ["--s1_json_dir",  s1_json]
            if s2_jpg:  args += ["--s2_jpg_dir",   s2_jpg]
            if s2_json: args += ["--s2_json_dir",  s2_json]
            if s1_resume: args += ["--resume", s1_resume]
            if s2_resume: args += ["--resume", s2_resume]
            if seasons_str: args += ["--seasons", seasons_str]
            if target_acc > 0:
                args += ["--target_acc",      str(target_acc),
                         "--target_acc_path", target_acc_path]
            args += ["--patience",   str(self._patience.value()),
                     "--test_split", str(self._test_split.value())]
        else:
            feat_mode = mode  # jpg / json / jpg_json
            args = [
                "--data_dir",     data_dir,
                "--save_path",    save_path,
                "--epochs",       str(self._ep.value()),
                "--batch_size",   str(self._bs.value()),
                "--lr",           str(self._lr.value()),
                "--workers",      str(self._nw.value()),
                "--device",       self._dev.currentText(),
                "--feat_mode",    feat_mode,
                "--img_mode",     img_mode,
                "--preproc_mode", preproc_mode,
                "--preproc_dev",  preproc_dev,
            ]
            if feat_mode == "jpg_json":
                if jpg_dir:  args += ["--jpg_dir",  jpg_dir]
                if json_dir: args += ["--json_dir", json_dir]
            if resume:
                args += ["--resume", resume]
            if target_acc > 0:
                args += ["--target_acc",      str(target_acc),
                         "--target_acc_path", target_acc_path]
            args += ["--patience",   str(self._patience.value()),
                     "--test_split", str(self._test_split.value())]

        self._worker = SubprocessTrainWorker(str(script), args)
        self._worker.line_ready.connect(self._app)
        self._worker.epoch_complete.connect(self._upd)
        self._worker.progress.connect(self._bar.setValue)
        self._worker.finished.connect(self._done)
        self._worker.error.connect(self._err)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning(): self._worker.stop()
        self._btn_tr.setEnabled(True); self._btn_st.setEnabled(False)
        self._app("[INFO] 중지 요청 — 현재 에포크 완료 후 종료됩니다.")

    # ─────────────────────────────────────────────────────────
    #  시그널 슬롯
    # ─────────────────────────────────────────────────────────
    def _app(self, msg: str):
        self._log.append(msg)
        sb = self._log.verticalScrollBar(); sb.setValue(sb.maximum())

    def _upd(self, ep: int, tl: float, ta: float, vl: float, va: float):
        self._tl.append(tl); self._vl.append(vl)
        self._ta.append(ta); self._va.append(va)
        xs = list(range(1, len(self._tl)+1))
        for ax, td, vd, t, y, yl in [
            (self._axL, self._tl, self._vl, "Loss",        "Loss",   None),
            (self._axA, self._ta, self._va, "Accuracy (%)","Acc (%)",(0,105)),
        ]:
            ax.clear(); ax.set_facecolor("#161B22")
            ax.plot(xs, td, color="#58A6FF", lw=1.5, marker="o", ms=3, label="Train")
            ax.plot(xs, vd, color="#F78166", lw=1.5, marker="s", ms=3, label="Val")
            ax.set_title(t, fontsize=9, color="#8B949E")
            ax.set_xlabel("Epoch", fontsize=8, color="#6E7681")
            ax.set_ylabel(y, fontsize=8, color="#6E7681")
            ax.tick_params(colors="#6E7681", labelsize=7)
            ax.legend(fontsize=7, facecolor="#161B22",
                      edgecolor="#30363D", labelcolor="#8B949E")
            ax.grid(True, alpha=0.12, color="#30363D")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
            if yl: ax.set_ylim(*yl)
        self._fig.tight_layout(pad=1.0); self._canvas.draw()

    def _done(self, path: str):
        self._btn_tr.setEnabled(True); self._btn_st.setEnabled(False)
        self._bar.setValue(100)
        self._app(f"\n✅ 학습 완료  →  {path}")
        self._update_ckpt_status()
        if self._worker: self._worker.deleteLater(); self._worker = None

    def _err(self, msg: str):
        self._btn_tr.setEnabled(True); self._btn_st.setEnabled(False)
        self._app(f"❌ {msg}")
        self._update_ckpt_status()
        if self._worker: self._worker.deleteLater(); self._worker = None
