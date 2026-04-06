import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Optional
import matplotlib; matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem,
    QProgressBar, QSplitter, QFileDialog, QHeaderView)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QColor

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

from datasets import PersonalColorDataset


# ── 인라인 워커 (workers.py 의존 없이 독립) ──────────────────
class DatasetLoaderWorker(QThread):
    """계절 분포 + 톤 세부 분포를 함께 반환"""
    finished = pyqtSignal(dict, dict)   # (dist, detail)
    error    = pyqtSignal(str)

    def __init__(self, folder_path: str):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            ds     = PersonalColorDataset(self.folder_path)
            dist   = ds.get_class_distribution()
            detail = ds.get_detailed_distribution()
            self.finished.emit(dist, detail)
        except Exception as e:
            self.error.emit(str(e))


# 계절 색상
_SC = {
    "Spring": "#FFB347", "Summer": "#87CEEB",
    "Autumn": "#CD853F", "Winter": "#6495ED",
}
# 톤 색상
_TC = {
    "Warm":   "#FDEBD0", "Bright": "#D5F5E3",
    "Light":  "#E8DAEF", "_root":  "#F2F3F4",
}


class DatasetTab(QWidget):
    dataset_loaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dir    = ""
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ── 폴더 선택 ─────────────────────────────────────────
        grp = QGroupBox("📁 데이터셋 폴더 선택")
        row = QHBoxLayout(grp)
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText(
            "Spring(또는 spring/fall)/Summer/Autumn/Winter 폴더 구조 "
            "— 하위 Warm/Bright/Light 폴더 자동 인식")
        b1 = QPushButton("폴더 열기")
        b1.setFixedWidth(100)
        b1.clicked.connect(self._browse)
        self._btn_load = QPushButton("📊 분석")
        self._btn_load.setFixedWidth(80)
        self._btn_load.setEnabled(False)
        self._btn_load.clicked.connect(self._load)
        row.addWidget(self._path_edit)
        row.addWidget(b1)
        row.addWidget(self._btn_load)
        root.addWidget(grp)

        sp = QSplitter(Qt.Orientation.Horizontal)

        # ── 계절별 분포 테이블 ────────────────────────────────
        tg = QGroupBox("📋 계절별 분포")
        tl = QVBoxLayout(tg)
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["클래스", "샘플 수", "비율(%)"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        tl.addWidget(self._table)
        sp.addWidget(tg)

        # ── 톤 세부 분포 테이블 ───────────────────────────────
        dg = QGroupBox("🎨 톤별 세부 분포  (Warm / Bright / Light)")
        dl = QVBoxLayout(dg)
        self._dtable = QTableWidget(0, 3)
        self._dtable.setHorizontalHeaderLabels(["계절", "톤", "이미지 수"])
        self._dtable.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._dtable.setAlternatingRowColors(True)
        dl.addWidget(self._dtable)
        sp.addWidget(dg)

        # ── 파이 차트 ─────────────────────────────────────────
        pg = QGroupBox("🥧 분포 차트")
        pl = QVBoxLayout(pg)
        self._fig    = Figure(figsize=(4, 4), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        pl.addWidget(self._canvas)
        sp.addWidget(pg)

        sp.setSizes([260, 280, 300])
        root.addWidget(sp, 1)

        self._status = QLabel("데이터셋 폴더를 선택하세요.")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._status)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setVisible(False)
        root.addWidget(self._bar)

    # ── 슬롯 ─────────────────────────────────────────────────
    def _browse(self):
        f = QFileDialog.getExistingDirectory(
            self, "데이터셋 폴더", "", QFileDialog.Option.ShowDirsOnly)
        if f:
            self._dir = f
            self._path_edit.setText(f)
            self._btn_load.setEnabled(True)

    def _load(self):
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        self._bar.setVisible(True)
        self._btn_load.setEnabled(False)
        self._status.setText("스캔 중...")
        self._worker = DatasetLoaderWorker(self._dir)
        self._worker.finished.connect(self._on_loaded)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_loaded(self, dist: dict, detail: dict):
        self._bar.setVisible(False)
        self._btn_load.setEnabled(True)
        total = sum(dist.values())
        self._status.setText(
            f"✅ {total}개 샘플  |  {len(dist)}개 클래스  "
            f"— 폴더명(fall/spring 등) 자동 정규화 적용")

        # ── 계절 테이블 ───────────────────────────────────────
        self._table.setRowCount(0)
        for cls, cnt in sorted(dist.items()):
            r = self._table.rowCount()
            self._table.insertRow(r)
            it = QTableWidgetItem(cls)
            it.setBackground(QColor(_SC.get(cls, "#ccc")))
            self._table.setItem(r, 0, it)
            self._table.setItem(r, 1, QTableWidgetItem(str(cnt)))
            self._table.setItem(r, 2, QTableWidgetItem(
                f"{cnt / max(total, 1) * 100:.1f}%"))

        # ── 톤 세부 테이블 ────────────────────────────────────
        self._dtable.setRowCount(0)
        from constants import SEASON_CLASSES
        tone_order = ["Warm", "Bright", "Light", "_root"]
        for cls in SEASON_CLASSES:
            if cls not in detail:
                continue
            tones = detail[cls]
            for tone in tone_order:
                if tone not in tones:
                    continue
                cnt = tones[tone]
                r   = self._dtable.rowCount()
                self._dtable.insertRow(r)
                ci = QTableWidgetItem(cls)
                ci.setBackground(QColor(_SC.get(cls, "#eee")))
                self._dtable.setItem(r, 0, ci)
                tone_label = "직접(톤없음)" if tone == "_root" else tone
                ti = QTableWidgetItem(tone_label)
                ti.setBackground(QColor(_TC.get(tone, "#fff")))
                self._dtable.setItem(r, 1, ti)
                self._dtable.setItem(r, 2, QTableWidgetItem(str(cnt)))

        self._draw_pie(dist)
        self.dataset_loaded.emit(self._dir)

    def _draw_pie(self, dist: dict):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        lb = list(dist.keys())
        sz = list(dist.values())
        co = [_SC.get(l, "#95A5A6") for l in lb]
        if sum(sz):
            _, _, au = ax.pie(
                sz, labels=lb, colors=co,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
            for t in au:
                t.set_fontsize(9)
        else:
            ax.text(0.5, 0.5, "데이터 없음",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title("계절별 분포", fontsize=11)
        self._canvas.draw()

    def _on_error(self, msg: str):
        self._bar.setVisible(False)
        self._btn_load.setEnabled(True)
        self._status.setText(f"❌ {msg}")