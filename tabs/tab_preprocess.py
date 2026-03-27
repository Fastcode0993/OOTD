import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import List, Optional, Dict
import cv2, numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSpinBox, QProgressBar, QSplitter, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QColor
from preprocessors import PersonalColorPreprocessor, AdvancedSkinPreprocessor
from workers import PreprocessWorker
from constants import SUPPORTED_EXT


class PreprocessTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dir = ""
        self._all_paths: List[str] = []
        self._std = PersonalColorPreprocessor()
        self._adv = AdvancedSkinPreprocessor()
        self._worker: Optional[PreprocessWorker] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        ctrl = QGroupBox("⚙️ 전처리 설정")
        crow = QHBoxLayout(ctrl)
        crow.addWidget(QLabel("미리보기 샘플:"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 200)
        self._n_spin.setValue(5)
        crow.addWidget(self._n_spin)
        self._adv_chk = QCheckBox("🔬 정밀 피부 추출 모드 (AdvancedSkinPreprocessor)")
        self._adv_chk.setToolTip("Face Oval + YCrCb/HSV 필터로 검은선/머리카락 제거")
        crow.addWidget(self._adv_chk)
        crow.addStretch()
        self._btn_run = QPushButton("▶ 전처리 실행")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._run)
        self._btn_stop = QPushButton("■ 중지")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        crow.addWidget(self._btn_run)
        crow.addWidget(self._btn_stop)
        root.addWidget(ctrl)

        sp = QSplitter(Qt.Orientation.Horizontal)

        grp1 = QGroupBox("📷 원본")
        l1 = QVBoxLayout(grp1)
        self._orig_lbl = QLabel("이미지 없음")
        self._orig_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_lbl.setMinimumSize(220, 220)
        self._orig_lbl.setStyleSheet("border:1px solid #ccc; background:#f9f9f9;")
        l1.addWidget(self._orig_lbl)
        sp.addWidget(grp1)

        grp2 = QGroupBox("🟢 볼 영역 / 피부 오버레이")
        l2 = QVBoxLayout(grp2)
        self._proc_lbl = QLabel("이미지 없음")
        self._proc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._proc_lbl.setMinimumSize(220, 220)
        self._proc_lbl.setStyleSheet("border:1px solid #ccc; background:#f9f9f9;")
        l2.addWidget(self._proc_lbl)
        sp.addWidget(grp2)

        grp3 = QGroupBox("🩹 피부 마스크 (Advanced 전용)")
        l3 = QVBoxLayout(grp3)
        self._mask_lbl = QLabel("표준 모드에서는\n마스크 미표시")
        self._mask_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mask_lbl.setMinimumSize(220, 220)
        self._mask_lbl.setStyleSheet("border:1px solid #ccc; background:#f9f9f9;")
        l3.addWidget(self._mask_lbl)
        sp.addWidget(grp3)

        grp4 = QGroupBox("🎨 추출 색상 정보")
        l4 = QVBoxLayout(grp4)
        self._swatch = QLabel()
        self._swatch.setFixedSize(90, 60)
        self._swatch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._swatch.setStyleSheet("border:2px solid #888; border-radius:8px; background:#eee;")
        l4.addWidget(self._swatch, alignment=Qt.AlignmentFlag.AlignHCenter)
        self._lab_lbl = QLabel("L*a*b*: -\n\nRGB: -")
        self._lab_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lab_lbl.setWordWrap(True)
        l4.addWidget(self._lab_lbl)
        self._face_lbl = QLabel("얼굴 감지: -")
        self._face_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l4.addWidget(self._face_lbl)
        self._ratio_lbl = QLabel("피부 면적 비율: -")
        self._ratio_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l4.addWidget(self._ratio_lbl)
        l4.addStretch()
        sp.addWidget(grp4)

        sp.setSizes([230, 230, 230, 180])
        root.addWidget(sp, 1)

        self._bar = QProgressBar()
        self._bar.setVisible(False)
        root.addWidget(self._bar)
        self._status = QLabel("데이터셋을 먼저 로드하세요.")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._status)

    def set_dataset_dir(self, path: str):
        self._dir = path
        self._btn_run.setEnabled(True)
        self._all_paths = [str(p) for p in Path(path).rglob("*") if p.suffix.lower() in SUPPORTED_EXT]
        self._status.setText(f"{len(self._all_paths)}개 이미지 발견  |  폴더: {Path(path).name}")

    def _run(self):
        selected = self._all_paths[:self._n_spin.value()]
        if not selected:
            self._status.setText("처리할 이미지가 없습니다.")
            return
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        pre = self._adv if self._adv_chk.isChecked() else self._std
        mn  = "정밀 피부 추출" if self._adv_chk.isChecked() else "표준 볼 영역"
        self._status.setText(f"[{mn} 모드] 전처리 중...")
        self._bar.setVisible(True)
        self._bar.setValue(0)
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._worker = PreprocessWorker(selected, pre)
        self._worker.progress.connect(self._on_prog)
        self._worker.image_ready.connect(self._show)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_err)
        self._worker.start()

    def _on_prog(self, p, fname):
        self._bar.setValue(p)
        self._status.setText(f"처리 중: {fname}  ({p}%)")

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._bar.setVisible(False)
        self._status.setText("중단됨")

    def _show(self, proc_img, meta):
        op = meta.get("image_path", "")
        if op:
            px = QPixmap(op)
            if not px.isNull():
                self._orig_lbl.setPixmap(px.scaled(220, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data.tobytes(), w, h, w*ch, QImage.Format.Format_RGB888)
        self._proc_lbl.setPixmap(QPixmap.fromImage(qi).scaled(220, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        sm = meta.get("skin_mask", None)
        if sm is not None:
            cm = np.zeros((*sm.shape, 3), dtype=np.uint8)
            cm[sm == 255] = [220, 120, 200]
            cm[sm == 0]   = [40, 40, 40]
            mh, mw = cm.shape[:2]
            qm = QImage(cm.data.tobytes(), mw, mh, mw*3, QImage.Format.Format_RGB888)
            self._mask_lbl.setPixmap(QPixmap.fromImage(qm).scaled(220, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self._mask_lbl.setText("표준 모드에서는\n마스크 미표시")

        lab  = meta.get("lab_mean", [0, 128, 128])
        rv   = meta.get("cheek_rgb", [200, 150, 150])
        det  = meta.get("face_detected", False)
        rat  = meta.get("skin_pixel_ratio", None)

        self._lab_lbl.setText(f"L*={lab[0]:.1f}  a*={lab[1]:.1f}  b*={lab[2]:.1f}\n\nR={rv[0]}  G={rv[1]}  B={rv[2]}")

        qc = QColor(int(rv[0]), int(rv[1]), int(rv[2]))
        self._swatch.setStyleSheet(f"background:{qc.name()}; border:2px solid #555; border-radius:8px;")

        self._face_lbl.setText("✅ 얼굴 감지됨" if det else "⚠️ 얼굴 미감지")

        if rat is not None:
            self._ratio_lbl.setText(f"피부 면적 비율: {rat*100:.1f}%")
        else:
            self._ratio_lbl.setText("피부 면적 비율: -")

    def _on_done(self, all_meta):
        self._bar.setVisible(False)
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status.setText(f"✅ {len(all_meta)}개 전처리 완료")

    def _on_err(self, msg):
        self._bar.setVisible(False)
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status.setText(f"❌ {msg}")
