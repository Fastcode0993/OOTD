import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import List, Optional
import cv2, numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSpinBox, QProgressBar, QSplitter, QCheckBox,
    QTextEdit, QMessageBox, QLineEdit, QFileDialog, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont
from preprocessors import PersonalColorPreprocessor, AdvancedSkinPreprocessor
from workers import PreprocessWorker, FullPreprocessWorker
from constants import SUPPORTED_EXT


class PreprocessTab(QWidget):
    # 전처리 완료 시 학습 탭으로 (전처리된_폴더, 모드명, output_format) 전달
    preprocess_done = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dir = ""
        self._all_paths: List[str] = []
        self._preprocessed_dir = ""
        self._std = PersonalColorPreprocessor()
        self._adv = AdvancedSkinPreprocessor()
        self._worker: Optional[PreprocessWorker] = None
        self._full_worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── ① 모드 선택 ──────────────────────────────────────
        mode_grp = QGroupBox("⚙️ 전처리 모드 선택")
        mode_lay = QVBoxLayout(mode_grp)

        self._std_chk = QCheckBox(
            "📍 표준 (Standard)  — MediaPipe 볼 영역 감지 + 논문 b* 보정")
        self._std_chk.setChecked(True)
        self._std_chk.setToolTip(
            "볼 좌우 랜드마크로 볼 영역을 추출하고\n"
            "논문 기준(b*=18.50)으로 색상 보정합니다.")
        mode_lay.addWidget(self._std_chk)

        self._adv_chk = QCheckBox(
            "🔬 정밀 (Advanced)  — Face Oval + YCrCb/HSV 피부 마스크 + b* 보정")
        self._adv_chk.setToolTip(
            "얼굴 전체 윤곽을 추출하고 눈/입술을 제외한\n"
            "정밀 피부 마스크를 생성합니다.\n"
            "머리카락·검은선 제거에 효과적입니다.")
        mode_lay.addWidget(self._adv_chk)

        # 라디오처럼 동작
        self._std_chk.toggled.connect(
            lambda v: self._adv_chk.setChecked(not v) if v else None)
        self._adv_chk.toggled.connect(
            lambda v: self._std_chk.setChecked(not v) if v else None)

        # ── 장치 선택 행 ──────────────────────────────────────
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("🖥️ 보정 연산 장치:"))
        self._dev_combo = QComboBox()
        self._dev_combo.setFixedWidth(160)
        self._dev_combo.setToolTip(
            "MediaPipe 얼굴 감지는 항상 CPU에서 실행됩니다.\n"
            "b* 색상 보정 연산만 GPU로 가속됩니다.\n\n"
            "GPU (CUDA): RTX 시리즈 등 NVIDIA GPU\n"
            "CPU       : GPU 없을 때 사용")

        import torch as _torch
        if _torch.cuda.is_available():
            gpu_name = _torch.cuda.get_device_name(0)
            vram     = _torch.cuda.get_device_properties(0).total_memory / 1e9
            self._dev_combo.addItem(f"GPU — {gpu_name} ({vram:.0f}GB)", "cuda")
            self._dev_combo.addItem("CPU", "cpu")
            self._dev_combo.setCurrentIndex(0)   # GPU 기본 선택
        else:
            self._dev_combo.addItem("CPU (CUDA 불가)", "cpu")

        dev_row.addWidget(self._dev_combo)
        dev_row.addStretch()
        mode_lay.addLayout(dev_row)

        root.addWidget(mode_grp)

        # ── ② 전체 전처리 (학습용) ───────────────────────────
        full_grp = QGroupBox(
            "🚀 전체 전처리 실행  (학습용 — 모든 이미지를 처리하여 저장)")
        full_lay = QVBoxLayout(full_grp)

        info = QLabel(
            "데이터셋 전체를 선택한 모드로 전처리하여 저장합니다.  "
            "완료 후 학습 탭에 자동 연결됩니다.")
        info.setStyleSheet("color:#3D5A73; font-size:11px;")
        full_lay.addWidget(info)

        # ── 출력 형식 선택 (JPG / JSON 별개) ─────────────────
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("💾 출력 형식:"))
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItem(
            "🖼️  JPG 모드  — 이미지 보정 저장만 (JSON 없음)", "jpg")
        self._fmt_combo.addItem(
            "📄  JSON 모드  — 피부 수치 피처 JSON 추출만 (이미지 저장 없음)", "json")
        self._fmt_combo.setCurrentIndex(0)
        self._fmt_combo.setToolTip(
            "JPG 모드:\n"
            "  b* 보정된 이미지만 저장합니다.\n"
            "  학습 탭에서 'JPG' 또는 'JPG+JSON' 모드로 사용 가능.\n\n"
            "JSON 모드:\n"
            "  이미지는 저장하지 않고 피부 색상 수치(Lab, b*, HSV 등)만\n"
            "  JSON 파일로 추출합니다.\n"
            "  학습 탭에서 'JSON' 또는 'JPG+JSON' 모드로 사용 가능.")
        fmt_row.addWidget(self._fmt_combo, 1)
        full_lay.addLayout(fmt_row)

        # ── 병렬 워커 수 선택 ─────────────────────────────────
        import os as _os
        _cpu_count = _os.cpu_count() or 4
        worker_row = QHBoxLayout()
        worker_row.addWidget(QLabel("⚙️ 병렬 워커:"))
        self._worker_combo = QComboBox()
        self._worker_combo.addItem("1  (순차 — 안정적)", 1)
        self._worker_combo.addItem("2  (2코어 병렬)", 2)
        self._worker_combo.addItem("4  (4코어 병렬)", 4)
        self._worker_combo.addItem("8  (8코어 병렬)", 8)
        self._worker_combo.addItem(f"최대 ({_cpu_count}코어 전부)", _cpu_count)
        self._worker_combo.setCurrentIndex(1)   # 기본값 2
        self._worker_combo.setToolTip(
            "전처리를 몇 개 CPU 코어에서 동시에 실행할지 선택합니다.\n\n"
            "각 워커가 독립 FaceMesh 인스턴스를 생성하여 병렬 처리합니다.\n\n"
            "1  (순차): 가장 안정적. 메모리 사용 최소.\n"
            "2~4       : 속도 2~3배 향상. 일반 PC 권장.\n"
            "최대      : 가장 빠르지만 메모리 많이 사용.\n"
            "            RAM 부족 시 오히려 느려질 수 있음.")
        worker_row.addWidget(self._worker_combo)
        worker_row.addStretch()
        full_lay.addLayout(worker_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("저장 폴더:"))
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText(
            "비워두면 자동 생성  (데이터셋폴더_preprocessed_std/)")
        out_row.addWidget(self._out_edit, 1)
        btn_pick = QPushButton("📂")
        btn_pick.setFixedWidth(32)
        btn_pick.clicked.connect(self._pick_out)
        out_row.addWidget(btn_pick)
        full_lay.addLayout(out_row)

        btn_row = QHBoxLayout()
        self._btn_full = QPushButton("🚀 전체 전처리 시작")
        self._btn_full.setEnabled(False)
        self._btn_full.setMinimumHeight(36)
        self._btn_full.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self._btn_full.clicked.connect(self._run_full)
        self._btn_full_stop = QPushButton("■ 중지")
        self._btn_full_stop.setEnabled(False)
        self._btn_full_stop.setMinimumHeight(36)
        self._btn_full_stop.clicked.connect(self._stop_full)
        btn_row.addWidget(self._btn_full, 1)
        btn_row.addWidget(self._btn_full_stop)
        full_lay.addLayout(btn_row)

        self._full_bar = QProgressBar()
        self._full_bar.setRange(0, 100)
        full_lay.addWidget(self._full_bar)

        self._full_status = QLabel("데이터셋을 먼저 로드하세요.")
        self._full_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._full_status.setStyleSheet("font-size:11px; color:#2C3E50;")
        full_lay.addWidget(self._full_status)

        root.addWidget(full_grp)

        # ── ③ 미리보기 ───────────────────────────────────────
        prev_grp = QGroupBox("👁️ 샘플 미리보기  (확인용)")
        prev_lay = QVBoxLayout(prev_grp)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("샘플 수:"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 20)
        self._n_spin.setValue(3)
        ctrl_row.addWidget(self._n_spin)
        ctrl_row.addStretch()
        self._btn_prev = QPushButton("▶ 미리보기")
        self._btn_prev.setEnabled(False)
        self._btn_prev.clicked.connect(self._run_preview)
        self._btn_prev_stop = QPushButton("■ 중지")
        self._btn_prev_stop.setEnabled(False)
        self._btn_prev_stop.clicked.connect(self._stop_preview)
        ctrl_row.addWidget(self._btn_prev)
        ctrl_row.addWidget(self._btn_prev_stop)
        prev_lay.addLayout(ctrl_row)

        sp = QSplitter(Qt.Orientation.Horizontal)

        for attr, title, note in [
            ("_orig_lbl",  "📷 원본", "이미지 없음"),
            ("_proc_lbl",  "🟢 오버레이", "이미지 없음"),
            ("_mask_lbl",  "🩹 피부 마스크 (정밀 전용)", "정밀 모드에서만 표시"),
        ]:
            g = QGroupBox(title)
            l = QVBoxLayout(g)
            lbl = QLabel(note)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(190, 190)
            lbl.setStyleSheet("border:1px solid #ccc; background:#f9f9f9;")
            l.addWidget(lbl)
            setattr(self, attr, lbl)
            sp.addWidget(g)

        grp4 = QGroupBox("🎨 색상 정보")
        l4 = QVBoxLayout(grp4)
        self._swatch = QLabel()
        self._swatch.setFixedSize(75, 48)
        self._swatch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._swatch.setStyleSheet(
            "border:2px solid #888; border-radius:8px; background:#eee;")
        l4.addWidget(self._swatch, alignment=Qt.AlignmentFlag.AlignHCenter)
        self._lab_lbl = QLabel("L*a*b*: -\n\nRGB: -")
        self._lab_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lab_lbl.setWordWrap(True)
        l4.addWidget(self._lab_lbl)
        self._face_lbl = QLabel("얼굴 감지: -")
        self._face_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l4.addWidget(self._face_lbl)
        self._ratio_lbl = QLabel("피부 비율: -")
        self._ratio_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l4.addWidget(self._ratio_lbl)
        l4.addStretch()
        sp.addWidget(grp4)

        sp.setSizes([190, 190, 190, 150])
        prev_lay.addWidget(sp)
        root.addWidget(prev_grp, 1)

        # ── ④ 로그 ───────────────────────────────────────────
        log_grp = QGroupBox("📋 전처리 로그")
        log_lay = QVBoxLayout(log_grp)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        self._log.setStyleSheet("background:#1E1E1E; color:#00FF7F;")
        self._log.setMaximumHeight(110)
        log_lay.addWidget(self._log)
        root.addWidget(log_grp)

    # ── 외부 인터페이스 ───────────────────────────────────────
    def set_dataset_dir(self, path: str):
        self._dir = path
        self._all_paths = [
            str(p) for p in Path(path).rglob("*")
            if p.suffix.lower() in SUPPORTED_EXT
        ]
        self._btn_full.setEnabled(True)
        self._btn_prev.setEnabled(True)
        self._full_status.setText(
            f"📁 {Path(path).name}  —  {len(self._all_paths)}개 이미지 발견")
        self._log_msg(f"📁 {path}  ({len(self._all_paths)}장)")

    def get_preprocessor(self):
        device = self._dev_combo.currentData()
        if self._adv_chk.isChecked():
            return AdvancedSkinPreprocessor(device=device)
        return PersonalColorPreprocessor(device=device)

    def get_mode_name(self) -> str:
        device = self._dev_combo.currentData()
        mode   = "정밀(Advanced)" if self._adv_chk.isChecked() else "표준(Standard)"
        dev    = "GPU" if device == "cuda" else "CPU"
        return f"{mode}/{dev}"

    def get_preprocessed_dir(self) -> str:
        """완료된 전처리 결과 폴더. 없으면 빈 문자열."""
        return self._preprocessed_dir

    # ── 저장 폴더 선택 ────────────────────────────────────────
    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", "")
        if d:
            self._out_edit.setText(d)

    def _get_out_dir(self) -> str:
        txt = self._out_edit.text().strip()
        if txt:
            return txt
        mode = "adv" if self._adv_chk.isChecked() else "std"
        return str(Path(self._dir).parent /
                   f"{Path(self._dir).name}_preprocessed_{mode}")

    # ── 전체 전처리 실행 ─────────────────────────────────────
    def _run_full(self):
        if not self._dir or not self._all_paths:
            QMessageBox.warning(self, "경고", "데이터셋을 먼저 로드하세요.")
            return

        out_dir    = self._get_out_dir()
        mode_name  = self.get_mode_name()
        out_fmt    = self._fmt_combo.currentData()   # "jpg" or "json"
        fmt_label  = self._fmt_combo.currentText()

        # ── 이전 실행 기록 확인 (이어하기 감지) ──────────────
        import json as _json
        meta_path  = Path(out_dir) / "preprocess_meta.json"
        resume_cnt = 0
        can_resume = False
        if meta_path.exists():
            try:
                prev = _json.loads(meta_path.read_text(encoding="utf-8"))
                prev_fmt = prev.get("output_format", "")
                if prev_fmt == out_fmt:
                    resume_cnt = len(prev.get("records", []))
                    can_resume = resume_cnt > 0
            except Exception:
                pass

        if can_resume:
            remain = len(self._all_paths) - resume_cnt
            reply = QMessageBox.question(
                self, "이어하기 감지",
                f"이전 전처리 기록이 있습니다.\n\n"
                f"  완료된 파일: {resume_cnt}개\n"
                f"  남은 파일:   {max(remain, 0)}개\n"
                f"  저장 폴더:   {out_dir}\n\n"
                f"이어서 진행하시겠습니까?\n"
                f"(아니오 선택 시 처음부터 다시 시작합니다)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.No:
                # 처음부터: 기존 결과 폴더 초기화
                import shutil as _shutil
                confirm = QMessageBox.warning(
                    self, "초기화 확인",
                    f"저장 폴더의 모든 전처리 결과를 삭제하고 처음부터 시작합니다.\n\n"
                    f"  {out_dir}\n\n계속하시겠습니까?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if confirm != QMessageBox.StandardButton.Yes:
                    return
                _shutil.rmtree(out_dir, ignore_errors=True)
                self._log_msg(f"🗑️  기존 전처리 결과 삭제됨: {out_dir}")
            # Yes → 이어하기 (workers가 알아서 스킵)
        else:
            reply = QMessageBox.question(
                self, "전체 전처리 시작",
                f"모드: {mode_name}\n"
                f"출력: {fmt_label}\n"
                f"이미지: {len(self._all_paths)}장\n"
                f"저장: {out_dir}\n\n"
                f"완료 후 학습 탭에 자동 연결됩니다.\n시작하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        self._btn_full.setEnabled(False)
        self._btn_full_stop.setEnabled(True)
        self._full_bar.setValue(0)
        device      = self._dev_combo.currentData()
        dev_label   = self._dev_combo.currentText()
        num_workers = self._worker_combo.currentData()
        self._log_msg(f"\n🚀 전체 전처리 [{mode_name}]  {len(self._all_paths)}장")
        if can_resume:
            self._log_msg(f"   ⏩ 이어하기: {resume_cnt}개 스킵 → 나머지부터 시작")
        self._log_msg(f"   💾 출력 형식: {fmt_label}")
        self._log_msg(f"   🖥️ 보정 장치: {dev_label}")
        self._log_msg(f"   ⚙️ 병렬 워커: {num_workers}{'개 (순차)' if num_workers == 1 else f'개 병렬'}")
        self._log_msg(f"   저장: {out_dir}")

        self._full_worker = FullPreprocessWorker(
            self._all_paths,
            out_dir,
            mode          = "advanced" if self._adv_chk.isChecked() else "standard",
            src_root      = self._dir,
            device        = device,
            output_format = out_fmt,
            num_workers   = num_workers)
        self._full_worker.progress.connect(self._on_full_prog)
        self._full_worker.log_message.connect(self._log_msg)
        self._full_worker.finished.connect(self._on_full_done)
        self._full_worker.error.connect(self._on_full_err)
        self._full_worker.start()

    def _stop_full(self):
        if self._full_worker and self._full_worker.isRunning():
            self._full_worker.stop()
        self._btn_full.setEnabled(True)
        self._btn_full_stop.setEnabled(False)
        self._full_status.setText("⏹ 중단됨")

    def _on_full_prog(self, pct: int, fname: str):
        self._full_bar.setValue(pct)
        self._full_status.setText(f"전처리 중: {fname}  ({pct}%)")

    def _on_full_done(self, out_dir: str, total: int, failed: int):
        self._btn_full.setEnabled(True)
        self._btn_full_stop.setEnabled(False)
        self._full_bar.setValue(100)
        self._preprocessed_dir = out_dir

        out_fmt   = self._fmt_combo.currentData()   # "jpg" or "json"
        fmt_label = "JPG" if out_fmt == "jpg" else "JSON"

        self._full_status.setText(f"✅ 완료  {total}장  →  {out_dir}")
        self._log_msg(f"\n✅ 완료!  [{fmt_label}] 저장: {total}장" +
                      (f"  실패: {failed}장" if failed else ""))
        self._log_msg(f"   ➡️  학습 탭에 자동 연결됨")

        self.preprocess_done.emit(out_dir, self.get_mode_name(), out_fmt)

        QMessageBox.information(
            self, "전처리 완료",
            f"✅ {total}장 완료!\n\n"
            f"모드: {self.get_mode_name()}\n"
            f"출력: {fmt_label}\n"
            f"저장: {out_dir}\n\n"
            f"학습 탭에 자동으로 연결되었습니다.")

    def _on_full_err(self, msg: str):
        self._btn_full.setEnabled(True)
        self._btn_full_stop.setEnabled(False)
        self._full_status.setText(f"❌ {msg}")
        self._log_msg(f"❌ {msg}")

    # ── 미리보기 ─────────────────────────────────────────────
    def _run_preview(self):
        selected = self._all_paths[:self._n_spin.value()]
        if not selected:
            return
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        self._btn_prev.setEnabled(False)
        self._btn_prev_stop.setEnabled(True)
        mode   = "advanced" if self._adv_chk.isChecked() else "standard"
        device = self._dev_combo.currentData()
        self._worker = PreprocessWorker(selected, mode=mode, device=device)
        self._worker.progress.connect(
            lambda p, f: self._full_status.setText(f"미리보기: {f}  ({p}%)"))
        self._worker.image_ready.connect(self._show)
        self._worker.finished.connect(lambda m: (
            self._btn_prev.setEnabled(True),
            self._btn_prev_stop.setEnabled(False),
            self._full_status.setText(f"미리보기 완료  {len(m)}장")))
        self._worker.error.connect(
            lambda e: self._full_status.setText(f"❌ {e}"))
        self._worker.start()

    def _stop_preview(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        self._btn_prev.setEnabled(True)
        self._btn_prev_stop.setEnabled(False)

    def _show(self, proc_img, meta):
        op = meta.get("image_path", "")
        if op:
            px = QPixmap(op)
            if not px.isNull():
                self._orig_lbl.setPixmap(px.scaled(
                    190, 190, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))

        rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data.tobytes(), w, h, w * ch, QImage.Format.Format_RGB888)
        self._proc_lbl.setPixmap(QPixmap.fromImage(qi).scaled(
            190, 190, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

        sm = meta.get("skin_mask", None)
        if sm is not None:
            cm = np.zeros((*sm.shape, 3), dtype=np.uint8)
            cm[sm == 255] = [220, 120, 200]
            cm[sm == 0]   = [40, 40, 40]
            mh, mw = cm.shape[:2]
            qm = QImage(cm.data.tobytes(), mw, mh, mw * 3,
                        QImage.Format.Format_RGB888)
            self._mask_lbl.setPixmap(QPixmap.fromImage(qm).scaled(
                190, 190, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        else:
            self._mask_lbl.setText("정밀 모드에서만 표시")

        lab = meta.get("lab_mean", [0, 128, 128])
        rv  = meta.get("cheek_rgb", [200, 150, 150])
        det = meta.get("face_detected", False)
        rat = meta.get("skin_pixel_ratio", None)

        self._lab_lbl.setText(
            f"L*={lab[0]:.1f}  a*={lab[1]:.1f}  b*={lab[2]:.1f}"
            f"\n\nR={rv[0]}  G={rv[1]}  B={rv[2]}")
        qc = QColor(int(rv[0]), int(rv[1]), int(rv[2]))
        self._swatch.setStyleSheet(
            f"background:{qc.name()}; border:2px solid #555; border-radius:8px;")
        self._face_lbl.setText("✅ 얼굴 감지됨" if det else "⚠️ 얼굴 미감지")
        self._ratio_lbl.setText(
            f"피부 비율: {rat * 100:.1f}%" if rat is not None else "피부 비율: -")

    def _log_msg(self, text: str):
        self._log.append(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())