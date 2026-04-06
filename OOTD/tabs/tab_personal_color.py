import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QProgressBar, QSpinBox, QComboBox,
    QTextEdit, QFileDialog, QCheckBox,
    QMessageBox, QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from constants import SEASON_CLASSES, SUPPORTED_EXT

# ─────────────────────────────────────────────────────────────
#  논문 기준값 (janal.txt)
#  원점: V₀=65.20 / b*₀=18.50 / S₀=0.33
# ─────────────────────────────────────────────────────────────
_B_TARGET    = 18.50          # 논문 b* 기준 (실제 스케일)
_B_TARGET_CV = float(_B_TARGET / 127.0 * 128.0 + 128.0)  # OpenCV b채널 ≈ 146.6
_V_TARGET    = 65.20
_S_TARGET    = 0.33


# ─────────────────────────────────────────────────────────────
#  4계절 × 3톤  변환 파라미터 (Lab 색공간)
#
#  폴더 구조:
#    {save_dir}/Spring/Warm/img.jpg
#    {save_dir}/Spring/Bright/img.jpg
#    {save_dir}/Spring/Light/img.jpg   ...
#
#  파라미터:
#    L_scale   : 명도 배율  (>1 밝게, <1 어둡게)
#    a_shift   : a* 이동   (+:붉은/마젠타, -:초록)
#    b_shift   : b* 이동   (+:노란/따뜻,   -:파란/쿨)
#    sat_scale : 채도 배율  (>1 선명, <1 뮤트)
#
#  논문 계절별 HSV-S 기준값 (볼 영역)
#    Spring=18.59 / Summer=12.50 / Autumn=27.14 / Winter=16.74
#  → b_shift / sat_scale 에 반영됨
# ─────────────────────────────────────────────────────────────
TONE_NAMES: List[str] = ["Warm", "Bright", "Light"]

# ─────────────────────────────────────────────────────────────
#  4계절 대표 파라미터 (각 계절의 Warm/Bright/Light 평균)
#  Stage 1 학습용 — 톤 구분 없이 계절 특성만 반영
#  각 계절의 3개 톤 파라미터를 평균하여 "순수 계절 색조" 생성
# ─────────────────────────────────────────────────────────────
SEASON_PARAMS: Dict[str, Dict] = {
    # Spring: L_scale avg(1.04,1.08,1.13)=1.083, a avg(7,5,3)=5, b avg(14,10,7)=10.3, sat avg(1.10,1.28,0.82)=1.07
    "Spring": dict(L_scale=1.08, a_shift=+5,  b_shift=+10, sat_scale=1.07),
    # Summer: L avg(1.02,1.05,1.11)=1.06, a avg(-2,-5,-3)=-3.3, b avg(-6,-13,-9)=-9.3, sat avg(0.90,1.12,0.68)=0.90
    "Summer": dict(L_scale=1.06, a_shift=-3,  b_shift=-9,  sat_scale=0.90),
    # Autumn: L avg(0.92,0.96,1.00)=0.96, a avg(10,7,4)=7, b avg(18,13,9)=13.3, sat avg(1.12,1.22,0.80)=1.05
    "Autumn": dict(L_scale=0.96, a_shift=+7,  b_shift=+13, sat_scale=1.05),
    # Winter: L avg(0.93,0.88,0.96)=0.92, a avg(3,-2,-1)=0, b avg(-8,-17,-10)=-11.7, sat avg(1.05,1.32,0.83)=1.07
    "Winter": dict(L_scale=0.92, a_shift= 0,  b_shift=-12, sat_scale=1.07),
}

SEASON_TONE_PARAMS: Dict[Tuple[str, str], Dict] = {
    # ── Spring  (웜톤, 중채도 S≈18.59) ──────────────────────
    ("Spring", "Warm"):   dict(L_scale=1.04, a_shift=+7,  b_shift=+14, sat_scale=1.10),
    ("Spring", "Bright"): dict(L_scale=1.08, a_shift=+5,  b_shift=+10, sat_scale=1.28),
    ("Spring", "Light"):  dict(L_scale=1.13, a_shift=+3,  b_shift=+7,  sat_scale=0.82),
    # ── Summer  (쿨톤, 저채도 S≈12.50) ──────────────────────
    ("Summer", "Warm"):   dict(L_scale=1.02, a_shift=-2,  b_shift=-6,  sat_scale=0.90),
    ("Summer", "Bright"): dict(L_scale=1.05, a_shift=-5,  b_shift=-13, sat_scale=1.12),
    ("Summer", "Light"):  dict(L_scale=1.11, a_shift=-3,  b_shift=-9,  sat_scale=0.68),
    # ── Autumn  (웜톤, 고채도 S≈27.14) ──────────────────────
    ("Autumn", "Warm"):   dict(L_scale=0.92, a_shift=+10, b_shift=+18, sat_scale=1.12),
    ("Autumn", "Bright"): dict(L_scale=0.96, a_shift=+7,  b_shift=+13, sat_scale=1.22),
    ("Autumn", "Light"):  dict(L_scale=1.00, a_shift=+4,  b_shift=+9,  sat_scale=0.80),
    # ── Winter  (쿨톤, 중채도 S≈16.74) ──────────────────────
    ("Winter", "Warm"):   dict(L_scale=0.93, a_shift=+3,  b_shift=-8,  sat_scale=1.05),
    ("Winter", "Bright"): dict(L_scale=0.88, a_shift=-2,  b_shift=-17, sat_scale=1.32),
    ("Winter", "Light"):  dict(L_scale=0.96, a_shift=-1,  b_shift=-10, sat_scale=0.83),
}


# ─────────────────────────────────────────────────────────────
#  b* 보정 함수 (논문 원점 b*₀=18.50 기준)
# ─────────────────────────────────────────────────────────────
def _normalize_lab_b(img_bgr: np.ndarray,
                     measured_b_cv: float) -> Tuple[np.ndarray, float]:
    """
    이미지 전체의 b* 채널을 논문 기준값(≈146.6 CV)으로 선형 이동합니다.

    Returns
    -------
    (보정된 BGR 이미지, delta)
      delta > 0 : 원본이 쿨하게 찍힘 → 워밍 보정
      delta < 0 : 원본이 웜하게 찍힘 → 쿨링 보정
    """
    delta = _B_TARGET_CV - measured_b_cv
    if abs(delta) < 0.5:
        return img_bgr.copy(), 0.0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, a, b = cv2.split(lab)
    b_corr = np.clip(b + delta, 0, 255)
    merged = cv2.merge([L, a, b_corr]).astype(np.uint8)
    return cv2.cvtColor(merged, cv2.COLOR_Lab2BGR), delta


def _measure_cheek_b(img_bgr: np.ndarray) -> float:
    """
    MediaPipe 없이도 빠르게 볼 영역 b* 추정:
    이미지 중앙 하단 1/4 영역(볼 근처)의 Lab b채널 평균을 반환합니다.
    MediaPipe가 있으면 preprocessors.get_cheek_lab_stats() 사용 권장.
    """
    h, w = img_bgr.shape[:2]
    # 중앙 하단 1/4 크롭 (볼 영역 근사)
    roi = img_bgr[h // 2: h * 3 // 4, w // 4: w * 3 // 4]
    if roi.size == 0:
        roi = img_bgr
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    return float(lab[:, :, 2].mean())



# ─────────────────────────────────────────────────────────────
#  GPU 배치 변환 — 보정된 원본 1장으로 12장을 한 번에 처리
#  LAB 텐서를 GPU에 올려두고 12개 파라미터를 연속 적용
# ─────────────────────────────────────────────────────────────
def _apply_all_gpu(work_bgr: np.ndarray,
                   save_dir: str, stem: str, suffix: str,
                   mode: str = "12class") -> Tuple[int, int]:
    """GPU에서 변환·저장  mode='4season'이면 4장, '12class'면 12장"""
    import torch
    saved, failed = 0, 0
    lab_np = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    t_base = torch.from_numpy(lab_np).cuda()

    if mode == "4season":
        targets = [(s, None, SEASON_PARAMS[s]) for s in SEASON_CLASSES]
    else:
        targets = [(s, t, SEASON_TONE_PARAMS[(s, t)])
                   for s in SEASON_CLASSES for t in TONE_NAMES]

    for item in targets:
        season, tone, p = item
        try:
            L  = (t_base[..., 0] * p["L_scale"]).clamp(0, 255)
            ac = ((t_base[..., 1] - 128.0) * p["sat_scale"]
                  + p["a_shift"] + 128.0).clamp(0, 255)
            bc = ((t_base[..., 2] - 128.0) * p["sat_scale"]
                  + p["b_shift"] + 128.0).clamp(0, 255)
            merged    = torch.stack([L, ac, bc], dim=-1).byte().cpu().numpy()
            converted = cv2.cvtColor(merged, cv2.COLOR_Lab2BGR)
            if tone:
                dst_dir = Path(save_dir) / season / tone
            else:
                dst_dir = Path(save_dir) / season
            dst_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_dir / (stem + suffix)), converted)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"[실패] {stem}{suffix} [{season}{'/' + tone if tone else ''}]: {e}")
    return saved, failed


def _apply_all_cpu(work_bgr: np.ndarray,
                   save_dir: str, stem: str, suffix: str,
                   mode: str = "12class") -> Tuple[int, int]:
    """CPU에서 변환·저장  mode='4season'이면 4장, '12class'면 12장"""
    lab = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L0, a0, b0 = cv2.split(lab)
    saved, failed = 0, 0

    if mode == "4season":
        targets = [(s, None, SEASON_PARAMS[s]) for s in SEASON_CLASSES]
    else:
        targets = [(s, t, SEASON_TONE_PARAMS[(s, t)])
                   for s in SEASON_CLASSES for t in TONE_NAMES]

    for item in targets:
        season, tone, p = item
        try:
            L  = np.clip(L0 * p["L_scale"], 0, 255)
            ac = np.clip((a0 - 128.0) * p["sat_scale"]
                         + p["a_shift"] + 128.0, 0, 255)
            bc = np.clip((b0 - 128.0) * p["sat_scale"]
                         + p["b_shift"] + 128.0, 0, 255)
            merged    = cv2.merge([L, ac, bc]).astype(np.uint8)
            converted = cv2.cvtColor(merged, cv2.COLOR_Lab2BGR)
            if tone:
                dst_dir = Path(save_dir) / season / tone
            else:
                dst_dir = Path(save_dir) / season
            dst_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_dir / (stem + suffix)), converted)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"[실패] {stem}{suffix} [{season}{'/' + tone if tone else ''}]: {e}")
    return saved, failed


# ─────────────────────────────────────────────────────────────
#  단일 이미지 처리 (ThreadPool worker 함수)
#  ① b* 보정(ROI 방식, MediaPipe 호출 없음)
#  → ② GPU/CPU 배치 변환(12장)  → ③ 저장
#
#  ★ MediaPipe FaceMesh는 SaveWorker.run()에서 스레드당 1개만
#    생성해 재사용합니다 (process_image_with_fm 참조)
# ─────────────────────────────────────────────────────────────
def _process_image(args: Tuple) -> Tuple[str, int, int, float, bool]:
    """
    MediaPipe 없이 중앙 ROI b* 추정 → 보정 → 12장 변환
    (빠른 경로: MediaPipe 초기화 비용 없음)

    Returns
    -------
    (img_path, saved, failed, delta, corrected)
    """
    img_path, save_dir, use_gpu, apply_correction, mode = args
    src = Path(img_path)
    n_out = 4 if mode == "4season" else 12
    if not src.exists():
        return img_path, 0, n_out, 0.0, False

    img_bgr = cv2.imread(str(src))
    if img_bgr is None:
        return img_path, 0, n_out, 0.0, False

    # ── ① b* 보정 (ROI 근사 — 빠름) ──────────────────────────
    delta     = 0.0
    corrected = False
    work_bgr  = img_bgr

    if apply_correction:
        b_cv             = _measure_cheek_b(img_bgr)   # MediaPipe 없이 추정
        work_bgr, delta  = _normalize_lab_b(img_bgr, b_cv)
        corrected        = abs(delta) >= 0.5

    # ── ② 배치 변환 & 저장 ────────────────────────────────────
    stem, suffix = src.stem, src.suffix
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 없음")
            saved, failed = _apply_all_gpu(work_bgr, save_dir, stem, suffix, mode)
        except Exception:
            saved, failed = _apply_all_cpu(work_bgr, save_dir, stem, suffix, mode)
    else:
        saved, failed = _apply_all_cpu(work_bgr, save_dir, stem, suffix, mode)

    return img_path, saved, failed, delta, corrected


# ─────────────────────────────────────────────────────────────
#  QThread 워커
# ─────────────────────────────────────────────────────────────
class SaveWorker(QThread):
    # (퍼센트, 파일명, delta, corrected)
    progress = pyqtSignal(int, str, float, bool)
    finished = pyqtSignal(int, int, int, int)   # saved, failed, corrected, skipped
    error    = pyqtSignal(str)

    def __init__(self, image_paths: List[str], save_dir: str,
                 use_gpu: bool, num_workers: int,
                 apply_correction: bool = True,
                 mode: str = "12class"):
        super().__init__()
        self.image_paths      = image_paths
        self.save_dir         = save_dir
        self.use_gpu          = use_gpu
        self.num_workers      = max(1, num_workers)
        self.apply_correction = apply_correction
        self.mode             = mode   # "4season" or "12class"
        self._running         = True

    def run(self):
        try:
            total_imgs     = len(self.image_paths)
            total_saved    = 0
            total_failed   = 0
            total_corrected = 0
            total_skipped  = 0
            done           = 0

            args_list = [
                (p, self.save_dir, self.use_gpu, self.apply_correction, self.mode)
                for p in self.image_paths
            ]

            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                futures = {
                    pool.submit(_process_image, a): a[0]
                    for a in args_list
                }
                for fut in as_completed(futures):
                    if not self._running:
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    img_path, saved, failed, delta, corrected = fut.result()
                    total_saved   += saved
                    total_failed  += failed
                    if corrected:
                        total_corrected += 1
                    else:
                        total_skipped   += 1
                    done += 1
                    pct = int(done / total_imgs * 100)
                    self.progress.emit(
                        pct, Path(img_path).name, delta, corrected)

            self.finished.emit(
                total_saved, total_failed, total_corrected, total_skipped)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────
#  탭 UI
# ─────────────────────────────────────────────────────────────
_SEASON_ICONS = {
    "Spring": "🌸", "Summer": "☀️", "Autumn": "🍂", "Winter": "❄️"}
_TONE_ICONS   = {"Warm": "🔥", "Bright": "✨", "Light": "🌤️"}
_TONE_DESC    = {
    "Warm":   "웜  — 따뜻하고 깊은 색조",
    "Bright": "브라이트  — 선명·화사한 색조",
    "Light":  "라이트  — 밝고 부드러운 색조",
}


class PersonalColorTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_paths: List[str]        = []
        self._save_worker: Optional[SaveWorker] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── ① 컨트롤 패널 ────────────────────────────────────
        ctrl = QGroupBox("🎨 데이터셋 생성  (이미지 1장 → 자동 변환)")
        crow = QHBoxLayout(ctrl)
        crow.setSpacing(10)

        self._folder_btn = QPushButton("📂 원본 폴더 선택")
        self._folder_btn.clicked.connect(self._select_folder)
        crow.addWidget(self._folder_btn)

        self._folder_lbl = QLabel("선택된 폴더 없음")
        self._folder_lbl.setStyleSheet("color:#555; font-size:11px;")
        crow.addWidget(self._folder_lbl, 1)

        # ── 생성 모드 선택 ──────────────────────────────────
        crow.addWidget(QLabel("📦 생성 모드:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("🌿 4계절  (1장 → 4장, Stage1 학습용)", "4season")
        self._mode_combo.addItem("🌈 12클래스  (1장 → 12장, Stage2 학습용)", "12class")
        self._mode_combo.setFixedWidth(260)
        self._mode_combo.setToolTip(
            "4계절 모드:\n"
            "  이미지 1장 → Spring / Summer / Autumn / Winter 4장\n"
            "  각 계절의 대표 색조(Warm/Bright/Light 평균)로 변환\n"
            "  Stage 1 학습용 순수 계절 데이터셋 생성\n\n"
            "12클래스 모드:\n"
            "  이미지 1장 → 4계절 × 3톤 = 12장\n"
            "  Stage 2 학습용 세분화 데이터셋 생성")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        crow.addWidget(self._mode_combo)

        crow.addWidget(QLabel("🖥️ 장치:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["CPU", "GPU (CUDA)"])
        self._device_combo.setFixedWidth(115)
        crow.addWidget(self._device_combo)

        crow.addWidget(QLabel("⚡ 스레드:"))
        self._workers_spin = QSpinBox()
        cpu_count = multiprocessing.cpu_count()
        self._workers_spin.setRange(1, cpu_count)
        self._workers_spin.setValue(min(4, cpu_count))
        self._workers_spin.setFixedWidth(55)
        self._workers_spin.setToolTip(
            f"동시 처리 스레드 수 (CPU 코어: {cpu_count})")
        crow.addWidget(self._workers_spin)

        self._btn_save = QPushButton("🚀 변환 & 저장")
        self._btn_save.setEnabled(False)
        self._btn_save.setMinimumHeight(36)
        self._btn_save.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self._btn_save.clicked.connect(self._start_save)
        crow.addWidget(self._btn_save)

        self._btn_stop = QPushButton("■ 중지")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setMinimumHeight(36)
        self._btn_stop.clicked.connect(self._stop_save)
        crow.addWidget(self._btn_stop)

        root.addWidget(ctrl)

        # ── ② b* 보정 옵션 패널 ──────────────────────────────
        corr_grp = QGroupBox(
            "🔬 논문 기준 b* 보정 설정  "
            "(다른 환경에서 수집한 데이터셋 정규화)")
        corr_lay = QHBoxLayout(corr_grp)
        corr_lay.setSpacing(14)

        self._corr_chk = QCheckBox(
            "b* 자동 보정 활성화  "
            "(볼 황색도를 논문 기준 b*₀ = 18.50 으로 정규화)")
        self._corr_chk.setChecked(True)
        self._corr_chk.setToolTip(
            "다른 조명·카메라 환경에서 촬영된 데이터셋도\n"
            "논문 표준(D50/5000K) 기준으로 보정하여 학습에 사용합니다.\n\n"
            "보정 공식:\n"
            "  delta = b*₀(18.50 → CV≈146.6) - 측정된 볼 b채널\n"
            "  delta > 0 : 쿨하게 찍힘 → 워밍 보정\n"
            "  delta < 0 : 웜하게 찍힘 → 쿨링 보정")
        corr_lay.addWidget(self._corr_chk, 1)

        # 기준값 표시
        ref_frame = QFrame()
        ref_frame.setStyleSheet(
            "background:#EAF4FB; border:1px solid #AED6F1; "
            "border-radius:5px; padding:4px;")
        ref_lay = QHBoxLayout(ref_frame)
        ref_lay.setSpacing(16)
        for label, val in [("b*₀", "18.50"), ("V₀", "65.20"), ("S₀", "0.33")]:
            lbl = QLabel(f"<b>{label}</b> = {val}")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("font-size:11px; color:#2471A3;")
            ref_lay.addWidget(lbl)
        corr_lay.addWidget(ref_frame)

        root.addWidget(corr_grp)

        # ── ③ 진행률 & 상태 ──────────────────────────────────
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setFormat("%p%  (%v / %m)")
        root.addWidget(self._bar)

        self._status = QLabel("폴더를 선택하고 변환을 시작하세요.")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("font-size:12px; color:#2C3E50;")
        root.addWidget(self._status)

        # ── ④ 폴더 구조 미리보기 (모드에 따라 동적 변경) ───────
        self._prev_grp = QGroupBox("📂 생성될 폴더 구조")
        self._prev_layout = QVBoxLayout(self._prev_grp)
        root.addWidget(self._prev_grp)
        self._rebuild_preview("4season")

        # ── ⑤ 로그 ───────────────────────────────────────────
        log_grp = QGroupBox("📋 변환 로그  (보정량 delta 실시간 표시)")
        log_layout = QVBoxLayout(log_grp)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        self._log.setStyleSheet("background:#1E1E1E; color:#00FF7F;")
        log_layout.addWidget(self._log)
        root.addWidget(log_grp, 1)

    # ── 슬롯 ─────────────────────────────────────────────────
    def _on_mode_changed(self, _=0):
        mode = self._mode_combo.currentData()
        self._rebuild_preview(mode)
        if self._all_paths:
            n = len(self._all_paths)
            out_n = 4 if mode == "4season" else 12
            tag   = "4계절" if mode == "4season" else "4계절 × 3톤"
            self._folder_lbl.setText(
                f"{n}개 이미지  →  변환 후 {n * out_n}장 ({tag})")

    def _rebuild_preview(self, mode: str):
        """모드에 따라 미리보기 그리드 재구성"""
        # 기존 위젯 제거
        while self._prev_layout.count():
            item = self._prev_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        grid = QGridLayout()
        grid.setSpacing(5)

        if mode == "4season":
            self._prev_grp.setTitle(
                "📂 생성될 폴더 구조  (4계절 = 4개 하위 폴더)  ← Stage 1 학습용")
            _season_colors = {
                "Spring": "#FFF0F5", "Summer": "#E8F4FD",
                "Autumn": "#FFF8E1", "Winter": "#F0F4FF",
            }
            for ci, season in enumerate(SEASON_CLASSES):
                p = SEASON_PARAMS[season]
                cell = QLabel(
                    f"{_SEASON_ICONS.get(season, '')} <b>{season}</b><br>"
                    f"<span style='font-size:9px;color:#555'>"
                    f"L×{p['L_scale']}  a{p['a_shift']:+d}  "
                    f"b{p['b_shift']:+d}  S×{p['sat_scale']}</span>")
                cell.setTextFormat(Qt.TextFormat.RichText)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setFixedHeight(56)
                cell.setStyleSheet(
                    f"background:{_season_colors.get(season, '#EEE')};"
                    f"border:1px solid #ccc;border-radius:6px;"
                    f"padding:4px;font-size:11px;")
                grid.addWidget(cell, 0, ci)

            info = QLabel(
                "✅ 각 계절의 Warm / Bright / Light 파라미터 평균값으로 대표 색조 생성\n"
                "→ 순수 계절 특성만 반영 — Stage 1 (계절 분류) 학습에 최적")
            info.setStyleSheet(
                "color:#1A5276;font-size:10px;padding:4px;"
                "background:#EAF4FB;border-radius:4px;")
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._prev_layout.addLayout(grid)
            self._prev_layout.addWidget(info)

        else:  # 12class
            self._prev_grp.setTitle(
                "📂 생성될 폴더 구조  (4계절 × 3톤 = 12개 하위 폴더)  ← Stage 2 학습용")
            for ci, season in enumerate(SEASON_CLASSES):
                hdr = QLabel(f"{_SEASON_ICONS.get(season, '')} <b>{season}</b>")
                hdr.setTextFormat(Qt.TextFormat.RichText)
                hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
                hdr.setStyleSheet(
                    "background:#E8F4FD;border:1px solid #AED6F1;"
                    "border-radius:4px;padding:4px;font-size:12px;")
                grid.addWidget(hdr, 0, ci)

            _tone_colors = {
                "Warm": "#FFF3E0", "Bright": "#E8F8E8", "Light": "#F3E5F5"}
            for ri, tone in enumerate(TONE_NAMES):
                for ci, season in enumerate(SEASON_CLASSES):
                    cell = QLabel(
                        f"{_TONE_ICONS.get(tone, '')} <b>{tone}</b><br>"
                        f"<span style='font-size:9px;color:#666'>"
                        f"{_TONE_DESC[tone]}</span>")
                    cell.setTextFormat(Qt.TextFormat.RichText)
                    cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    cell.setFixedHeight(52)
                    cell.setStyleSheet(
                        f"background:{_tone_colors[tone]};"
                        f"border:1px solid #ccc;"
                        f"border-radius:4px;padding:3px;font-size:11px;")
                    grid.addWidget(cell, ri + 1, ci)

            info = QLabel(
                "✅ 4계절 × 3톤(Warm/Bright/Light) 세분화 변환\n"
                "→ Stage 2 (톤 분류) 학습에 최적")
            info.setStyleSheet(
                "color:#196F3D;font-size:10px;padding:4px;"
                "background:#EAFAF1;border-radius:4px;")
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._prev_layout.addLayout(grid)
            self._prev_layout.addWidget(info)

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "원본 폴더 선택", "")
        if not folder:
            return
        self._all_paths = [
            str(p) for p in Path(folder).rglob("*")
            if p.suffix.lower() in SUPPORTED_EXT
        ]
        n    = len(self._all_paths)
        mode = self._mode_combo.currentData()
        out_n = 4 if mode == "4season" else 12
        tag   = "4계절" if mode == "4season" else "4계절 × 3톤"

        self._folder_lbl.setText(
            f"{Path(folder).name}  ({n}개  →  {n * out_n}장 [{tag}])")
        self._btn_save.setEnabled(n > 0)
        self._bar.setMaximum(n)
        self._bar.setValue(0)
        self._bar.setFormat(f"%v / {n}장  (%p%)")
        self._log_msg(f"📁 {folder}")
        self._log_msg(
            f"   이미지 {n}장  →  {n * out_n}장 생성 예정  [{tag}]")

        corr_on = self._corr_chk.isChecked()
        if corr_on:
            self._log_msg(
                f"   🔬 b* 보정 활성화  "
                f"(기준 b*₀={_B_TARGET:.2f} → CV≈{_B_TARGET_CV:.1f})")
        else:
            self._log_msg("   ⚠️  b* 보정 비활성화 — 원본 그대로 변환")

    def _start_save(self):
        if not self._all_paths:
            QMessageBox.warning(self, "경고", "폴더를 먼저 선택하세요.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "저장 경로 선택", "")
        if not save_dir:
            return

        use_gpu          = (self._device_combo.currentIndex() == 1)
        num_workers      = self._workers_spin.value()
        apply_correction = self._corr_chk.isChecked()
        mode             = self._mode_combo.currentData()
        out_n            = 4 if mode == "4season" else 12
        tag              = "4계절" if mode == "4season" else "4계절 × 3톤"

        if use_gpu:
            try:
                import torch as _torch
                cuda_ok = _torch.cuda.is_available()
                if cuda_ok:
                    self._log_msg(
                        f"   🖥️ GPU: {_torch.cuda.get_device_name(0)}")
            except Exception:
                cuda_ok = False
            if not cuda_ok:
                QMessageBox.warning(
                    self, "GPU 없음",
                    "PyTorch CUDA를 찾을 수 없습니다.\n"
                    "CPU 모드로 전환합니다.")
                use_gpu = False
                self._device_combo.setCurrentIndex(0)

        self._btn_save.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._bar.setValue(0)

        device_str = ("GPU (CUDA)" if use_gpu
                      else f"CPU × {num_workers} 스레드")
        corr_str   = "보정 ON" if apply_correction else "보정 OFF"
        self._status.setText(
            f"⚙️ 변환 중  [{tag}]  [{device_str}]  [{corr_str}]  ...")
        self._log_msg(
            f"\n🚀 변환 시작  모드={tag}  장치={device_str}  {corr_str}  저장={save_dir}")
        self._log_msg(
            f"   {len(self._all_paths)}장 × {out_n}변형 = "
            f"{len(self._all_paths) * out_n}장 처리 예정\n")

        self._save_worker = SaveWorker(
            self._all_paths, save_dir, use_gpu, num_workers,
            apply_correction, mode)
        self._save_worker.progress.connect(self._on_progress)
        self._save_worker.finished.connect(self._on_finished)
        self._save_worker.error.connect(self._on_error)
        self._save_worker.start()

    def _stop_save(self):
        if self._save_worker and self._save_worker.isRunning():
            self._save_worker.stop()
        self._btn_save.setEnabled(bool(self._all_paths))
        self._btn_stop.setEnabled(False)
        self._status.setText("⏹ 중단됨")
        self._log_msg("⏹ 사용자가 중단했습니다.")

    def _on_progress(self, pct: int, filename: str,
                     delta: float, corrected: bool):
        self._bar.setValue(pct)
        if corrected:
            direction = "워밍↑" if delta > 0 else "쿨링↓"
            corr_info = f"delta={delta:+.1f}({direction})"
            self._status.setText(
                f"🔬 {filename}  보정:{corr_info}  → 12장 생성  ({pct}%)")
            self._log_msg(
                f"  ✅ {filename}  {corr_info}  → 12장  [{pct}%]")
        else:
            self._status.setText(
                f"✅ {filename}  보정없음  → 12장 생성  ({pct}%)")
            self._log_msg(
                f"  ✅ {filename}  보정없음(기준 근접)  → 12장  [{pct}%]")

    def _on_finished(self, total_saved: int, total_failed: int,
                     total_corrected: int, total_skipped: int):
        self._btn_save.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._bar.setValue(self._bar.maximum())

        mode  = self._mode_combo.currentData()
        n_imgs = len(self._all_paths)
        out_n  = 4 if mode == "4season" else 12
        tag    = "4계절" if mode == "4season" else "4계절 × 3톤"
        struct = "4개 폴더 (Stage 1 학습용)" if mode == "4season" \
                 else "12개 폴더 (Stage 2 학습용)"

        if mode == "4season":
            detail = "\n".join(
                f"  {_SEASON_ICONS.get(s, '')} {s}/"
                for s in SEASON_CLASSES)
        else:
            detail = "\n".join(
                f"  {_SEASON_ICONS.get(s, '')} {s}:  "
                + "  /  ".join(
                    f"{_TONE_ICONS.get(t, '')} {t}" for t in TONE_NAMES)
                for s in SEASON_CLASSES)

        msg = (
            f"✅ 총 {total_saved}장 저장 완료!\n"
            f"({n_imgs}장 × {tag} = {n_imgs * out_n}장)\n\n"
            f"📂 폴더 구조: {struct}\n\n"
            f"🔬 b* 보정 적용: {total_corrected}장\n"
            f"⬜ 보정 불필요(기준 근접): {total_skipped}장\n\n"
            f"{detail}"
        )
        if total_failed:
            msg += f"\n\n⚠️ 실패: {total_failed}장"

        self._status.setText(
            f"🎉 완료! [{tag}]  {total_saved}장 저장  "
            f"(보정 {total_corrected}장)")
        self._log_msg(
            f"\n🎉 완료! [{tag}]  총 {total_saved}장 저장됨.\n"
            f"   보정 적용: {total_corrected}장  /  "
            f"보정 불필요: {total_skipped}장")
        if total_failed:
            self._log_msg(f"⚠️ 실패: {total_failed}장")
        QMessageBox.information(self, "완료", msg)

    def _on_error(self, msg: str):
        self._btn_save.setEnabled(bool(self._all_paths))
        self._btn_stop.setEnabled(False)
        self._status.setText(f"❌ 오류: {msg}")
        self._log_msg(f"❌ 오류: {msg}")
        QMessageBox.critical(self, "오류", msg)

    def _log_msg(self, text: str):
        self._log.append(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())