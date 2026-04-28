"""
ai/preprocessing.py
────────────────────
MediaPipe FaceMesh 기반 정밀 얼굴 크롭 + 피부 세그멘테이션 →
EfficientNet-B2 입력 텐서(260×260) + 16차원 피처 반환.

OOTD/preprocessors.py AdvancedSkinPreprocessor 와 동일한 로직.
MediaPipe 미설치 시 Haarcascade + YCrCb/HSV fallback 사용.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── MediaPipe 가용 여부 ────────────────────────────────────────────────────
# TensorFlow + protobuf 버전 충돌 시 mediapipe.tasks 임포트가 실패함.
# optional_dependencies.py 가 tensorflow.tools.docs 를 가져오는데
# protobuf 버전 불일치로 TF가 깨진 경우 stub으로 우회.
def _patch_tf_stub() -> None:
    import sys, types
    if "tensorflow" not in sys.modules:
        class _DocCtrl:
            def __getattr__(self, name):
                return lambda f=None, *a, **kw: (f if callable(f) else lambda fn: fn)
        tf_stub   = types.ModuleType("tensorflow")
        tf_tools  = types.ModuleType("tensorflow.tools")
        tf_docs   = types.ModuleType("tensorflow.tools.docs")
        tf_docs.doc_controls = _DocCtrl()
        tf_stub.tools = tf_tools
        sys.modules.setdefault("tensorflow",            tf_stub)
        sys.modules.setdefault("tensorflow.tools",      tf_tools)
        sys.modules.setdefault("tensorflow.tools.docs", tf_docs)

try:
    _patch_tf_stub()
    import mediapipe as mp
    _MP_FACE_MESH = mp.solutions.face_mesh
    _MEDIAPIPE_OK = True
    logger.info(f"MediaPipe FaceMesh available (v{mp.__version__}).")
except Exception as e:
    _MEDIAPIPE_OK = False
    logger.warning(f"MediaPipe 미설치/오류 → Haarcascade fallback 사용. ({e})")

# ── MediaPipe FaceMesh 랜드마크 인덱스 (OOTD/constants.py 와 동일) ─────────
_FACE_OVAL = [
    10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,
    172,58,132,93,234,127,162,21,54,103,67,109,
]
_LEFT_EYE       = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
_RIGHT_EYE      = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_LEFT_EYEBROW   = [70,63,105,66,107,55,65,52,53,46]
_RIGHT_EYEBROW  = [300,293,334,296,336,285,295,282,283,276]
_LIPS           = [
    61,146,91,181,84,17,314,405,321,375,291,
    308,324,318,402,317,14,87,178,88,95,
    185,40,39,37,0,267,269,270,409,415,
    310,311,312,13,82,81,42,183,78,
]
_LEFT_CHEEK  = [50,101,118,117,123,187,207,206,205,36,93]
_RIGHT_CHEEK = [280,330,347,346,352,411,427,426,425,266,323]

# ── 피부 세그멘테이션 상수 (OOTD AdvancedSkinPreprocessor 동일) ──────────
_YCR_MIN = np.array([0,   133,  77], dtype=np.uint8)
_YCR_MAX = np.array([255, 173, 127], dtype=np.uint8)
_Y_DARK  = 40
_DILATE  = 13

# ── b* 논문 기준값 (993명 표본: b*₀=18.50) ───────────────────────────────
_B_CV_TARGET = float(18.50 / 127.0 * 128.0 + 128.0)  # ≈ 146.6

# ── 모델 입력 크기 ─────────────────────────────────────────────────────────
TARGET_SIZE    = (260, 260)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])
FEAT_DIM = 16


@dataclass
class FaceDetectionResult:
    success:     bool
    face_crop:   Optional[np.ndarray]   = None
    tensor:      Optional[torch.Tensor] = None   # (1,3,260,260)
    meta_tensor: Optional[torch.Tensor] = None   # (1,16)
    bbox:        Optional[Tuple[int,int,int,int]] = None
    quality:     Optional[Dict[str, float]] = None
    message:     str = ""


# ══════════════════════════════════════════════════════════════════════════
#  유틸: 마스크 생성
# ══════════════════════════════════════════════════════════════════════════

def _hull_mask(indices, lm, h: int, w: int) -> np.ndarray:
    pts = np.array(
        [(int(np.clip(lm[i].x * w, 0, w - 1)),
          int(np.clip(lm[i].y * h, 0, h - 1)))
         for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    return mask


def _skin_mask_from_landmarks(rgb: np.ndarray, lm, h: int, w: int) -> np.ndarray:
    """
    OOTD AdvancedSkinPreprocessor._skin_mask 와 동일.
    Face Oval - (눈·눈썹·입술 + dilation) ∩ YCrCb ∩ HSV
    """
    # 얼굴 외곽
    face = _hull_mask(_FACE_OVAL, lm, h, w)

    # 눈·눈썹·입술 제외
    excl = np.zeros((h, w), dtype=np.uint8)
    for grp in (_LEFT_EYE, _RIGHT_EYE, _LEFT_EYEBROW, _RIGHT_EYEBROW, _LIPS):
        excl = cv2.bitwise_or(excl, _hull_mask(grp, lm, h, w))
    k_ex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_DILATE, _DILATE))
    excl = cv2.dilate(excl, k_ex, iterations=1)
    face = cv2.bitwise_and(face, cv2.bitwise_not(excl))

    # YCrCb 피부 범위
    ycc     = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    sc      = cv2.inRange(ycc, _YCR_MIN, _YCR_MAX)
    dark    = (ycc[:, :, 0] < _Y_DARK).astype(np.uint8) * 255
    sc      = cv2.bitwise_and(sc, cv2.bitwise_not(dark))

    # HSV 피부 범위
    hsv     = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ha      = cv2.inRange(hsv, np.array([0,   20, 50]), np.array([25,  170, 255]))
    hb      = cv2.inRange(hsv, np.array([160, 20, 50]), np.array([180, 170, 255]))
    sh      = cv2.bitwise_or(ha, hb)

    # 교집합 + 형태학적 정제
    comb    = cv2.bitwise_and(face, cv2.bitwise_and(sc, sh))
    k7      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    comb    = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k7, iterations=2)
    comb    = cv2.morphologyEx(comb, cv2.MORPH_OPEN,  k7, iterations=1)
    return comb


def _b_correct_cpu(bgr: np.ndarray, delta: float) -> np.ndarray:
    """OpenCV LAB b채널을 delta만큼 이동 (b* 보정)."""
    if abs(delta) < 0.5:
        return bgr.copy()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    b = np.clip(b + delta, 0, 255)
    merged = cv2.merge([L, a, b]).astype(np.uint8)
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════
#  16차원 피처 추출 (OOTD datasets.py _extract_features 동일)
# ══════════════════════════════════════════════════════════════════════════

def _extract_features_from_skin(skin_px_bgr: np.ndarray,
                                 skin_ratio: float,
                                 b_delta: float) -> torch.Tensor:
    """피부 픽셀 배열(N,3) BGR → 16차원 텐서 (1,16)."""
    if len(skin_px_bgr) == 0:
        return torch.zeros(1, 16, dtype=torch.float32)

    skin_img = skin_px_bgr.reshape(-1, 1, 3).astype(np.uint8)

    # LAB
    lab_px  = cv2.cvtColor(skin_img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab_mean = lab_px.mean(axis=0)
    lab_std  = lab_px.std(axis=0)
    L_cv, a_cv, b_cv = float(lab_mean[0]), float(lab_mean[1]), float(lab_mean[2])
    L_std, a_std, b_std = float(lab_std[0]), float(lab_std[1]), float(lab_std[2])

    # HSV 채도
    hsv_px = cv2.cvtColor(skin_img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hsv_s_mean = float(hsv_px[:, 1].mean() / 255.0 * 100.0)

    # 실제 Lab 스케일 변환
    L_real = L_cv / 255.0 * 100.0
    a_real = a_cv - 128.0
    b_real = b_cv - 128.0
    b_star = (b_cv - 128.0) / 128.0 * 127.0

    # ITA
    try:
        ita = math.degrees(math.atan2(L_real - 50.0, b_real)) if b_real != 0.0 else 0.0
    except Exception:
        ita = 0.0

    a_abs    = abs(a_real)
    b_var    = b_std ** 2
    chroma_c = (a_std / (a_abs + 1e-6)) if a_abs > 0.5 else 0.0
    lum_c    = (L_std / (L_real + 1e-6)) if L_real > 1.0 else 0.0

    raw = [
        L_real  / 100.0,
        a_real  / 128.0,
        b_real  / 128.0,
        L_std   / 50.0,
        a_std   / 64.0,
        b_std   / 64.0,
        max(-1.0, min(1.0, b_star / 127.0)),
        min(1.0, hsv_s_mean / 100.0),
        min(1.0, float(skin_ratio)),
        float(abs(b_delta) >= 0.5),           # corrected flag
        max(-1.0, min(1.0, ita / 90.0)),
        min(1.0, a_abs / 128.0),
        min(1.0, b_var / 4096.0),
        min(1.0, chroma_c / 5.0),
        min(1.0, lum_c),
        min(1.0, abs(b_delta) / 50.0),        # b_delta_abs
    ]
    return torch.tensor([raw], dtype=torch.float32)


def _quality_from_skin(skin_px_bgr: np.ndarray,
                       skin_ratio: float,
                       b_cv: float,
                       b_delta: float) -> Dict[str, float]:
    """조명/노출 상태를 비교하기 위한 전처리 진단값."""
    q: Dict[str, float] = {
        "skin_ratio": round(float(skin_ratio), 4),
        "b_cv": round(float(b_cv), 3),
        "b_delta": round(float(b_delta), 3),
        "l_mean": 0.0,
        "s_mean": 0.0,
        "highlight_ratio": 0.0,
        "shadow_ratio": 0.0,
    }
    if len(skin_px_bgr) == 0:
        return q

    lab_px = cv2.cvtColor(
        skin_px_bgr.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3).astype(np.float32)
    hsv_px = cv2.cvtColor(
        skin_px_bgr.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2HSV,
    ).reshape(-1, 3).astype(np.float32)

    q["l_mean"] = round(float(lab_px[:, 0].mean() / 255.0 * 100.0), 3)
    q["s_mean"] = round(float(hsv_px[:, 1].mean() / 255.0 * 100.0), 3)
    q["highlight_ratio"] = round(float((skin_px_bgr.max(axis=1) >= 245).mean()), 4)
    q["shadow_ratio"] = round(float((skin_px_bgr.max(axis=1) <= 35).mean()), 4)
    return q


# ══════════════════════════════════════════════════════════════════════════
#  메인 전처리기
# ══════════════════════════════════════════════════════════════════════════

class FacePreprocessor:
    """싱글톤. MediaPipe 우선, 없으면 Haarcascade fallback."""

    _instance: Optional["FacePreprocessor"] = None

    def __new__(cls) -> "FacePreprocessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        # MediaPipe FaceMesh (인스턴스별 독립 — QThread 안전)
        self._face_mesh = None
        if _MEDIAPIPE_OK:
            try:
                self._face_mesh = _MP_FACE_MESH.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                logger.info("FacePreprocessor: MediaPipe FaceMesh initialized.")
            except Exception as e:
                logger.warning(f"FaceMesh 초기화 실패 → Haarcascade fallback: {e}")
                self._face_mesh = None

        # Haarcascade fallback
        base = cv2.data.haarcascades
        self._face_cascade = cv2.CascadeClassifier(
            base + "haarcascade_frontalface_default.xml")
        self._eye_cascade  = cv2.CascadeClassifier(
            base + "haarcascade_eye.xml")
        if not _MEDIAPIPE_OK or self._face_mesh is None:
            logger.info("FacePreprocessor: Haarcascade mode active.")

    # ── 공개 인터페이스 ───────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> FaceDetectionResult:
        if bgr_frame is None or bgr_frame.size == 0:
            return FaceDetectionResult(success=False, message="빈 프레임")

        if self._face_mesh is not None:
            return self._process_mediapipe(bgr_frame)
        return self._process_haarcascade(bgr_frame)

    def close(self) -> None:
        if self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass

    # ── MediaPipe 경로 (OOTD AdvancedSkinPreprocessor 동일) ──────────────

    def _process_mediapipe(self, bgr: np.ndarray) -> FaceDetectionResult:
        h, w = bgr.shape[:2]
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            # 재시도: 해상도 리사이즈 후 검출
            scale = 640 / max(h, w)
            if scale < 1.0:
                small  = cv2.resize(rgb, (int(w * scale), int(h * scale)))
                res2   = self._face_mesh.process(small)
                if not res2.multi_face_landmarks:
                    return FaceDetectionResult(success=False, message="얼굴을 찾을 수 없습니다.")
                res = res2
                # 랜드마크 좌표는 정규화값(0~1)이므로 재스케일 불필요
            else:
                return FaceDetectionResult(success=False, message="얼굴을 찾을 수 없습니다.")

        lm = res.multi_face_landmarks[0].landmark

        # ── 볼 영역 b* 측정 (보정 기준) ──────────────────────────────
        b_cv_measured = self._measure_cheek_b(bgr, lm, h, w)
        b_delta       = _B_CV_TARGET - b_cv_measured

        # ── b* 보정 (논문 기준값으로 정규화) ─────────────────────────
        work_bgr = bgr
        if abs(b_delta) >= 0.5:
            work_bgr = _b_correct_cpu(bgr, b_delta)

        # ── 피부 마스크 ───────────────────────────────────────────────
        rgb_work = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
        sm       = _skin_mask_from_landmarks(rgb_work, lm, h, w)

        face_area    = int(_hull_mask(_FACE_OVAL, lm, h, w).sum() / 255)
        skin_px_cnt  = int(sm.sum() / 255)
        skin_ratio   = skin_px_cnt / max(face_area, 1)

        # ── 피부 픽셀 추출 ────────────────────────────────────────────
        skin_px_bgr = work_bgr[sm == 255]   # (N, 3)

        # ── 16차원 피처 ───────────────────────────────────────────────
        meta_tensor = _extract_features_from_skin(skin_px_bgr, skin_ratio, b_delta)
        quality = _quality_from_skin(skin_px_bgr, skin_ratio, b_cv_measured, b_delta)

        # ── 얼굴 크롭 (Face Oval bbox + 15% 패딩) ────────────────────
        xs = [lm[i].x for i in _FACE_OVAL]
        ys = [lm[i].y for i in _FACE_OVAL]
        fw_n = max(xs) - min(xs)
        fh_n = max(ys) - min(ys)
        pad  = 0.15
        cx1  = max(0, int((min(xs) - fw_n * pad) * w))
        cy1  = max(0, int((min(ys) - fh_n * pad) * h))
        cx2  = min(w, int((max(xs) + fw_n * pad) * w))
        cy2  = min(h, int((max(ys) + fh_n * pad) * h))

        if cx2 <= cx1 or cy2 <= cy1:
            return FaceDetectionResult(success=False, message="얼굴 크롭 bbox 무효")

        # 비피부 픽셀 → 검정 (OOTD 학습 데이터와 동일)
        crop_bgr  = work_bgr[cy1:cy2, cx1:cx2].copy()
        crop_mask = sm[cy1:cy2, cx1:cx2]
        crop_bgr[crop_mask == 0] = 0

        logger.info(
            f"[MediaPipe] crop={crop_bgr.shape[1]}×{crop_bgr.shape[0]} "
            f"skin={skin_ratio:.1%} b_cv={b_cv_measured:.1f} "
            f"b_delta={b_delta:+.1f} L={quality['l_mean']:.1f} "
            f"S={quality['s_mean']:.1f} "
            f"highlight={quality['highlight_ratio']:.1%} "
            f"shadow={quality['shadow_ratio']:.1%}")

        face_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor   = _transform(face_rgb).unsqueeze(0)

        return FaceDetectionResult(
            success=True,
            face_crop=crop_bgr,
            tensor=tensor,
            meta_tensor=meta_tensor,
            bbox=(cx1, cy1, cx2 - cx1, cy2 - cy1),
            quality=quality,
            message="OK",
        )

    @staticmethod
    def _measure_cheek_b(bgr: np.ndarray, lm, h: int, w: int) -> float:
        """볼 영역 LAB b채널 평균 (b* 보정 기준값 산출)."""
        lp = bgr[_hull_mask(_LEFT_CHEEK,  lm, h, w) == 255]
        rp = bgr[_hull_mask(_RIGHT_CHEEK, lm, h, w) == 255]
        cp = (np.vstack([lp, rp]) if lp.size and rp.size
              else (lp if lp.size else rp))
        if not cp.size:
            return _B_CV_TARGET  # 측정 불가 → 보정 없음
        lab = cv2.cvtColor(
            cp.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        return float(lab[:, 2].mean())

    # ── Haarcascade fallback ──────────────────────────────────────────────

    def _process_haarcascade(self, bgr: np.ndarray) -> FaceDetectionResult:
        h, w = bgr.shape[:2]
        gray  = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

        face = None
        for p in [
            dict(scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)),
            dict(scaleFactor=1.05, minNeighbors=3, minSize=(50, 50)),
        ]:
            faces = self._face_cascade.detectMultiScale(gray, **p)
            if len(faces) > 0:
                face = max(faces, key=lambda r: r[2] * r[3])
                break

        if face is None:
            return FaceDetectionResult(success=False, message="얼굴을 찾을 수 없습니다.")

        fx, fy, fw, fh = face
        x1 = max(0, fx - int(fw * 0.20))
        y1 = max(0, fy - int(fh * 0.10))
        x2 = min(w, fx + fw + int(fw * 0.20))
        y2 = min(h, fy + fh + int(fh * 0.35))
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return FaceDetectionResult(success=False, message="얼굴 크롭 실패")

        sm         = self._ycrcb_skin_mask(crop)
        skin_px    = crop[sm == 255]
        face_area  = crop.shape[0] * crop.shape[1]
        skin_ratio = len(skin_px) / max(face_area, 1)

        b_cv_measured = float(
            cv2.cvtColor(
                skin_px.reshape(-1, 1, 3).astype(np.uint8),
                cv2.COLOR_BGR2LAB
            ).reshape(-1, 3)[:, 2].mean()
        ) if len(skin_px) > 0 else _B_CV_TARGET
        b_delta = _B_CV_TARGET - b_cv_measured

        meta_tensor = _extract_features_from_skin(skin_px, skin_ratio, b_delta)
        quality = _quality_from_skin(skin_px, skin_ratio, b_cv_measured, b_delta)
        logger.info(
            f"[Haarcascade] crop={crop.shape[1]}×{crop.shape[0]} "
            f"skin={skin_ratio:.1%} b_cv={b_cv_measured:.1f} "
            f"b_delta={b_delta:+.1f} L={quality['l_mean']:.1f} "
            f"S={quality['s_mean']:.1f} "
            f"highlight={quality['highlight_ratio']:.1%} "
            f"shadow={quality['shadow_ratio']:.1%}")

        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor   = _transform(face_rgb).unsqueeze(0)
        return FaceDetectionResult(
            success=True, face_crop=crop, tensor=tensor,
            meta_tensor=meta_tensor, bbox=(x1, y1, x2 - x1, y2 - y1),
            quality=quality,
            message="OK (Haarcascade)",
        )

    def _ycrcb_skin_mask(self, bgr_crop: np.ndarray) -> np.ndarray:
        h, w = bgr_crop.shape[:2]
        ycr  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
        dark = (ycr[:, :, 0] < _Y_DARK).astype(np.uint8) * 255
        sc   = cv2.inRange(ycr, _YCR_MIN, _YCR_MAX)
        sc   = cv2.bitwise_and(sc, cv2.bitwise_not(dark))
        hsv  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        ha   = cv2.inRange(hsv, np.array([0,   20, 50]), np.array([25,  170, 255]))
        hb   = cv2.inRange(hsv, np.array([160, 20, 50]), np.array([180, 170, 255]))
        skin = cv2.bitwise_and(sc, cv2.bitwise_or(ha, hb))
        skin[:int(h * 0.38), :] = 0
        skin[int(h * 0.88):, :] = 0
        eyes = self._eye_cascade.detectMultiScale(
            cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(eyes) > 0:
            excl = np.zeros((h, w), dtype=np.uint8)
            for ex, ey, ew, eh in eyes:
                p = int(max(ew, eh) * 0.4)
                excl[max(0,ey-p):min(h,ey+eh+p), max(0,ex-p):min(w,ex+ew+p)] = 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_DILATE, _DILATE))
            skin = cv2.bitwise_and(skin, cv2.bitwise_not(cv2.dilate(excl, k)))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k7, iterations=2)
        return cv2.morphologyEx(skin, cv2.MORPH_OPEN, k7, iterations=1)


# ── 하위 호환 함수 (infer.py 에서 직접 호출 시 대비) ─────────────────────
def extract_16dim_features(bgr_crop: np.ndarray,
                            skin_mask: Optional[np.ndarray] = None) -> torch.Tensor:
    if skin_mask is not None and skin_mask.sum() > 0:
        skin_px = bgr_crop[skin_mask == 255]
        ratio   = len(skin_px) / max(bgr_crop.shape[0] * bgr_crop.shape[1], 1)
        b_cv    = float(cv2.cvtColor(
            skin_px.reshape(-1,1,3).astype(np.uint8),
            cv2.COLOR_BGR2LAB).reshape(-1,3)[:,2].mean()) if len(skin_px) > 0 else _B_CV_TARGET
        return _extract_features_from_skin(skin_px, ratio, _B_CV_TARGET - b_cv)
    else:
        lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        return _extract_features_from_skin(
            bgr_crop.reshape(-1, 3), 1.0, _B_CV_TARGET - float(lab[:, 2].mean()))
