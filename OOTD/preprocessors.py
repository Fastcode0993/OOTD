import abc
import logging
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from constants import (
    LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS,
    FACE_OVAL_LANDMARKS,
    LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS,
    LEFT_EYEBROW_LANDMARKS, RIGHT_EYEBROW_LANDMARKS,
    LIPS_LANDMARKS,
)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe 미설치")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════
#  논문 기준값 (janal.txt)
#  V₀=65.20  b*₀=18.50  S₀=0.33  ← 993명 표본 기반 원점
# ═════════════════════════════════════════════════════════════════
_B_TARGET = 18.50
_V_TARGET = 65.20
_S_TARGET = 0.33
_B_TARGET_CV = float(_B_TARGET / 127.0 * 128.0 + 128.0)  # ≈ 146.6


# ─────────────────────────────────────────────────────────────
#  GPU 가속 b* 보정 — 배치 처리
#  이미지 리스트를 GPU에 올려 한번에 Lab 변환 + b채널 이동
# ─────────────────────────────────────────────────────────────
class GPULabCorrector:
    """
    GPU(CUDA)로 b* 보정을 배치 처리합니다.
    MediaPipe는 CPU에서 실행되며, 보정 연산만 GPU에서 수행합니다.

    사용법:
        corrector = GPULabCorrector(device="cuda")
        corrected_bgr = corrector.correct(bgr_img, measured_b_cv)
        corrected_list = corrector.correct_batch([(bgr1, b1), (bgr2, b2), ...])
    """
    def __init__(self, device: str = "cuda"):
        self.device = self._resolve_device(device)
        logging.info(f"[GPULabCorrector] 장치: {self.device}")

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "cuda":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            logging.warning("CUDA 불가 → CPU 폴백")
            return "cpu"
        return "cpu"

    def correct(self, img_bgr: np.ndarray, measured_b_cv: float) -> np.ndarray:
        """단일 이미지 b* 보정 (GPU or CPU)"""
        delta = _B_TARGET_CV - measured_b_cv
        if abs(delta) < 0.5:
            return img_bgr.copy()

        if self.device == "cuda" and TORCH_AVAILABLE:
            return self._correct_gpu(img_bgr, delta)
        return self._correct_cpu(img_bgr, delta)

    def correct_batch(self, items: list) -> list:
        """
        배치 b* 보정.
        items: [(bgr_ndarray, measured_b_cv), ...]
        returns: [corrected_bgr_ndarray, ...]
        GPU 모드에서는 전체를 한 번에 처리해 속도 극대화.
        """
        if self.device == "cuda" and TORCH_AVAILABLE:
            return self._correct_batch_gpu(items)
        return [self._correct_cpu(bgr, _B_TARGET_CV - b)
                for bgr, b in items]

    # ── CPU 보정 (기존 방식) ──────────────────────────────────
    @staticmethod
    def _correct_cpu(img_bgr: np.ndarray, delta: float) -> np.ndarray:
        if abs(delta) < 0.5:
            return img_bgr.copy()
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        L, a, b = cv2.split(lab)
        b = np.clip(b + delta, 0, 255)
        merged = cv2.merge([L, a, b]).astype(np.uint8)
        return cv2.cvtColor(merged, cv2.COLOR_Lab2BGR)

    # ── GPU 단일 보정 ─────────────────────────────────────────
    def _correct_gpu(self, img_bgr: np.ndarray, delta: float) -> np.ndarray:
        try:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
            t   = torch.from_numpy(lab).to(self.device)  # (H,W,3)
            t[..., 2] = (t[..., 2] + delta).clamp(0, 255)
            result = t.cpu().numpy().astype(np.uint8)
            return cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        except Exception as e:
            logging.warning(f"GPU 보정 실패 → CPU 폴백: {e}")
            return self._correct_cpu(img_bgr, delta)

    # ── GPU 배치 보정 ─────────────────────────────────────────
    def _correct_batch_gpu(self, items: list) -> list:
        """
        서로 다른 크기의 이미지를 GPU에서 개별 처리.
        (배치 스택 대신 GPU 텐서 연산만 활용 — 크기 불일치 회피)
        """
        results = []
        try:
            for bgr, b_cv in items:
                delta = _B_TARGET_CV - b_cv
                if abs(delta) < 0.5:
                    results.append(bgr.copy())
                    continue
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
                t   = torch.from_numpy(lab).to(self.device)
                t[..., 2] = (t[..., 2] + delta).clamp(0, 255)
                out = t.cpu().numpy().astype(np.uint8)
                results.append(cv2.cvtColor(out, cv2.COLOR_Lab2BGR))
        except Exception as e:
            logging.warning(f"GPU 배치 보정 실패 → CPU 폴백: {e}")
            results = [self._correct_cpu(bgr, _B_TARGET_CV - b)
                       for bgr, b in items]
        return results

    def info(self) -> str:
        if self.device == "cuda" and TORCH_AVAILABLE:
            try:
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                return f"GPU ({name}, {vram:.1f}GB)"
            except Exception:
                pass
        return "CPU"


def normalize_lab_b(img_bgr: np.ndarray,
                    measured_b_cv: float,
                    device: str = "cpu") -> np.ndarray:
    """
    b* 채널을 논문 기준(b*₀=18.50)으로 보정합니다.
    device="cuda" 시 GPU 가속 적용.
    """
    corrector = GPULabCorrector(device)
    delta = _B_TARGET_CV - measured_b_cv
    return corrector._correct_gpu(img_bgr, delta) if device == "cuda" \
        else corrector._correct_cpu(img_bgr, delta)


def get_cheek_lab_stats(img_bgr: np.ndarray,
                        lm, h: int, w: int) -> Dict:
    """
    MediaPipe 랜드마크로 볼 영역 픽셀을 추출하고
    OpenCV LAB 통계 및 HSV-S 값을 반환합니다.

    Returns
    -------
    {
      "cheek_rgb"    : [R, G, B],
      "lab_mean"     : [L_cv, a_cv, b_cv],   # OpenCV 스케일
      "lab_std"      : [L_std, a_std, b_std],
      "hsv_s_mean"   : float,                 # HSV S 평균 (0~100)
      "b_star"       : float,                 # 실제 b* 값 (논문 스케일)
      "b_cv"         : float,                 # OpenCV b채널 평균
    }
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _region(indices):
        pts = [(int(np.clip(lm[i].x * w, 0, w - 1)),
                int(np.clip(lm[i].y * h, 0, h - 1))) for i in indices]
        if not pts:
            return np.empty((0, 3), dtype=np.uint8)
        arr  = np.array(pts, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(arr), 255)
        return rgb[mask == 255]

    lp = _region(LEFT_CHEEK_LANDMARKS)
    rp = _region(RIGHT_CHEEK_LANDMARKS)
    cp = (np.vstack([lp, rp]) if lp.size and rp.size
          else (lp if lp.size else rp))

    out: Dict = {
        "cheek_rgb": [200, 150, 150],
        "lab_mean":  [0., 128., 128.],
        "lab_std":   [0., 0., 0.],
        "hsv_s_mean": 0.0,
        "b_star":    0.0,
        "b_cv":      128.0,
    }
    if not cp.size:
        return out

    mr  = cp.mean(axis=0).astype(np.uint8)
    out["cheek_rgb"] = mr.tolist()

    # LAB
    lab_px = cv2.cvtColor(
        cp[:, ::-1].reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab_mean = lab_px.mean(axis=0)
    lab_std  = lab_px.std(axis=0)
    out["lab_mean"] = lab_mean.tolist()
    out["lab_std"]  = lab_std.tolist()
    out["b_cv"]     = float(lab_mean[2])

    # b* 논문 스케일 변환 (OpenCV b∈[0,255], 중심128 → b*∈[-127,127])
    out["b_star"] = float((lab_mean[2] - 128.0) / 128.0 * 127.0)

    # HSV-S (0~100 스케일)
    hsv_px = cv2.cvtColor(
        cp[:, ::-1].reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    out["hsv_s_mean"] = float(hsv_px[:, 1].mean() / 255.0 * 100.0)

    return out


class BasePreprocessor(abc.ABC):
    @abc.abstractmethod
    def preprocess(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        ...

    @abc.abstractmethod
    def get_feature_vector(self, metadata: Dict) -> np.ndarray:
        ...

    def get_face_crop(self, image_path: str,
                      pad: float = 0.15) -> Tuple[Optional[np.ndarray], Dict]:
        """
        얼굴 영역만 크롭하여 BGR ndarray 반환.
        Face Oval 랜드마크 기준 bbox + pad 여백 적용.
        얼굴 미감지 시 원본 반환.

        Parameters
        ----------
        image_path : 원본 이미지 경로
        pad        : bbox 대비 여백 비율 (0.15 = 15%)

        Returns
        -------
        (cropped_bgr, meta)   — meta["face_detected"] 로 성공 여부 확인
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Load failed: {image_path}")

        meta: Dict = {"face_detected": False, "image_path": image_path}

        if not MEDIAPIPE_AVAILABLE:
            return bgr, meta

        # 임시 FaceMesh (get_face_crop 전용 — 인스턴스 재사용 금지)
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            with mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.4) as fm:
                res = fm.process(rgb)
        except Exception as e:
            logging.warning(f"FaceMesh 실패: {e}")
            return bgr, meta

        if not res.multi_face_landmarks:
            return bgr, meta

        lm = res.multi_face_landmarks[0].landmark
        xs = [lm[i].x for i in FACE_OVAL_LANDMARKS]
        ys = [lm[i].y for i in FACE_OVAL_LANDMARKS]

        # ★ 패딩을 얼굴 크기 기준으로 계산
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)
        pad_x  = face_w * pad
        pad_y  = face_h * pad
        x1 = max(0, int((min(xs) - pad_x) * w))
        y1 = max(0, int((min(ys) - pad_y) * h))
        x2 = min(w, int((max(xs) + pad_x) * w))
        y2 = min(h, int((max(ys) + pad_y) * h))

        if x2 <= x1 or y2 <= y1:
            return bgr, meta

        cropped = bgr[y1:y2, x1:x2]
        meta["face_detected"] = True
        meta["crop_box"] = [x1, y1, x2, y2]
        return cropped, meta


class PersonalColorPreprocessor(BasePreprocessor):
    """표준 볼 영역 전처리 + 논문 b* 보정 (GPU 가속 지원)"""

    def __init__(self, apply_correction: bool = True, device: str = "cpu"):
        self.apply_correction = apply_correction
        self.device     = device
        self._corrector = GPULabCorrector(device)
        # ★ 인스턴스별로 독립 FaceMesh 생성
        #   클래스 변수 공유 금지 — QThread에서 graph가 None이 되는 문제 방지
        self._face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

    @staticmethod
    def _region_pixels(indices, rgb, lm, h, w):
        pts = [(int(np.clip(lm[i].x * w, 0, w - 1)),
                int(np.clip(lm[i].y * h, 0, h - 1))) for i in indices]
        if not pts:
            return np.empty((0, 3), dtype=np.uint8)
        arr  = np.array(pts, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(arr), 255)
        return rgb[mask == 255]

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Load failed: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        meta: Dict = {
            "face_detected":  False,
            "lab_mean":       [0., 128., 128.],
            "lab_std":        [0., 0., 0.],
            "cheek_rgb":      [200, 150, 150],
            "hsv_s_mean":     0.0,
            "b_star":         0.0,
            "b_cv":           128.0,
            "b_delta":        0.0,   # 보정량 (양수=워밍, 음수=쿨링)
            "corrected":      False,
            "image_path":     image_path,
            "mode":           "standard",
        }

        if not MEDIAPIPE_AVAILABLE or self._face_mesh is None:
            logging.warning(f"[Standard] MediaPipe 불가 → 크롭 불가: {image_path}")
            meta["face_crop_bgr"] = bgr.copy()          # ★ 원본이라도 키 보장
            return cv2.resize(bgr, (224, 224)), meta

        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            logging.warning(
                f"[Standard] 얼굴 미감지 → 원본 저장: {image_path} "
                f"(size={w}x{h})")
            meta["face_crop_bgr"] = bgr.copy()          # ★ 원본이라도 키 보장
            return cv2.resize(bgr, (224, 224)), meta

        lm = res.multi_face_landmarks[0].landmark
        meta["face_detected"] = True

        # ── 볼 영역 색상 추출 ──────────────────────────────────
        stats = get_cheek_lab_stats(bgr, lm, h, w)
        meta.update(stats)

        # ── b* 보정 (GPU or CPU) ───────────────────────────────
        work_bgr = bgr
        if self.apply_correction and stats["b_cv"] > 0:
            delta = _B_TARGET_CV - stats["b_cv"]
            meta["b_delta"] = round(delta, 2)
            if abs(delta) >= 0.5:
                work_bgr = self._corrector.correct(bgr, stats["b_cv"])
                meta["corrected"] = True
                logging.debug(
                    f"[보정/{self._corrector.device}] {image_path} "
                    f"b_cv={stats['b_cv']:.1f} delta={delta:+.1f}")

        # ── 얼굴 크롭 영역 계산 (Face Oval 기준 + 얼굴 크기 대비 15% 패딩) ─
        _pad_ratio = 0.15
        xs = [lm[i].x for i in FACE_OVAL_LANDMARKS]
        ys = [lm[i].y for i in FACE_OVAL_LANDMARKS]
        # ★ 패딩을 *얼굴 크기* 기준으로 계산 (기존: 이미지 크기 기준 → 크롭 무효)
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)
        pad_x  = face_w * _pad_ratio
        pad_y  = face_h * _pad_ratio
        cx1 = max(0, int((min(xs) - pad_x) * w))
        cy1 = max(0, int((min(ys) - pad_y) * h))
        cx2 = min(w, int((max(xs) + pad_x) * w))
        cy2 = min(h, int((max(ys) + pad_y) * h))
        if cx2 > cx1 and cy2 > cy1:
            meta["face_crop_bgr"] = work_bgr[cy1:cy2, cx1:cx2].copy()
            meta["crop_box"] = [cx1, cy1, cx2, cy2]
            crop_shape = meta["face_crop_bgr"].shape
            logging.info(
                f"[Standard] 크롭 성공: {image_path} "
                f"원본={w}x{h} → 크롭={crop_shape[1]}x{crop_shape[0]} "
                f"bbox=[{cx1},{cy1},{cx2},{cy2}]")
            # Standard 모드도 skin_pixel_ratio 근사 제공
            # 볼 영역 픽셀수 / 크롭 면적
            face_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            rgb_work  = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
            lp = self._region_pixels(LEFT_CHEEK_LANDMARKS,  rgb_work, lm, h, w)
            rp = self._region_pixels(RIGHT_CHEEK_LANDMARKS, rgb_work, lm, h, w)
            meta["skin_pixel_ratio"] = min(1.0, (lp.shape[0] + rp.shape[0]) / face_area)
        else:
            logging.warning(
                f"[Standard] 크롭 bbox 무효 → 원본 사용: {image_path} "
                f"bbox=[{cx1},{cy1},{cx2},{cy2}]")
            meta["face_crop_bgr"]    = work_bgr.copy()
            meta["skin_pixel_ratio"] = 0.0

        # ── 오버레이 (볼 영역 시각화) ──────────────────────────
        out = work_bgr.copy()
        ov  = out.copy()
        for idx_list in (LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS):
            pts = np.array(
                [(int(np.clip(lm[i].x * w, 0, w - 1)),
                  int(np.clip(lm[i].y * h, 0, h - 1)))
                 for i in idx_list], dtype=np.int32)
            cv2.fillConvexPoly(ov, cv2.convexHull(pts), (0, 220, 100))
        out = cv2.addWeighted(ov, 0.35, out, 0.65, 0)

        return cv2.resize(out, (224, 224)), meta

    def get_feature_vector(self, metadata: Dict) -> np.ndarray:
        m = np.array(metadata.get("lab_mean", [0, 128, 128]), dtype=np.float32)
        s = np.array(metadata.get("lab_std",  [0, 0, 0]),     dtype=np.float32)
        return np.concatenate([
            m / np.array([100., 255., 255.]),
            s / np.array([50., 128., 128.])
        ])

    def __del__(self):
        if self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass


class AdvancedSkinPreprocessor(BasePreprocessor):
    """정밀 피부 마스크 전처리 + 논문 b* 보정 (GPU 가속 지원)"""
    _YCR_MIN = np.array([0,   133,  77], dtype=np.uint8)
    _YCR_MAX = np.array([255, 173, 127], dtype=np.uint8)
    _Y_DARK  = 40
    _DILATE  = 13

    def __init__(self, apply_correction: bool = True, device: str = "cpu"):
        self.apply_correction = apply_correction
        self.device     = device
        self._corrector = GPULabCorrector(device)
        # ★ 인스턴스별로 독립 FaceMesh 생성 (QThread 공유 금지)
        self._face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

    @staticmethod
    def _hull_mask(indices, lm, h, w):
        pts = np.array(
            [(int(np.clip(lm[i].x * w, 0, w - 1)),
              int(np.clip(lm[i].y * h, 0, h - 1)))
             for i in indices], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(pts) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        return mask

    def _skin_mask(self, rgb, lm, h, w):
        face = self._hull_mask(FACE_OVAL_LANDMARKS, lm, h, w)
        excl = np.zeros((h, w), dtype=np.uint8)
        for g in (LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS,
                  LEFT_EYEBROW_LANDMARKS, RIGHT_EYEBROW_LANDMARKS,
                  LIPS_LANDMARKS):
            excl = cv2.bitwise_or(excl, self._hull_mask(g, lm, h, w))
        k_ex = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._DILATE, self._DILATE))
        excl = cv2.dilate(excl, k_ex, iterations=1)
        face = cv2.bitwise_and(face, cv2.bitwise_not(excl))
        ycc  = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        sc   = cv2.inRange(ycc, self._YCR_MIN, self._YCR_MAX)
        dk   = (ycc[:, :, 0] < self._Y_DARK).astype(np.uint8) * 255
        sc   = cv2.bitwise_and(sc, cv2.bitwise_not(dk))
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ha   = cv2.inRange(hsv, np.array([0,   20, 50]),
                                np.array([25,  170, 255]))
        hb   = cv2.inRange(hsv, np.array([160, 20, 50]),
                                np.array([180, 170, 255]))
        sh   = cv2.bitwise_or(ha, hb)
        comb = cv2.bitwise_and(face, cv2.bitwise_and(sc, sh))
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k, iterations=2)
        comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN,  k, iterations=1)
        return comb

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Load failed: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        meta: Dict = {
            "face_detected":    False,
            "lab_mean":         [0., 128., 128.],
            "lab_std":          [0., 0., 0.],
            "cheek_rgb":        [200, 150, 150],
            "skin_pixel_ratio": 0.0,
            "skin_mask":        None,
            "hsv_s_mean":       0.0,
            "b_star":           0.0,
            "b_cv":             128.0,
            "b_delta":          0.0,
            "corrected":        False,
            "image_path":       image_path,
            "mode":             "advanced",
        }

        if not MEDIAPIPE_AVAILABLE or self._face_mesh is None:
            logging.warning(f"[Advanced] MediaPipe 불가 → 크롭 불가: {image_path}")
            meta["face_crop_bgr"] = bgr.copy()          # ★ 원본이라도 키 보장
            return cv2.resize(bgr, (224, 224)), meta

        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            logging.warning(
                f"[Advanced] 얼굴 미감지 → 원본 저장: {image_path} "
                f"(size={w}x{h})")
            meta["face_crop_bgr"] = bgr.copy()          # ★ 원본이라도 키 보장
            return cv2.resize(bgr, (224, 224)), meta

        lm = res.multi_face_landmarks[0].landmark
        meta["face_detected"] = True

        # ── 볼 영역 b* 먼저 측정 (보정 기준값 산출) ───────────
        stats = get_cheek_lab_stats(bgr, lm, h, w)
        meta.update({k: v for k, v in stats.items()
                     if k != "skin_mask"})

        # ── b* 보정 (GPU or CPU) ───────────────────────────────
        work_bgr = bgr
        if self.apply_correction and stats["b_cv"] > 0:
            delta = _B_TARGET_CV - stats["b_cv"]
            meta["b_delta"] = round(delta, 2)
            if abs(delta) >= 0.5:
                work_bgr = self._corrector.correct(bgr, stats["b_cv"])
                meta["corrected"] = True
                logging.debug(
                    f"[보정/{self._corrector.device}] {image_path} "
                    f"b_cv={stats['b_cv']:.1f} delta={delta:+.1f}")

        # ── 보정된 이미지로 피부 마스크 재계산 ────────────────
        rgb_corr = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
        sm = self._skin_mask(rgb_corr, lm, h, w)
        meta["skin_mask"] = sm
        fa = int(self._hull_mask(FACE_OVAL_LANDMARKS, lm, h, w).sum() / 255)
        meta["skin_pixel_ratio"] = int(sm.sum() / 255) / max(fa, 1)

        sp = rgb_corr[sm == 255]
        if sp.size:
            mr  = sp.mean(axis=0).astype(np.uint8)
            meta["cheek_rgb"] = mr.tolist()
            lab = cv2.cvtColor(
                mr.reshape(1, 1, 3).astype(np.uint8),
                cv2.COLOR_RGB2LAB)
            meta["lab_mean"] = lab[0, 0].tolist()
            la = cv2.cvtColor(
                sp[:, ::-1].reshape(-1, 1, 3).astype(np.uint8),
                cv2.COLOR_BGR2LAB).reshape(-1, 3)
            meta["lab_std"] = la.std(axis=0).tolist()

        # ── 얼굴 크롭 영역 계산 (Face Oval 기준 + 얼굴 크기 대비 15% 패딩) ─
        _pad_ratio = 0.15
        xs = [lm[i].x for i in FACE_OVAL_LANDMARKS]
        ys = [lm[i].y for i in FACE_OVAL_LANDMARKS]
        # ★ 패딩을 *얼굴 크기* 기준으로 계산 (기존: 이미지 크기 기준 → 크롭 무효)
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)
        pad_x  = face_w * _pad_ratio
        pad_y  = face_h * _pad_ratio
        cx1 = max(0, int((min(xs) - pad_x) * w))
        cy1 = max(0, int((min(ys) - pad_y) * h))
        cx2 = min(w, int((max(xs) + pad_x) * w))
        cy2 = min(h, int((max(ys) + pad_y) * h))
        if cx2 > cx1 and cy2 > cy1:
            # ★ 피부 마스크를 크롭 영역에 적용 — 비피부 픽셀 제거
            crop_bgr  = work_bgr[cy1:cy2, cx1:cx2].copy()
            crop_mask = sm[cy1:cy2, cx1:cx2]
            crop_bgr[crop_mask == 0] = 0          # 비피부 → 검정
            meta["face_crop_bgr"] = crop_bgr
            meta["crop_box"] = [cx1, cy1, cx2, cy2]
            crop_shape = crop_bgr.shape
            skin_px  = int(crop_mask.sum() / 255)
            total_px = crop_shape[0] * crop_shape[1]
            logging.info(
                f"[Advanced] 크롭+마스크 성공: {image_path} "
                f"원본={w}x{h} → 크롭={crop_shape[1]}x{crop_shape[0]} "
                f"피부픽셀={skin_px}/{total_px} "
                f"({skin_px/max(total_px,1)*100:.1f}%) "
                f"bbox=[{cx1},{cy1},{cx2},{cy2}]")
        else:
            logging.warning(
                f"[Advanced] 크롭 bbox 무효 → 원본+마스크 사용: {image_path} "
                f"bbox=[{cx1},{cy1},{cx2},{cy2}]")
            full_masked = work_bgr.copy()
            full_masked[sm == 0] = 0
            meta["face_crop_bgr"] = full_masked

        # ── 오버레이 ───────────────────────────────────────────
        out = work_bgr.copy()
        ov  = out.copy()
        ov[sm == 255] = (
            ov[sm == 255].astype(np.float32) * 0.4
            + np.array([100, 80, 230], dtype=np.float32) * 0.6
        ).clip(0, 255).astype(np.uint8)
        out = cv2.addWeighted(ov, 0.55, out, 0.45, 0)
        ctrs, _ = cv2.findContours(
            sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ctrs, -1, (0, 200, 50), 1)

        return cv2.resize(out, (224, 224)), meta

    def get_feature_vector(self, metadata: Dict) -> np.ndarray:
        m  = np.array(metadata.get("lab_mean", [0, 128, 128]), dtype=np.float32)
        s  = np.array(metadata.get("lab_std",  [0, 0, 0]),     dtype=np.float32)
        sr = np.array([metadata.get("skin_pixel_ratio", 0.)],   dtype=np.float32)
        return np.concatenate([
            m  / np.array([100., 255., 255.]),
            s  / np.array([50., 128., 128.]),
            sr
        ])

    def __del__(self):
        if self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass