import abc
import logging
from typing import Dict, List, Tuple

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


class BasePreprocessor(abc.ABC):
    @abc.abstractmethod
    def preprocess(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        ...

    @abc.abstractmethod
    def get_feature_vector(self, metadata: Dict) -> np.ndarray:
        ...


class PersonalColorPreprocessor(BasePreprocessor):
    def __init__(self):
        self._face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5)

    @staticmethod
    def _region_pixels(indices, rgb, lm, h, w):
        pts = [(int(np.clip(lm[i].x*w,0,w-1)), int(np.clip(lm[i].y*h,0,h-1))) for i in indices]
        if not pts:
            return np.empty((0,3), dtype=np.uint8)
        arr  = np.array(pts, dtype=np.int32)
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(arr), 255)
        return rgb[mask==255]

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Load failed: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        meta: Dict = {
            "face_detected": False, "lab_mean": [0.,128.,128.],
            "lab_std": [0.,0.,0.], "cheek_rgb": [200,150,150],
            "image_path": image_path, "mode": "standard",
        }
        if not MEDIAPIPE_AVAILABLE or self._face_mesh is None:
            return cv2.resize(bgr,(224,224)), meta
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return cv2.resize(bgr,(224,224)), meta
        lm = res.multi_face_landmarks[0].landmark
        meta["face_detected"] = True
        lp = self._region_pixels(LEFT_CHEEK_LANDMARKS,  rgb, lm, h, w)
        rp = self._region_pixels(RIGHT_CHEEK_LANDMARKS, rgb, lm, h, w)
        cp = np.vstack([lp,rp]) if lp.size and rp.size else (lp if lp.size else rp)
        if cp.size:
            mr = cp.mean(axis=0).astype(np.uint8)
            meta["cheek_rgb"] = mr.tolist()
            lab = cv2.cvtColor(mr.reshape(1,1,3).astype(np.uint8), cv2.COLOR_RGB2LAB)
            meta["lab_mean"] = lab[0,0].tolist()
            la = cv2.cvtColor(cp[:,::-1].reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1,3)
            meta["lab_std"] = la.std(axis=0).tolist()
        out = bgr.copy(); ov = out.copy()
        for idx_list in (LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS):
            pts = np.array([(int(np.clip(lm[i].x*w,0,w-1)),int(np.clip(lm[i].y*h,0,h-1))) for i in idx_list], dtype=np.int32)
            cv2.fillConvexPoly(ov, cv2.convexHull(pts), (0,220,100))
        out = cv2.addWeighted(ov, 0.35, out, 0.65, 0)
        return cv2.resize(out,(224,224)), meta

    def get_feature_vector(self, metadata: Dict) -> np.ndarray:
        m = np.array(metadata.get("lab_mean",[0,128,128]), dtype=np.float32)
        s = np.array(metadata.get("lab_std", [0,0,0]),    dtype=np.float32)
        return np.concatenate([m/np.array([100.,255.,255.]), s/np.array([50.,128.,128.])])

    def __del__(self):
        if self._face_mesh:
            try: self._face_mesh.close()
            except Exception: pass


class AdvancedSkinPreprocessor(BasePreprocessor):
    _YCR_MIN = np.array([0,  133, 77],  dtype=np.uint8)
    _YCR_MAX = np.array([255,173, 127], dtype=np.uint8)
    _Y_DARK  = 40
    _DILATE  = 13

    def __init__(self):
        self._face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5)

    @staticmethod
    def _hull_mask(indices, lm, h, w):
        pts = np.array([(int(np.clip(lm[i].x*w,0,w-1)),int(np.clip(lm[i].y*h,0,h-1))) for i in indices], dtype=np.int32)
        mask = np.zeros((h,w), dtype=np.uint8)
        if len(pts)>=3:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        return mask

    def _skin_mask(self, rgb, lm, h, w):
        face = self._hull_mask(FACE_OVAL_LANDMARKS, lm, h, w)
        excl = np.zeros((h,w), dtype=np.uint8)
        for g in (LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS,
                  LEFT_EYEBROW_LANDMARKS, RIGHT_EYEBROW_LANDMARKS, LIPS_LANDMARKS):
            excl = cv2.bitwise_or(excl, self._hull_mask(g, lm, h, w))
        k_ex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._DILATE, self._DILATE))
        excl = cv2.dilate(excl, k_ex, iterations=1)
        face = cv2.bitwise_and(face, cv2.bitwise_not(excl))
        ycc  = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        sc   = cv2.inRange(ycc, self._YCR_MIN, self._YCR_MAX)
        dk   = (ycc[:,:,0] < self._Y_DARK).astype(np.uint8)*255
        sc   = cv2.bitwise_and(sc, cv2.bitwise_not(dk))
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ha   = cv2.inRange(hsv, np.array([0,20,50]),   np.array([25,170,255]))
        hb   = cv2.inRange(hsv, np.array([160,20,50]), np.array([180,170,255]))
        sh   = cv2.bitwise_or(ha, hb)
        comb = cv2.bitwise_and(face, cv2.bitwise_and(sc, sh))
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
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
            "face_detected": False, "lab_mean": [0.,128.,128.],
            "lab_std": [0.,0.,0.], "cheek_rgb": [200,150,150],
            "skin_pixel_ratio": 0.0, "skin_mask": None,
            "image_path": image_path, "mode": "advanced",
        }
        if not MEDIAPIPE_AVAILABLE or self._face_mesh is None:
            return cv2.resize(bgr,(224,224)), meta
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return cv2.resize(bgr,(224,224)), meta
        lm = res.multi_face_landmarks[0].landmark
        meta["face_detected"] = True
        sm = self._skin_mask(rgb, lm, h, w)
        meta["skin_mask"] = sm
        fa = int(self._hull_mask(FACE_OVAL_LANDMARKS, lm, h, w).sum()/255)
        meta["skin_pixel_ratio"] = int(sm.sum()/255) / max(fa, 1)
        sp = rgb[sm==255]
        if sp.size:
            mr = sp.mean(axis=0).astype(np.uint8)
            meta["cheek_rgb"] = mr.tolist()
            lab = cv2.cvtColor(mr.reshape(1,1,3).astype(np.uint8), cv2.COLOR_RGB2LAB)
            meta["lab_mean"] = lab[0,0].tolist()
            la = cv2.cvtColor(sp[:,::-1].reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1,3)
            meta["lab_std"] = la.std(axis=0).tolist()
        out = bgr.copy(); ov = out.copy()
        ov[sm==255] = (ov[sm==255].astype(np.float32)*0.4 + np.array([100,80,230],dtype=np.float32)*0.6).clip(0,255).astype(np.uint8)
        out = cv2.addWeighted(ov, 0.55, out, 0.45, 0)
        ctrs, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ctrs, -1, (0,200,50), 1)
        return cv2.resize(out,(224,224)), meta

    def get_feature_vector(self, metadata: Dict) -> np.ndarray:
        m  = np.array(metadata.get("lab_mean",[0,128,128]), dtype=np.float32)
        s  = np.array(metadata.get("lab_std", [0,0,0]),     dtype=np.float32)
        sr = np.array([metadata.get("skin_pixel_ratio",0.)], dtype=np.float32)
        return np.concatenate([m/np.array([100.,255.,255.]), s/np.array([50.,128.,128.]), sr])

    def __del__(self):
        if self._face_mesh:
            try: self._face_mesh.close()
            except Exception: pass
