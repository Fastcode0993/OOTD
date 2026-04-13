"""
ai/infer.py
───────────
PyTorch 모델로 퍼스널 컬러를 추론하고 결과를 딕셔너리로 반환.
라즈베리파이5 CPU에서 torch.no_grad() + 단일 스레드로 최적화.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .model_loader import ModelLoader, PERSONAL_COLOR_CLASSES
from .preprocessing import FacePreprocessor, FaceDetectionResult

logger = logging.getLogger(__name__)

# ── 컬러 코드 → 한국어/영어 레이블 매핑 ─────────────────────────────────
_LABEL_MAP: Dict[str, Dict[str, str]] = {
    "spring_warm_light":   {"ko": "봄 웜 라이트",   "en": "Spring Warm Light"},
    "spring_warm_bright":  {"ko": "봄 웜 브라이트",  "en": "Spring Warm Bright"},
    "spring_warm_deep":    {"ko": "봄 웜 딥",        "en": "Spring Warm Deep"},
    "summer_cool_light":   {"ko": "여름 쿨 라이트",  "en": "Summer Cool Light"},
    "summer_cool_muted":   {"ko": "여름 쿨 뮤트",    "en": "Summer Cool Muted"},
    "summer_cool_bright":  {"ko": "여름 쿨 브라이트","en": "Summer Cool Bright"},
    "autumn_warm_deep":    {"ko": "가을 웜 딥",      "en": "Autumn Warm Deep"},
    "autumn_warm_muted":   {"ko": "가을 웜 뮤트",    "en": "Autumn Warm Muted"},
    "autumn_warm_light":   {"ko": "가을 웜 라이트",  "en": "Autumn Warm Light"},
    "winter_cool_deep":    {"ko": "겨울 쿨 딥",      "en": "Winter Cool Deep"},
    "winter_cool_bright":  {"ko": "겨울 쿨 브라이트","en": "Winter Cool Bright"},
    "winter_cool_muted":   {"ko": "겨울 쿨 뮤트",    "en": "Winter Cool Muted"},
}

# 퍼스널 컬러별 대표 팔레트 (DB 없이도 빠른 응답 가능하도록 캐시)
_PALETTE_CACHE: Dict[str, List[str]] = {
    "spring_warm_light":   ["#FDDBB4", "#F9A87A", "#F47C5A", "#F9C784", "#E8855C"],
    "spring_warm_bright":  ["#FFAA44", "#FF7733", "#FFD700", "#FF9933", "#FFBB55"],
    "spring_warm_deep":    ["#C47A1E", "#A0602A", "#8B5E3C", "#B87333", "#A0522D"],
    "summer_cool_light":   ["#E8D5E8", "#C9A7C9", "#B8A0C8", "#D4B8D4", "#E0C8E0"],
    "summer_cool_muted":   ["#B0A8B9", "#9890A8", "#8888A0", "#A0A0B8", "#C0B8C8"],
    "summer_cool_bright":  ["#FF69B4", "#87CEEB", "#FF4499", "#66BBEE", "#FF66AA"],
    "autumn_warm_deep":    ["#8B4513", "#A0522D", "#6B4C2A", "#8B6914", "#7A4A2A"],
    "autumn_warm_muted":   ["#B5A642", "#9B8B5A", "#A09060", "#8B7D50", "#C4A870"],
    "autumn_warm_light":   ["#FFCBA4", "#FFB87A", "#F4A460", "#DEB887", "#D2A679"],
    "winter_cool_deep":    ["#1C1C3A", "#2D2D5A", "#8B0000", "#003366", "#2C2C2C"],
    "winter_cool_bright":  ["#0033CC", "#CC0066", "#0099CC", "#9900CC", "#003399"],
    "winter_cool_muted":   ["#708090", "#6A5ACD", "#7B7B7B", "#4A4A6A", "#8888AA"],
}


def run_inference(bgr_frame) -> Dict[str, Any]:
    """
    OpenCV BGR 프레임을 받아 퍼스널 컬러 진단 결과를 반환.

    Returns
    -------
    dict with keys:
        success          bool
        personal_color   str   (코드, e.g. "spring_warm_light")
        label_ko         str
        label_en         str
        confidence       float (0~1)
        top3             list of {color, confidence}
        recommended_colors list[str]   (hex 코드)
        raw_scores       str  (JSON)
        face_detected    bool
        inference_ms     float
        message          str
    """
    t0 = time.perf_counter()

    # ── 전처리 ─────────────────────────────────────────────────────────
    preprocessor = FacePreprocessor()
    det: FaceDetectionResult = preprocessor.process(bgr_frame)

    if not det.success:
        return _error_result(det.message)

    # ── 추론 ───────────────────────────────────────────────────────────
    loader = ModelLoader()
    if not loader.is_loaded:
        loader.load()

    model = loader.model

    with torch.no_grad():                        # 그래디언트 비활성화 → 메모리·속도 최적화
        logits = model(det.tensor)               # (1, NUM_CLASSES)
        probs  = F.softmax(logits, dim=1)[0]     # (NUM_CLASSES,)

    # ── 상위 3개 결과 추출 ─────────────────────────────────────────────
    top3_indices = torch.topk(probs, k=3).indices.tolist()
    top3_probs   = torch.topk(probs, k=3).values.tolist()

    best_idx        = top3_indices[0]
    personal_color  = PERSONAL_COLOR_CLASSES[best_idx]
    confidence      = float(top3_probs[0])

    top3 = [
        {
            "color":      PERSONAL_COLOR_CLASSES[idx],
            "label_ko":   _LABEL_MAP[PERSONAL_COLOR_CLASSES[idx]]["ko"],
            "confidence": float(p),
        }
        for idx, p in zip(top3_indices, top3_probs)
    ]

    raw_scores = json.dumps({
        cls: round(float(probs[i]), 6)
        for i, cls in enumerate(PERSONAL_COLOR_CLASSES)
    })

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Inference done: {personal_color} ({confidence:.1%}) in {elapsed_ms:.1f} ms"
    )

    return {
        "success":           True,
        "personal_color":    personal_color,
        "label_ko":          _LABEL_MAP[personal_color]["ko"],
        "label_en":          _LABEL_MAP[personal_color]["en"],
        "confidence":        confidence,
        "top3":              top3,
        "recommended_colors": _PALETTE_CACHE.get(personal_color, []),
        "raw_scores":        raw_scores,
        "face_detected":     True,
        "inference_ms":      round(elapsed_ms, 2),
        "message":           "OK",
    }


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────
def _error_result(message: str) -> Dict[str, Any]:
    return {
        "success":           False,
        "personal_color":    None,
        "label_ko":          None,
        "label_en":          None,
        "confidence":        0.0,
        "top3":              [],
        "recommended_colors": [],
        "raw_scores":        "{}",
        "face_detected":     False,
        "inference_ms":      0.0,
        "message":           message,
    }
