"""
ai/infer.py
───────────
계층적 퍼스널 컬러 추론.
Stage 1: 4-class 계절 분류 → Stage 2: 3-class 톤 분류 (계절 조건부).
최종 신뢰도 = season_conf × tone_conf.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .model_loader import ModelLoader, TONE_LABELS, SEASONS
from .preprocessing import FacePreprocessor, FaceDetectionResult

logger = logging.getLogger(__name__)

# ── 퍼스널 컬러 코드 → 한국어/영어 레이블 ─────────────────────────────────
_LABEL_MAP: Dict[str, Dict[str, str]] = {
    "spring_warm_light":   {"ko": "봄 웜 라이트",    "en": "Spring Warm Light"},
    "spring_warm_bright":  {"ko": "봄 웜 브라이트",  "en": "Spring Warm Bright"},
    "spring_warm_deep":    {"ko": "봄 웜 딥",         "en": "Spring Warm Deep"},
    "summer_cool_light":   {"ko": "여름 쿨 라이트",   "en": "Summer Cool Light"},
    "summer_cool_muted":   {"ko": "여름 쿨 뮤트",     "en": "Summer Cool Muted"},
    "summer_cool_bright":  {"ko": "여름 쿨 브라이트", "en": "Summer Cool Bright"},
    "autumn_warm_deep":    {"ko": "가을 웜 딥",       "en": "Autumn Warm Deep"},
    "autumn_warm_muted":   {"ko": "가을 웜 뮤트",     "en": "Autumn Warm Muted"},
    "autumn_warm_light":   {"ko": "가을 웜 라이트",   "en": "Autumn Warm Light"},
    "winter_cool_deep":    {"ko": "겨울 쿨 딥",       "en": "Winter Cool Deep"},
    "winter_cool_bright":  {"ko": "겨울 쿨 브라이트", "en": "Winter Cool Bright"},
    "winter_cool_muted":   {"ko": "겨울 쿨 뮤트",     "en": "Winter Cool Muted"},
}

# 퍼스널 컬러별 대표 색상 팔레트
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
    계층적 2-stage 추론.

    Stage 1 — 4계절 분류:
        season_probs = softmax(stage1_logits)   → (4,)
        season_idx   = argmax(season_probs)

    Stage 2 — 예측된 계절의 3톤 분류:
        tone_logits  = stage2_logits[season_idx]  → (3,)
        tone_probs   = softmax(tone_logits)
        tone_idx     = argmax(tone_probs)

    최종 신뢰도 = season_conf × tone_conf

    Returns
    -------
    dict:
        success, personal_color, label_ko, label_en,
        season, season_confidence, tone_confidence, confidence,
        top3, recommended_colors, raw_scores,
        face_detected, inference_ms, message
    """
    t0 = time.perf_counter()

    # ── 전처리 ─────────────────────────────────────────────────────────
    preprocessor = FacePreprocessor()
    det: FaceDetectionResult = preprocessor.process(bgr_frame)

    if not det.success:
        return _error_result(det.message)

    # ── 모델 로드 확인 ──────────────────────────────────────────────────
    loader = ModelLoader()
    if not loader.is_loaded:
        loader.load()

    model = loader.model

    # ── 계층적 추론 ──────────────────────────────────────────────────────
    with torch.no_grad():
        s1_logits, s2_logits = model(det.tensor, det.meta_tensor)
        # s1_logits : (1, 4)
        # s2_logits : (1, 4, 3)

        season_probs = F.softmax(s1_logits, dim=1)[0]   # (4,)
        season_idx   = int(torch.argmax(season_probs))
        season_conf  = float(season_probs[season_idx])

        # 예측된 계절의 톤 헤드 결과
        tone_logits = s2_logits[0, season_idx, :]        # (3,)
        tone_probs  = F.softmax(tone_logits, dim=0)
        tone_idx    = int(torch.argmax(tone_probs))
        tone_conf   = float(tone_probs[tone_idx])

    personal_color = TONE_LABELS[season_idx][tone_idx]
    confidence     = season_conf * tone_conf             # 최종 결합 신뢰도

    # ── TOP 3 — 모든 12 조합의 결합 확률 정렬 ──────────────────────────
    all_probs: List[tuple] = []
    with torch.no_grad():
        for s_idx in range(4):
            s_prob  = float(season_probs[s_idx])
            t_probs = F.softmax(s2_logits[0, s_idx, :], dim=0)
            for t_idx in range(3):
                color  = TONE_LABELS[s_idx][t_idx]
                prob   = s_prob * float(t_probs[t_idx])
                all_probs.append((color, prob))

    all_probs.sort(key=lambda x: x[1], reverse=True)

    top3 = [
        {
            "color":      c,
            "label_ko":   _LABEL_MAP[c]["ko"],
            "confidence": round(p, 6),
        }
        for c, p in all_probs[:3]
    ]

    # 전체 raw scores
    raw_scores = json.dumps({c: round(p, 6) for c, p in all_probs})

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[Hierarchical] {SEASONS[season_idx]} ({season_conf:.1%}) → "
        f"{personal_color} ({tone_conf:.1%}) | total {confidence:.1%} | "
        f"{elapsed_ms:.1f} ms"
    )

    return {
        "success":            True,
        "personal_color":     personal_color,
        "label_ko":           _LABEL_MAP[personal_color]["ko"],
        "label_en":           _LABEL_MAP[personal_color]["en"],
        "season":             SEASONS[season_idx],
        "season_confidence":  round(season_conf, 6),
        "tone_confidence":    round(tone_conf, 6),
        "confidence":         round(confidence, 6),
        "top3":               top3,
        "recommended_colors": _PALETTE_CACHE.get(personal_color, []),
        "raw_scores":         raw_scores,
        "face_detected":      True,
        "inference_ms":       round(elapsed_ms, 2),
        "message":            "OK",
    }


def _error_result(message: str) -> Dict[str, Any]:
    return {
        "success":            False,
        "personal_color":     None,
        "label_ko":           None,
        "label_en":           None,
        "season":             None,
        "season_confidence":  0.0,
        "tone_confidence":    0.0,
        "confidence":         0.0,
        "top3":               [],
        "recommended_colors": [],
        "raw_scores":         "{}",
        "face_detected":      False,
        "inference_ms":       0.0,
        "message":            message,
    }
