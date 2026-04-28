"""
ai/infer.py
───────────
계층적 2-stage 추론 (OOTD/workers.py _infer_hierarchical 과 동일한 흐름).

Stage 1: EfficientNet-B2 + 16-dim feat → 4계절 분류
Stage 2: 예측 계절 전용 EfficientNet-B2 + 16-dim feat → 3톤 분류
최종 신뢰도 = season_conf × tone_conf
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .model_loader import (
    ModelLoader, SEASON_CLASSES, TONE_CLASSES,
    UNDERTONE_MAP, SEASON_KO, TONE_KO,
    _LABEL_MAP, _PALETTE,
)
from .preprocessing import FacePreprocessor, FaceDetectionResult

logger = logging.getLogger(__name__)


def run_inference(bgr_frame) -> Dict[str, Any]:
    """
    BGR 이미지 → 퍼스널 컬러 추론 결과 dict.

    Returns
    -------
    dict:
        success, personal_color, label_ko, label_en,
        season, tone, season_confidence, tone_confidence, confidence,
        top3, recommended_colors, raw_scores,
        face_detected, inference_ms, message
    """
    t0 = time.perf_counter()

    # ── 전처리 ─────────────────────────────────────────────────────────────
    preprocessor = FacePreprocessor()
    det: FaceDetectionResult = preprocessor.process(bgr_frame)

    if not det.success:
        return _error_result(det.message)

    img_t  = det.tensor       # (1,3,260,260)
    feat_t = det.meta_tensor  # (1,16)

    # ── 모델 준비 ───────────────────────────────────────────────────────────
    loader = ModelLoader()
    if not loader.is_loaded:
        loader.load()

    # ── Stage 1: 계절 분류 ────────────────────────────────────────────────
    with torch.no_grad():
        s1_logits   = loader.stage1(img_t, feat_t)      # (1,4)
        s1_probs    = F.softmax(s1_logits, dim=1)[0]    # (4,)
        season_idx  = int(torch.argmax(s1_probs))
        pred_season = SEASON_CLASSES[season_idx]
        season_conf = float(s1_probs[season_idx])

    # ── Stage 2: 예측 계절 전용 톤 분류 ─────────────────────────────────
    with torch.no_grad():
        s2_logits = loader.stage2(pred_season)(img_t, feat_t)  # (1,3)
        s2_probs  = F.softmax(s2_logits, dim=1)[0]             # (3,)
        tone_idx  = int(torch.argmax(s2_probs))
        pred_tone = TONE_CLASSES[tone_idx]
        tone_conf = float(s2_probs[tone_idx])

    personal_color = f"{pred_season}_{pred_tone}"
    confidence     = season_conf * tone_conf

    # ── TOP 3: 12클래스 전체 결합 확률 ──────────────────────────────────
    all_probs: List[tuple] = []
    with torch.no_grad():
        for si, season in enumerate(SEASON_CLASSES):
            s_prob   = float(s1_probs[si])
            t_out    = loader.stage2(season)(img_t, feat_t)
            t_probs  = F.softmax(t_out, dim=1)[0]
            for ti, tone in enumerate(TONE_CLASSES):
                color = f"{season}_{tone}"
                prob  = s_prob * float(t_probs[ti])
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

    raw_scores = json.dumps({c: round(p, 6) for c, p in all_probs})

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[Hierarchical] {pred_season}({season_conf:.1%}) → "
        f"{pred_tone}({tone_conf:.1%}) | "
        f"{personal_color} conf={confidence:.1%} | {elapsed_ms:.1f}ms"
    )
    if det.quality:
        q = det.quality
        logger.info(
            "[PreprocessQuality] "
            f"skin={q.get('skin_ratio', 0):.1%} "
            f"b_cv={q.get('b_cv', 0):.1f} "
            f"b_delta={q.get('b_delta', 0):+.1f} "
            f"L={q.get('l_mean', 0):.1f} "
            f"S={q.get('s_mean', 0):.1f} "
            f"highlight={q.get('highlight_ratio', 0):.1%} "
            f"shadow={q.get('shadow_ratio', 0):.1%}"
        )

    undertone  = UNDERTONE_MAP[pred_season]
    season_ko  = SEASON_KO[pred_season]
    tone_ko    = TONE_KO[pred_tone]

    return {
        "success":            True,
        "personal_color":     personal_color,
        "label_ko":           _LABEL_MAP[personal_color]["ko"],
        "label_en":           _LABEL_MAP[personal_color]["en"],
        # ── 3단계 진단 정보 ──────────────────────────────────
        "season":             pred_season,      # "Spring"
        "season_ko":          season_ko,        # "봄"
        "undertone":          undertone,        # "웜톤" / "쿨톤"
        "tone":               pred_tone,        # "Warm"
        "tone_ko":            tone_ko,          # "웜"
        # ────────────────────────────────────────────────────
        "season_confidence":  round(season_conf, 6),
        "tone_confidence":    round(tone_conf, 6),
        "confidence":         round(confidence, 6),
        "top3":               top3,
        "recommended_colors": _PALETTE.get(personal_color, []),
        "raw_scores":         raw_scores,
        "face_detected":      True,
        "preprocess_quality": det.quality or {},
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
        "tone":               None,
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
