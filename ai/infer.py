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

# ── 품질 게이트 임계값 ─────────────────────────────────────────────────────
_QUALITY_SKIN_MIN      = 0.08   # 최소 피부 비율 (이하: 거부)
_QUALITY_SKIN_WARN     = 0.15   # 경고 피부 비율
_QUALITY_HIGHLIGHT_MAX = 0.45   # 최대 하이라이트 비율 (초과: 거부)
_QUALITY_HIGHLIGHT_WARN = 0.28  # 경고 하이라이트 비율
_QUALITY_SHADOW_MAX    = 0.45   # 최대 그림자 비율 (초과: 거부)
_QUALITY_SHADOW_WARN   = 0.28   # 경고 그림자 비율
_QUALITY_L_MIN         = 22.0   # 최소 밝기 L (이하: 거부)
_QUALITY_L_MAX         = 90.0   # 최대 밝기 L (초과: 거부)

_TEMPERATURE  = 1.2    # Softmax temperature scaling (>1 → 분포 완화)
_CONF_GAP_WARN = 0.08  # top1-top2 확률 차이가 이 이하이면 low_confidence 플래그


def _quality_gate(quality: dict) -> tuple:
    """
    전처리 품질 검사.
    Returns (reject: bool, warning: str | None)
    """
    sr  = quality.get("skin_ratio",      1.0)
    hi  = quality.get("highlight_ratio", 0.0)
    sh  = quality.get("shadow_ratio",    0.0)
    lm  = quality.get("l_mean",          50.0)

    if sr < _QUALITY_SKIN_MIN:
        return True, "얼굴을 인식할 수 없습니다. 정면을 바라보고 조명을 밝게 해주세요."
    if hi > _QUALITY_HIGHLIGHT_MAX:
        return True, "이미지가 과다 노출되었습니다. 직사광선을 피해 주세요."
    if sh > _QUALITY_SHADOW_MAX:
        return True, "이미지가 너무 어둡습니다. 밝은 환경에서 촬영해 주세요."
    if lm < _QUALITY_L_MIN:
        return True, "조명이 너무 어둡습니다. 밝은 곳에서 다시 촬영해 주세요."
    if lm > _QUALITY_L_MAX:
        return True, "조명이 너무 밝습니다. 조명을 줄이거나 위치를 조정해 주세요."

    # 경고 (처리는 계속)
    warns = []
    if sr < _QUALITY_SKIN_WARN:
        warns.append("피부 인식 면적이 좁습니다")
    if hi > _QUALITY_HIGHLIGHT_WARN:
        warns.append("밝은 조명 영향 감지")
    if sh > _QUALITY_SHADOW_WARN:
        warns.append("그림자 영향 감지")
    return False, ("  ·  ".join(warns) if warns else None)


def _scaled_softmax(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Temperature scaling 적용 softmax."""
    return torch.nn.functional.softmax(logits / T, dim=1)


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

    # ── 품질 게이트 ────────────────────────────────────────────────────────
    quality = det.quality or {}
    reject, quality_warning = _quality_gate(quality)
    if reject:
        return _error_result(quality_warning or "이미지 품질이 낮습니다. 다시 시도해주세요.")

    img_t  = det.tensor       # (1,3,260,260)
    feat_t = det.meta_tensor  # (1,16)

    # ── 모델 준비 ───────────────────────────────────────────────────────────
    loader = ModelLoader()
    if not loader.is_loaded:
        loader.load()

    # ── Stage 1: 계절 분류 (Temperature Scaling 적용) ────────────────────
    with torch.no_grad():
        s1_logits   = loader.stage1(img_t, feat_t)              # (1,4)
        s1_probs    = _scaled_softmax(s1_logits, _TEMPERATURE)[0]  # (4,)
        season_idx  = int(torch.argmax(s1_probs))
        pred_season = SEASON_CLASSES[season_idx]
        season_conf = float(s1_probs[season_idx])

    # ── Stage 2: 예측 계절 전용 톤 분류 (Temperature Scaling 적용) ──────
    with torch.no_grad():
        s2_logits = loader.stage2(pred_season)(img_t, feat_t)       # (1,3)
        s2_probs  = _scaled_softmax(s2_logits, _TEMPERATURE)[0]     # (3,)
        tone_idx  = int(torch.argmax(s2_probs))
        pred_tone = TONE_CLASSES[tone_idx]
        tone_conf = float(s2_probs[tone_idx])

    personal_color = f"{pred_season}_{pred_tone}"
    confidence     = season_conf * tone_conf

    # ── TOP 3: 12클래스 전체 결합 확률 (Temperature Scaling 적용) ───────
    all_probs: List[tuple] = []
    with torch.no_grad():
        for si, season in enumerate(SEASON_CLASSES):
            s_prob  = float(s1_probs[si])
            t_out   = loader.stage2(season)(img_t, feat_t)
            t_probs = _scaled_softmax(t_out, _TEMPERATURE)[0]
            for ti, tone in enumerate(TONE_CLASSES):
                color = f"{season}_{tone}"
                prob  = s_prob * float(t_probs[ti])
                all_probs.append((color, prob))

    all_probs.sort(key=lambda x: x[1], reverse=True)

    # 12클래스 확률 합산으로 정규화 (결합 확률이 1이 되도록)
    prob_sum = sum(p for _, p in all_probs) or 1.0
    all_probs = [(c, p / prob_sum) for c, p in all_probs]

    # 정규화된 확률로 최종 신뢰도 재계산
    top1_prob = all_probs[0][1]
    top2_prob = all_probs[1][1] if len(all_probs) > 1 else 0.0
    decision_margin = top1_prob - top2_prob
    low_confidence  = (confidence < 0.18) or (decision_margin < _CONF_GAP_WARN)

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
        f"{personal_color} conf={confidence:.1%} "
        f"margin={decision_margin:.3f} low={low_confidence} | {elapsed_ms:.1f}ms"
    )
    if quality:
        q = quality
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
    if quality_warning:
        logger.info(f"[QualityWarning] {quality_warning}")

    undertone  = UNDERTONE_MAP[pred_season]
    season_ko  = SEASON_KO[pred_season]
    tone_ko    = TONE_KO[pred_tone]

    return {
        "success":            True,
        "personal_color":     personal_color,
        "label_ko":           _LABEL_MAP[personal_color]["ko"],
        "label_en":           _LABEL_MAP[personal_color]["en"],
        # ── 3단계 진단 정보 ──────────────────────────────────
        "season":             pred_season,
        "season_ko":          season_ko,
        "undertone":          undertone,
        "tone":               pred_tone,
        "tone_ko":            tone_ko,
        # ── 신뢰도 ──────────────────────────────────────────
        "season_confidence":  round(season_conf, 6),
        "tone_confidence":    round(tone_conf, 6),
        "confidence":         round(confidence, 6),
        "decision_margin":    round(decision_margin, 6),
        "low_confidence":     low_confidence,
        "quality_warning":    quality_warning,
        # ── 결과 ────────────────────────────────────────────
        "top3":               top3,
        "recommended_colors": _PALETTE.get(personal_color, []),
        "raw_scores":         raw_scores,
        "face_detected":      True,
        "preprocess_quality": quality,
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
