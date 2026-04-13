"""
server/routes/analyze.py
────────────────────────
POST /analyze 엔드포인트.
멀티파트로 이미지와 session_id를 받아 AI 추론 후 결과를 반환.
간단한 API Key 인증 포함.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ── 프로젝트 루트를 sys.path에 추가 (서버에서 ai/ 모듈 임포트) ─────────
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.infer import run_inference
from server.services.recommendation import (
    get_color_type_info,
    get_recommendations,
    save_diagnosis,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── API Key 설정 (환경변수 또는 기본값) ──────────────────────────────────
_API_KEY = os.getenv("KIOSK_API_KEY", "kiosk-dev-key-2024")


# ── 인증 의존성 ───────────────────────────────────────────────────────────
def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── /analyze 엔드포인트 ──────────────────────────────────────────────────
@router.post("/analyze")
async def analyze(
    image_file: UploadFile = File(..., description="얼굴 이미지 (JPEG/PNG)"),
    session_id: str        = Form(..., description="클라이언트 UUID"),
    _auth: None            = Depends(verify_api_key),
) -> JSONResponse:
    """
    1. 업로드된 이미지를 OpenCV 배열로 변환
    2. AI 추론 (MediaPipe + PyTorch)
    3. DB 저장
    4. 추천 아이템 조회
    5. 전체 결과 반환
    """
    # ── 이미지 읽기 ────────────────────────────────────────────────────
    try:
        raw_bytes = await image_file.read()
        np_arr    = np.frombuffer(raw_bytes, dtype=np.uint8)
        bgr_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr_frame is None:
            raise ValueError("이미지 디코딩 실패")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 오류: {e}")

    # ── AI 추론 ────────────────────────────────────────────────────────
    infer_result = run_inference(bgr_frame)

    if not infer_result["success"]:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": infer_result["message"]},
        )

    personal_color = infer_result["personal_color"]
    captured_at    = datetime.now(timezone.utc).isoformat()

    # ── DB 저장 ────────────────────────────────────────────────────────
    save_diagnosis(
        session_id     = session_id,
        captured_at    = captured_at,
        personal_color = personal_color,
        confidence     = infer_result["confidence"],
        raw_scores     = infer_result["raw_scores"],
    )

    # ── 추천 아이템 & 메타 조회 ────────────────────────────────────────
    color_info   = get_color_type_info(personal_color) or {}
    reco_items   = get_recommendations(personal_color)

    return JSONResponse(content={
        "success":           True,
        "session_id":        session_id,
        "personal_color":    personal_color,
        "label_ko":          infer_result["label_ko"],
        "label_en":          infer_result["label_en"],
        "confidence":        infer_result["confidence"],
        "top3":              infer_result["top3"],
        "recommended_colors": infer_result["recommended_colors"],
        "description_ko":    color_info.get("description_ko", ""),
        "recommendations":   reco_items,
        "captured_at":       captured_at,
        "inference_ms":      infer_result["inference_ms"],
    })
