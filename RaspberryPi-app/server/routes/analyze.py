"""
server/routes/analyze.py
────────────────────────
POST /analyze 엔드포인트.
AI 추론 후 클라우드 서버에 진단 결과를 저장하고 QR 코드를 반환.
오프라인 환경에서는 로컬 session_id를 QR 대체 식별자로 사용.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.infer import run_inference
from ai.model_loader import _DESCRIPTION_KO
from server.services.recommendation import (
    get_color_type_info,
    get_recommendations,
    save_diagnosis,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_API_KEY          = os.getenv("KIOSK_API_KEY",    "kiosk-dev-key-2024")  # 로컬 FastAPI 인증
_CLOUD_API_KEY    = os.getenv("CLOUD_API_KEY",    _API_KEY)              # 클라우드 인증 (별도 설정 가능)
_CLOUD_API_URL    = os.getenv("CLOUD_API_URL",    "https://personalootd.kro.kr/api")
_CLOUD_APP_BASE   = os.getenv("CLOUD_APP_BASE",   "https://personalootd.kro.kr")


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


async def _sync_to_cloud(payload: dict) -> Optional[str]:
    """
    클라우드 Node.js 서버에 진단 결과를 저장하고 qr_code를 반환.
    실패 시 None 반환 (오프라인 허용).
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{_CLOUD_API_URL}/save-diagnosis",
                json=payload,
                headers={"x-api-key": _CLOUD_API_KEY},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data.get("qr_code")
    except Exception as e:
        logger.warning(f"Cloud sync failed (offline mode): {e}")
    return None


@router.post("/analyze")
async def analyze(
    image_file: UploadFile = File(..., description="얼굴 이미지 (JPEG/PNG)"),
    session_id: str        = Form(..., description="클라이언트 UUID"),
    _auth: None            = Depends(verify_api_key),
) -> JSONResponse:
    """
    1. 이미지 디코딩
    2. 계층적 AI 추론 (Stage1: 계절, Stage2: 톤)
    3. 로컬 SQLite 저장
    4. 클라우드 서버 동기화 → QR 코드 획득
    5. 추천 아이템 조회
    6. 전체 결과 반환
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

    # ── 로컬 SQLite 저장 ───────────────────────────────────────────────
    save_diagnosis(
        session_id     = session_id,
        captured_at    = captured_at,
        personal_color = personal_color,
        confidence     = infer_result["confidence"],
        raw_scores     = infer_result["raw_scores"],
    )

    # ── 클라우드 동기화 → QR 코드 ──────────────────────────────────────
    color_info = get_color_type_info(personal_color) or {}
    if not color_info.get("description_ko"):
        color_info["description_ko"] = _DESCRIPTION_KO.get(personal_color, "")
    cloud_payload = {
        "session_id":         session_id,
        "season":             (infer_result.get("season") or "").lower(),
        "tone":               personal_color.split("_", 1)[1].lower() if "_" in personal_color else "",
        "personal_color":     personal_color.lower(),
        "label_ko":           infer_result["label_ko"],
        "label_en":           infer_result["label_en"],
        "season_confidence":  infer_result.get("season_confidence", 0.0),
        "tone_confidence":    infer_result.get("tone_confidence", 0.0),
        "final_confidence":   infer_result["confidence"],
        "color_palette": {
            "primary":   infer_result["recommended_colors"][:3],
            "secondary": infer_result["recommended_colors"][3:5],
            "accent":    [],
            "description": color_info.get("description_ko", ""),
        },
        "captured_at": captured_at,
    }

    cloud_qr_code = await _sync_to_cloud(cloud_payload)

    # QR URL 결정: 클라우드 동기화 성공 시 클라우드 URL, 실패 시 로컬 세션 기반
    if cloud_qr_code:
        qr_url = f"{_CLOUD_APP_BASE}/result/{cloud_qr_code}"
    else:
        qr_url = f"{_CLOUD_APP_BASE}/result/{session_id}"

    # ── 추천 아이템 조회 ───────────────────────────────────────────────
    reco_items = get_recommendations(personal_color)

    return JSONResponse(content={
        "success":            True,
        "session_id":         session_id,
        "personal_color":     personal_color,
        "label_ko":           infer_result["label_ko"],
        "label_en":           infer_result["label_en"],
        "season":             infer_result.get("season"),
        "season_ko":          infer_result.get("season_ko", ""),
        "undertone":          infer_result.get("undertone", ""),
        "tone":               infer_result.get("tone", ""),
        "tone_ko":            infer_result.get("tone_ko", ""),
        "season_confidence":  infer_result.get("season_confidence", 0.0),
        "tone_confidence":    infer_result.get("tone_confidence", 0.0),
        "confidence":         infer_result["confidence"],
        "top3":               infer_result["top3"],
        "recommended_colors": infer_result["recommended_colors"],
        "description_ko":     color_info.get("description_ko", ""),
        "recommendations":    reco_items,
        "cloud_qr_code":      cloud_qr_code,
        "qr_url":             qr_url,
        "captured_at":        captured_at,
        "inference_ms":       infer_result["inference_ms"],
    })
