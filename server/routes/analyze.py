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
import socket
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
_CLOUD_API_KEY    = os.getenv("CLOUD_API_KEY",    "kiosk-b75288ebc4c4c30f6559da5bd606e71b")  # 클라우드 인증
_CLOUD_API_URL    = os.getenv("CLOUD_API_URL",    "https://personalootd.kro.kr/api")
_CLOUD_APP_BASE   = os.getenv("CLOUD_APP_BASE",   "https://personalootd.kro.kr")
_LOCAL_RESULT_BASE = os.getenv("LOCAL_RESULT_BASE") or os.getenv("KIOSK_PUBLIC_BASE")


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


async def _sync_to_cloud(payload: dict) -> Optional[dict]:
    """
    클라우드 Node.js 서버에 진단 결과를 저장하고 QR 정보를 반환.
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
                    return {
                        "qr_code": data.get("qr_code"),
                        "scan_url": data.get("scan_url"),
                    }
            logger.warning(
                "Cloud sync returned %s from %s: %s",
                resp.status_code,
                _CLOUD_API_URL,
                resp.text[:300],
            )
    except Exception as e:
        logger.warning(f"Cloud sync failed (offline mode): {e}")
    return None


def _detect_lan_host() -> str:
    """스마트폰이 같은 네트워크에서 접근할 수 있는 키오스크 IP를 추정한다."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _local_result_url(session_id: str) -> str:
    """
    클라우드 sync 실패 시 사용할 로컬 결과 페이지 URL.
    운영 환경에서는 LOCAL_RESULT_BASE 또는 KIOSK_PUBLIC_BASE를
    예: http://192.168.0.23:8000 형태로 지정하면 된다.
    """
    base = (_LOCAL_RESULT_BASE or "").strip().rstrip("/")
    if not base:
        port = int(os.getenv("PORT", "8000"))
        base = f"http://{_detect_lan_host()}:{port}"
    return f"{base}/result/{session_id}"


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

    # 추천 아이템을 먼저 만든 뒤 클라우드에도 함께 저장한다.
    reco_items = get_recommendations(personal_color)

    # ── 클라우드 동기화 → QR 코드 ──────────────────────────────────────
    color_info = get_color_type_info(personal_color) or {}
    if not color_info.get("description_ko"):
        color_info["description_ko"] = _DESCRIPTION_KO.get(personal_color, "")
    cloud_payload = {
        "session_id":         session_id,
        "season":             (infer_result.get("season") or "").lower(),
        "undertone":          infer_result.get("undertone", ""),
        "tone":               infer_result.get("tone", ""),
        "tone_ko":            infer_result.get("tone_ko", ""),
        "depth":              infer_result.get("tone", ""),
        "personal_color":     personal_color.lower(),
        "label_ko":           infer_result["label_ko"],
        "label_en":           infer_result["label_en"],
        "season_confidence":  infer_result.get("season_confidence", 0.0),
        "tone_confidence":    infer_result.get("tone_confidence", 0.0),
        "final_confidence":   infer_result["confidence"],
        "top3":               infer_result.get("top3", []),
        "decision_margin":    infer_result.get("decision_margin", 0.0),
        "low_confidence":     infer_result.get("low_confidence", False),
        "quality_warning":    infer_result.get("quality_warning"),
        "color_palette": {
            "primary":   infer_result["recommended_colors"][:3],
            "secondary": infer_result["recommended_colors"][3:5],
            "accent":    [],
            "description": color_info.get("description_ko", ""),
        },
        "recommendations": reco_items,
        "captured_at": captured_at,
    }

    cloud_result = await _sync_to_cloud(cloud_payload)
    cloud_qr_code = (cloud_result or {}).get("qr_code")
    cloud_scan_url = (cloud_result or {}).get("scan_url")

    # QR URL 결정: 클라우드 동기화 성공 시 클라우드 URL, 실패 시 로컬 결과 페이지.
    if cloud_qr_code:
        qr_url = cloud_scan_url or f"{_CLOUD_APP_BASE}/result/{cloud_qr_code}"
        qr_mode = "cloud"
    else:
        qr_url = _local_result_url(session_id)
        qr_mode = "local"

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
        "decision_margin":    infer_result.get("decision_margin", 0.0),
        "low_confidence":     infer_result.get("low_confidence", False),
        "quality_warning":    infer_result.get("quality_warning"),
        "top3":               infer_result["top3"],
        "recommended_colors": infer_result["recommended_colors"],
        "description_ko":     color_info.get("description_ko", ""),
        "recommendations":    reco_items,
        "cloud_qr_code":      cloud_qr_code,
        "qr_url":             qr_url,
        "qr_mode":            qr_mode,
        "cloud_synced":       cloud_qr_code is not None,
        "captured_at":        captured_at,
        "inference_ms":       infer_result["inference_ms"],
    })
