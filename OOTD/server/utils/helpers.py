"""
server/utils/helpers.py
────────────────────────
공통 유틸리티 함수 모음.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── 세션 ID 생성 ──────────────────────────────────────────────────────────
def new_session_id() -> str:
    """UUID4 기반 세션 ID 생성."""
    return str(uuid.uuid4())


# ── 이미지 저장 ───────────────────────────────────────────────────────────
def save_face_image(
    bgr_frame: np.ndarray,
    session_id: str,
    base_dir: str = "data/captures",
) -> Optional[str]:
    """
    얼굴 크롭 이미지를 디스크에 저장하고 경로를 반환.
    저장 실패 시 None 반환.
    """
    try:
        save_dir = Path(base_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{session_id}.jpg"
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return str(filepath)
    except Exception:
        return None


# ── 타임스탬프 ────────────────────────────────────────────────────────────
def utc_now_iso() -> str:
    """현재 UTC 시각을 ISO-8601 문자열로 반환."""
    return datetime.now(timezone.utc).isoformat()


# ── API Key 해시 검증 ─────────────────────────────────────────────────────
def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()
