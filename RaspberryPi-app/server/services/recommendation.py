"""
server/services/recommendation.py
──────────────────────────────────
퍼스널 컬러 코드를 받아 SQLite에서 추천 아이템과
컬러 팔레트를 조회하여 반환한다.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "db" / "kiosk.db"


# ── DB 연결 헬퍼 ──────────────────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── 퍼스널 컬러 메타 조회 ─────────────────────────────────────────────────
def get_color_type_info(color_code: str) -> Optional[Dict[str, Any]]:
    """color_types 테이블에서 컬러 유형 메타 정보를 반환."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM color_types WHERE code = ?", (color_code,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["palette_hex"] = json.loads(result.get("palette_hex", "[]"))
        return result
    except Exception as e:
        logger.error(f"get_color_type_info error: {e}")
        return None


# ── 추천 아이템 조회 ──────────────────────────────────────────────────────
def get_recommendations(
    color_code: str,
    categories: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    카테고리별로 추천 아이템을 묶어서 반환.

    Returns
    -------
    {
      "fashion": [...],
      "makeup":  [...],
      "hair":    [...],
      "interior":[...],
    }
    """
    if categories is None:
        categories = ["fashion", "makeup", "hair", "interior"]

    result: Dict[str, List] = {c: [] for c in categories}

    placeholders = ",".join("?" for _ in categories)
    query = (
        "SELECT * FROM recommended_items "
        "WHERE color_type_code = ? AND category IN (" + placeholders + ")"
    )

    try:
        with _get_conn() as conn:
            rows = conn.execute(query, [color_code, *categories]).fetchall()
        for row in rows:
            cat = row["category"]
            if cat in result:
                result[cat].append(dict(row))
    except Exception as e:
        logger.error(f"get_recommendations error: {e}")

    return result


# ── 진단 결과 저장 ────────────────────────────────────────────────────────
def save_diagnosis(
    session_id: str,
    captured_at: str,
    personal_color: str,
    confidence: float,
    raw_scores: str,
    image_path: Optional[str] = None,
) -> bool:
    """diagnosis_results 테이블에 진단 결과를 저장."""

    # personal_color 코드 파싱 (예: "spring_warm_light" → season/tone/depth)
    parts = personal_color.split("_")
    season = parts[0].capitalize() if len(parts) > 0 else "Unknown"
    tone   = parts[1].capitalize() if len(parts) > 1 else "Unknown"
    depth  = parts[2].capitalize() if len(parts) > 2 else "Unknown"

    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO diagnosis_results
                    (session_id, captured_at, personal_color,
                     color_season, color_tone, color_depth,
                     confidence, raw_scores, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, captured_at, personal_color,
                 season, tone, depth,
                 confidence, raw_scores, image_path),
            )
        return True
    except Exception as e:
        logger.error(f"save_diagnosis error: {e}")
        return False


# ── 진단 결과 조회 ────────────────────────────────────────────────────────
def get_diagnosis(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM diagnosis_results WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"get_diagnosis error: {e}")
        return None
