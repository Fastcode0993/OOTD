"""
server/services/recommendation.py
──────────────────────────────────
퍼스널 컬러 코드를 받아 SQLite에서 추천 아이템과
컬러 팔레트를 조회하여 반환한다.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "db" / "kiosk.db"
CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "recommendation_catalog.json"

_DEFAULT_CATEGORIES = ["fashion", "makeup", "hair", "interior"]
_SEASONS = {"spring", "summer", "autumn", "winter"}
_TONES = {"warm", "bright", "light"}
_NAVER_SHOPPING_URL = "https://openapi.naver.com/v1/search/shop.json"
_SHOPPING_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


# ── DB 연결 헬퍼 ──────────────────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _load_catalog() -> Dict[str, Any]:
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"recommendation catalog load error: {e}")
        return {}


def _normalize_color_code(color_code: str) -> Tuple[str, str]:
    """Return (season, tone) from values such as Spring_Warm or spring_warm_light."""
    parts = [p.strip().lower() for p in (color_code or "").replace("-", "_").split("_") if p.strip()]
    season = next((p for p in parts if p in _SEASONS), "")
    tone = next((p for p in parts if p in _TONES), "")
    return season, tone


def _score_item(season: str, tone: str, item: Dict[str, Any], catalog: Dict[str, Any]) -> float:
    weights = catalog.get("algorithm", {}).get("weights", {})
    profile = catalog.get("season_profiles", {}).get(season, {})
    tone_info = catalog.get("tone_modifiers", {}).get(tone, {})

    text = " ".join(
        str(item.get(k, "")).lower()
        for k in ("recommended_shade_en", "color_name_en", "tip_ko", "tip_en")
    )
    season_families = [x.lower() for x in profile.get("families", [])]
    boost_families = [x.lower() for x in tone_info.get("boost_families", [])]

    color_family_fit = 1.0 if any(f in text for f in season_families) else 0.75
    tone_fit = 1.0 if not tone or any(f in text for f in boost_families) else 0.82
    evidence_weight = min(1.0, 0.65 + 0.1 * len(item.get("evidence", [])))

    score = (
        weights.get("season_fit", 0.4)
        + weights.get("undertone_fit", 0.25)
        + weights.get("tone_fit", 0.15) * tone_fit
        + weights.get("color_family_fit", 0.12) * color_family_fit
        + weights.get("evidence_weight", 0.08) * evidence_weight
    )
    return round(score, 3)


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").replace("&quot;", '"').replace("&amp;", "&")


def _shopping_query(item: Dict[str, Any], category: str) -> str:
    parts = [
        item.get("brand", ""),
        item.get("item_name_ko") or item.get("item_name_en", ""),
    ]
    shade = item.get("recommended_shade_ko") or item.get("color_name_ko", "")
    if shade:
        parts.append(shade.split("/")[0].split(",")[0])
    if category == "fashion":
        parts.append("패션")
    elif category == "makeup":
        parts.append("화장품")
    return " ".join(str(p).strip() for p in parts if str(p).strip())


def _naver_shopping_search(query: str) -> Optional[Dict[str, Any]]:
    """
    네이버 쇼핑 검색 API로 실제 판매 상품 1개를 조회한다.
    NAVER_SHOPPING_CLIENT_ID / NAVER_SHOPPING_CLIENT_SECRET 없으면 비활성화.
    """
    client_id = os.getenv("NAVER_SHOPPING_CLIENT_ID") or os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_SHOPPING_CLIENT_SECRET") or os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret or not query:
        return None

    if query in _SHOPPING_CACHE:
        return _SHOPPING_CACHE[query]

    try:
        resp = httpx.get(
            _NAVER_SHOPPING_URL,
            params={"query": query, "display": 3, "sort": "sim"},
            headers={
                "X-Naver-Client-Id": client_id,
                "X-Naver-Client-Secret": client_secret,
            },
            timeout=2.5,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            _SHOPPING_CACHE[query] = None
            return None

        first = items[0]
        price = first.get("lprice") or ""
        product = {
            "live_product": True,
            "product_source": "naver_shopping",
            "product_query": query,
            "product_title": _strip_html(first.get("title", "")),
            "product_mall": first.get("mallName", ""),
            "product_price": int(price) if str(price).isdigit() else None,
            "product_image": first.get("image", ""),
            "source_url": first.get("link", ""),
        }
        _SHOPPING_CACHE[query] = product
        return product
    except Exception as e:
        logger.warning(f"Naver shopping search failed query={query!r}: {e}")
        _SHOPPING_CACHE[query] = None
        return None


def _enrich_with_live_product(item: Dict[str, Any], category: str) -> Dict[str, Any]:
    enriched = dict(item)
    query = _shopping_query(enriched, category)
    live = _naver_shopping_search(query)
    if not live:
        enriched.setdefault("live_product", False)
        return enriched

    enriched.update(live)
    if live.get("product_title"):
        enriched["item_name_ko"] = live["product_title"]
    if live.get("product_mall"):
        enriched["brand"] = live["product_mall"]
    return enriched


def _catalog_recommendations(
    color_code: str,
    categories: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if categories is None:
        categories = _DEFAULT_CATEGORIES

    result: Dict[str, List[Dict[str, Any]]] = {c: [] for c in categories}
    catalog = _load_catalog()
    season, tone = _normalize_color_code(color_code)
    profile = catalog.get("season_profiles", {}).get(season)
    if not profile:
        return result

    tone_info = catalog.get("tone_modifiers", {}).get(tone, {})
    for category in categories:
        if category not in ("fashion", "makeup"):
            continue
        items = []
        for item in profile.get(category, []):
            enriched = dict(item)
            enriched["category"] = category
            enriched["score"] = _score_item(season, tone, enriched, catalog)
            enriched["match_basis_ko"] = (
                f"{profile.get('label_ko', season)} 타입의 {profile.get('undertone', '')} undertone과 "
                f"{tone_info.get('label_ko', tone) or tone} 톤 보정 기준을 반영했습니다."
            )
            if tone_info.get("tip_ko"):
                enriched["tone_tip_ko"] = tone_info["tip_ko"]
            enriched = _enrich_with_live_product(enriched, category)
            items.append(enriched)
        result[category] = sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    return result


def _merge_recommendations(
    db_items: Dict[str, List[Dict[str, Any]]],
    fallback_items: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    merged = {k: list(v) for k, v in fallback_items.items()}
    for category, items in db_items.items():
        if items:
            merged[category] = items
    return merged


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
        categories = _DEFAULT_CATEGORIES

    result: Dict[str, List[Dict[str, Any]]] = {c: [] for c in categories}

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

    return _merge_recommendations(result, _catalog_recommendations(color_code, categories))


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
