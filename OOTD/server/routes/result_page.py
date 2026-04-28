"""
server/routes/result_page.py
─────────────────────────────
GET /result/{session_id}  →  퍼스널 컬러 진단 결과 HTML 페이지.
개발 환경에서 QR 코드 스캔 시 이 페이지가 열린다.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)
router = APIRouter()

DB_PATH = Path(__file__).resolve().parents[2] / "db" / "kiosk.db"


def _query_diagnosis(session_id: str) -> dict | None:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM diagnosis_results WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"result_page DB error: {e}")
        return None


def _palette_html(personal_color: str) -> str:
    try:
        from ai.model_loader import _PALETTE
        colors = _PALETTE.get(personal_color, [])
    except Exception:
        colors = []
    if not colors:
        return ""
    swatches = "".join(
        f'<div style="width:52px;height:52px;border-radius:10px;background:{c};'
        f'box-shadow:0 2px 8px rgba(0,0,0,.4);"></div>'
        for c in colors
    )
    return f'<div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">{swatches}</div>'


def _description(personal_color: str) -> str:
    try:
        from ai.model_loader import _DESCRIPTION_KO
        return _DESCRIPTION_KO.get(personal_color, "")
    except Exception:
        return ""


def _label_ko(personal_color: str) -> str:
    try:
        from ai.model_loader import _LABEL_MAP
        return _LABEL_MAP.get(personal_color, {}).get("ko", personal_color)
    except Exception:
        return personal_color


def _top3_html(raw_scores: str) -> str:
    try:
        scores: dict = json.loads(raw_scores or "{}")
        from ai.model_loader import _LABEL_MAP
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        rows = ""
        for i, (color, prob) in enumerate(ranked):
            lbl = _LABEL_MAP.get(color, {}).get("ko", color)
            accent = "#C9A96E" if i == 0 else "#666"
            crown = "👑 " if i == 0 else ""
            rows += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:10px 18px;margin:6px 0;border-radius:10px;'
                f'background:#231E1A;border:1px solid {accent};">'
                f'<span style="color:{accent};font-size:15px;">{crown}{i+1}위  {lbl}</span>'
                f'<span style="color:#888;font-size:14px;">{prob*100:.1f}%</span>'
                f'</div>'
            )
        return rows
    except Exception:
        return ""


@router.get("/result/{session_id}", response_class=HTMLResponse)
async def result_page(session_id: str) -> HTMLResponse:
    data = _query_diagnosis(session_id)

    if data is None:
        return HTMLResponse(
            content=f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
            <title>결과 없음</title>
            <style>body{{background:#1A1512;color:#C9A96E;font-family:sans-serif;
            display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}}
            </style></head><body><h2>진단 결과를 찾을 수 없습니다.<br>
            <small style="color:#888;font-size:14px;">{session_id}</small></h2></body></html>""",
            status_code=404,
        )

    pc       = data.get("personal_color", "")
    conf     = float(data.get("confidence", 0)) * 100
    captured = data.get("captured_at", "")[:19].replace("T", " ")
    label    = _label_ko(pc)
    desc     = _description(pc)
    palette  = _palette_html(pc)
    top3     = _top3_html(data.get("raw_scores", "{}"))

    # personal_color "Summer_Warm" → 계절·톤 파싱
    season_ko_map = {"Spring": "봄", "Summer": "여름", "Autumn": "가을", "Winter": "겨울"}
    tone_ko_map   = {"Warm": "웜", "Bright": "브라이트", "Light": "라이트"}
    undertone_map = {"Spring": "웜톤", "Summer": "쿨톤", "Autumn": "웜톤", "Winter": "쿨톤"}
    parts     = pc.split("_", 1)
    season    = parts[0] if parts else ""
    tone      = parts[1] if len(parts) > 1 else ""
    season_ko  = season_ko_map.get(season, season)
    tone_ko    = tone_ko_map.get(tone, tone)
    undertone  = undertone_map.get(season, "")
    season_tone = f"{season_ko} - {tone_ko}" if tone_ko else season_ko

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>퍼스널 컬러 진단 결과</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #1A1512;
      color: #F5EFE6;
      font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
      min-height: 100vh;
      padding: 32px 20px 60px;
    }}
    .container {{ max-width: 480px; margin: 0 auto; }}
    .tag {{
      text-align: center;
      color: #C9A96E;
      font-size: 12px;
      letter-spacing: 5px;
      margin-bottom: 8px;
    }}
    h1 {{
      text-align: center;
      font-size: 28px;
      font-family: Georgia, serif;
      color: #F5EFE6;
      margin-bottom: 4px;
    }}
    .subtitle {{
      text-align: center;
      color: #C9A96E;
      font-size: 13px;
      margin-bottom: 28px;
    }}
    .badges {{
      display: flex;
      gap: 14px;
      justify-content: center;
      margin-bottom: 16px;
    }}
    .badge {{
      background: #231E1A;
      border-radius: 12px;
      padding: 10px 18px;
      text-align: center;
      min-width: 120px;
    }}
    .badge-hint {{
      font-size: 11px;
      color: #C9A96E;
      letter-spacing: 2px;
      margin-bottom: 4px;
    }}
    .badge-val {{
      font-size: 20px;
      font-family: Georgia, serif;
      font-weight: bold;
    }}
    .conf {{
      text-align: center;
      color: #C9A96E;
      font-size: 17px;
      margin-bottom: 20px;
    }}
    .desc {{
      text-align: center;
      color: #C8BFB0;
      font-size: 15px;
      line-height: 1.7;
      margin-bottom: 28px;
    }}
    .section-title {{
      color: #C9A96E;
      font-size: 12px;
      letter-spacing: 3px;
      text-align: center;
      margin-bottom: 14px;
    }}
    .palette {{ margin-bottom: 28px; }}
    .top3 {{ margin-bottom: 32px; }}
    .footer {{
      text-align: center;
      color: #444;
      font-size: 12px;
      margin-top: 32px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="tag">✦  PERSONAL COLOR  ✦</div>
    <h1>{label}</h1>
    <div class="subtitle">진단일  {captured}</div>

    <div class="badges">
      <div class="badge" style="border:1px solid #C9A96E;">
        <div class="badge-hint">계절 · 톤</div>
        <div class="badge-val" style="color:#F5EFE6;">{season_tone}</div>
      </div>
      <div class="badge" style="border:1px solid #C4858A;">
        <div class="badge-hint">웜 / 쿨</div>
        <div class="badge-val" style="color:#F5EFE6;">{undertone}</div>
      </div>
    </div>

    <div class="conf">적합도  {conf:.1f}%</div>
    <div class="desc">{desc}</div>

    <div class="section-title palette">어울리는 색상 팔레트</div>
    <div class="palette">{palette}</div>

    <div class="section-title">TOP 3 결과</div>
    <div class="top3">{top3}</div>

    <div class="footer">Personal Color Kiosk &nbsp;·&nbsp; {session_id[:8]}...</div>
  </div>
</body>
</html>"""

    return HTMLResponse(content=html)
