"""
server/app.py
─────────────
FastAPI 서버 실행 진입점.
시작 시 SQLite DB 초기화 및 AI 모델 pre-load 수행.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── 프로젝트 루트 sys.path 등록 ──────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ai.model_loader import ModelLoader
from server.routes.analyze import router as analyze_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH    = _root / "db" / "kiosk.db"
SCHEMA_PATH = _root / "db" / "schema.sql"


# ── DB 초기화 ─────────────────────────────────────────────────────────────
def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    logger.info(f"DB initialized at {DB_PATH}")


# ── 라이프스팬 (시작/종료 훅) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Kiosk Server Starting ===")
    _init_db()

    # AI 모델 미리 로드 (첫 요청 지연 방지)
    model_path = str(_root / "ai" / "models" / "hierarchical.pt")
    loader = ModelLoader()
    loader.load(model_path)

    yield  # ── 서버 실행 중 ──

    logger.info("=== Kiosk Server Shutting Down ===")


# ── FastAPI 앱 생성 ───────────────────────────────────────────────────────
app = FastAPI(
    title="Personal Color Kiosk API",
    version="1.0.0",
    description="퍼스널 컬러 진단 키오스크 백엔드 API",
    lifespan=lifespan,
)

# CORS — localhost 전용 (키오스크 로컬 통신만 허용)
_ALLOWED_ORIGINS = [
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "x-api-key"],
)

# 라우터 등록
app.include_router(analyze_router, prefix="/api/v1", tags=["Analyze"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": ModelLoader().is_loaded}


# ── 직접 실행 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "server.app:app",
        host="127.0.0.1",
        port=port,
        reload=False,         # 프로덕션 환경
        workers=1,            # 라즈베리파이5 단일 프로세스
        log_level="info",
    )
