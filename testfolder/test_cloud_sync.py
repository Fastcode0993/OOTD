"""
클라우드 동기화 연결 진단 스크립트.
  python testfolder/test_cloud_sync.py
"""
import os, sys, json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import httpx

CLOUD_API_URL = os.getenv("CLOUD_API_URL", "https://personalootd.kro.kr/api")
API_KEY       = os.getenv("KIOSK_API_KEY", "kiosk-b75288ebc4c4c30f6559da5bd606e71b")

def check(label, fn):
    print(f"\n[{label}]")
    try:
        fn()
    except Exception as e:
        print(f"  ✘ 예외: {e}")

def test_health():
    r = httpx.get(f"{CLOUD_API_URL}/health", timeout=8)
    print(f"  status: {r.status_code}")
    print(f"  body  : {r.text[:200]}")

def test_save_no_auth():
    r = httpx.post(f"{CLOUD_API_URL}/save-diagnosis",
                   json={"season":"Summer","personal_color":"Summer_Warm"},
                   timeout=8)
    print(f"  status (인증없음): {r.status_code}  → 401이면 인증 미들웨어 동작 중")

def test_save_with_auth():
    payload = {
        "session_id": "test-session-diag-001",
        "season": "summer",
        "personal_color": "summer_warm",
        "label_ko": "여름 웜",
        "label_en": "Summer Warm",
        "season_confidence": 0.85,
        "tone_confidence": 0.90,
        "final_confidence": 0.765,
        "color_palette": {"primary":["#D4A5A5"],"secondary":[],"accent":[],"description":""},
        "captured_at": "2026-04-20T12:00:00Z",
    }
    r = httpx.post(f"{CLOUD_API_URL}/save-diagnosis",
                   json=payload,
                   headers={"x-api-key": API_KEY},
                   timeout=8)
    print(f"  status: {r.status_code}")
    print(f"  body  : {r.text[:300]}")
    if r.status_code == 200:
        data = r.json()
        qr = data.get("qr_code","")
        print(f"\n  ✔ 성공! QR코드: {qr}")
        print(f"  결과 URL: https://personalootd.kro.kr/result/{qr}")

print("=" * 55)
print(f"  CLOUD_API_URL : {CLOUD_API_URL}")
print(f"  KIOSK_API_KEY : {API_KEY}")
print("=" * 55)

check("1. 헬스체크", test_health)
check("2. 인증없이 저장 시도 (401 확인)", test_save_no_auth)
check("3. API키 포함 저장 시도 (핵심)", test_save_with_auth)
