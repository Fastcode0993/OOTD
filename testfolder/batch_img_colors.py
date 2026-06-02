"""
testfolder/batch_img_colors.py

Read every image in testfolder/img and print the personal-color result.

Usage:
  python testfolder/batch_img_colors.py
  python testfolder/batch_img_colors.py --img-dir path/to/images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMG_DIR = Path(__file__).resolve().parent / "img"
MODEL_PATH = ROOT / "ai" / "models" / "final_hierarchical.pt"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.infer import run_inference
from ai.model_loader import ModelLoader


def load_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def image_files(img_dir: Path) -> list[Path]:
    return sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def short_top3(result: dict) -> str:
    items = []
    for item in result.get("top3", [])[:3]:
        label = item.get("label_ko") or item.get("color", "-")
        conf = float(item.get("confidence", 0.0)) * 100
        items.append(f"{label} {conf:.1f}%")
    return " / ".join(items)


def season_summary(result: dict) -> str:
    raw = result.get("raw_scores") or "{}"
    try:
        scores = json.loads(raw)
    except Exception:
        return "-"
    totals = {"Spring": 0.0, "Summer": 0.0, "Autumn": 0.0, "Winter": 0.0}
    for color, score in scores.items():
        season = str(color).split("_", 1)[0]
        if season in totals:
            totals[season] += float(score)
    return " ".join(f"{k[:2]}={v * 100:.1f}%" for k, v in totals.items())


def quality_summary(result: dict) -> str:
    q = result.get("preprocess_quality") or {}
    if not q:
        return "-"
    return (
        f"skin={float(q.get('skin_ratio', 0)) * 100:.1f}% "
        f"L={float(q.get('l_mean', 0)):.1f} "
        f"S={float(q.get('s_mean', 0)):.1f} "
        f"b_delta={float(q.get('b_delta', 0)):+.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="img 폴더 이미지별 퍼스널 컬러 추론")
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR, help="이미지 폴더 경로")
    parser.add_argument("--debug", action="store_true", help="계절 확률 합계와 전처리 품질값까지 출력")
    args = parser.parse_args()

    img_dir = args.img_dir.resolve()
    if not img_dir.exists():
        print(f"이미지 폴더가 없습니다: {img_dir}")
        return 1

    files = image_files(img_dir)
    if not files:
        print(f"이미지가 없습니다: {img_dir}")
        print(f"지원 확장자: {', '.join(sorted(SUPPORTED_EXTS))}")
        return 1

    print(f"모델 로딩: {MODEL_PATH}")
    ModelLoader().load(str(MODEL_PATH))
    print(f"이미지 {len(files)}개 분석 시작: {img_dir}\n")

    print(f"{'파일명':<34} {'결과':<16} {'신뢰도':>8}  TOP3")
    print("-" * 92)

    for path in files:
        bgr = load_bgr(path)
        if bgr is None:
            print(f"{path.name:<34} {'로드 실패':<16}")
            continue

        result = run_inference(bgr)
        if not result.get("success"):
            msg = result.get("message", "분석 실패")
            print(f"{path.name:<34} {'실패':<16} {'-':>8}  {msg}")
            continue

        label = result.get("label_ko", "-")
        confidence = float(result.get("confidence", 0.0)) * 100
        print(f"{path.name:<34} {label:<16} {confidence:7.1f}%  {short_top3(result)}")
        if args.debug:
            print(f"{'':<34} {'계절합계':<16} {'':>8}  {season_summary(result)}")
            print(f"{'':<34} {'품질값':<16} {'':>8}  {quality_summary(result)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
