"""
inference_cli.py
────────────────
커맨드라인에서 가상 이미지를 빠르게 생성 & 추론하는 도구.

사용법:
  python inference_cli.py --mode dummy --tone warm_light
  python inference_cli.py --mode pil --tone cool_deep
  python inference_cli.py --mode file --image test_face.jpg
  python inference_cli.py --mode all  # 모든 테스트 한 번에
"""

import sys
from pathlib import Path
import argparse
import json

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw

# 프로젝트 모듈
from project.ai.infer import run_inference
from project.ai.model_loader import ModelLoader


# ========================================================================
# 이미지 생성 함수
# ========================================================================
def create_dummy_image(tone="warm_light", width=640, height=480):
    """단색 피부톤 이미지 생성"""
    skin_colors = {
        "warm_light":   (180, 160, 140),
        "warm_deep":    (120, 100, 80),
        "cool_light":   (160, 140, 140),
        "cool_deep":    (100, 80, 100),
    }
    color = skin_colors.get(tone, (160, 140, 120))
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), 150, color, -1)
    noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def create_pil_image(tone="warm_light", width=640, height=480):
    """PIL로 얼굴 형태 생성"""
    skin_colors_rgb = {
        "warm_light":   (217, 180, 105),
        "warm_deep":    (139, 110, 65),
        "cool_light":   (198, 166, 143),
        "cool_deep":    (138, 110, 125),
    }
    color = skin_colors_rgb.get(tone, (200, 150, 100))
    img = Image.new('RGB', (width, height), (220, 220, 220))
    draw = ImageDraw.Draw(img)
    center_x, center_y, radius = width // 2, height // 2, 150
    bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
    draw.ellipse(bbox, fill=color)
    eye_y = center_y - 60
    draw.ellipse([center_x - 80, eye_y - 20, center_x - 50, eye_y + 20], fill=(50, 50, 50))
    draw.ellipse([center_x + 50, eye_y - 20, center_x + 80, eye_y + 20], fill=(50, 50, 50))
    mouth_y = center_y + 60
    draw.arc([center_x - 60, mouth_y, center_x + 60, mouth_y + 40], 0, 180, fill=(200, 100, 100), width=3)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ========================================================================
# 추론 래퍼
# ========================================================================
def infer_and_print(bgr_image, test_name="Test", save_output=False):
    """추론 실행 및 결과 출력"""
    result = run_inference(bgr_image)
    
    print(f"\n📊 {test_name}")
    print("─" * 60)
    
    if result["success"]:
        print(f"✓ {result['label_ko']:<20} (신뢰도: {result['confidence']:.1%})")
        print(f"  영문명: {result['label_en']}")
        print(f"  추론시간: {result['inference_ms']:.2f}ms")
        print(f"\n  🥇 TOP 3:")
        for i, top in enumerate(result['top3'], 1):
            print(f"     {i}. {top['label_ko']:<25} {top['confidence']:.1%}")
        print(f"\n  🎨 팔레트: {' | '.join(result['recommended_colors'][:3])}")
    else:
        print(f"✗ 실패: {result['message']}")
    
    if save_output:
        output_path = f"result_{test_name.replace(' ', '_')}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  💾 저장됨: {output_path}")
    
    return result


# ========================================================================
# 메인
# ========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="가상 이미지로 퍼스널 컬러 추론 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python inference_cli.py --mode dummy --tone warm_light
  python inference_cli.py --mode pil --tone cool_deep
  python inference_cli.py --mode file --image test_face.jpg
  python inference_cli.py --mode all --save
        """
    )
    
    parser.add_argument('--mode', choices=['dummy', 'pil', 'file', 'all'], 
                        default='dummy', help='이미지 생성 모드')
    parser.add_argument('--tone', choices=['warm_light', 'warm_deep', 'cool_light', 'cool_deep'],
                        default='warm_light', help='피부톤 (dummy/pil 모드)')
    parser.add_argument('--image', help='이미지 파일 경로 (file 모드)')
    parser.add_argument('--save', action='store_true', help='결과를 JSON으로 저장')
    
    args = parser.parse_args()
    
    # 모델 로드
    print("🔧 모델 로딩 중...", end=" ")
    loader = ModelLoader()
    model_path = str(_ROOT / "project" / "ai" / "models" / "personal_color.pt")
    loader.load(model_path)
    print("✓\n")
    
    # 모드별 실행
    if args.mode == 'dummy':
        img = create_dummy_image(args.tone)
        infer_and_print(img, f"단색 {args.tone}", args.save)
    
    elif args.mode == 'pil':
        img = create_pil_image(args.tone)
        infer_and_print(img, f"PIL {args.tone}", args.save)
    
    elif args.mode == 'file':
        if not args.image:
            print("❌ --image 인자가 필요합니다")
            return
        try:
            img = cv2.imread(args.image)
            if img is None:
                print(f"❌ 이미지를 로드할 수 없습니다: {args.image}")
                return
            infer_and_print(img, f"파일 {args.image}", args.save)
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    elif args.mode == 'all':
        tones = ['warm_light', 'warm_deep', 'cool_light', 'cool_deep']
        print("=" * 60)
        print("🧪 전체 테스트 실행")
        print("=" * 60)
        for tone in tones:
            img = create_dummy_image(tone)
            infer_and_print(img, f"단색-{tone}", args.save)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
