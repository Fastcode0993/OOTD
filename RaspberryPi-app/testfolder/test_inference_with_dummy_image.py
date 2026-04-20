"""
test_inference_with_dummy_image.py (수정)
────────────────────────────────────────
img/ 폴더 안의 모든 얼굴 이미지를 자동으로 로드하여
AI 모델로 퍼스널 컬러를 추론하고 결과를 표시한다.

사용법:
  python test_inference_with_dummy_image.py
  
폴더 구조:
  project/
  ├─ img/
  │  ├─ face1.jpg
  │  ├─ face2.jpg
  │  └─ ...
  └─ test_inference_with_dummy_image.py
"""

import sys
from pathlib import Path
import logging

# 프로젝트 루트 추가 (testfolder의 상위 디렉토리)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

# 프로젝트 모듈 임포트
from ai.model_loader import ModelLoader
from ai.preprocessing import FacePreprocessor
from ai.infer import run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ========================================================================
# 이미지 폴더 설정
# ========================================================================
IMG_DIR = Path(__file__).parent / "img"
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


# ========================================================================
# 이미지 로드 및 검증
# ========================================================================
def get_image_files():
    """
    img/ 폴더에서 모든 이미지 파일 찾기.
    
    Returns
    -------
    list[Path]
        이미지 파일 경로 리스트
    """
    if not IMG_DIR.exists():
        logger.warning(f"📁 {IMG_DIR} 폴더가 없습니다. 생성하고 이미지를 넣어주세요.")
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        return []
    
    image_files = []
    for fmt in SUPPORTED_FORMATS:
        image_files.extend(IMG_DIR.glob(f"*{fmt}"))
        image_files.extend(IMG_DIR.glob(f"*{fmt.upper()}"))
    
    # 중복 제거 및 정렬
    image_files = sorted(set(image_files))
    return image_files


def load_image(image_path):
    """
    이미지 파일을 OpenCV BGR 형식으로 로드.
    
    Parameters
    ----------
    image_path : Path
        이미지 파일 경로
        
    Returns
    -------
    tuple[bool, np.ndarray or str]
        (성공 여부, BGR 이미지 또는 에러 메시지)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, f"파일을 읽을 수 없습니다: {image_path.name}"
        logger.info(f"✓ 로드: {image_path.name} ({img.shape[1]}×{img.shape[0]})")
        return True, img
    except Exception as e:
        return False, f"로드 오류: {e}"


# ========================================================================
# 추론 및 결과 표시
# ========================================================================
def run_inference_on_image(bgr_image, image_name="Image"):
    """
    이미지로 추론 실행 및 결과 출력.
    
    Parameters
    ----------
    bgr_image : np.ndarray
        BGR 형식의 이미지
    image_name : str
        이미지 이름 (로그용)
        
    Returns
    -------
    dict
        추론 결과
    """
    result = run_inference(bgr_image)
    
    # 결과 출력
    print(f"\n{'='*70}")
    print(f"📸 이미지: {image_name}")
    print(f"{'='*70}")
    
    if result["success"]:
        print(f"✓ 추론 성공")
        print(f"\n  🎨 퍼스널 컬러")
        print(f"  ├─ 한국어: {result['label_ko']:<20} (신뢰도: {result['confidence']*100:5.1f}%)")
        print(f"  └─ 영문:   {result['label_en']:<20}")
        
        print(f"\n  ⏱️  추론 시간: {result['inference_ms']:.2f}ms")
        
        print(f"\n  🥇 TOP 3 결과:")
        for i, top in enumerate(result['top3'], 1):
            bar_length = int(top['confidence'] * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"     {i}. {top['label_ko']:<20} {top['confidence']*100:5.1f}%  [{bar}]")
        
        print(f"\n  🎨 추천 색상 팔레트:")
        # 터미널에 색상 표시 (ANSI 배경색)
        for i, hex_color in enumerate(result['recommended_colors'][:5]):
            # RGB를 0-255로 정규화
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            print(f"     [{i+1}] {hex_color}  (RGB: {r:3d}, {g:3d}, {b:3d})")
        
        return result
    else:
        print(f"✗ 추론 실패: {result['message']}")
        return result


# ========================================================================
# 메인 프로세스
# ========================================================================
def main():
    print("\n" + "="*70)
    print("🚀 퍼스널 컬러 키오스크 — img 폴더 배치 추론")
    print("="*70)
    
    # 모델 로드
    print("\n[1/3] 모델 로딩 중...")
    try:
        loader = ModelLoader()
        model_path = str(_ROOT / "ai" / "models" / "final_hierarchical.pt")
        loader.load(model_path)
        print("✓ 모델 로드 완료\n")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return
    
    # 이미지 파일 찾기
    print("[2/3] img 폴더에서 이미지 검색 중...")
    image_files = get_image_files()
    
    if not image_files:
        print(f"\n⚠️  {IMG_DIR}에 이미지가 없습니다.")
        print(f"   다음 형식의 이미지를 넣어주세요:")
        print(f"   {', '.join(sorted(SUPPORTED_FORMATS))}\n")
        return
    
    print(f"✓ {len(image_files)}개 이미지 발견\n")
    
    # 추론 실행
    print("[3/3] 추론 실행 중...")
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n({idx}/{len(image_files)}) 처리 중...", end=" ")
        
        # 이미지 로드
        success, img_or_error = load_image(image_path)
        if not success:
            print(f"✗ {img_or_error}")
            results.append({
                "file": image_path.name,
                "success": False,
                "error": img_or_error
            })
            continue
        
        # 추론
        result = run_inference_on_image(img_or_error, image_path.name)
        result["file"] = image_path.name
        results.append(result)
    
    # 최종 요약
    print("\n\n" + "="*70)
    print("📊 최종 요약")
    print("="*70)
    
    success_count = sum(1 for r in results if r.get("success", False))
    failed_count = len(results) - success_count
    
    print(f"\n  총 처리: {len(results)}개")
    print(f"  ✓ 성공: {success_count}개")
    print(f"  ✗ 실패: {failed_count}개\n")
    
    if success_count > 0:
        print("  🎨 결과 요약:")
        for r in results:
            if r.get("success"):
                print(f"     • {r['file']:<30} → {r['label_ko']:<20} ({r['confidence']*100:.1f}%)")
    
    # 결과를 텍스트 파일로 저장
    save_results_to_file(results)
    
    print("\n" + "="*70 + "\n")


def save_results_to_file(results):
    """
    추론 결과를 텍스트 파일로 저장.
    
    Parameters
    ----------
    results : list[dict]
        추론 결과 리스트
    """
    output_file = IMG_DIR / "inference_results.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("퍼스널 컬러 추론 결과 — 배치 처리\n")
            f.write("="*70 + "\n\n")
            
            for result in results:
                f.write(f"📸 {result['file']}\n")
                f.write("-" * 70 + "\n")
                
                if result.get("success"):
                    f.write(f"  퍼스널 컬러: {result['label_ko']:<20} ({result['label_en']})\n")
                    f.write(f"  신뢰도:     {result['confidence']*100:.1f}%\n")
                    f.write(f"  추론시간:   {result['inference_ms']:.2f}ms\n")
                    f.write(f"\n  TOP 3:\n")
                    for i, top in enumerate(result['top3'], 1):
                        f.write(f"    {i}. {top['label_ko']:<20} {top['confidence']*100:.1f}%\n")
                    f.write(f"\n  색상 팔레트:\n")
                    for i, color in enumerate(result['recommended_colors'][:5], 1):
                        f.write(f"    [{i}] {color}\n")
                else:
                    f.write(f"  ✗ 오류: {result.get('error', '알 수 없음')}\n")
                
                f.write("\n")
        
        print(f"  💾 결과 저장: {output_file}")
    except Exception as e:
        print(f"  ✗ 결과 저장 실패: {e}")


if __name__ == "__main__":
    main()