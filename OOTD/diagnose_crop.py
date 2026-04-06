"""
얼굴 크롭 문제 진단 스크립트
사용법: python diagnose_crop.py <이미지_경로_또는_폴더>

이 스크립트는 다음을 확인합니다:
1. MediaPipe 설치 및 FaceMesh 초기화
2. 얼굴 감지 성공 여부 (multi_face_landmarks)
3. Face Oval 랜드마크 좌표 범위
4. 크롭 bbox vs 원본 크기 비교
5. 실제 크롭 이미지 shape 확인
"""

import sys
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s | %(message)s"
)

def diagnose_single(image_path: str):
    import cv2
    import numpy as np

    print(f"\n{'='*60}")
    print(f"  진단: {image_path}")
    print(f"{'='*60}")

    # 1) 이미지 로드
    bgr = cv2.imread(image_path)
    if bgr is None:
        print("❌ 이미지 로드 실패!")
        return
    h, w = bgr.shape[:2]
    print(f"✓ 이미지 로드 성공: {w}x{h} (WxH)")

    # 2) MediaPipe 확인
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe 설치됨: {mp.__version__}")
    except ImportError:
        print("❌ MediaPipe 미설치! → pip install mediapipe")
        return

    # 3) FaceMesh 초기화
    try:
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("✓ FaceMesh 초기화 성공")
    except Exception as e:
        print(f"❌ FaceMesh 초기화 실패: {e}")
        return

    # 4) 얼굴 감지
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)

    if not res.multi_face_landmarks:
        print("❌ 얼굴 미감지! (multi_face_landmarks is None)")
        print("   → 이것이 원인일 가능성이 높습니다.")
        print("   → face_crop_bgr가 설정되지 않아 원본이 저장됩니다.")
        print()
        print("  해결 방법:")
        print("  1) min_detection_confidence를 낮춰보세요 (0.3 등)")
        print("  2) 이미지 해상도가 너무 크면 리사이즈 후 감지")
        print("  3) 이미지에 실제로 정면 얼굴이 있는지 확인")

        # 낮은 confidence로 재시도
        print("\n  [재시도] min_detection_confidence=0.3...")
        fm2 = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
        )
        res2 = fm2.process(rgb)
        if res2.multi_face_landmarks:
            print("  ✓ confidence=0.3에서 얼굴 감지 성공!")
            print("  → min_detection_confidence를 0.3으로 낮추면 해결됩니다.")
        else:
            print("  ❌ confidence=0.3에서도 미감지")

            # 리사이즈 후 재시도
            if max(h, w) > 1920:
                scale = 1920 / max(h, w)
                small = cv2.resize(rgb, None, fx=scale, fy=scale)
                print(f"\n  [재시도] 리사이즈 {w}x{h} → {small.shape[1]}x{small.shape[0]}...")
                res3 = fm2.process(small)
                if res3.multi_face_landmarks:
                    print("  ✓ 리사이즈 후 감지 성공!")
                    print("  → preprocess() 호출 전에 이미지를 축소하면 해결됩니다.")
                else:
                    print("  ❌ 리사이즈 후에도 미감지")
        fm2.close()
        fm.close()
        return

    print("✓ 얼굴 감지 성공!")
    lm = res.multi_face_landmarks[0].landmark
    print(f"  랜드마크 수: {len(lm)}")

    # 5) FACE_OVAL_LANDMARKS 확인
    FACE_OVAL_LANDMARKS = [
        10,338,297,332,284,251,389,356,454,323,361,288,
        397,365,379,378,400,377,152,148,176,149,150,136,
        172,58,132,93,234,127,162,21,54,103,67,109,
    ]

    xs = [lm[i].x for i in FACE_OVAL_LANDMARKS]
    ys = [lm[i].y for i in FACE_OVAL_LANDMARKS]

    print(f"\n  Face Oval 범위 (정규화 좌표):")
    print(f"    X: {min(xs):.4f} ~ {max(xs):.4f}  (폭: {max(xs)-min(xs):.4f})")
    print(f"    Y: {min(ys):.4f} ~ {max(ys):.4f}  (높이: {max(ys)-min(ys):.4f})")

    # 6) 크롭 bbox 계산
    _pad = 0.15
    cx1 = max(0, int((min(xs) - _pad) * w))
    cy1 = max(0, int((min(ys) - _pad) * h))
    cx2 = min(w, int((max(xs) + _pad) * w))
    cy2 = min(h, int((max(ys) + _pad) * h))

    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    print(f"\n  크롭 bbox (pad={_pad}):")
    print(f"    [{cx1}, {cy1}, {cx2}, {cy2}]")
    print(f"    크롭 크기: {crop_w}x{crop_h}")
    print(f"    원본 크기: {w}x{h}")
    print(f"    크롭/원본 비율: {crop_w/w:.2%} x {crop_h/h:.2%}")

    if cx2 <= cx1 or cy2 <= cy1:
        print("  ❌ 크롭 bbox가 무효! (cx2<=cx1 또는 cy2<=cy1)")
    elif crop_w == w and crop_h == h:
        print("  ⚠️ 크롭 크기 == 원본 크기! 크롭이 무의미합니다.")
        print("     → 얼굴이 이미지 전체를 차지하거나 패딩이 너무 큽니다.")
    elif crop_w / w > 0.95 and crop_h / h > 0.95:
        print("  ⚠️ 크롭이 원본의 95% 이상 → 거의 효과 없음")
    else:
        print("  ✓ 크롭이 원본보다 작음 → 정상!")

    # 7) 실제 크롭
    cropped = bgr[cy1:cy2, cx1:cx2].copy()
    print(f"\n  실제 크롭된 배열 shape: {cropped.shape}")

    # 결과 저장
    out_dir = os.path.dirname(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    diag_path = os.path.join(out_dir, f"_DIAG_{base}.jpg")
    cv2.imwrite(diag_path, cropped)
    print(f"  → 크롭 결과 저장: {diag_path}")

    fm.close()
    print(f"\n{'='*60}")


def main():
    if len(sys.argv) < 2:
        print("사용법: python diagnose_crop.py <이미지_경로_또는_폴더>")
        print("  이미지 파일 또는 폴더를 지정하세요.")
        sys.exit(1)

    target = sys.argv[1]
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if os.path.isfile(target):
        diagnose_single(target)
    elif os.path.isdir(target):
        files = []
        for root, dirs, fnames in os.walk(target):
            for f in fnames:
                if os.path.splitext(f)[1].lower() in EXTS:
                    files.append(os.path.join(root, f))
        if not files:
            print("폴더에 이미지 파일이 없습니다.")
            sys.exit(1)
        # 최대 5개만 진단
        for f in files[:5]:
            diagnose_single(f)
        if len(files) > 5:
            print(f"\n(총 {len(files)}개 중 5개만 진단함)")
    else:
        print(f"경로를 찾을 수 없습니다: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
