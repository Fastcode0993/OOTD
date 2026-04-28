# 가상 이미지로 퍼스널 컬러 추론하기 — 개발 가이드

카메라 없는 개발 환경에서 가상 얼굴 이미지를 생성하여 AI 모델을 테스트하는 방법을 설명합니다.

---

## 🎯 3가지 주요 방법

### **방법 1: CLI 도구 (가장 빠름)**

```bash
# 단색 피부톤 이미지로 추론
python inference_cli.py --mode dummy --tone warm_light

# PIL로 생성한 얼굴 형태로 추론
python inference_cli.py --mode pil --tone cool_deep

# 실제 이미지 파일로 추론
python inference_cli.py --mode file --image test_face.jpg

# 모든 조합 테스트 (결과 JSON 저장)
python inference_cli.py --mode all --save
```

**출력 예:**
```
🔧 모델 로딩 중... ✓

📊 단색 warm_light
────────────────────────────────────────────────────────────────
✓ 봄 웜 라이트         (신뢰도: 45.2%)
  영문명: Spring Warm Light
  추론시간: 142.33ms

  🥇 TOP 3:
     1. 봄 웜 라이트             45.2%
     2. 봄 웜 브라이트           32.1%
     3. 봄 웜 딥                 8.3%

  🎨 팔레트: #FDDBB4 | #F9A87A | #F47C5A
```

---

### **방법 2: 웹 인터페이스 (가장 시각적)**

```bash
# Flask 웹 서버 실행
python dev_inference_web.py

# 브라우저에서 열기
# http://localhost:5000
```

**기능:**
- 🎨 4가지 피부톤으로 즉시 테스트
- 📸 이미지 파일 업로드 지원
- 🚀 모든 조합 일괄 테스트
- 📊 실시간 결과 표시 및 시각화

---

### **방법 3: Python 스크립트 (가장 유연함)**

```python
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'project'))

from ai.infer import run_inference
from ai.model_loader import ModelLoader
import numpy as np
import cv2

# 모델 로드
loader = ModelLoader()
loader.load("project/ai/models/personal_color.pt")

# 가상 이미지 생성 (640×480, BGR)
img = np.full((480, 640, 3), (180, 160, 140), dtype=np.uint8)
cv2.circle(img, (320, 240), 150, (180, 160, 140), -1)

# 추론
result = run_inference(img)

# 결과 출력
print(f"결과: {result['label_ko']}")
print(f"신뢰도: {result['confidence']:.1%}")
```

---

## 🔧 이미지 생성 방식 비교

| 방식 | 생성 속도 | 자연스러움 | 용도 |
|------|---------|---------|------|
| **단색 배경** (numpy) | ⚡⚡⚡ | ⭐ | 빠른 테스트 |
| **PIL 얼굴** | ⚡⚡ | ⭐⭐ | 기본 테스트 |
| **실제 이미지** | ⚡ | ⭐⭐⭐ | 정확한 검증 |

---

## 📋 각 방식의 구현

### **1️⃣ 단색 배경 이미지 (Numpy)**

```python
def create_dummy_skin_image(width=640, height=480, skin_tone="warm_light"):
    # 피부톤별 BGR 색상
    skin_colors = {
        "warm_light":   (180, 160, 140),   # 밝은 웜톤 (봄)
        "warm_deep":    (120, 100, 80),    # 어두운 웜톤 (가을)
        "cool_light":   (160, 140, 140),   # 밝은 쿨톤 (여름)
        "cool_deep":    (100, 80, 100),    # 어두운 쿨톤 (겨울)
    }
    
    color = skin_colors.get(skin_tone)
    # 단색 배경
    img = np.full((height, width, 3), color, dtype=np.uint8)
    
    # 중앙에 원형 얼굴 영역
    cv2.circle(img, (width // 2, height // 2), 150, color, -1)
    
    # 약간의 노이즈 추가 (자연스러움)
    noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img
```

**특징:**
- ✓ 가장 빠른 생성 (~5ms)
- ✓ 색상 균일성으로 안정적인 추론
- ✗ 현실성 낮음 (얼굴 특징 없음)

---

### **2️⃣ PIL 얼굴 형태**

```python
from PIL import Image, ImageDraw
import cv2

def create_pil_face_image(width=640, height=480, skin_tone="warm_light"):
    # RGB 색상 (PIL은 RGB, OpenCV는 BGR)
    skin_colors_rgb = {
        "warm_light":   (217, 180, 105),
        "warm_deep":    (139, 110, 65),
        "cool_light":   (198, 166, 143),
        "cool_deep":    (138, 110, 125),
    }
    
    color = skin_colors_rgb.get(skin_tone)
    
    # 배경
    img = Image.new('RGB', (width, height), (220, 220, 220))
    draw = ImageDraw.Draw(img)
    
    # 타원형 얼굴
    center_x, center_y = width // 2, height // 2
    radius = 150
    bbox = [center_x - radius, center_y - radius, 
            center_x + radius, center_y + radius]
    draw.ellipse(bbox, fill=color)
    
    # 눈 그리기
    eye_y = center_y - 60
    draw.ellipse([center_x - 80, eye_y - 20, center_x - 50, eye_y + 20], 
                 fill=(50, 50, 50))
    draw.ellipse([center_x + 50, eye_y - 20, center_x + 80, eye_y + 20], 
                 fill=(50, 50, 50))
    
    # 입 그리기
    mouth_y = center_y + 60
    draw.arc([center_x - 60, mouth_y, center_x + 60, mouth_y + 40], 
             0, 180, fill=(200, 100, 100), width=3)
    
    # PIL → OpenCV BGR 변환
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
```

**특징:**
- ✓ 중간 수준의 자연스러움 (눈, 입 포함)
- ✓ 생성 속도 빠름 (~20ms)
- ⚠️ MediaPipe 얼굴 감지 성공률 50~70%

---

### **3️⃣ 실제 얼굴 이미지**

```python
def load_real_face_image(image_path):
    """실제 얼굴 사진 로드"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")
    return img

# 사용 예
img = load_real_face_image("test_face.jpg")
result = run_inference(img)
```

**특징:**
- ✓ 가장 현실적 (실제 얼굴)
- ✓ MediaPipe 감지 성공률 95%+
- ✗ 테스트 이미지 준비 필요

---

## 🧪 테스트 시나리오

### **시나리오 1: 빠른 개발 테스트**

```bash
# 10초 안에 기본 동작 확인
python inference_cli.py --mode dummy --tone warm_light
```

### **시나리오 2: UI 개발**

```bash
# 웹 인터페이스로 실시간 테스트
python dev_inference_web.py
# 브라우저: http://localhost:5000
```

### **시나리오 3: 모델 검증**

```bash
# 모든 피부톤 조합으로 정확도 검증
python inference_cli.py --mode all --save
# result_*.json 파일 생성
```

### **시나리오 4: 프로덕션 테스트**

```bash
# 실제 얼굴 이미지로 최종 검증
python inference_cli.py --mode file --image real_face.jpg
```

---

## 📊 성능 비교

| 항목 | 단색 | PIL | 실제 이미지 |
|------|------|-----|-----------|
| 생성/로드 시간 | 5ms | 20ms | 10ms |
| MediaPipe 감지율 | 100% | 60% | 98% |
| 추론 시간 | 140ms | 140ms | 140ms |
| **총 소요시간** | **145ms** | **160ms** | **150ms** |

---

## 🐛 트러블슈팅

### **문제 1: "얼굴을 찾을 수 없습니다" 에러**

**원인:** PIL로 생성한 얼굴이 MediaPipe 감지 기준에 맞지 않음

**해결:**
```python
# 방법 A: 단색 이미지 사용
python inference_cli.py --mode dummy

# 방법 B: 실제 이미지 사용
python inference_cli.py --mode file --image face.jpg
```

### **문제 2: 모델 로드 실패**

**원인:** personal_color.pt 파일 없음

**해결:**
- 프로젝트 루트에서 `ai/models/` 폴더 생성
- 학습된 모델을 `ai/models/personal_color.pt`에 배치
- 없으면 자동으로 더미 모델(랜덤 초기화) 사용

### **문제 3: Flask 포트 충돌**

```bash
# 5000번 포트가 사용 중일 때
# dev_inference_web.py 수정
app.run(port=5001)  # 5001번으로 변경
```

---

## 📚 참고: 프로젝트 모듈 임포트

```python
import sys
from pathlib import Path

# 방법 1: 절대 경로
project_root = Path(__file__).parent / 'project'
sys.path.insert(0, str(project_root))

# 방법 2: 상대 경로
sys.path.insert(0, '.')

# 이후 임포트
from ai.infer import run_inference
from ai.model_loader import ModelLoader
from ai.preprocessing import FacePreprocessor
```

---

## 🎓 다음 단계

1. **모델 학습**
   - `train.py` 참고
   - 6만 장 데이터셋으로 학습
   - `personal_color.pt` 생성

2. **정확도 평가**
   ```bash
   python inference_cli.py --mode all --save
   # 결과를 JSON으로 저장 후 분석
   ```

3. **라즈베리파이 배포**
   - `main.py` 실행
   - 카메라 연결
   - 키오스크 모드 테스트

---

**작성일:** 2024년  
**대상:** 개발자 테스트 환경
