# 퍼스널 컬러 진단 키오스크 — 구현 설명서

---

## 목차

1. [프로젝트 전체 구조 요약](#1-프로젝트-전체-구조-요약)
2. [모듈별 기술 선택 근거](#2-모듈별-기술-선택-근거)
3. [UI/UX 디자인 설계](#3-uiux-디자인-설계)
4. [라즈베리파이5 성능 최적화 전략](#4-라즈베리파이5-성능-최적화-전략)
5. [데이터 흐름 전체 다이어그램](#5-데이터-흐름-전체-다이어그램)
6. [설치 및 실행 방법](#6-설치-및-실행-방법)
7. [트러블슈팅 가이드](#7-트러블슈팅-가이드)

---

## 1. 프로젝트 전체 구조 요약

```
project/
├─ kiosk/
│  ├─ main.py                  ← 진입점: 서버 서브프로세스 실행 + PyQt6 앱 시작
│  ├─ camera.py                ← QThread 기반 UVC 카메라 캡처 루프
│  ├─ ui.py                    ← QMainWindow + QStackedWidget 화면 전환 관리자
│  └─ screens/
│     ├─ idle_screen.py        ← 대기 화면 (터치 시작)
│     ├─ guide_screen.py       ← 카메라 미리보기 + 얼굴 정렬 가이드
│     ├─ analysis_screen.py    ← AI 분석 중 로딩 애니메이션
│     ├─ result_screen.py      ← 진단 결과 + 색상 팔레트
│     ├─ recommendation_screen.py ← 카테고리별 추천 아이템
│     └─ qr_screen.py          ← QR 코드 저장 + 자동 홈 복귀
├─ server/
│  ├─ app.py                   ← FastAPI 앱, 라이프스팬 훅, DB 초기화
│  ├─ routes/analyze.py        ← POST /api/v1/analyze 엔드포인트
│  ├─ services/recommendation.py ← SQLite 조회/저장 서비스 레이어
│  └─ utils/helpers.py         ← 공통 유틸리티
├─ ai/
│  ├─ model_loader.py          ← ModelLoader 싱글톤, TorchScript 변환
│  ├─ preprocessing.py         ← MediaPipe FaceMesh → Crop → Tensor
│  ├─ infer.py                 ← torch.no_grad() 추론, TOP3 결과 반환
│  └─ train.py                 ← 참고용 학습 스크립트
├─ db/
│  └─ schema.sql               ← SQLite 스키마 + 시드 데이터 (12종 컬러)
└─ requirements.txt
```

### 화면 흐름

```
[대기 화면]
    ↓ 터치
[촬영 가이드]  ←──────────────────────────────┐
    ↓ 촬영 버튼                               │ 다시 촬영
[분석 화면 (로딩)]                             │
    ↓ API 응답                                │
[결과 화면]  ────────────────────────────────→┘
    ↓ 다음
[추천 아이템]
    ↓ QR 저장
[QR 화면]
    ↓ 30초 자동 귀환 or 홈 버튼
[대기 화면]
```

---

## 2. 모듈별 기술 선택 근거

### 2-1. Kiosk UI — PyQt6

**왜 PyQt6인가?**

| 대안 | 탈락 이유 |
|------|----------|
| Tkinter | 터치 이벤트 처리·고DPI 스케일링이 부족하고 커스텀 페인팅이 번거로움 |
| Kivy | 라즈베리파이 ARM64 에서 의존성 충돌이 잦고 GPU 없을 때 성능이 나쁨 |
| Electron/웹 | Node.js 런타임 오버헤드, OpenCV 연동이 복잡 |
| **PyQt6** | ARM64 네이티브 빌드 지원, QThread·pyqtSignal 로 카메라/AI 스레드 안전 분리, QPainter 로 타원 오버레이·스피너 등 커스텀 그래픽 직접 구현 가능 |

**핵심 설계 패턴:**
- `QStackedWidget` — 6개 화면을 하나의 위젯 스택으로 관리. 화면 전환 시 객체 재생성 없이 인덱스만 변경 → 메모리·렌더링 효율적
- `QThread` + `pyqtSignal` — 카메라 캡처(`CameraThread`)와 API 호출(`_AnalyzeWorker`)을 메인 GUI 스레드와 완전히 분리, UI 블로킹 제로
- `QPropertyAnimation` — CSS transition 없이 Qt 네이티브 애니메이션으로 힌트 레이블 펄스 효과 구현

### 2-2. 카메라 제어 — OpenCV + V4L2

```python
# camera.py 핵심 설정
self._cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)   # 버퍼 2프레임 → 지연 최소화
```

- **V4L2 백엔드** 명시: 라즈베리파이 리눅스에서 UVC 카메라와 가장 안정적으로 통신
- **캡처 해상도 1920×1080** (4K 원본에서 다운샘플): IMX415 센서의 넓은 색영역을 살리면서 처리 부하 절감
- **미리보기는 960×540** (절반)으로 스케일: PyQt6 위젯에 렌더링할 때 CPU 복사 비용 감소
- `CAP_PROP_BUFFERSIZE=2`: 카메라 드라이버 내부 버퍼를 2프레임으로 제한 → 오래된 프레임 대신 최신 프레임 표시

### 2-3. 얼굴 전처리 — MediaPipe FaceMesh

**왜 MediaPipe인가?**

MediaPipe FaceMesh는 468개 랜드마크를 CPU 전용으로도 40ms 이내에 처리합니다. OpenCV DNN 기반 face detector 대비 정확도와 속도 모두 우수하며, 특히 조명 변화에 강합니다.

```python
# preprocessing.py
self._face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,    # 단일 프레임 → 영상 추적 오버헤드 제거
    max_num_faces=1,           # 1명만 → 불필요한 탐색 생략
    refine_landmarks=False,    # 눈/입술 세밀 랜드마크 불필요 → 속도 우선
    min_detection_confidence=0.6,
)
```

**얼굴 Crop 로직:**
1. 468개 랜드마크의 min/max x, y 로 바운딩박스 계산
2. 가로 20%, 세로 25% 패딩 추가 → 이마·턱 포함
3. CLAHE(Contrast Limited Adaptive Histogram Equalization) 로 조명 편차 보정 (LAB L채널)
4. ImageNet 정규화 후 `(1, 3, 224, 224)` 텐서 반환

### 2-4. AI 추론 — PyTorch + MobileNetV3-Small

**모델 선택: MobileNetV3-Small**

| 모델 | Top-1 정확도 | 연산량(MFLOPs) | 라즈베리파이5 추론 |
|------|-------------|--------------|----------------|
| ResNet-50 | 76.1% | 4,100 | ~2,000ms |
| EfficientNet-B0 | 77.1% | 390 | ~600ms |
| **MobileNetV3-Small** | 67.7% | **56** | **~150ms** |

퍼스널 컬러 12분류는 ImageNet 1000분류보다 훨씬 단순하므로 MobileNetV3-Small의 정확도로 충분하며, 라즈베리파이5에서 약 150ms 추론이 가능합니다.

**추론 최적화 코드:**
```python
# infer.py
with torch.no_grad():          # ① 역전파 그래프 미생성 → 메모리 절약
    logits = model(det.tensor)
    probs  = F.softmax(logits, dim=1)[0]
```

**TorchScript 변환 (model_loader.py):**
```python
model = torch.jit.script(model)  # ② JIT 컴파일 → ARM CPU 최적화 코드 생성
```

TorchScript는 Python GIL 없이 실행 가능한 IR로 컴파일되어, 라즈베리파이5의 4코어를 더 효율적으로 활용합니다.

**다단계 학습 구조 (train.py):**
```
에폭 1~3:  분류 헤드만 학습 (backbone 동결) → 빠른 수렴
에폭 4~30: 전체 레이어 파인튜닝 (lr × 0.1) → 정교한 피부톤 특징 학습
```
Label Smoothing(0.1)과 Cosine Annealing LR 스케줄러를 적용해 과적합 방지.

### 2-5. Backend — FastAPI + SQLite

**FastAPI 선택 이유:**
- `async` 기반 → 단일 프로세스에서 I/O 대기 없이 다음 요청 처리
- `python-multipart` 로 이미지 파일 업로드 처리 내장
- 자동 OpenAPI 문서 (`/docs`)로 개발·디버깅 편의

**SQLite 선택 이유:**
- 라즈베리파이5 내장 스토리지에서 별도 DB 서버 불필요
- WAL(Write-Ahead Logging) 모드: 읽기/쓰기 동시성 향상
- 키오스크 특성상 단일 사용자 순차 세션 → PostgreSQL 수준의 동시성 불필요

**API Key 인증:**
```python
# FastAPI Depends 패턴으로 모든 엔드포인트에 자동 적용
def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, ...)
```

---

## 3. UI/UX 디자인 설계

### 3-1. 디자인 컨셉: "럭셔리 뷰티 스튜디오"

10.1인치 터치 디스플레이(1280×800)에서 사용자가 자신의 퍼스널 컬러를 진단받는 경험을 **고급 뷰티 브랜드 팝업스토어**처럼 연출했습니다.

### 3-2. 컬러 시스템

```
--dark-bg:    #1A1512   (딥 샴페인 블랙  — 배경 기조)
--dark-card:  #231E1A   (카드 배경)
--gold:       #C9A96E   (주 강조색 — 럭셔리 골드)
--rose-gold:  #C4858A   (보조 강조색)
--cream:      #F5EFE6   (주요 텍스트)
--success:    #7EC8A4   (얼굴 인식 성공 상태)
```

배경에 어두운 샴페인 계열을 사용함으로써:
1. 피부톤 컬러 팔레트가 선명하게 대비되어 직관적으로 인식됨
2. 화면 외부의 키오스크 하우징(보통 검정/실버)과 자연스럽게 연결됨
3. 골드 강조색이 럭셔리 브랜드 아이덴티티를 강조

### 3-3. 타이포그래피

| 용도 | 폰트 | 이유 |
|------|------|------|
| 디스플레이 타이틀 | Georgia (세리프) | 우아함·클래식 뷰티 브랜드 연상 |
| UI 본문/안내 | Noto Sans KR | 한글 가독성 최우선, ARM 렌더링 안정 |
| 강조 레이블 | letter-spacing 4~6px | 고급 패키지 인쇄물 느낌 |

### 3-4. 터치 최적화 (10.1인치, 1280×800)

| 요소 | 크기 | 근거 |
|------|------|------|
| 주요 CTA 버튼 | 340×72px 이상 | 손가락 터치 최소 44px, 여유 있게 72px |
| 탭 버튼 | 높이 46px | 손가락 너비 고려 |
| 카드 간격 | 14px 이상 | 오터치 방지 |
| 폰트 최소 | 13px | 시야거리 30~50cm 기준 |

### 3-5. 화면별 UX 설계 원칙

**대기 화면 (IdleScreen)**
- 풀스크린 터치: 버튼 뿐만 아니라 화면 어디를 눌러도 시작 → 진입 장벽 최소화
- 힌트 레이블 펄스 애니메이션: 조작 가능함을 무의식적으로 인지

**촬영 가이드 (GuideScreen)**
- 얼굴 타원 오버레이: 어디에 서야 하는지 즉시 이해
  - 얼굴 미감지: 점선 골드 타원
  - 얼굴 감지 성공: 실선 그린(#7EC8A4) 타원 + 상태 메시지 변경
- 어두운 비네팅 마스크: 중앙(얼굴 영역)에 시선 집중 유도

**분석 화면 (AnalysisScreen)**
- 골드 아크 스피너: 단순 회전이 아닌 270° 부채꼴 → 모던한 느낌
- 4단계 진행 메시지: "얼굴 특징점 분석 → 피부톤 추출 → AI 추론 → 결과 준비"
  - 사용자가 기다리는 동안 무슨 일이 일어나는지 알 수 있어 불안감 해소

**결과 화면 (ResultScreen)**
- 퍼스널 컬러 이름을 48px 대형 텍스트로 → 즉각적인 정보 인지
- 색상 팔레트 스와치 5개: 색이름 없이 시각적으로 어울리는 컬러 전달
- TOP3 카드: 1위 골드 강조, 나머지 그레이 → 시각적 위계 명확

**추천 화면 (RecommendationScreen)**
- 카테고리 탭 (패션/메이크업/헤어/인테리어): 수평 스크롤 없이 탭 전환
- 아이템 카드: 색상 스와치 + 아이템명 + 컬러명 + 팁 → 한눈에 파악
- 3열 그리드: 1280px 폭에서 각 카드 280px, 3개 = 840px + 여백 균형

---

## 4. 라즈베리파이5 성능 최적화 전략

### 4-1. CPU/메모리 최적화 총괄

```
┌─────────────────────────────────────────────────────┐
│  라즈베리파이5  (ARM Cortex-A76 × 4, 8GB LPDDR4X)  │
│                                                     │
│  Core 0-1: PyQt6 GUI + CameraThread                │
│  Core 2:   uvicorn FastAPI (single worker)          │
│  Core 3:   _AnalyzeWorker (AI 추론 시에만 활성화)  │
└─────────────────────────────────────────────────────┘
```

### 4-2. 카메라 최적화

```python
# camera.py
self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 내부 버퍼 최소화

# 미리보기 다운샘플: 1920×1080 → 960×540 (CPU 복사 비용 75% 절감)
preview = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
```

- `INTER_AREA`: 다운샘플링 시 가장 품질이 좋은 보간법 (모아레 없음)
- QThread에서 프레임 처리: GUI 스레드 블로킹 없이 30fps 유지

### 4-3. MediaPipe 최적화

```python
FaceMesh(
    static_image_mode=True,    # 트래킹 알고리즘 비활성화 → 단일 프레임 최적
    refine_landmarks=False,    # 478 → 468개 랜드마크로 줄여 연산 절감
    max_num_faces=1,           # 1명만 탐지 → 조기 종료
)
```

- **싱글톤 패턴**: `FacePreprocessor._instance` — MediaPipe 세션을 매 요청마다 생성하지 않고 재사용
- 분석 요청은 촬영 버튼을 눌렀을 때만 발생 (영상 스트림 실시간 처리 아님) → CPU 부하 집중 시점 예측 가능

### 4-4. PyTorch 추론 최적화

```python
# model_loader.py
model = torch.jit.script(model)   # TorchScript JIT 컴파일

# infer.py
with torch.no_grad():             # 역전파 그래프 미생성 → ~40% 메모리 절약
    logits = model(det.tensor)    # (1, 12) 출력
```

- **싱글톤 ModelLoader**: 앱 시작 시 1회만 로드, 이후 메모리 상주
- **CPU 스레드 수 제한** (필요 시 추가 가능):
  ```python
  torch.set_num_threads(2)  # 코어 2개로 제한, GUI 스레드에 여유 부여
  ```
- `torch.float32` 유지: 라즈베리파이5는 float16 SIMD 가속이 제한적이므로 float32가 더 안정적

### 4-5. FastAPI 서버 최적화

```python
# app.py
uvicorn.run(..., workers=1)       # 단일 워커 — 포크 오버헤드 없음
```

- `asynccontextmanager` 라이프스팬: 서버 시작 시 DB 초기화 + 모델 pre-load → 첫 요청 지연 제거
- SQLite WAL 모드: 읽기 쿼리와 쓰기 쿼리가 동시 실행 가능
- httpx 동기 클라이언트 (`_AnalyzeWorker` QThread 내부): PyQt6 이벤트 루프와 asyncio 이벤트 루프 충돌 회피

### 4-6. 메모리 사용량 예측

| 컴포넌트 | 예상 메모리 |
|---------|------------|
| PyQt6 UI + 화면 스택 | ~80MB |
| OpenCV 카메라 버퍼 | ~30MB |
| MediaPipe FaceMesh | ~50MB |
| MobileNetV3-Small 모델 | ~8MB |
| FastAPI + SQLite | ~60MB |
| **합계** | **~228MB** |

8GB RAM 대비 약 3% 사용 → 여유롭게 운용 가능.

### 4-7. 전체 처리 시간 예측 (라즈베리파이5 기준)

```
카메라 스냅샷 캡처    :  ~33ms  (30fps 기준)
JPEG 인코딩           :  ~20ms
httpx POST 전송       :  ~5ms   (loopback)
MediaPipe FaceMesh    :  ~80ms
CLAHE 피부 보정       :  ~10ms
PyTorch 추론          :  ~150ms (MobileNetV3-Small, TorchScript)
DB 저장 + 조회        :  ~15ms
응답 전송             :  ~5ms
────────────────────────────────
총 분석 소요 시간      :  약 320ms ~ 500ms
```

사용자 체감 대기시간 0.5초 이내 — 분석 화면의 로딩 애니메이션이 자연스럽게 커버.

---

## 5. 데이터 흐름 전체 다이어그램

```
[USB U30CAM]
     │ V4L2 30fps
     ▼
[CameraThread]  ──frame_ready──▶  [GuideScreen 미리보기]
     │
     │ take_snapshot() (촬영 버튼 클릭)
     ▼
[BGR ndarray 1920×1080]
     │
     ▼
[_AnalyzeWorker (QThread)]
     │ cv2.imencode → JPEG bytes
     │ httpx.post /api/v1/analyze
     ▼
[FastAPI /analyze]
     │ verify_api_key
     │ cv2.imdecode → BGR ndarray
     ▼
[ai/infer.run_inference()]
     │
     ├─▶ [FacePreprocessor.process()]
     │        MediaPipe FaceMesh
     │        Crop + CLAHE + Normalize
     │        → Tensor (1,3,224,224)
     │
     └─▶ [ModelLoader.model(tensor)]
              torch.no_grad()
              TorchScript forward
              softmax → TOP3
     │
     ▼
[recommendation.save_diagnosis()]  →  [SQLite kiosk.db]
[recommendation.get_color_type_info()]
[recommendation.get_recommendations()]
     │
     ▼
[JSON Response]
     │
     ▼
[_AnalyzeWorker.finished signal]
     │
     ▼
[KioskWindow._on_analyze_finished()]
     │
     ├─▶ [ResultScreen.set_result()]   →  퍼스널 컬러 + 팔레트 표시
     ├─▶ [RecommendationScreen.set_data()] → 추천 아이템 표시
     └─▶ [QRScreen.set_session()]      →  QR 코드 생성 + 자동 홈 복귀
```

---

## 6. 설치 및 실행 방법

### 6-1. 환경 준비 (라즈베리파이5 기준)

```bash
# 1. 시스템 패키지
sudo apt update && sudo apt install -y \
    python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 \
    libxcb-xinerama0 libxcb-cursor0 \
    fonts-noto-cjk \
    v4l-utils

# 2. Python 가상환경
python3 -m venv .venv
source .venv/bin/activate

# 3. 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 4. AI 모델 디렉터리 생성
mkdir -p ai/models
# personal_color.pt 파일을 ai/models/ 에 배치
# (없으면 랜덤 초기화 더미 모델로 동작)
```

### 6-2. 실행

```bash
# 전체화면 키오스크 모드
python -m kiosk.main

# 개발용 창 모드
python -m kiosk.main --windowed

# 서버만 단독 실행 (API 테스트)
python server/app.py

# API 테스트
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "x-api-key: kiosk-dev-key-2024" \
  -F "image_file=@test_face.jpg" \
  -F "session_id=test-001"
```

### 6-3. 환경 변수

```bash
KIOSK_API_KEY=my-secret-key   # API 인증 키 (기본: kiosk-dev-key-2024)
PORT=8000                      # FastAPI 포트 (기본: 8000)
```

### 6-4. 자동 시작 설정 (systemd)

```ini
# /etc/systemd/system/kiosk.service
[Unit]
Description=Personal Color Kiosk
After=graphical.target

[Service]
User=pi
WorkingDirectory=/home/pi/project
ExecStart=/home/pi/project/.venv/bin/python -m kiosk.main
Restart=always
RestartSec=5
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/pi/.Xauthority

[Install]
WantedBy=graphical.target
```

```bash
sudo systemctl enable kiosk
sudo systemctl start kiosk
```

---

## 7. 트러블슈팅 가이드

| 증상 | 원인 | 해결 |
|------|------|------|
| 카메라 화면 안 나옴 | V4L2 장치 없음 | `v4l2-ctl --list-devices` 로 인덱스 확인 후 `CameraThread(camera_index=N)` 변경 |
| 얼굴 인식 안 됨 | 조명 부족 / 각도 | `min_detection_confidence=0.5` 로 낮추거나 조명 보강 |
| AI 추론 느림 | 모델 미변환 | TorchScript 실패 시 eager mode 로 fallback, `torch.set_num_threads(3)` 추가 |
| Qt 폰트 깨짐 (한글) | Noto Sans KR 미설치 | `sudo apt install fonts-noto-cjk` 실행 |
| 서버 포트 충돌 | 기존 프로세스 잔존 | `sudo fuser -k 8000/tcp` 실행 후 재시작 |
| QR 생성 실패 | qrcode 미설치 | `pip install qrcode[pil]` |
| 전체화면에서 커서 보임 | Qt 커서 설정 무시 | `DISPLAY=:0 unclutter -idle 1 &` 를 시작 스크립트에 추가 |

---

*문서 버전: 1.0 | 작성일: 2024 | 대상 플랫폼: Raspberry Pi 5 + ED-HMI3010-101C (1280×800)*
