# OOTD - Personal Color Diagnosis System

ED-HMI3010-101C 기반 라즈베리 파이 5 통합 퍼스널 컬러 진단 시스템

## 시스템 개요

이 시스템은 산업용 HMI 단말기 (ED-HMI3010-101C) 에 내장된 라즈베리 파이 5 8GB 를 활용하여 퍼스널 컬러를 진단하는 AI 기반 솔루션입니다.

### 하드웨어 사양
- **디스플레이**: 10.1 인치 터치스크린 (1280x800)
- **카메라**: USB U30CAM (IMX415, 4K)
- **프로세서**: Raspberry Pi 5 8GB
- **인터페이스**: USB 3.0

## 기능

### 1. 카메라 최적화
- 4K 해상도 (3840x2160) 로 원본 프레임 캡처
- 추론용 (224x224) 과 화면 출력용 (Low-res) 으로 분리 처리
- HMI 성능 하락 방지용 비동기 프레임 큐

### 2. 얼굴 분석 (MediaPipe Face Mesh)
- 양볼과 이마 좌표 추출
- 해당 구역의 평균 RGB/Lab 값 계산
- 얼굴 오вал, 양볼, 이마 영역 시각화

### 3. 멀티모달 추론
- **CNN**: EfficientNet-B0 를 활용한 이미지 기반 계절 분류
- **MLP**: 피부색 수치 (Lab) 기반 계절 분류
- 결합 예측 (이미지 60% + 피부색 40%)

### 4. 확률 해석 규칙
- Confidence ≥ 0.8: 진단 완료
- Confidence < 0.6: RE-TAKE 메시지
- 0.6 ≤ Confidence < 0.8: 재촬영 권장

### 5. HMI 프론트엔드
- 터치 최적화된 버튼 디자인 (최소 60px)
- 계절별 배경색 적용 (봄-파스텔 옐로우, 여름-파스텔 블루 등)
- 도넛 차트로 신뢰도 시각화
- Session ID 기반 결과 관리

## 프로젝트 구조

```
program/
├── main_hmi.py              # 메인 애플리케이션
├── camera_manager.py        # 카메라 관리 클래스
├── api_server.py            # FastAPI 백엔드 서버
├── frontend/
│   └── index.html           # React 프론트엔드
├── requirements.txt         # 의존성 목록
├── constants.py             # 상수 정의
├── server_preprocess.py     # MediaPipe 전처리
├── Dockerfile               # Docker 이미지
└── README.md               # 이 파일
```

## 설치 및 실행

### 1. 의존성 설치

```bash
cd program
pip install -r requirements.txt
```

### 2. 시스템 실행

```bash
python main_hmi.py
```

### 3. 웹 프론트엔드 실행

```bash
# API 서버 먼저 시작
cd program
python api_server.py

# 별도 브라우저에서
http://localhost:8000/frontend/index.html
```

### 4. Docker 로 배포 (Debian 11)

```bash
# Docker 이미지 빌드
docker build -t hmi-color-diagnosis .

# 컨테이너 실행
docker run -p 8000:8000 hmi-color-diagnosis
```

## 키보드 단축키

| 키 | 기능 |
|---|---|
| F1 | 진단 시작 |
| F2 | 결과 보기 |
| F3 | 카메라 미리보기 |
| ESC | 종료 |

## API 엔드포인트

### POST /analyze
이미지 분석 및 계절 톤 추론

**Request:**
```
Content-Type: multipart/form-data
file: 이미지 파일
session_id: 세션 ID (선택)
```

**Response:**
```json
{
  "session_id": "sess_xxx",
  "season": "Spring",
  "confidence": 0.85,
  "season_korean": "봄",
  "palette": [...],
  "message": "진단 완료",
  "re_take": false
}
```

### GET /results/{session_id}
특정 세션의 진단 결과 조회

### GET /results
최근 진단 결과 목록 조회

## 데이터베이스

SQLite 로 진단 결과를 저장합니다.

```sql
CREATE TABLE diagnosis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    season TEXT NOT NULL,
    confidence REAL NOT NULL,
    cheek_rgb TEXT,
    cheek_lab TEXT,
    forehead_rgb TEXT,
    forehead_lab TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 기술 스택

- **Python 3.12+**
- **OpenCV 4.8+**
- **MediaPipe 0.10+**
- **FastAPI 0.104+**
- **PyTorch 2.0+**
- **EfficientNet-B0**

## 라이선스

MIT License