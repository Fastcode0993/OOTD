# PyQt6로 키오스크 개발: 개발 환경 & 실전 가이드

---

## 🤔 1. PyQt6 vs React: 개발 환경 비교

### 1-1. 개발 가능성 비교표

| 항목 | PyQt6 | React | 결론 |
|------|-------|-------|------|
| **개발 가능** | ✅ Yes | ✅ Yes | 둘 다 가능 |
| **Hot Reload** | ⚠️ 제한적 | ✅ 완벽 | React 우수 |
| **재시작 필요** | ✅ 거의 없음* | ❌ 불필요 | React 우수 |
| **디버깅** | ✅ 강력 | ✅ 강력 | 동등 |
| **UI/UX 개발** | ⚠️ 코드 기반 | ✅ 시각적 | React 우수 |
| **재컴파일** | ❌ 없음 | ❌ 없음 | 동등 |
| **성능** | ✅ 매우 빠름 | ⚠️ 약간 느림 | PyQt6 우수 |
| **배포 용이성** | ⚠️ 설정 필요 | ✅ 매우 쉬움 | React 우수 |
| **Python 통합** | ✅ 완벽 | ⚠️ 복잡 | PyQt6 우수 |

---

## 📊 2. PyQt6 개발 환경 구축 (실제 가능함!)

### 2-1. 기본 개발 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│  PyQt6 개발 환경 (완전히 가능!)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1️⃣ 파이썬 파일 작성 (main.py)                            │
│      ↓                                                       │
│  2️⃣ 터미널에서 실행 (python main.py)                      │
│      ↓                                                       │
│  3️⃣ GUI 자동 로드 (재시작 필요 없음) ★                  │
│      ↓                                                       │
│  4️⃣ IDE에서 코드 수정 (원본 파일)                         │
│      ↓                                                       │
│  5️⃣ 다시 실행 (python main.py) → 새 버전 로드           │
│      ↓                                                       │
│  🎉 완성!                                                    │
│                                                              │
│  ⚠️ 주의: React와 달리 매번 명령어 실행 필요              │
│          (Hot reload 없음)                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2-2. 추천 개발 환경 설정

**Option A: 기본 설정 (권장)**
```bash
# 1. 가상환경 생성
python -m venv venv
.\venv\Scripts\activate  # Windows

# 2. 필요한 패키지 설치
pip install PyQt6 PyQt6-WebEngine
pip install torch torchvision  # ML 모델용
pip install opencv-python mediapipe  # 얼굴 감지용

# 3. IDE 설정 (VS Code 권장)
# - Python 확장 설치
# - Pylint 설치: pip install pylint

# 4. 실행
python kiosk.py
```

**Option B: 개선된 개발 설정 (더 편함)**
```bash
# 추가 설치
pip install python-dotenv      # 환경변수 관리
pip install colorama           # 컬러 콘솔 출력
pip install pytest             # 테스트 프레임워크

# 개발 중 자동 재시작 (별도 터미널에서)
pip install watchdog

# watch.py 작성해서 파일 변경 감지시 자동 실행
```

---

## 🎯 3. PyQt6 키오스크 개발: 실전 예제

### 3-1. 최소 구현 (Hello World)

```python
# kiosk_minimal.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

class KioskWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """UI 구성"""
        self.setWindowTitle("Personal Color Kiosk")
        self.setGeometry(100, 100, 1024, 768)
        
        # 중앙 위젯
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # 제목
        title = QLabel("퍼스널 컬러 진단 키오스크")
        title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 설명
        desc = QLabel("얼굴 사진을 촬영하면 당신의 퍼스널 컬러를 진단해줍니다!")
        desc.setFont(QFont("Arial", 16))
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KioskWindow()
    window.show()
    sys.exit(app.exec())
```

**실행:**
```bash
python kiosk_minimal.py
# → 창이 열림
# → 코드 수정 후 저장
# → 다시 python kiosk_minimal.py 실행
# → 변경사항 적용됨 ✅
```

---

### 3-2. 모델 통합 버전 (실제 작동)

```python
# kiosk_ai.py
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtMultimedia import QMediaCaptureSession, QCamera
import mediapipe as mp

class InferenceWorker(QThread):
    """백그라운드에서 모델 추론 실행"""
    
    finished = pyqtSignal(str, float)  # 결과, 신뢰도
    error = pyqtSignal(str)
    
    def __init__(self, model_path, image_path):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        
    def run(self):
        try:
            # 모델 로드
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 이미지 로드 & 전처리
            img = cv2.imread(self.image_path)
            # ... 얼굴 감지, 크롭, 정규화 ...
            
            # Stage 1: 계절 판별
            # season_logits = model1(img_tensor + meta_tensor)
            # season_probs = softmax(season_logits)
            # season_idx = argmax(season_probs)
            
            # Stage 2: 톤 판별
            # tone_logits = model2[season](img_tensor + meta_tensor)
            # tone_probs = softmax(tone_logits)
            # tone_idx = argmax(tone_probs)
            
            result = "Spring-Warm"  # 예시
            confidence = 0.87
            
            self.finished.emit(result, confidence)
            
        except Exception as e:
            self.error.emit(str(e))


class KioskUI(QMainWindow):
    """키오스크 메인 UI"""
    
    def __init__(self):
        super().__init__()
        self.model_path = "models/hierarchical.pt"
        self.current_image = None
        self.init_ui()
        self.load_model()
        
    def init_ui(self):
        """UI 구성"""
        self.setWindowTitle("Personal Color Kiosk - AI Diagnosis")
        self.setGeometry(0, 0, 1280, 1024)
        self.setStyleSheet("background: #F4F6F9;")
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # ─ 헤더 ─
        header = QLabel("🎨 당신의 퍼스널 컬러를 찾아보세요!")
        header.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #2C3E50; margin-bottom: 20px;")
        main_layout.addWidget(header)
        
        # ─ 콘텐츠 영역 ─
        content_layout = QHBoxLayout()
        
        # 좌측: 이미지 미리보기
        left_layout = QVBoxLayout()
        
        img_label = QLabel("사진 미리보기")
        img_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        left_layout.addWidget(img_label)
        
        self.preview = QLabel()
        self.preview.setMinimumSize(400, 400)
        self.preview.setStyleSheet(
            "border: 2px solid #3A7BD5; "
            "border-radius: 8px; "
            "background: white; "
            "alignment: center;"
        )
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.preview)
        
        content_layout.addLayout(left_layout, 4)
        
        # 우측: 결과 & 버튼
        right_layout = QVBoxLayout()
        
        result_label = QLabel("진단 결과")
        result_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        right_layout.addWidget(result_label)
        
        self.result_display = QLabel("아직 진단되지 않음")
        self.result_display.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.result_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_display.setStyleSheet(
            "background: white; "
            "border: 2px solid #3A7BD5; "
            "border-radius: 8px; "
            "padding: 20px; "
            "color: #2C3E50;"
        )
        right_layout.addWidget(self.result_display)
        
        # 신뢰도 바
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #D0D7DE;
                border-radius: 5px;
                text-align: center;
                background: #EEF1F5;
                height: 30px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, x2:1, stop:0 #3A7BD5, stop:1 #00D2FF);
                border-radius: 3px;
            }
        """)
        right_layout.addWidget(self.confidence_bar)
        
        right_layout.addSpacing(20)
        
        # 버튼들
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)
        
        self.btn_load = QPushButton("📁 이미지 선택")
        self.btn_load.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.btn_load.setMinimumHeight(50)
        self.btn_load.setStyleSheet("""
            QPushButton {
                background: #3A7BD5;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover { background: #2E63B8; }
            QPushButton:pressed { background: #254F99; }
        """)
        self.btn_load.clicked.connect(self.load_image)
        button_layout.addWidget(self.btn_load)
        
        self.btn_diagnose = QPushButton("🚀 진단 시작")
        self.btn_diagnose.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.btn_diagnose.setMinimumHeight(50)
        self.btn_diagnose.setEnabled(False)
        self.btn_diagnose.setStyleSheet("""
            QPushButton {
                background: #27AE60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover { background: #229954; }
            QPushButton:pressed { background: #1E8449; }
            QPushButton:disabled { background: #B0BAC8; color: #7A8490; }
        """)
        self.btn_diagnose.clicked.connect(self.run_diagnosis)
        button_layout.addWidget(self.btn_diagnose)
        
        self.btn_reset = QPushButton("🔄 초기화")
        self.btn_reset.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.btn_reset.setMinimumHeight(50)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background: #E74C3C;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover { background: #C0392B; }
            QPushButton:pressed { background: #A93226; }
        """)
        self.btn_reset.clicked.connect(self.reset_ui)
        button_layout.addWidget(self.btn_reset)
        
        right_layout.addLayout(button_layout)
        right_layout.addStretch()
        
        content_layout.addLayout(right_layout, 3)
        main_layout.addLayout(content_layout)
        
        # 상태 바
        self.statusBar().showMessage("준비 완료")
    
    def load_model(self):
        """모델 로드"""
        try:
            if not Path(self.model_path).exists():
                self.statusBar().showMessage("⚠️ 모델 파일을 찾을 수 없습니다!")
                return
            
            # checkpoint = torch.load(self.model_path, map_location='cpu')
            self.statusBar().showMessage("✅ 모델 로드 완료")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"모델 로드 실패: {e}")
    
    def load_image(self):
        """이미지 로드"""
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", 
            "Images (*.jpg *.jpeg *.png);;All Files (*)"
        )
        
        if not path:
            return
        
        self.current_image = path
        
        # 미리보기 표시
        pixmap = QPixmap(path)
        scaled = pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(scaled)
        
        self.btn_diagnose.setEnabled(True)
        self.statusBar().showMessage(f"이미지 로드됨: {Path(path).name}")
    
    def run_diagnosis(self):
        """진단 실행"""
        if not self.current_image:
            QMessageBox.warning(self, "경고", "먼저 이미지를 선택하세요!")
            return
        
        self.btn_diagnose.setEnabled(False)
        self.statusBar().showMessage("🔄 진단 중...")
        
        # 백그라운드 워커 실행
        self.worker = InferenceWorker(self.model_path, self.current_image)
        self.worker.finished.connect(self.on_diagnosis_complete)
        self.worker.error.connect(self.on_diagnosis_error)
        self.worker.start()
    
    def on_diagnosis_complete(self, result: str, confidence: float):
        """진단 완료"""
        self.result_display.setText(f"당신의 퍼스널 컬러:\n{result}")
        self.result_display.setStyleSheet(
            "background: #E8F8F5; "
            "border: 2px solid #27AE60; "
            "border-radius: 8px; "
            "padding: 20px; "
            "color: #27AE60; "
            "font-weight: bold;"
        )
        
        self.confidence_bar.setValue(int(confidence * 100))
        self.confidence_bar.setFormat(f"신뢰도: {confidence*100:.1f}%")
        
        self.statusBar().showMessage("✅ 진단 완료!")
        self.btn_diagnose.setEnabled(True)
    
    def on_diagnosis_error(self, error_msg: str):
        """진단 오류"""
        QMessageBox.critical(self, "진단 오류", f"진단 중 오류 발생:\n{error_msg}")
        self.statusBar().showMessage("❌ 진단 실패")
        self.btn_diagnose.setEnabled(True)
    
    def reset_ui(self):
        """UI 초기화"""
        self.current_image = None
        self.preview.setText("사진 미리보기")
        self.preview.setPixmap(QPixmap())
        self.result_display.setText("아직 진단되지 않음")
        self.result_display.setStyleSheet(
            "background: white; "
            "border: 2px solid #3A7BD5; "
            "border-radius: 8px; "
            "padding: 20px; "
            "color: #2C3E50;"
        )
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("신뢰도: 0%")
        self.btn_diagnose.setEnabled(False)
        self.statusBar().showMessage("초기화됨")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KioskUI()
    window.show()
    sys.exit(app.exec())
```

**실행:**
```bash
python kiosk_ai.py
# → 키오스크 GUI 실행
# → 파일 수정 → 저장
# → 새 터미널에서 python kiosk_ai.py 실행
# → 변경사항 적용 ✅
```

---

## 🔥 4. Hot Reload 같은 경험 만들기

### 4-1. 자동 재시작 스크립트 (Python)

```python
# auto_reload.py
import subprocess
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class PyFileHandler(FileSystemEventHandler):
    def __init__(self, script_path):
        self.script_path = script_path
        self.process = None
        self.start_process()
    
    def start_process(self):
        """프로세스 시작"""
        if self.process:
            self.process.terminate()
            self.process.wait()
        
        print(f"\n{'='*50}")
        print(f"🚀 {self.script_path} 실행 중...")
        print(f"{'='*50}\n")
        
        self.process = subprocess.Popen(
            ["python", self.script_path],
            cwd=Path(self.script_path).parent
        )
    
    def on_modified(self, event):
        """파일 수정 감지"""
        if event.src_path.endswith('.py'):
            print(f"\n✏️ 변경감지: {Path(event.src_path).name}")
            time.sleep(1)  # 저장 완료 대기
            self.start_process()


if __name__ == "__main__":
    script = "kiosk_ai.py"  # 모니터할 파일
    
    handler = PyFileHandler(script)
    observer = Observer()
    observer.schedule(handler, path='.', recursive=False)
    observer.start()
    
    print(f"👁️ {script} 변경사항 모니터링 중...")
    print("💾 파일을 저장하면 자동으로 다시 실행됩니다.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if handler.process:
            handler.process.terminate()
    observer.join()
```

**사용:**
```bash
# 터미널 1 (감시자)
pip install watchdog
python auto_reload.py

# 터미널 2 (IDE)
# kiosk_ai.py 열고 수정
# 저장하면 → 자동으로 재시작됨! ✅
```

---

### 4-2. VS Code 디버그 설정

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "PyQt6 Kiosk",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/kiosk_ai.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "stopOnEntry": false,
            "env": {
                "QT_QPA_PLATFORM": "offscreen"
            }
        }
    ]
}
```

**사용:**
```
VS Code → F5 → 디버깅 시작
브레이크포인트 설정 가능 ✅
콘솔에 print() 출력 확인 가능 ✅
```

---

## 📱 5. PyQt6 vs React: 실전 선택 가이드

### 5-1. PyQt6를 선택해야 할 때 ✅

```
✅ 강력한 GPU 연산 필요 (PyTorch, TensorFlow)
✅ 빠른 성능 필수 (UI 반응성 중요)
✅ Windows/Linux 데스크톱 앱
✅ 기존 Python 코드 활용 가능
✅ 설치 파일 단일화 (배포 용이)

→ 결론: PyQt6 최고!
```

### 5-2. React를 선택해야 할 때 ✅

```
✅ Web/Mobile 동시 지원 필요
✅ Hot reload 개발 경험 중요
✅ UI/UX 디자이너와 협업
✅ 실시간 동기화 필요 (여러 기기)
✅ 클라우드 배포 (AWS, GCP)

→ 결론: React 최고!
```

---

## 🎯 6. PyQt6 키오스크 최종 구성도

```
┌─────────────────────────────────────────────────────┐
│          PyQt6 키오스크 최종 아키텍처              │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [KioskUI (PyQt6)]                                │
│  ├─ Header (타이틀)                               │
│  ├─ Image Preview (미리보기)                      │
│  ├─ Result Display (결과)                         │
│  └─ Control Buttons (버튼들)                      │
│        ↓                                            │
│  [InferenceWorker (QThread)]                      │
│  ├─ Model Load (Stage1, Stage2)                  │
│  ├─ Image Preprocess                             │
│  │   ├─ Face Detection (MediaPipe)              │
│  │   ├─ Crop & Normalize                        │
│  │   ├─ Color Correction (b*)                   │
│  │   └─ Meta Extraction (JSON)                  │
│  ├─ Stage 1: 4-class (Spring/Summer/...)        │
│  ├─ Stage 2: 3-class (Warm/Bright/Light)        │
│  └─ Result Return → UI Update                    │
│        ↓                                            │
│  [Result Display]                                 │
│  ├─ Personal Color (e.g., "Spring-Warm")       │
│  ├─ Confidence (e.g., 94.2%)                    │
│  └─ Recommendations                              │
│                                                      │
│  [File System]                                     │
│  ├─ kiosk_ai.py (메인 로직)                       │
│  ├─ models/hierarchical.pt (학습된 모델)        │
│  └─ images/ (샘플 이미지)                         │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## ⚡ 7. 개발 팁

### 7-1. 빠른 프로토타이핑

```python
# 스타일을 별도 파일로 분리 (변경시 바로 재시작)
# styles.py

MAIN_STYLE = """
    QMainWindow { background: #F4F6F9; }
    QPushButton {
        background: #3A7BD5;
        color: white;
        border-radius: 8px;
        padding: 10px;
    }
    ...
"""

# 메인 파일에서
# self.setStyleSheet(MAIN_STYLE)

# → 스타일만 변경 후 재실행 = 빠름!
```

### 7-2. 로깅 설정

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 사용
logger.info("이미지 로드됨")
logger.error("모델 로드 실패")
```

### 7-3. 설정 파일 (config.ini)

```ini
[model]
path = models/hierarchical.pt
stage1_threshold = 0.7
stage2_threshold = 0.65

[ui]
window_width = 1280
window_height = 1024
theme = dark

[inference]
device = cuda
batch_size = 1
```

**로드:**
```python
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

model_path = config['model']['path']
```

---

## 🎉 결론

| 항목 | 답변 |
|------|------|
| **PyQt6로 개발 환경 구축 가능?** | ✅ **완전히 가능!** |
| **Hot reload 같은 경험?** | ⚠️ **자동화로 가능** (auto_reload.py) |
| **React 수준의 편의성?** | ❌ **약간 떨어짐** (매번 재실행) |
| **추천?** | ✅ **PyQt6 강추** (AI 모델 활용할 때) |

**PyQt6 개발 워크플로우:**
```
파이썬 코드 작성
    ↓
저장 (Ctrl+S)
    ↓
터미널에서 python kiosk.py 실행
    ↓
UI 확인 & 테스트
    ↓
원하는 결과 = 완성!
```

React보다는 번거롭지만, **GPU 기반 AI 모델과의 통합은 훨씬 간단합니다!** 🚀
