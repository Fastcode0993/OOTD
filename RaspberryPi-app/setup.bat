@echo off
REM ================================================================
REM  setup.bat — Windows (노트북/개발환경) 의존성 설치
REM  사용법: setup.bat
REM          setup.bat --venv   (가상환경 생성 후 설치)
REM ================================================================

setlocal

set PYTHON=python
set PIP=%PYTHON% -m pip

REM ── 가상환경 옵션 ──────────────────────────────────────────────
if "%1"=="--venv" (
    echo [1/5] 가상환경 생성 중...
    %PYTHON% -m venv .venv
    call .venv\Scripts\activate.bat
    set PIP=pip
)

echo.
echo ================================================================
echo   Personal Color Kiosk — Windows 의존성 설치
echo ================================================================

REM ── pip 업그레이드 ─────────────────────────────────────────────
echo [Step 1] pip 업그레이드...
%PIP% install --upgrade pip

REM ── protobuf 먼저 고정 (mediapipe 0.10.14 호환) ───────────────
REM    주의: tensorflow가 설치된 환경에서는 TF가 protobuf>=6 요구하여
REM          충돌하지만, 키오스크 코드는 tensorflow를 사용하지 않음.
REM          preprocessing.py 의 stub 우회로 양쪽 공존 가능.
echo [Step 2] protobuf 버전 고정 (mediapipe 호환)...
%PIP% install "protobuf>=4.25.0,<5.0.0"

REM ── MediaPipe (solutions API 마지막 지원 버전) ─────────────────
echo [Step 3] MediaPipe 설치 (0.10.14)...
%PIP% install "mediapipe==0.10.14"

REM ── 나머지 의존성 ─────────────────────────────────────────────
echo [Step 4] 나머지 패키지 설치...
%PIP% install ^
    "torch>=2.6.0" ^
    "torchvision>=0.21.0" ^
    "opencv-python>=4.9.0" ^
    "numpy>=1.26.0" ^
    "Pillow>=10.3.0" ^
    "fastapi>=0.111.0" ^
    "uvicorn[standard]>=0.29.0" ^
    "python-multipart>=0.0.9" ^
    "aiofiles>=23.2.1" ^
    "httpx>=0.27.0" ^
    "qrcode[pil]>=7.4.2" ^
    "scikit-learn>=1.4.0" ^
    "pandas>=2.2.0"

REM ── PyQt6 (Windows pip 지원) ───────────────────────────────────
echo [Step 5] PyQt6 설치...
%PIP% install "PyQt6>=6.6.0"

REM ── 설치 확인 ─────────────────────────────────────────────────
echo.
echo [확인] MediaPipe FaceMesh 동작 테스트...
%PYTHON% -c "from ai.preprocessing import FacePreprocessor, _MEDIAPIPE_OK; print('  MediaPipe:', 'OK' if _MEDIAPIPE_OK else 'Fallback(Haarcascade)')"

echo.
echo ================================================================
echo   설치 완료!
echo   실행: python testfolder/dev_kiosk.py
echo ================================================================
endlocal
