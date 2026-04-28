#!/usr/bin/env bash
# ================================================================
#  setup.sh — Linux / Raspberry Pi 5 의존성 설치
#  사용법:
#    bash setup.sh          # 시스템 Python3
#    bash setup.sh --venv   # 가상환경 생성 후 설치
# ================================================================

set -e

PYTHON=python3
PIP="$PYTHON -m pip"

# ── 가상환경 옵션 ──────────────────────────────────────────────
if [ "$1" = "--venv" ]; then
    echo "[1/6] 가상환경 생성 중..."
    $PYTHON -m venv .venv
    source .venv/bin/activate
    PIP=pip
fi

echo ""
echo "================================================================"
echo "  Personal Color Kiosk — Linux/RPi5 의존성 설치"
echo "================================================================"

# ── 시스템 패키지 (RPi / Debian 계열) ─────────────────────────
if command -v apt-get &>/dev/null; then
    echo "[Step 1] 시스템 패키지 설치..."
    sudo apt-get update -qq
    sudo apt-get install -y \
        python3-pip \
        python3-dev \
        libatlas-base-dev \
        libjpeg-dev \
        libopenblas-dev \
        libhdf5-dev \
        libgl1 \
        libglib2.0-0 \
        python3-pyqt6 \
        qt6-base-dev 2>/dev/null || true
else
    echo "[Step 1] apt 없음 — 시스템 패키지 건너뜀"
fi

# ── pip 업그레이드 ─────────────────────────────────────────────
echo "[Step 2] pip 업그레이드..."
$PIP install --upgrade pip

# ── protobuf 고정 (mediapipe 0.10.14 호환) ─────────────────────
echo "[Step 3] protobuf 버전 고정..."
$PIP install "protobuf>=4.25.0,<5.0.0"

# ── MediaPipe ─────────────────────────────────────────────────
# RPi5 (ARM64 Linux): mediapipe 0.10.14 공식 wheel 제공됨
echo "[Step 4] MediaPipe 설치 (0.10.14)..."
$PIP install "mediapipe==0.10.14"

# ── PyTorch (RPi5 ARM64 전용 wheel) ───────────────────────────
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "[Step 5] PyTorch ARM64 설치..."
    # Raspberry Pi OS (Bookworm) 기준
    $PIP install \
        "torch>=2.0.0" \
        "torchvision>=0.15.0" \
        --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    $PIP install torch torchvision
else
    echo "[Step 5] PyTorch (x86_64) 설치..."
    $PIP install "torch>=2.6.0" "torchvision>=0.21.0"
fi

# ── 나머지 의존성 ─────────────────────────────────────────────
echo "[Step 6] 나머지 패키지 설치..."
$PIP install \
    "opencv-python-headless>=4.9.0" \
    "numpy>=1.26.0" \
    "Pillow>=10.3.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    "aiofiles>=23.2.1" \
    "httpx>=0.27.0" \
    "qrcode[pil]>=7.4.2" \
    "scikit-learn>=1.4.0" \
    "pandas>=2.2.0"

# PyQt6 (pip 미지원 시 시스템 패키지 사용)
$PIP install "PyQt6>=6.6.0" 2>/dev/null || \
    echo "  PyQt6 pip 설치 실패 → 시스템 패키지(python3-pyqt6) 사용"

# ── 설치 확인 ─────────────────────────────────────────────────
echo ""
echo "[확인] MediaPipe FaceMesh 동작 테스트..."
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from ai.preprocessing import FacePreprocessor, _MEDIAPIPE_OK
print('  MediaPipe:', 'OK' if _MEDIAPIPE_OK else 'Fallback(Haarcascade)')
"

echo ""
echo "================================================================"
echo "  설치 완료!"
echo "  실행: python3 testfolder/dev_kiosk.py     # 개발 테스트"
echo "        python3 kiosk/main.py               # 실제 키오스크"
echo "================================================================"
