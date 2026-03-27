퍼스널 컬러 데이터 windows 기준 GUI 툴 
# 파이썬 가상화 할것. 
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip

#개별
pip install PyQt6 PyQt6-WebEngine
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy matplotlib pillow
#통합
pip install PyQt6 torch torchvision scikit-learn opencv-python mediapipe pandas matplotlib numpy Pillow
