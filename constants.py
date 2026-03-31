"""
HMI 퍼스널 컬러 진단 시스템 - 상수 정의
계절별 팔레트, 클래스 매핑, 임계값 등
"""

# 4 계절 클래스
SEASON_CLASSES = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_KOREAN = {"Spring": "봄", "Summer": "여름", "Autumn": "가을", "Winter": "겨울"}

# 8 계절 (더 세분화된 버전)
EIGHT_SEASONS = [
    "Spring_Light", "Spring_Medium", "Spring_Dark",
    "Summer_Light", "Summer_Medium", "Summer_Dark",
    "Autumn_Light", "Autumn_Medium", "Autumn_Dark",
    "Winter_Light", "Winter_Medium", "Winter_Dark"
]

# 계절별 팔레트 (HEX 색상)
SEASON_PALETTES = {
    "Spring": [
        {"name": "봄의 꽃", "hex": "#FFB6C1", "rgb": [255, 182, 193]},
        {"name": "새싹", "hex": "#90EE90", "rgb": [144, 238, 144]},
        {"name": "하늘", "hex": "#87CEEB", "rgb": [135, 206, 235]},
        {"name": "노란색", "hex": "#FFD700", "rgb": [255, 215, 0]},
        {"name": "분홍", "hex": "#FF69B4", "rgb": [255, 105, 180]},
    ],
    "Summer": [
        {"name": "파란색", "hex": "#4169E1", "rgb": [65, 105, 225]},
        {"name": "초록", "hex": "#32CD32", "rgb": [50, 205, 50]},
        {"name": "보라", "hex": "#9370DB", "rgb": [147, 112, 219]},
        {"name": "주황", "hex": "#FF8C00", "rgb": [255, 140, 0]},
        {"name": "연두", "hex": "#98FB98", "rgb": [152, 251, 152]},
    ],
    "Autumn": [
        {"name": "주황", "hex": "#FFA500", "rgb": [255, 165, 0]},
        {"name": "빨강", "hex": "#DC143C", "rgb": [220, 20, 60]},
        {"name": "노랑", "hex": "#FFD700", "rgb": [255, 215, 0]},
        {"name": "갈색", "hex": "#8B4513", "rgb": [139, 69, 19]},
        {"name": "주황갈색", "hex": "#D2691E", "rgb": [210, 105, 30]},
    ],
    "Winter": [
        {"name": "흰색", "hex": "#FFFFFF", "rgb": [255, 255, 255]},
        {"name": "보라", "hex": "#9370DB", "rgb": [147, 112, 219]},
        {"name": "파랑", "hex": "#4169E1", "rgb": [65, 105, 225]},
        {"name": "회색", "hex": "#808080", "rgb": [128, 128, 128]},
        {"name": "연보라", "hex": "#E6E6FA", "rgb": [230, 230, 250]},
    ],
}

# 피부색 임계값 (Lab 색상 공간)
PERSONAL_COLOR_THRESHOLDS = {
    "light": {"L_min": 50, "L_max": 80},
    "medium": {"L_min": 80, "L_max": 100},
    "dark": {"L_min": 100, "L_max": 130},
}

# 얼굴 랜드마크 인덱스 (MediaPipe Face Mesh)
FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_CHEEK_LANDMARKS = [50, 101, 118, 117, 123, 187, 207, 206, 205, 36, 93]
RIGHT_CHEEK_LANDMARKS = [280, 330, 347, 346, 352, 411, 427, 426, 425, 266, 323]
FOREHEAD_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
]

# HMI 설정
HMI_WIDTH = 1280
HMI_HEIGHT = 800

# 서버 URL
SERVER_URL = "http://localhost:8000"

# 확률 해석 규칙
CONFIDENCE_THRESHOLDS = {
    "high": 0.8,   # 진단 완료
    "low": 0.6,    # RE-TAKE 권장
}

# 메시지 템플릿
MESSAGES = {
    "success": "진단 완료",
    "re_take": "RE-TAKE: 명확한 얼굴을 다시 촬영해주세요",
    "low_confidence": "저렴한 확률로 진단되었습니다. 재촬영 권장",
}