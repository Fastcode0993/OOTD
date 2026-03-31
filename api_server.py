"""
HMI AI 진단 API 서버 (FastAPI)
멀티모달 추론: 이미지 (CNN) + 피부색 수치 (MLP) 결합
"""

import os
import sqlite3
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from server_preprocess import FaceMeshProcessor, FaceAnalysisResult
from constants import SEASON_CLASSES, PERSONAL_COLOR_THRESHOLDS, EIGHT_SEASONS, SEASON_PALETTES

# FastAPI 앱 생성
app = FastAPI(
    title="HMI AI 진단 서버",
    description="ED-HMI3010-101C 기반 퍼스널 컬러 진단 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 데이터베이스 설정
DB_PATH = os.path.join(os.path.dirname(__file__), "diagnosis.db")


def init_database():
    """diagnosis_results 테이블 생성"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnosis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            season TEXT NOT NULL,
            confidence REAL NOT NULL,
            cheek_rgb TEXT,
            cheek_lab TEXT,
            forehead_rgb TEXT,
            forehead_lab TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


# 모델 정의
class EfficientNetB0(nn.Module):
    """EfficientNet-B0 를 활용한 계절 분류 모델"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        # EfficientNet-B0 로드 (이미지 분류용)
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier[1] = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone.classifier[2] = nn.Flatten()
        
        # 피부색 기반 MLP (Lab 값 처리)
        self.skin_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes)
        )
        
        # 결합 레이어
        self.combine = nn.Linear(1000 + 8, num_classes)
    
    def forward(self, x: torch.Tensor, lab_values: torch.Tensor) -> torch.Tensor:
        # 이미지 경로에서 로드
        if isinstance(x, str):
            image = Image.open(x).convert('RGB')
            x = self.backbone.preprocess(image).unsqueeze(0)
        elif isinstance(x, np.ndarray):
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = Image.fromarray(x)
            x = self.backbone.preprocess(x).unsqueeze(0)
        
        image_features = self.backbone(x)
        skin_features = self.skin_mlp(lab_values)
        
        # 결합
        combined = torch.cat([image_features, skin_features], dim=1)
        output = self.combine(combined)
        
        return output


# 전역 모델 인스턴스
model = None


def load_model(model_path: Optional[str] = None):
    """모델 로드"""
    global model
    if model is not None:
        return model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # EfficientNet-B0 가중치 로드
    num_classes = 4  # 봄, 여름, 가을, 겨울
    model = EfficientNetB0(num_classes=num_classes)
    model = model.to(device)
    model.eval()
    
    # 가중치 파일 로드 (best_model.pt 사용)
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded model from: {model_path}")
    else:
        # 기본 가중치 사용 (이미지넷 가중치 기반)
        print("Using default EfficientNet-B0 weights")
    
    return model


# Pydantic 모델
class DiagnosisRequest(BaseModel):
    session_id: str
    cheek_rgb: List[int]
    cheek_lab: List[float]
    forehead_rgb: List[int]
    forehead_lab: List[float]


class DiagnosisResponse(BaseModel):
    session_id: str
    season: str
    confidence: float
    season_korean: str
    palette: List[Dict[str, str]]
    message: str
    re_take: bool = False


# API 엔드포인트
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    init_database()
    load_model()


@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "service": "HMI AI 진단 서버",
        "version": "1.0.0"
    }


@app.post("/analyze", response_model=DiagnosisResponse)
async def analyze_image(file: UploadFile = File(...), session_id: str = "auto"):
    """
    이미지 분석 및 계절 톤 추론
    
    - MediaPipe 로 얼굴 분석 (468 개 랜드마크)
    - 양볼과 이마의 RGB/Lab 값 추출
    - 이미지 (CNN) + 피부색 수치 (MLP) 결합
    - Confidence 0.8 이상: 정상
    - Confidence 0.6 미만: RE-TAKE 메시지
    """
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model()
    model.eval()
    
    try:
        # 이미지 로드 및 전처리
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if bgr_image is None:
            raise HTTPException(status_code=400, detail="이미지 디코딩 실패")
        
        # MediaPipe 로 얼굴 분석 (서버에서 수행)
        processor = FaceMeshProcessor()
        analysis_result = processor.analyze(bgr_image)
        
        if not analysis_result or not analysis_result.face_detected:
            raise HTTPException(status_code=400, detail="얼굴을 감지할 수 없습니다")
        
        # 224x224 로 리사이즈 (CNN 추론용)
        inference_image = cv2.resize(bgr_image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # CNN 예측
        with torch.no_grad():
            input_tensor = torch.from_numpy(inference_image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            image_probs = model.backbone(input_tensor)
            image_probs = torch.softmax(image_probs, dim=1)
            image_confidence, image_pred = torch.max(image_probs, 1)
        
        # 피부색 수치 (MLP) 예측 (서버에서 추출한 값 사용)
        cheek_rgb = np.array(analysis_result.cheek_rgb, dtype=np.float32) / 255.0
        cheek_lab = np.array(analysis_result.cheek_lab, dtype=np.float32)
        forehead_rgb = np.array(analysis_result.forehead_rgb, dtype=np.float32) / 255.0
        forehead_lab = np.array(analysis_result.forehead_lab, dtype=np.float32)
        
        # 평균 Lab 값 사용
        avg_lab = (cheek_lab + forehead_lab) / 2
        
        skin_input = torch.tensor([avg_lab], dtype=torch.float32).to(device)
        skin_probs = model.skin_mlp(skin_input)
        skin_probs = torch.softmax(skin_probs, dim=1)
        skin_confidence, skin_pred = torch.max(skin_probs, 1)
        
        # 결합 예측 (이미지 + 피부색 가중 평균)
        combined_confidence = (image_confidence.item() * 0.6 + skin_confidence.item() * 0.4)
        combined_pred = 0 if image_confidence.item() > skin_confidence.item() else 1
        
        # 확률 해석 규칙 적용
        if combined_confidence >= 0.8:
            message = "진단 완료"
            re_take = False
        elif combined_confidence < 0.6:
            message = "RE-TAKE: 명확한 얼굴을 다시 촬영해주세요"
            re_take = True
        else:
            message = "저렴한 확률로 진단되었습니다. 재촬영 권장"
            re_take = True
        
        # 한국어 계절 이름 매핑
        season_map = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}
        season_korean_map = {
            "Spring": "봄", "Summer": "여름", "Autumn": "가을", "Winter": "겨울"
        }
        
        predicted_season = season_map[combined_pred]
        
        # 결과 저장 (SQLite)
        save_result(session_id, predicted_season, combined_confidence.item(), 
                   analysis_result.cheek_rgb, analysis_result.cheek_lab,
                   analysis_result.forehead_rgb, analysis_result.forehead_lab)
        
        # 계절별 팔레트 반환
        palette = SEASON_PALETTES.get(predicted_season, [])
        
        return DiagnosisResponse(
            session_id=session_id,
            season=predicted_season,
            confidence=combined_confidence.item(),
            season_korean=season_korean_map[predicted_season],
            palette=palette,
            message=message,
            re_take=re_take
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")


@app.post("/preprocess", response_model=FaceAnalysisResult)
async def preprocess_image(file: UploadFile = File(...)):
    """
    이미지 전처리 API (MediaPipe 얼굴 분석)
    
    HMI 가 보낸 이미지를 받아 468 개 랜드마크와 피부색 데이터를 추출
    
    Request:
        - file: 이미지 파일 (4K 또는 원본)
    
    Response:
        - face_detected: 얼굴 감지 여부
        - landmarks: 468 개 좌표
        - cheek_rgb: 양볼 평균 RGB
        - cheek_lab: 양볼 평균 Lab
        - forehead_rgb: 이마 평균 RGB
        - forehead_lab: 이마 평균 Lab
        - face_confidence: 얼굴 감지 신뢰도
    """
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    try:
        # 이미지 로드
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if bgr_image is None:
            raise HTTPException(status_code=400, detail="이미지 디코딩 실패")
        
        # MediaPipe 로 얼굴 분석 (서버에서 수행)
        processor = FaceMeshProcessor()
        analysis_result = processor.analyze(bgr_image)
        
        if not analysis_result or not analysis_result.face_detected:
            raise HTTPException(status_code=400, detail="얼굴을 감지할 수 없습니다")
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"전처리 실패: {str(e)}")


def save_result(session_id: str, season: str, confidence: float,
                cheek_rgb: List[int], cheek_lab: List[float],
                forehead_rgb: List[int], forehead_lab: List[float]):
    """진단 결과 SQLite 에 저장"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO diagnosis_results 
            (session_id, season, confidence, cheek_rgb, cheek_lab, forehead_rgb, forehead_lab)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, season, confidence, 
              str(cheek_rgb), str(cheek_lab), str(forehead_rgb), str(forehead_lab)))
        conn.commit()
    except sqlite3.IntegrityError:
        # session_id 가 이미 존재하면 업데이트
        cursor.execute('''
            UPDATE diagnosis_results 
            SET season=?, confidence=?, cheek_rgb=?, cheek_lab=?, 
                    forehead_rgb=?, forehead_lab=?
            WHERE session_id=?
        ''', (season, confidence, str(cheek_rgb), str(cheek_lab), 
              str(forehead_rgb), str(forehead_lab), session_id))
        conn.commit()
    
    conn.close()


@app.get("/results/{session_id}")
async def get_result(session_id: str):
    """특정 세션의 진단 결과 조회"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT session_id, season, confidence, cheek_rgb, cheek_lab, 
               forehead_rgb, forehead_lab, created_at
        FROM diagnosis_results
        WHERE session_id = ?
    ''', (session_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        raise HTTPException(status_code=404, detail="결과가 없습니다")
    
    return {
        "session_id": row[0],
        "season": row[1],
        "confidence": row[2],
        "cheek_rgb": eval(row[3]),
        "cheek_lab": eval(row[4]),
        "forehead_rgb": eval(row[5]),
        "forehead_lab": eval(row[6]),
        "created_at": row[7]
    }


@app.get("/results")
async def get_all_results(limit: int = 10):
    """최근 진단 결과 목록 조회"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT session_id, season, confidence, created_at
        FROM diagnosis_results
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "session_id": row[0],
            "season": row[1],
            "confidence": row[2],
            "created_at": row[3]
        }
        for row in rows
    ]


# WebSocket 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 연결 (실시간 카메라 프레임 전송)"""
    await websocket.accept()
    
    try:
        while True:
            # 클라이언트에서 데이터 수신 (프레임 요청)
            data = await websocket.receive_text()
            
            # 서버에서 카메라 프레임 전송 (실제 구현 필요)
            # HMI 에서 WebSocket 으로 프레임 받기 위한 엔드포인트
            frame_data = await websocket.receive_text()
            
    except Exception as e:
        print(f"WebSocket 에러: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
