# QR코드 기반 퍼스널 컬러 진단 시스템 완벽 가이드 📱

---

## 🏗️ 전체 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  💻 키오스크 (오프라인 - 진단 전담)                    │
│  ├─ PyQt6 / React (터치스크린)                         │
│  ├─ 사진 촬영                                           │
│  ├─ AI 추론 (hierarchical.pt)                         │
│  ├─ 퍼스널 컬러 결과 표시 (색상 팔레트)               │
│  └─ QR코드 생성 & 표시                                │
└────────────────────┬────────────────────────────────────┘
                     │ (오프라인에서 QR코드만 표시)
                     │
┌────────────────────┴────────────────────────────────────┐
│  ☁️ 클라우드 서버 (Node.js + Express)                  │
│  ├─ REST API 엔드포인트                                │
│  ├─ QR 코드 검증                                        │
│  ├─ 진단 데이터 조회                                   │
│  └─ 색상 정보 제공                                     │
└────────────────────┬────────────────────────────────────┘
                     │
├────────────────────┴────────────────────────────────────┤
│  📦 데이터베이스 (MongoDB)                              │
│  ├─ 진단 결과 저장                                     │
│  │   ├─ user_id                                       │
│  │   ├─ season / tone                                 │
│  │   ├─ confidence                                    │
│  │   ├─ color_palette                                │
│  │   ├─ qr_code (고유값)                             │
│  │   └─ timestamp                                     │
│  │                                                     │
│  └─ 사용자 정보 (선택)                                │
└─────────────────────────────────────────────────────────┘
                     │
└────────────────────┬────────────────────────────────────┐
│  📱 모바일 앱 (React Native / Flutter)                │
│  ├─ QR 스캔 기능                                      │
│  ├─ 진단 결과 조회                                    │
│  ├─ 색상 상세 정보 표시                               │
│  └─ PDF/이미지 다운로드                               │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 사용자 여정 (Flow)

```
┌─────────────────────────────────────────────────────┐
│  1️⃣  키오스크에서 (오프라인)                      │
├─────────────────────────────────────────────────────┤
│  [키오스크 터치스크린]                             │
│  ├─ "시작하기" 버튼 클릭                           │
│  ├─ 카메라로 얼굴 사진 촬영                        │
│  │  └─ 또는 기존 사진 선택                         │
│  │                                                 │
│  ├─ AI 분석 중... (2-3초)                         │
│  │  ├─ MediaPipe: 얼굴 감지                       │
│  │  ├─ 색상 메타데이터 추출                       │
│  │  ├─ Stage1: 4계절 판별                         │
│  │  └─ Stage2: 톤 판별 (3가지)                    │
│  │                                                 │
│  └─ 결과 화면 (5초간 표시)                        │
│     ├─ 🌸 Spring-Warm                             │
│     ├─ 신뢰도: 87% × 74% = 64%                    │
│     ├─ 색상 팔레트 시각화 (RGB 박스들)           │
│     │                                             │
│     └─ QR코드 생성 & 표시 (10초)                │
│        ├─ Unique ID: abc123def456              │
│        ├─ QR코드 이미지 (500×500px)            │
│        └─ "앱에서 QR을 찍으세요!" 메시지       │
│                                                 │
│     [화면 예시]                                  │
│     ┌─────────────────────────────┐             │
│     │  당신의 퍼스널 컬러        │             │
│     │  Spring - Warm              │             │
│     │  신뢰도: 64%                │             │
│     │                              │             │
│     │  ▓▓▓▓▓ ▓▓▓▓▓ ▓▓▓▓▓         │ 색상팔레트  │
│     │  ▓▓▓▓▓ ▓▓▓▓▓ ▓▓▓▓▓         │             │
│     │                              │             │
│     │  [QR코드 이미지]             │             │
│     │  ┌──────────────┐           │             │
│     │  │              │           │             │
│     │  │   (QR코드)   │           │             │
│     │  │              │           │             │
│     │  └──────────────┘           │             │
│     │                              │             │
│     │  앱에서 QR을 찍으세요!      │             │
│     └─────────────────────────────┘             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  2️⃣  모바일 앱에서 (온라인)                        │
├─────────────────────────────────────────────────────┤
│  [사용자가 집에 가서]                              │
│  ├─ 앱 실행                                        │
│  ├─ "QR 스캔" 버튼 클릭                           │
│  │  └─ 카메라 권한 요청                           │
│  │                                                 │
│  ├─ 키오스크에서 받은 QR코드 스캔                  │
│  │  └─ (핸드폰 카메라로 사진 촬영)               │
│  │                                                 │
│  └─ [서버에 요청]                                 │
│     POST /api/result/{qr_code}                    │
│     └─ QR코드 → Unique ID 추출                   │
│        └─ MongoDB에서 해당 결과 조회              │
│                                                 │
│  [결과 화면]                                      │
│  ├─ Spring-Warm (확대된 정보)                    │
│  │                                                 │
│  ├─ 📊 상세 색상 정보                            │
│  │  ├─ 계절 신뢰도: 87%                         │
│  │  ├─ 톤 신뢰도: 74%                           │
│  │  └─ 최종 신뢰도: 64%                         │
│  │                                                 │
│  ├─ 🎨 색상 팔레트 (확대)                        │
│  │  ├─ 주요 색상: #FF8C69, #FFB366, ...         │
│  │  ├─ 보조 색상: #CD853F, #DAA520, ...         │
│  │  └─ 악센트: #32CD32, #90EE90                 │
│  │                                                 │
│  ├─ 📋 상세 테이블                               │
│  │  ├─ 색상명     | HEX코드    | RGB             │
│  │  ├─ Coral      | #FF8C69    | (255,140,105)  │
│  │  ├─ Peach      | #FFAA88    | (255,170,136)  │
│  │  ├─ Orange     | #FFA500    | (255,165,0)    │
│  │  └─ ...                                        │
│  │                                                 │
│  ├─ 💄 추천 화장품 (클릭 가능)                    │
│  │  ├─ MISSHA M Cushion (Light Warm Beige)     │
│  │  ├─ MAC Lipstick (Coral)                    │
│  │  └─ Maybelline Eyeshadow                    │
│  │                                                 │
│  ├─ 👗 추천 패션 색상                             │
│  │  ├─ 상의: 코랄, 복숭아, 크림                 │
│  │  ├─ 하의: 베이지, 카멜, 올리브               │
│  │  └─ 외출: 카멜, 탠, 라이트브라운             │
│  │                                                 │
│  └─ 📥 다운로드 옵션                             │
│     ├─ 📥 이미지 (PNG)                           │
│     ├─ 📄 PDF 보고서                            │
│     └─ 📦 전체 다운로드 (ZIP)                   │
└─────────────────────────────────────────────────────┘
```

---

## 💻 Backend: Node.js + Express + MongoDB

### 설치

```bash
npm init -y
npm install express mongoose dotenv cors qrcode
npm install --save-dev nodemon

# 디렉토리 구조
backend/
├── server.js           # 메인 서버
├── models/
│   ├── Diagnosis.js    # MongoDB 스키마
│   └── User.js         # 사용자 정보 (선택)
├── routes/
│   └── api.js          # API 라우트
├── controllers/
│   └── diagnosisController.js
├── config/
│   └── db.js           # MongoDB 연결
└── .env                # 환경 변수
```

### 1️⃣ MongoDB 스키마 (models/Diagnosis.js)

```javascript
// models/Diagnosis.js

const mongoose = require('mongoose');

const diagnosisSchema = new mongoose.Schema({
  qr_code: {
    type: String,
    unique: true,
    required: true,
    index: true
  },
  
  // 진단 결과
  season: {
    type: String,
    enum: ['Spring', 'Summer', 'Autumn', 'Winter'],
    required: true
  },
  tone: {
    type: String,
    enum: ['Warm', 'Bright', 'Light'],
    required: true
  },
  
  // 신뢰도
  season_confidence: Number,
  tone_confidence: Number,
  final_confidence: Number,
  
  // 색상 팔레트
  color_palette: {
    primary: [String],      // ["#FF8C69", "#FFB366", ...]
    secondary: [String],
    accent: [String],
    description: String
  },
  
  // 추천사항
  recommendations: {
    cosmetics: {
      cushion: [Object],
      lipstick: [Object],
      eyeshadow: [Object],
      blush: [Object]
    },
    fashion: {
      colors: [String],
      brands: [String],
      styles: [String]
    }
  },
  
  // 메타데이터
  user_info: {
    name: String,
    email: String,
    phone: String
  },
  
  created_at: {
    type: Date,
    default: Date.now
  },
  
  // 원본 사진 URL (선택)
  image_url: String
});

module.exports = mongoose.model('Diagnosis', diagnosisSchema);
```

### 2️⃣ 서버 설정 (server.js)

```javascript
// server.js

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const app = express();

// 미들웨어
app.use(express.json());
app.use(cors());

// MongoDB 연결
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => console.log('MongoDB 연결됨'))
  .catch(err => console.log('MongoDB 연결 실패:', err));

// 라우트
app.use('/api', require('./routes/api'));

// 서버 시작
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`서버 실행 중: http://localhost:${PORT}`);
});
```

### 3️⃣ API 라우트 (routes/api.js)

```javascript
// routes/api.js

const express = require('express');
const router = express.Router();
const Diagnosis = require('../models/Diagnosis');
const QRCode = require('qrcode');
const { v4: uuidv4 } = require('uuid');

// ─────────────────────────────────────────────────────
//  1. 진단 결과 저장 (키오스크에서 호출)
// ─────────────────────────────────────────────────────

router.post('/save-diagnosis', async (req, res) => {
  /**
   * 키오스크에서 호출
   * 
   * Request Body:
   * {
   *   "season": "Spring",
   *   "tone": "Warm",
   *   "season_confidence": 0.87,
   *   "tone_confidence": 0.74,
   *   "final_confidence": 0.6438,
   *   "color_palette": {
   *     "primary": ["#FF8C69", ...],
   *     "secondary": [...],
   *     "accent": [...]
   *   },
   *   "recommendations": {...},
   *   "user_info": {
   *     "name": "홍길동",
   *     "email": "hong@example.com"
   *   }
   * }
   */
  
  try {
    // QR코드용 Unique ID 생성
    const qr_code = uuidv4().substring(0, 8).toUpperCase();
    
    // 진단 결과 저장
    const diagnosis = new Diagnosis({
      qr_code,
      ...req.body,
      created_at: new Date()
    });
    
    await diagnosis.save();
    
    // 응답
    res.json({
      success: true,
      qr_code,
      message: '진단 결과가 저장되었습니다.'
    });
    
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// ─────────────────────────────────────────────────────
//  2. 진단 결과 조회 (모바일 앱에서 호출)
// ─────────────────────────────────────────────────────

router.get('/result/:qr_code', async (req, res) => {
  /**
   * 모바일 앱에서 QR코드 스캔 후 호출
   * 
   * URL: /api/result/ABC1D2E3
   * 
   * Response:
   * {
   *   "qr_code": "ABC1D2E3",
   *   "season": "Spring",
   *   "tone": "Warm",
   *   "final_confidence": 0.6438,
   *   "color_palette": {...},
   *   "recommendations": {...},
   *   "created_at": "2024-01-15T10:30:00Z"
   * }
   */
  
  try {
    const { qr_code } = req.params;
    
    // QR코드로 진단 결과 조회
    const diagnosis = await Diagnosis.findOne({ qr_code });
    
    if (!diagnosis) {
      return res.status(404).json({
        success: false,
        error: '진단 결과를 찾을 수 없습니다.'
      });
    }
    
    res.json({
      success: true,
      data: diagnosis
    });
    
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// ─────────────────────────────────────────────────────
//  3. QR코드 생성 (키오스크에서 호출)
// ─────────────────────────────────────────────────────

router.get('/generate-qr/:qr_code', async (req, res) => {
  /**
   * QR코드 이미지 생성
   * 
   * URL: /api/generate-qr/ABC1D2E3
   * 
   * Response: QR코드 이미지 (PNG)
   */
  
  try {
    const { qr_code } = req.params;
    
    // QR코드 데이터: 앱에서 인식할 수 있는 URL 형태
    const qr_data = `https://yourapp.com/scan/${qr_code}`;
    
    // QR코드 생성
    const qr_image = await QRCode.toDataURL(qr_data, {
      errorCorrectionLevel: 'H',
      type: 'image/png',
      quality: 0.95,
      margin: 1,
      width: 500
    });
    
    res.json({
      success: true,
      qr_code,
      qr_image,  // Base64 encoded PNG
      url: qr_data
    });
    
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// ─────────────────────────────────────────────────────
//  4. 색상 정보 상세 조회
// ─────────────────────────────────────────────────────

router.get('/result/:qr_code/colors', async (req, res) => {
  /**
   * 모바일 앱에서 색상 테이블 표시용
   * 
   * Response:
   * {
   *   "primary": [
   *     {
   *       "name": "Coral",
   *       "hex": "#FF8C69",
   *       "rgb": "255, 140, 105",
   *       "usage": "주요 색상"
   *     },
   *     ...
   *   ],
   *   "secondary": [...],
   *   "accent": [...]
   * }
   */
  
  try {
    const { qr_code } = req.params;
    const diagnosis = await Diagnosis.findOne({ qr_code });
    
    if (!diagnosis) {
      return res.status(404).json({ error: '결과를 찾을 수 없습니다.' });
    }
    
    // 색상 이름 매핑
    const color_names = {
      '#FF8C69': { name: 'Coral', usage: '주요 색상' },
      '#FFB366': { name: 'Peach', usage: '주요 색상' },
      '#FFA500': { name: 'Orange', usage: '주요 색상' },
      '#CD853F': { name: 'Terracotta', usage: '보조 색상' },
      // ... 더 많은 색상들
    };
    
    // HEX to RGB 변환
    const hexToRgb = (hex) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? 
        `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` 
        : '0, 0, 0';
    };
    
    // 응답 구성
    const colors_detailed = {
      primary: diagnosis.color_palette.primary.map(hex => ({
        name: color_names[hex]?.name || 'Color',
        hex,
        rgb: hexToRgb(hex),
        usage: 'Primary Color'
      })),
      secondary: diagnosis.color_palette.secondary.map(hex => ({
        name: color_names[hex]?.name || 'Color',
        hex,
        rgb: hexToRgb(hex),
        usage: 'Secondary Color'
      })),
      accent: diagnosis.color_palette.accent.map(hex => ({
        name: color_names[hex]?.name || 'Color',
        hex,
        rgb: hexToRgb(hex),
        usage: 'Accent Color'
      }))
    };
    
    res.json({
      success: true,
      season: diagnosis.season,
      tone: diagnosis.tone,
      colors: colors_detailed
    });
    
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// ─────────────────────────────────────────────────────
//  5. PDF 다운로드 URL 생성
// ─────────────────────────────────────────────────────

router.get('/result/:qr_code/pdf', async (req, res) => {
  /**
   * PDF 생성 및 다운로드 링크 제공
   * 
   * Response:
   * {
   *   "pdf_url": "https://yourcloud.com/pdf/ABC1D2E3.pdf",
   *   "filename": "personal_color_ABC1D2E3.pdf"
   * }
   */
  
  try {
    const { qr_code } = req.params;
    const diagnosis = await Diagnosis.findOne({ qr_code });
    
    if (!diagnosis) {
      return res.status(404).json({ error: '결과를 찾을 수 없습니다.' });
    }
    
    // PDF 생성 로직 (별도 함수 사용)
    // const pdf_buffer = await generatePDF(diagnosis);
    // const pdf_url = await uploadToCloud(pdf_buffer, qr_code);
    
    res.json({
      success: true,
      pdf_url: `https://yourcloud.com/pdf/${qr_code}.pdf`,
      filename: `personal_color_${qr_code}.pdf`
    });
    
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;
```

### 4️⃣ 환경 변수 (.env)

```bash
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/personal_color
PORT=5000
NODE_ENV=development
```

---

## 📱 Frontend: React 모바일 앱

### 설치

```bash
npx create-react-app mobile-app
cd mobile-app
npm install axios react-qr-reader html2pdf.js

# 또는 React Native (권장)
npx react-native init PersonalColorApp
npm install react-native-qr-scanner axios
```

### 1️⃣ QR 스캔 화면 (QRScanner.jsx)

```jsx
// components/QRScanner.jsx

import React, { useState, useRef } from 'react';
import axios from 'axios';

export default function QRScanner() {
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const videoRef = useRef(null);

  // ─────────────────────────────────────────────────────
  //  QR 스캔
  // ─────────────────────────────────────────────────────

  const startScanning = async () => {
    setScanning(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }
      });
      videoRef.current.srcObject = stream;
    } catch (error) {
      alert('카메라 접근 권한이 필요합니다.');
    }
  };

  // QR코드 인식 (jsQR 또는 라이브러리 사용)
  const handleQRDetected = async (qrCode) => {
    setLoading(true);
    try {
      // QR코드에서 QR_CODE 값 추출
      // https://yourapp.com/scan/ABC1D2E3 → ABC1D2E3
      const match = qrCode.match(/scan\/([A-Z0-9]+)/);
      if (!match) {
        alert('유효하지 않은 QR코드입니다.');
        setLoading(false);
        return;
      }

      const qr_code = match[1];

      // 서버에서 결과 조회
      const response = await axios.get(
        `http://localhost:5000/api/result/${qr_code}`
      );

      if (response.data.success) {
        setResult(response.data.data);
        setScanning(false);
      }
    } catch (error) {
      alert('결과를 조회할 수 없습니다: ' + error.message);
    }
    setLoading(false);
  };

  // ─────────────────────────────────────────────────────
  //  스캔 화면
  // ─────────────────────────────────────────────────────

  if (scanning && !result) {
    return (
      <div className="qr-scanner-container">
        <div className="scanner-header">
          <h1>QR코드 스캔</h1>
          <p>키오스크에서 받은 QR코드를 카메라에 맞춰주세요.</p>
        </div>

        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover'
          }}
        />

        <div className="scanner-overlay">
          <div className="scan-frame" />
          <p>QR코드를 프레임 안에 맞춰주세요</p>
        </div>

        <button
          className="close-btn"
          onClick={() => setScanning(false)}
        >
          ✕ 취소
        </button>

        <style jsx>{`
          .qr-scanner-container {
            position: relative;
            width: 100%;
            height: 100vh;
            background: black;
          }

          .scanner-header {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 20px;
            text-align: center;
            z-index: 10;
          }

          .scanner-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 5;
          }

          .scan-frame {
            width: 250px;
            height: 250px;
            border: 3px solid #00ff00;
            border-radius: 10px;
            box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
          }

          .close-btn {
            position: absolute;
            top: 100px;
            right: 20px;
            background: rgba(255, 0, 0, 0.8);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 20;
          }
        `}</style>
      </div>
    );
  }

  // ─────────────────────────────────────────────────────
  //  홈 화면
  // ─────────────────────────────────────────────────────

  return (
    <div className="home-container">
      <h1>🎨 퍼스널 컬러 앱</h1>
      <p>키오스크에서 진단받은 퍼스널 컬러를 확인하세요.</p>

      <button
        onClick={startScanning}
        className="scan-button"
      >
        📱 QR코드 스캔
      </button>

      <style jsx>{`
        .home-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 20px;
        }

        h1 {
          font-size: 36px;
          margin-bottom: 10px;
        }

        p {
          font-size: 16px;
          margin-bottom: 30px;
          text-align: center;
        }

        .scan-button {
          padding: 15px 40px;
          font-size: 18px;
          background: white;
          color: #667eea;
          border: none;
          border-radius: 10px;
          cursor: pointer;
          font-weight: bold;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .scan-button:active {
          transform: scale(0.95);
        }
      `}</style>
    </div>
  );
}
```

### 2️⃣ 결과 상세 화면 (ResultDetail.jsx)

```jsx
// components/ResultDetail.jsx

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import html2pdf from 'html2pdf.js';

export default function ResultDetail({ qr_code, diagnosis }) {
  const [colorDetails, setColorDetails] = useState(null);
  const [activeTab, setActiveTab] = useState('info');
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    loadColorDetails();
  }, [qr_code]);

  // ─────────────────────────────────────────────────────
  //  색상 상세 정보 로드
  // ─────────────────────────────────────────────────────

  const loadColorDetails = async () => {
    try {
      const response = await axios.get(
        `http://localhost:5000/api/result/${qr_code}/colors`
      );
      setColorDetails(response.data.colors);
    } catch (error) {
      console.error('색상 정보 로드 실패:', error);
    }
  };

  // ─────────────────────────────────────────────────────
  //  PDF 다운로드
  // ─────────────────────────────────────────────────────

  const downloadPDF = async () => {
    setDownloading(true);
    try {
      const element = document.getElementById('pdf-content');
      const opt = {
        margin: 10,
        filename: `personal_color_${qr_code}.pdf`,
        image: { type: 'png', quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { orientation: 'portrait', unit: 'mm', format: 'a4' }
      };
      html2pdf().set(opt).from(element).save();
    } catch (error) {
      alert('PDF 다운로드 실패');
    }
    setDownloading(false);
  };

  // ─────────────────────────────────────────────────────
  //  UI
  // ─────────────────────────────────────────────────────

  return (
    <div className="result-container">
      {/* 헤더 */}
      <div className="header">
        <h1>{diagnosis.season} - {diagnosis.tone}</h1>
        <p>신뢰도: {(diagnosis.final_confidence * 100).toFixed(1)}%</p>
      </div>

      {/* 탭 */}
      <div className="tabs">
        <button
          className={activeTab === 'info' ? 'active' : ''}
          onClick={() => setActiveTab('info')}
        >
          ℹ️ 기본정보
        </button>
        <button
          className={activeTab === 'colors' ? 'active' : ''}
          onClick={() => setActiveTab('colors')}
        >
          🎨 색상표
        </button>
        <button
          className={activeTab === 'recommendations' ? 'active' : ''}
          onClick={() => setActiveTab('recommendations')}
        >
          💄 추천
        </button>
      </div>

      {/* 콘텐츠 */}
      <div className="content" id="pdf-content">

        {/* 기본 정보 탭 */}
        {activeTab === 'info' && (
          <div className="info-section">
            <h2>당신의 퍼스널 컬러</h2>

            {/* 색상 팔레트 시각화 */}
            <div className="color-palette">
              <h3>색상 팔레트</h3>
              
              <div className="palette-group">
                <h4>주요 색상 (Primary)</h4>
                <div className="color-grid">
                  {diagnosis.color_palette.primary.map((hex, i) => (
                    <div key={i} className="color-box">
                      <div
                        className="color-swatch"
                        style={{ backgroundColor: hex }}
                      />
                      <p>{hex}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="palette-group">
                <h4>보조 색상 (Secondary)</h4>
                <div className="color-grid">
                  {diagnosis.color_palette.secondary.map((hex, i) => (
                    <div key={i} className="color-box">
                      <div
                        className="color-swatch"
                        style={{ backgroundColor: hex }}
                      />
                      <p>{hex}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="palette-group">
                <h4>악센트 색상 (Accent)</h4>
                <div className="color-grid">
                  {diagnosis.color_palette.accent.map((hex, i) => (
                    <div key={i} className="color-box">
                      <div
                        className="color-swatch"
                        style={{ backgroundColor: hex }}
                      />
                      <p>{hex}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 신뢰도 표 */}
            <div className="confidence-table">
              <h3>신뢰도</h3>
              <table>
                <tbody>
                  <tr>
                    <td>계절 신뢰도</td>
                    <td>{(diagnosis.season_confidence * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td>톤 신뢰도</td>
                    <td>{(diagnosis.tone_confidence * 100).toFixed(1)}%</td>
                  </tr>
                  <tr className="final">
                    <td>최종 신뢰도</td>
                    <td>{(diagnosis.final_confidence * 100).toFixed(1)}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 색상표 탭 */}
        {activeTab === 'colors' && colorDetails && (
          <div className="colors-section">
            <h2>색상 상세 정보</h2>

            <div className="color-table-container">
              <h3>주요 색상</h3>
              <table className="color-table">
                <thead>
                  <tr>
                    <th>색상명</th>
                    <th>HEX 코드</th>
                    <th>RGB</th>
                    <th>미리보기</th>
                  </tr>
                </thead>
                <tbody>
                  {colorDetails.primary.map((color, i) => (
                    <tr key={i}>
                      <td>{color.name}</td>
                      <td className="hex">{color.hex}</td>
                      <td className="rgb">{color.rgb}</td>
                      <td>
                        <div
                          className="preview"
                          style={{ backgroundColor: color.hex }}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h3>보조 색상</h3>
              <table className="color-table">
                <thead>
                  <tr>
                    <th>색상명</th>
                    <th>HEX 코드</th>
                    <th>RGB</th>
                    <th>미리보기</th>
                  </tr>
                </thead>
                <tbody>
                  {colorDetails.secondary.map((color, i) => (
                    <tr key={i}>
                      <td>{color.name}</td>
                      <td className="hex">{color.hex}</td>
                      <td className="rgb">{color.rgb}</td>
                      <td>
                        <div
                          className="preview"
                          style={{ backgroundColor: color.hex }}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 추천 탭 */}
        {activeTab === 'recommendations' && (
          <div className="recommendations-section">
            <h2>맞춤 추천</h2>

            <div className="recommendation-group">
              <h3>💄 추천 화장품</h3>
              <ul>
                {diagnosis.recommendations?.cosmetics?.cushion?.map((item, i) => (
                  <li key={i}>{item.brand} - {item.product}</li>
                ))}
              </ul>
            </div>

            <div className="recommendation-group">
              <h3>👗 추천 패션 색상</h3>
              <ul>
                {diagnosis.recommendations?.fashion?.colors?.map((color, i) => (
                  <li key={i}>{color}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>

      {/* 다운로드 버튼 */}
      <div className="download-section">
        <button
          onClick={downloadPDF}
          disabled={downloading}
          className="download-btn"
        >
          {downloading ? '다운로드 중...' : '📥 PDF 다운로드'}
        </button>
      </div>

      {/* 스타일 */}
      <style jsx>{`
        .result-container {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
          background: white;
        }

        .header {
          text-align: center;
          padding: 30px 0;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-radius: 10px;
          margin-bottom: 20px;
        }

        .header h1 {
          font-size: 32px;
          margin: 0;
        }

        .tabs {
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
          border-bottom: 2px solid #eee;
        }

        .tabs button {
          padding: 10px 20px;
          border: none;
          background: none;
          cursor: pointer;
          font-weight: bold;
          border-bottom: 3px solid transparent;
        }

        .tabs button.active {
          color: #667eea;
          border-bottom-color: #667eea;
        }

        .color-palette {
          margin-bottom: 30px;
        }

        .palette-group {
          margin-bottom: 20px;
        }

        .color-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
          gap: 10px;
          margin-bottom: 15px;
        }

        .color-box {
          text-align: center;
        }

        .color-swatch {
          width: 100%;
          aspect-ratio: 1;
          border-radius: 8px;
          border: 2px solid #ddd;
          margin-bottom: 5px;
        }

        .color-table {
          width: 100%;
          border-collapse: collapse;
          margin-bottom: 20px;
        }

        .color-table th,
        .color-table td {
          border: 1px solid #ddd;
          padding: 10px;
          text-align: left;
        }

        .color-table th {
          background: #f0f0f0;
          font-weight: bold;
        }

        .preview {
          width: 50px;
          height: 50px;
          border-radius: 5px;
          border: 1px solid #ddd;
        }

        .confidence-table {
          background: #f9f9f9;
          padding: 15px;
          border-radius: 8px;
          margin: 20px 0;
        }

        .confidence-table table {
          width: 100%;
          border-collapse: collapse;
        }

        .confidence-table td {
          padding: 10px;
          border-bottom: 1px solid #ddd;
        }

        .confidence-table td:first-child {
          font-weight: bold;
          width: 50%;
        }

        .confidence-table td:last-child {
          text-align: right;
          color: #667eea;
          font-weight: bold;
        }

        .confidence-table tr.final {
          background: #e8f0ff;
          font-size: 16px;
        }

        .download-section {
          text-align: center;
          padding: 20px 0;
        }

        .download-btn {
          padding: 12px 30px;
          font-size: 16px;
          background: #667eea;
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-weight: bold;
        }

        .download-btn:disabled {
          background: #ccc;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
```

---

## 🖥️ 키오스크: PyQt6 QR코드 생성

```python
# kiosk_with_qr.py

import sys
import requests
import qrcode
from io import BytesIO
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont

class DiagnosisThread(QThread):
    """AI 진단을 백그라운드에서 실행"""
    finished = pyqtSignal(dict)
    
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
    
    def run(self):
        # hierarchical.pt 모델로 추론
        diagnosis = {
            "season": "Spring",
            "tone": "Warm",
            "season_confidence": 0.87,
            "tone_confidence": 0.74,
            "final_confidence": 0.6438,
            "color_palette": {
                "primary": ["#FF8C69", "#FFB366", "#FFA500"],
                "secondary": ["#CD853F", "#DAA520", "#FF6347"],
                "accent": ["#32CD32", "#90EE90"]
            },
            "recommendations": {...},
            "user_info": {
                "name": "사용자",
                "email": ""
            }
        }
        self.finished.emit(diagnosis)

class KioskWithQR(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("퍼스널 컬러 진단 키오스크")
        self.setGeometry(0, 0, 1280, 1024)
        self.setStyleSheet("background: #F4F6F9;")
        
        self.api_url = "http://localhost:5000/api"
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # 제목
        title = QLabel("🎨 퍼스널 컬러 진단 시스템")
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 진단 버튼
        btn = QPushButton("📸 진단 시작")
        btn.setFont(QFont("Arial", 20))
        btn.setMinimumHeight(80)
        btn.clicked.connect(self.start_diagnosis)
        layout.addWidget(btn)
    
    def start_diagnosis(self):
        """진단 시작"""
        
        # 실제로는 카메라로 촬영 후 AI 분석
        self.worker = DiagnosisThread("image.jpg")
        self.worker.finished.connect(self.show_result)
        self.worker.start()
    
    def show_result(self, diagnosis):
        """진단 결과 표시"""
        
        # 현재 UI 클리어
        for i in reversed(range(self.centralWidget().layout().count())): 
            self.centralWidget().layout().itemAt(i).widget().setParent(None)
        
        layout = self.centralWidget().layout()
        
        # 결과 표시
        result_label = QLabel(
            f"🌸 {diagnosis['season']}-{diagnosis['tone']}\n"
            f"신뢰도: {diagnosis['final_confidence']*100:.1f}%"
        )
        result_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(result_label)
        
        # 색상 팔레트 표시
        color_widget = QWidget()
        color_layout = QVBoxLayout(color_widget)
        
        color_label = QLabel("주요 색상:")
        color_layout.addWidget(color_label)
        
        color_grid = QWidget()
        grid_layout = QVBoxLayout(color_grid)
        
        for hex_color in diagnosis['color_palette']['primary']:
            color_box = QLabel()
            color_box.setStyleSheet(f"background-color: {hex_color};")
            color_box.setMinimumHeight(50)
            grid_layout.addWidget(color_box)
        
        color_layout.addWidget(color_grid)
        layout.addWidget(color_widget)
        
        # 서버에 결과 저장
        self.save_diagnosis(diagnosis)
    
    def save_diagnosis(self, diagnosis):
        """진단 결과를 서버에 저장하고 QR코드 생성"""
        
        try:
            # 1. 서버에 결과 저장
            response = requests.post(
                f"{self.api_url}/save-diagnosis",
                json=diagnosis
            )
            
            data = response.json()
            if data['success']:
                qr_code = data['qr_code']
                
                # 2. QR코드 생성
                self.show_qr_code(qr_code)
        
        except Exception as e:
            print(f"오류: {e}")
    
    def show_qr_code(self, qr_code):
        """QR코드 표시"""
        
        try:
            # 서버에서 QR코드 이미지 받기
            response = requests.get(
                f"{self.api_url}/generate-qr/{qr_code}"
            )
            
            if response.status_code == 200:
                data = response.json()
                qr_image_base64 = data['qr_image']
                
                # Base64 → 이미지 변환
                import base64
                qr_bytes = base64.b64decode(qr_image_base64.split(',')[1])
                qr_pixmap = QPixmap()
                qr_pixmap.loadFromData(qr_bytes)
                
                # UI에 QR코드 표시
                layout = self.centralWidget().layout()
                
                qr_label = QLabel("앱에서 QR을 찍으세요!")
                qr_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
                layout.addWidget(qr_label)
                
                qr_image = QLabel()
                qr_image.setPixmap(qr_pixmap)
                qr_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(qr_image)
        
        except Exception as e:
            print(f"QR코드 표시 오류: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KioskWithQR()
    window.show()
    sys.exit(app.exec())
```

---

## 🚀 배포 가이드

### 1️⃣ MongoDB Atlas (클라우드 DB)

```bash
# 1. https://www.mongodb.com/cloud/atlas 접속
# 2. 계정 생성
# 3. 클러스터 생성
# 4. Connection String 복사
# 5. .env에 MONGODB_URI 입력
```

### 2️⃣ Node.js 서버 배포

```bash
# Heroku 배포 (권장)
npm install -g heroku-cli
heroku login
heroku create personal-color-api
git push heroku main

# 또는 Railway
npm install -g railway
railway login
railway init
railway up
```

### 3️⃣ React 앱 배포

```bash
# Vercel 배포
npm install -g vercel
vercel

# 또는 Netlify
npm run build
# Netlify에 build 폴더 드래그 & 드롭
```

---

## 📊 최종 아키텍처 정리

```
┌──────────────────────────────────────────┐
│  키오스크 (오프라인)                     │
│  PyQt6 / React                           │
│  - 사진 촬영                             │
│  - AI 분석 (hierarchical.pt)            │
│  - 결과 표시                             │
│  - QR코드 생성 & 전송                    │
└────────┬─────────────────────────────────┘
         │ HTTP POST
         │ /save-diagnosis
         ▼
┌──────────────────────────────────────────┐
│  Node.js + Express 서버                 │
│  - REST API                              │
│  - QR코드 관리                           │
│  - 데이터 조회                           │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  MongoDB Atlas                           │
│  - 진단 결과 저장                        │
│  - 사용자 정보                           │
└──────────────────────────────────────────┘
         ▲
         │ HTTP GET
         │ /result/{qr_code}
         │
┌────────┴─────────────────────────────────┐
│  모바일 앱 (온라인)                      │
│  React / React Native                    │
│  - QR 스캔                               │
│  - 결과 조회                             │
│  - 색상 상세정보                         │
│  - PDF 다운로드                          │
└──────────────────────────────────────────┘
```

---

## ✅ 체크리스트

### 백엔드
- [ ] Node.js + Express 설치
- [ ] MongoDB Atlas 계정 생성
- [ ] models/Diagnosis.js 생성
- [ ] routes/api.js 구현
- [ ] QR코드 생성 API
- [ ] 결과 조회 API
- [ ] 로컬 테스트
- [ ] Heroku/Railway 배포

### 프론트엔드 (React)
- [ ] QR 스캔 컴포넌트
- [ ] 결과 상세 화면
- [ ] 색상 테이블
- [ ] PDF 다운로드
- [ ] 반응형 디자인
- [ ] 로컬 테스트
- [ ] Vercel/Netlify 배포

### 키오스크
- [ ] PyQt6 UI
- [ ] AI 추론 통합
- [ ] QR코드 생성
- [ ] 서버 통신
- [ ] 테스트

---

## 🎯 최종 요약

```
완벽한 3단계 시스템:

1️⃣  키오스크 (오프라인)
    → 사진 촬영 → AI 분석 → QR코드 생성

2️⃣  클라우드 서버
    → MongoDB에 데이터 저장
    → REST API로 제공

3️⃣  모바일 앱 (온라인)
    → QR 스캔 → 결과 조회
    → 색상 정보 표시 & 다운로드

= 완벽한 오프라인 + 온라인 통합 시스템! 🚀
```

이제 모든 준비가 완료되었습니다! 🎉
