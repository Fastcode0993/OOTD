const crypto   = require('crypto');
const express  = require('express');
const rateLimit = require('express-rate-limit');
const QRCode   = require('qrcode');
const Diagnosis = require('../models/Diagnosis');

const router = express.Router();

const CLIENT_URL    = process.env.CLIENT_URL    || 'http://localhost:3000';
const KIOSK_API_KEY = process.env.KIOSK_API_KEY || 'kiosk-dev-key-2024';
const QR_EXPIRE_DAYS = 30;

// ── 32자 대문자 영숫자 QR 코드 생성 (A-Z0-9, crypto.randomInt 사용) ───────
const QR_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
function generateQrCode() {
  return Array.from({ length: 32 }, () => QR_CHARS[crypto.randomInt(QR_CHARS.length)]).join('');
}

// ── 속도 제한: 결과 조회 1분에 15회 초과 시 차단 ─────────────────────────
const resultRateLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 15,
  standardHeaders: 'draft-7',
  legacyHeaders: false,
  message: { success: false, error: '요청이 너무 많습니다. 잠시 후 다시 시도해주세요.' },
  keyGenerator: (req) => req.ip,
});

// ── 인증 미들웨어 (키오스크 전용 엔드포인트) ─────────────────────────────
function kioskAuth(req, res, next) {
  const key = req.headers['x-api-key'] || req.headers['x-kiosk-key'];
  if (!key || key !== KIOSK_API_KEY) {
    return res.status(401).json({ success: false, error: '인증 실패' });
  }
  next();
}

// ─────────────────────────────────────────────────────────────────────────
//  1. 진단 결과 저장 (키오스크에서 호출)
//  POST /api/save-diagnosis
// ─────────────────────────────────────────────────────────────────────────
router.post('/save-diagnosis', kioskAuth, async (req, res) => {
  try {
    const {
      session_id,
      season,
      personal_color,
      label_ko,
      label_en,
      season_confidence,
      tone_confidence,
      final_confidence,
      color_palette,
      recommendations,
      captured_at,
    } = req.body;

    if (!season || !personal_color) {
      return res.status(400).json({ success: false, error: '필수 필드 누락 (season, personal_color)' });
    }

    // 32자 대문자 영숫자 QR 코드 생성 (crypto.randomInt — 암호학적 안전 난수)
    const qr_code  = generateQrCode();
    const now      = captured_at ? new Date(captured_at) : new Date();
    const expires  = new Date(now.getTime() + QR_EXPIRE_DAYS * 24 * 60 * 60 * 1000);

    const diagnosis = new Diagnosis({
      qr_code,
      session_id:        session_id || null,
      season,
      personal_color,
      label_ko:          label_ko   || '',
      label_en:          label_en   || '',
      season_confidence: season_confidence || 0,
      tone_confidence:   tone_confidence   || 0,
      final_confidence:  final_confidence  || 0,
      color_palette:     color_palette     || { primary: [], secondary: [], accent: [] },
      recommendations:   recommendations   || {},
      captured_at:       now,
      expires_at:        expires,
    });

    await diagnosis.save();

    res.json({
      success:  true,
      qr_code,
      scan_url: `${CLIENT_URL}/result/${qr_code}`,
      message:  '진단 결과가 저장되었습니다.',
    });

  } catch (err) {
    if (err.code === 11000) {
      return res.status(409).json({ success: false, error: '중복된 session_id' });
    }
    console.error('save-diagnosis error:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────
//  2. 진단 결과 조회 (모바일에서 QR 스캔 후 호출)
//  GET /api/result/:qr_code
// ─────────────────────────────────────────────────────────────────────────
router.get('/result/:qr_code', resultRateLimiter, async (req, res) => {
  try {
    const { qr_code } = req.params;
    const diagnosis = await Diagnosis.findOneAndUpdate(
      { qr_code },
      { $inc: { scan_count: 1 } },
      { new: true },
    );

    if (!diagnosis) {
      return res.status(404).json({ success: false, error: '진단 결과를 찾을 수 없습니다.' });
    }

    // 만료 체크 (30일)
    if (diagnosis.expires_at && new Date() > diagnosis.expires_at) {
      return res.status(410).json({
        success: false,
        error: 'QR 코드가 만료되었습니다.',
        expired_at: diagnosis.expires_at,
      });
    }

    res.json({ success: true, data: diagnosis });

  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────
//  3. 세션 ID로 조회 (로컬 세션 QR 스캔 대응)
//  GET /api/result/session/:session_id
// ─────────────────────────────────────────────────────────────────────────
router.get('/result/session/:session_id', resultRateLimiter, async (req, res) => {
  try {
    const { session_id } = req.params;
    const diagnosis = await Diagnosis.findOne({ session_id });

    if (!diagnosis) {
      return res.status(404).json({ success: false, error: '결과를 찾을 수 없습니다.' });
    }

    // 만료 체크
    if (diagnosis.expires_at && new Date() > diagnosis.expires_at) {
      return res.status(410).json({
        success: false,
        error: 'QR 코드가 만료되었습니다.',
        expired_at: diagnosis.expires_at,
      });
    }

    res.json({ success: true, data: diagnosis });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────
//  4. QR 코드 이미지 생성
//  GET /api/qr/:qr_code
// ─────────────────────────────────────────────────────────────────────────
router.get('/qr/:qr_code', async (req, res) => {
  try {
    const { qr_code } = req.params;
    const url = `${CLIENT_URL}/result/${qr_code}`;

    const qr_image = await QRCode.toDataURL(url, {
      errorCorrectionLevel: 'H',
      type: 'image/png',
      quality: 0.95,
      margin: 1,
      width: 400,
      color: { dark: '#000000', light: '#ffffff' },
    });

    res.json({ success: true, qr_code, qr_image, url });

  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────
//  5. 색상 상세 정보 (HEX → 색상명, RGB 변환 포함)
//  GET /api/result/:qr_code/colors
// ─────────────────────────────────────────────────────────────────────────
router.get('/result/:qr_code/colors', async (req, res) => {
  try {
    const { qr_code } = req.params;
    const diagnosis = await Diagnosis.findOne({ qr_code });

    if (!diagnosis) {
      return res.status(404).json({ success: false, error: '결과를 찾을 수 없습니다.' });
    }

    const hexToRgb = (hex) => {
      const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return r
        ? `${parseInt(r[1], 16)}, ${parseInt(r[2], 16)}, ${parseInt(r[3], 16)}`
        : '0, 0, 0';
    };

    // 색상명 룩업 테이블 (대표 색상)
    const COLOR_NAMES = {
      '#FDDBB4': '복숭아', '#F9A87A': '살구', '#F47C5A': '코랄',
      '#FFAA44': '오렌지', '#FF7733': '탠저린', '#FFD700': '골드',
      '#C47A1E': '카멜', '#A0522D': '테라코타', '#8B5E3C': '브라운',
      '#E8D5E8': '라벤더', '#C9A7C9': '모브', '#B8A0C8': '퍼플',
      '#B0A8B9': '그레이', '#9890A8': '모브그레이', '#8888A0': '슬레이트',
      '#FF69B4': '핫핑크', '#87CEEB': '스카이블루', '#FF4499': '로즈',
      '#8B4513': '다크브라운', '#A0522D': '세도나', '#6B4C2A': '카카오',
      '#B5A642': '올리브', '#9B8B5A': '카키', '#A09060': '샌드',
      '#FFCBA4': '피치', '#FFB87A': '애프리콧', '#F4A460': '샌디베이지',
      '#1C1C3A': '네이비블랙', '#2D2D5A': '딥네이비', '#8B0000': '버건디',
      '#0033CC': '로열블루', '#CC0066': '퓨시아', '#0099CC': '코발트',
      '#708090': '슬레이트그레이', '#6A5ACD': '슬레이트블루', '#7B7B7B': '그레이',
    };

    const mapColors = (hexList, usage) =>
      (hexList || []).map((hex) => ({
        name:  COLOR_NAMES[hex] || '색상',
        hex,
        rgb:   hexToRgb(hex),
        usage,
      }));

    res.json({
      success: true,
      season:  diagnosis.season,
      personal_color: diagnosis.personal_color,
      label_ko: diagnosis.label_ko,
      colors: {
        primary:   mapColors(diagnosis.color_palette?.primary,   '주요 색상'),
        secondary: mapColors(diagnosis.color_palette?.secondary, '보조 색상'),
        accent:    mapColors(diagnosis.color_palette?.accent,    '악센트'),
      },
    });

  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────
//  6. 헬스 체크
//  GET /api/health
// ─────────────────────────────────────────────────────────────────────────
router.get('/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

module.exports = router;
