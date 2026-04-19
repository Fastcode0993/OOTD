const mongoose = require('mongoose');

const diagnosisSchema = new mongoose.Schema({
  qr_code: {
    type: String,
    unique: true,
    required: true,
    index: true,
  },

  session_id: {
    type: String,
    unique: true,
    sparse: true,
  },

  // 계층적 모델 결과
  season: {
    type: String,
    enum: ['spring', 'summer', 'autumn', 'winter'],
    required: true,
  },
  personal_color: {
    type: String,
    required: true,
  },
  label_ko: String,
  label_en: String,

  // 신뢰도
  season_confidence:  { type: Number, min: 0, max: 1 },
  tone_confidence:    { type: Number, min: 0, max: 1 },
  final_confidence:   { type: Number, min: 0, max: 1 },

  // 색상 팔레트
  color_palette: {
    primary:     [String],
    secondary:   [String],
    accent:      [String],
    description: String,
  },

  // 카테고리별 추천 (키오스크 로컬 DB 기준)
  recommendations: {
    fashion:  [mongoose.Schema.Types.Mixed],
    makeup:   [mongoose.Schema.Types.Mixed],
    hair:     [mongoose.Schema.Types.Mixed],
    interior: [mongoose.Schema.Types.Mixed],
  },

  captured_at: { type: Date, default: Date.now },
  created_at:  { type: Date, default: Date.now },
  expires_at:  { type: Date, required: true, index: true },

  // QR 조회 횟수 (통계용)
  scan_count: { type: Number, default: 0 },
});

module.exports = mongoose.model('Diagnosis', diagnosisSchema);
