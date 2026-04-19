import React from 'react';
import { useNavigate } from 'react-router-dom';

const S = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px 24px',
    background: 'linear-gradient(160deg, #1a1512 0%, #231e1a 100%)',
  },
  badge: {
    color: '#c9a96e',
    fontSize: 12,
    letterSpacing: 6,
    marginBottom: 24,
    textTransform: 'uppercase',
  },
  title: {
    fontFamily: 'Georgia, serif',
    fontSize: 36,
    color: '#f5efe6',
    textAlign: 'center',
    marginBottom: 8,
    lineHeight: 1.3,
  },
  subtitle: {
    fontSize: 15,
    color: '#9a8a7a',
    textAlign: 'center',
    marginBottom: 48,
    lineHeight: 1.6,
  },
  paletteRow: {
    display: 'flex',
    gap: 8,
    marginBottom: 48,
  },
  swatch: (color) => ({
    width: 44,
    height: 44,
    borderRadius: 22,
    background: color,
    border: '2px solid rgba(255,255,255,0.1)',
  }),
  scanBtn: {
    width: '100%',
    maxWidth: 320,
    padding: '18px 0',
    background: '#c9a96e',
    color: '#1a1512',
    border: 'none',
    borderRadius: 40,
    fontSize: 18,
    fontWeight: 700,
    cursor: 'pointer',
    letterSpacing: 1,
    marginBottom: 16,
    fontFamily: 'Georgia, serif',
  },
  hint: {
    fontSize: 13,
    color: '#6a5a4a',
    textAlign: 'center',
    lineHeight: 1.5,
  },
  divider: {
    width: 40,
    height: 1,
    background: '#3a3028',
    margin: '32px auto',
  },
  steps: {
    width: '100%',
    maxWidth: 320,
  },
  step: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 14,
    marginBottom: 20,
  },
  stepNum: {
    width: 28,
    height: 28,
    borderRadius: 14,
    background: '#3a3028',
    border: '1px solid #c9a96e',
    color: '#c9a96e',
    fontSize: 13,
    fontWeight: 700,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  stepText: {
    fontSize: 14,
    color: '#c8bfb0',
    lineHeight: 1.5,
    paddingTop: 4,
  },
};

const DEMO_COLORS = ['#FDDBB4', '#F9A87A', '#F47C5A', '#F9C784', '#E8855C'];

const STEPS = [
  '키오스크에서 퍼스널 컬러 진단을 받으세요.',
  '결과 화면의 QR 코드를 스마트폰으로 스캔하세요.',
  '상세한 색상 정보와 추천을 확인하세요.',
];

export default function Home() {
  const navigate = useNavigate();

  return (
    <div style={S.container}>
      <p style={S.badge}>Personal Color</p>
      <h1 style={S.title}>나만의 컬러를<br />확인하세요</h1>
      <p style={S.subtitle}>
        키오스크에서 진단받은<br />
        퍼스널 컬러 결과를 스마트폰에서 확인하세요.
      </p>

      <div style={S.paletteRow}>
        {DEMO_COLORS.map((c) => (
          <div key={c} style={S.swatch(c)} />
        ))}
      </div>

      <button style={S.scanBtn} onClick={() => navigate('/scan')}>
        QR 코드 스캔
      </button>
      <p style={S.hint}>키오스크 화면의 QR 코드를 찍으세요.</p>

      <div style={S.divider} />

      <div style={S.steps}>
        {STEPS.map((text, i) => (
          <div key={i} style={S.step}>
            <div style={S.stepNum}>{i + 1}</div>
            <p style={S.stepText}>{text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
