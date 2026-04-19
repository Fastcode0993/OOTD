import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Scanner } from '@yudiel/react-qr-scanner';

const S = {
  container: {
    minHeight: '100vh',
    background: '#000',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
  },
  header: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 20,
    padding: '20px 16px 16px',
    background: 'linear-gradient(to bottom, rgba(0,0,0,0.8), transparent)',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  backBtn: {
    background: 'rgba(255,255,255,0.15)',
    border: '1px solid rgba(255,255,255,0.3)',
    color: '#fff',
    padding: '8px 16px',
    borderRadius: 20,
    cursor: 'pointer',
    fontSize: 14,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 17,
    fontWeight: 600,
  },
  scannerWrap: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 64,
  },
  scannerBox: {
    width: '100%',
    maxWidth: 360,
    borderRadius: 16,
    overflow: 'hidden',
    border: '3px solid #c9a96e',
  },
  hint: {
    position: 'absolute',
    bottom: 80,
    left: 0,
    right: 0,
    textAlign: 'center',
    color: '#fff',
    fontSize: 14,
    zIndex: 20,
    background: 'rgba(0,0,0,0.5)',
    padding: '8px 20px',
    margin: '0 auto',
    width: 'fit-content',
    borderRadius: 20,
  },
  status: {
    position: 'absolute',
    bottom: 36,
    left: 0,
    right: 0,
    textAlign: 'center',
    color: '#c9a96e',
    fontSize: 14,
    zIndex: 20,
    padding: '0 24px',
  },
  errorBox: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    background: '#1a1512',
  },
  errorTitle: {
    color: '#f5efe6',
    fontSize: 22,
    marginBottom: 12,
    textAlign: 'center',
  },
  errorText: {
    color: '#9a8a7a',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 1.6,
    marginBottom: 32,
  },
  retryBtn: {
    padding: '14px 40px',
    background: '#c9a96e',
    color: '#1a1512',
    border: 'none',
    borderRadius: 30,
    fontSize: 16,
    fontWeight: 700,
    cursor: 'pointer',
  },
};

export default function Scan() {
  const navigate = useNavigate();
  const [status, setStatus]     = useState('QR 코드를 프레임 안에 맞춰주세요');
  const [camError, setCamError] = useState(false);
  const [done, setDone]         = useState(false);

  const handleScan = useCallback((results) => {
    if (done || !results || results.length === 0) return;
    const decodedText = results[0].rawValue;
    if (!decodedText) return;

    setDone(true);
    setStatus('QR 코드 인식됨! 결과를 불러오는 중...');

    let code = decodedText.trim();
    const match = code.match(/\/result\/([A-Z0-9a-z-]+)/);
    if (match) code = match[1];

    navigate(`/result/${code}`);
  }, [done, navigate]);

  const handleError = useCallback((err) => {
    console.error('QR scanner error:', err);
    setCamError(true);
  }, []);

  if (camError) {
    return (
      <div style={S.errorBox}>
        <h2 style={S.errorTitle}>카메라를 사용할 수 없습니다</h2>
        <p style={S.errorText}>
          카메라 접근 권한을 허용하거나<br />
          브라우저 설정을 확인해주세요.
        </p>
        <button style={S.retryBtn} onClick={() => window.location.reload()}>
          다시 시도
        </button>
      </div>
    );
  }

  return (
    <div style={S.container}>
      <div style={S.header}>
        <button style={S.backBtn} onClick={() => navigate(-1)}>← 뒤로</button>
        <span style={S.headerTitle}>QR 코드 스캔</span>
      </div>

      <div style={S.scannerWrap}>
        <div style={S.scannerBox}>
          <Scanner
            onScan={handleScan}
            onError={handleError}
            constraints={{ facingMode: 'environment' }}
            scanDelay={300}
            styles={{ container: { width: '100%' } }}
          />
        </div>
      </div>

      <p style={S.hint}>QR 코드를 프레임 안에 맞춰주세요</p>
      <div style={S.status}>{status}</div>
    </div>
  );
}
