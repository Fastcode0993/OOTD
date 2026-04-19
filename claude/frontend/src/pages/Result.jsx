import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import ColorPalette from '../components/ColorPalette';

const API_BASE = process.env.REACT_APP_API_URL || '/api';

const S = {
  page: {
    minHeight: '100vh',
    background: '#1a1512',
    paddingBottom: 40,
  },
  heroSection: {
    padding: '0 0 32px',
    textAlign: 'center',
    position: 'relative',
    overflow: 'hidden',
  },
  heroBg: (color) => ({
    position: 'absolute',
    inset: 0,
    background: `radial-gradient(ellipse at top, ${color}33 0%, transparent 70%)`,
    pointerEvents: 'none',
  }),
  backRow: {
    display: 'flex',
    alignItems: 'center',
    padding: '16px 16px 0',
  },
  backBtn: {
    background: 'transparent',
    border: 'none',
    color: '#c9a96e',
    fontSize: 15,
    cursor: 'pointer',
    padding: '4px 0',
  },
  badge: {
    color: '#c9a96e',
    fontSize: 11,
    letterSpacing: 5,
    marginTop: 24,
    marginBottom: 12,
  },
  colorName: {
    fontFamily: 'Georgia, serif',
    fontSize: 32,
    color: '#f5efe6',
    marginBottom: 8,
    lineHeight: 1.2,
    padding: '0 24px',
  },
  confidenceRow: {
    display: 'flex',
    gap: 12,
    justifyContent: 'center',
    marginBottom: 8,
    flexWrap: 'wrap',
    padding: '0 16px',
  },
  confBadge: (color) => ({
    background: '#231e1a',
    border: `1px solid ${color}55`,
    color,
    padding: '4px 12px',
    borderRadius: 12,
    fontSize: 12,
  }),
  desc: {
    color: '#9a8a7a',
    fontSize: 14,
    lineHeight: 1.6,
    padding: '12px 24px 0',
  },
  section: {
    margin: '0 16px 24px',
    background: '#231e1a',
    borderRadius: 16,
    padding: '20px',
    border: '1px solid #3a3028',
  },
  sectionTitle: {
    color: '#c9a96e',
    fontSize: 12,
    letterSpacing: 3,
    marginBottom: 16,
    textTransform: 'uppercase',
  },
  top3Row: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  top3Item: (isFirst) => ({
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '10px 14px',
    background: isFirst ? '#2e2820' : '#1e1a16',
    borderRadius: 10,
    border: `1px solid ${isFirst ? '#c9a96e' : '#3a3028'}`,
  }),
  rankNum: (isFirst) => ({
    color: isFirst ? '#c9a96e' : '#6a5a4a',
    fontWeight: 700,
    fontSize: 14,
    minWidth: 20,
  }),
  itemLabel: {
    flex: 1,
    color: '#f5efe6',
    fontSize: 14,
  },
  itemConf: {
    color: '#9a8a7a',
    fontSize: 13,
  },
  recoGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: 10,
  },
  recoCard: {
    background: '#1e1a16',
    borderRadius: 10,
    padding: '12px',
    border: '1px solid #3a3028',
  },
  recoSwatch: (hex) => ({
    width: '100%',
    height: 48,
    borderRadius: 6,
    background: hex,
    marginBottom: 8,
  }),
  recoName: {
    color: '#f5efe6',
    fontSize: 13,
    fontWeight: 600,
    marginBottom: 2,
  },
  recoColorName: {
    color: '#c9a96e',
    fontSize: 11,
  },
  recoTip: {
    color: '#9a8a7a',
    fontSize: 11,
    marginTop: 4,
    lineHeight: 1.4,
  },
  tabRow: {
    display: 'flex',
    overflowX: 'auto',
    padding: '0 16px',
    marginBottom: 16,
    gap: 8,
    scrollbarWidth: 'none',
  },
  tab: (active) => ({
    padding: '8px 18px',
    borderRadius: 20,
    border: `1px solid ${active ? '#c9a96e' : '#3a3028'}`,
    background: active ? '#c9a96e20' : 'transparent',
    color: active ? '#c9a96e' : '#6a5a4a',
    fontSize: 13,
    cursor: 'pointer',
    whiteSpace: 'nowrap',
    flexShrink: 0,
  }),
  loadingBox: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  loadingText: { color: '#c9a96e', fontSize: 15 },
  spinner: {
    width: 40,
    height: 40,
    border: '3px solid #3a3028',
    borderTop: '3px solid #c9a96e',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  errorBox: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    gap: 16,
    textAlign: 'center',
  },
  errorTitle: { color: '#f5efe6', fontSize: 20 },
  errorText: { color: '#9a8a7a', fontSize: 14, lineHeight: 1.6 },
  retryBtn: {
    padding: '12px 32px',
    background: '#c9a96e',
    color: '#1a1512',
    border: 'none',
    borderRadius: 24,
    fontSize: 15,
    fontWeight: 700,
    cursor: 'pointer',
  },
};

const RECO_TABS = ['fashion', 'makeup', 'hair', 'interior'];
const RECO_LABELS = { fashion: '패션', makeup: '메이크업', hair: '헤어', interior: '인테리어' };

export default function Result() {
  const { code }   = useParams();
  const navigate   = useNavigate();
  const [data, setData]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState('');
  const [activeTab, setActiveTab] = useState('fashion');

  useEffect(() => {
    const fetchResult = async () => {
      try {
        setLoading(true);
        // 8자 대문자 QR 코드 vs UUID session_id 구분
        let endpoint;
        if (/^[A-Z0-9]{8}$/.test(code)) {
          endpoint = `${API_BASE}/result/${code}`;
        } else {
          endpoint = `${API_BASE}/result/session/${code}`;
        }
        const resp = await axios.get(endpoint);
        setData(resp.data.data);
      } catch (err) {
        setError(err.response?.data?.error || '결과를 불러올 수 없습니다.');
      } finally {
        setLoading(false);
      }
    };
    fetchResult();
  }, [code]);

  if (loading) {
    return (
      <div style={S.loadingBox}>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        <div style={S.spinner} />
        <p style={S.loadingText}>결과를 불러오는 중...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div style={S.errorBox}>
        <h2 style={S.errorTitle}>결과를 찾을 수 없어요</h2>
        <p style={S.errorText}>{error || 'QR 코드를 다시 확인해주세요.'}</p>
        <button style={S.retryBtn} onClick={() => navigate('/scan')}>
          다시 스캔
        </button>
      </div>
    );
  }

  const primaryColor = data.color_palette?.primary?.[0] || '#c9a96e';
  const recoItems    = data.recommendations?.[activeTab] || [];

  return (
    <div style={S.page}>
      {/* 히어로 섹션 */}
      <div style={S.heroSection}>
        <div style={S.heroBg(primaryColor)} />
        <div style={S.backRow}>
          <button style={S.backBtn} onClick={() => navigate('/')}>← 처음으로</button>
        </div>
        <p style={S.badge}>PERSONAL COLOR RESULT</p>
        <h1 style={S.colorName}>{data.label_ko || data.personal_color}</h1>
        <div style={S.confidenceRow}>
          <span style={S.confBadge('#c9a96e')}>
            계절 {((data.season_confidence || 0) * 100).toFixed(0)}%
          </span>
          <span style={S.confBadge('#c4858a')}>
            톤 {((data.tone_confidence || 0) * 100).toFixed(0)}%
          </span>
          <span style={S.confBadge('#7ec8a4')}>
            최종 {((data.final_confidence || 0) * 100).toFixed(1)}%
          </span>
        </div>
        {data.color_palette?.description && (
          <p style={S.desc}>{data.color_palette.description}</p>
        )}
      </div>

      {/* 색상 팔레트 */}
      <div style={S.section}>
        <p style={S.sectionTitle}>✦  어울리는 색상 팔레트</p>
        <ColorPalette palette={data.color_palette} />
      </div>

      {/* TOP 3 */}
      {data.top3?.length > 0 && (
        <div style={S.section}>
          <p style={S.sectionTitle}>✦  진단 순위</p>
          <div style={S.top3Row}>
            {data.top3.slice(0, 3).map((item, i) => (
              <div key={i} style={S.top3Item(i === 0)}>
                <span style={S.rankNum(i === 0)}>{i === 0 ? '👑' : i + 1}</span>
                <span style={S.itemLabel}>{item.label_ko || item.color}</span>
                <span style={S.itemConf}>{(item.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 추천 아이템 */}
      {Object.keys(data.recommendations || {}).some(
        (k) => data.recommendations[k]?.length > 0
      ) && (
        <div style={{ marginBottom: 24 }}>
          <div style={{ padding: '0 16px 12px' }}>
            <p style={{ color: '#c9a96e', fontSize: 12, letterSpacing: 3, textTransform: 'uppercase' }}>
              ✦  추천 아이템
            </p>
          </div>
          <div style={S.tabRow}>
            {RECO_TABS.map((tab) => (
              <button
                key={tab}
                style={S.tab(activeTab === tab)}
                onClick={() => setActiveTab(tab)}
              >
                {RECO_LABELS[tab]}
              </button>
            ))}
          </div>
          <div style={{ padding: '0 16px' }}>
            {recoItems.length > 0 ? (
              <div style={S.recoGrid}>
                {recoItems.map((item, i) => (
                  <div key={i} style={S.recoCard}>
                    <div style={S.recoSwatch(item.color_hex || '#888')} />
                    <p style={S.recoName}>{item.item_name_ko || item.item_name_en}</p>
                    <p style={S.recoColorName}>{item.color_name_ko || item.color_hex}</p>
                    {item.tip_ko && <p style={S.recoTip}>{item.tip_ko}</p>}
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#6a5a4a', fontSize: 13, textAlign: 'center', padding: 20 }}>
                추천 아이템이 없습니다.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
