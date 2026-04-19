import React, { useState } from 'react';

const S = {
  group: { marginBottom: 16 },
  groupTitle: {
    color: '#9a8a7a',
    fontSize: 11,
    letterSpacing: 2,
    marginBottom: 10,
    textTransform: 'uppercase',
  },
  swatchRow: { display: 'flex', gap: 8, flexWrap: 'wrap' },
  swatch: (hex, selected) => ({
    width: 52,
    height: 52,
    borderRadius: 10,
    background: hex,
    border: `2px solid ${selected ? '#c9a96e' : 'transparent'}`,
    cursor: 'pointer',
    transition: 'transform 0.15s',
    transform: selected ? 'scale(1.12)' : 'scale(1)',
    flexShrink: 0,
  }),
  tooltip: {
    marginTop: 10,
    padding: '8px 14px',
    background: '#2e2820',
    borderRadius: 8,
    border: '1px solid #3a3028',
  },
  hexText: {
    color: '#c9a96e',
    fontSize: 15,
    fontWeight: 700,
    letterSpacing: 1,
  },
  rgbText: {
    color: '#9a8a7a',
    fontSize: 12,
    marginTop: 2,
  },
};

function hexToRgb(hex) {
  const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return r
    ? `rgb(${parseInt(r[1], 16)}, ${parseInt(r[2], 16)}, ${parseInt(r[3], 16)})`
    : '';
}

function SwatchGroup({ title, colors }) {
  const [selected, setSelected] = useState(null);

  if (!colors || colors.length === 0) return null;

  return (
    <div style={S.group}>
      <p style={S.groupTitle}>{title}</p>
      <div style={S.swatchRow}>
        {colors.map((hex, i) => (
          <div
            key={i}
            style={S.swatch(hex, selected === hex)}
            onClick={() => setSelected(selected === hex ? null : hex)}
            title={hex}
          />
        ))}
      </div>
      {selected && (
        <div style={S.tooltip}>
          <p style={S.hexText}>{selected.toUpperCase()}</p>
          <p style={S.rgbText}>{hexToRgb(selected)}</p>
        </div>
      )}
    </div>
  );
}

export default function ColorPalette({ palette }) {
  if (!palette) return null;

  return (
    <div>
      <SwatchGroup title="주요 색상"  colors={palette.primary}   />
      <SwatchGroup title="보조 색상"  colors={palette.secondary} />
      {palette.accent?.length > 0 && (
        <SwatchGroup title="악센트" colors={palette.accent} />
      )}
    </div>
  );
}
