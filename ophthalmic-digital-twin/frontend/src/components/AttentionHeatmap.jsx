/**
 * AttentionHeatmap — feature x feature self-attention grid from the transformer.
 * Plain SVG, blue intensity scale. Helps show which measurements the model
 * related to each other when staging this patient.
 */

import React from 'react'

const card = {
  background: 'var(--panel)',
  border: '1px solid var(--border)',
  borderRadius: '10px',
  padding: '16px',
}

function cellColor(v, max) {
  const t = max > 0 ? Math.min(1, v / max) : 0
  // white -> clinical blue
  const r = Math.round(255 - t * (255 - 37))
  const g = Math.round(255 - t * (255 - 99))
  const b = Math.round(255 - t * (255 - 235))
  return `rgb(${r},${g},${b})`
}

export default function AttentionHeatmap({ data, featureNames }) {
  const matrix = data?.attention_heatmap
  if (!matrix || matrix.length === 0 || (matrix.length === 1 && matrix[0].length === 1)) {
    return (
      <div style={{ ...card, color: 'var(--text-muted)', fontSize: '13px' }}>
        Attention map appears after staging a patient.
      </div>
    )
  }

  const n = matrix.length
  const flat = matrix.flat()
  const max = Math.max(...flat)
  const size = Math.min(360, 28 * n)
  const cell = size / n
  const names = (featureNames || []).slice(0, n)

  return (
    <div style={card}>
      <div style={{ fontSize: '11px', letterSpacing: '0.06em', textTransform: 'uppercase',
                    color: 'var(--text-muted)', fontWeight: 600, marginBottom: '10px' }}>
        Feature attention map
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg width={size + 4} height={size + 4}>
          {matrix.map((row, i) =>
            row.map((v, j) => (
              <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1}
                    rx={2} fill={cellColor(v, max)}>
                <title>{`${names[i] || i} → ${names[j] || j}: ${v.toFixed(3)}`}</title>
              </rect>
            ))
          )}
        </svg>
      </div>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px' }}>
        Darker = stronger learned association between two measurements. Hover a cell for values.
      </div>
    </div>
  )
}
