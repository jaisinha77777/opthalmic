/**
 * InferencePanel — staged result for the tabular clinical model.
 * Shows the predicted stage, calibrated confidence, reliability flag (MC-Dropout),
 * per-class probabilities, and the top contributing features (attention/SHAP).
 */

import React from 'react'
import { colorForIndex, SEVERITY_ORDER } from '../severity'

const card = {
  background: 'var(--panel)',
  border: '1px solid var(--border)',
  borderRadius: '10px',
  padding: '16px',
}
const label = {
  fontSize: '11px',
  letterSpacing: '0.06em',
  textTransform: 'uppercase',
  color: 'var(--text-muted)',
  marginBottom: '8px',
  fontWeight: 600,
}

function Bar({ name, value, color, active }) {
  return (
    <div style={{ marginBottom: '7px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '3px' }}>
        <span style={{ color: active ? color : 'var(--text)', fontWeight: active ? 700 : 400 }}>{name}</span>
        <span style={{ fontFamily: 'var(--mono)', color: 'var(--text-muted)' }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: '7px', background: '#eef1f5', borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${value * 100}%`, background: color, borderRadius: '4px' }} />
      </div>
    </div>
  )
}

export default function InferencePanel({ data }) {
  if (!data) {
    return (
      <div style={{ ...card, color: 'var(--text-muted)', fontSize: '13px' }}>
        Enter patient measurements and run <b>Stage patient</b> to see the model's
        severity assessment, calibrated confidence, and feature attribution.
      </div>
    )
  }

  const stageColor = colorForIndex(data.severity_index)
  const reliable = data.reliable
  const topScore = data.top_features?.[0]?.[1] || 1

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={card}>
        <div style={label}>Predicted stage (Hodapp-Parrish-Anderson)</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '22px', fontWeight: 700, color: stageColor }}>
            {data.prediction_label}
          </span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: '13px', color: 'var(--text-muted)' }}>
            {(data.confidence * 100).toFixed(1)}% confidence
          </span>
          <span style={{
            fontSize: '11px', fontWeight: 700, padding: '3px 9px', borderRadius: '20px',
            color: reliable ? '#15803d' : '#b91c1c',
            background: reliable ? '#dcfce7' : '#fee2e2',
          }}>
            {reliable ? 'RELIABLE' : 'LOW CONFIDENCE — REVIEW'}
          </span>
        </div>
        <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
          total uncertainty {data.total_uncertainty?.toFixed(3)} · aleatoric {data.aleatoric_variance?.toFixed(3)}
        </div>
      </div>

      <div style={card}>
        <div style={label}>Class probabilities</div>
        {(data.class_labels || []).map((name, i) => (
          <Bar key={name} name={name} value={data.probabilities[i]}
               color={colorForIndex(SEVERITY_ORDER.indexOf(name))}
               active={i === data.prediction} />
        ))}
      </div>

      {data.top_features?.length > 0 && (
        <div style={card}>
          <div style={label}>Top contributing features (attention)</div>
          {data.top_features.slice(0, 8).map(([name, score]) => (
            <div key={name} style={{ marginBottom: '6px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '2px' }}>
                <span>{name}</span>
                <span style={{ fontFamily: 'var(--mono)', color: 'var(--text-muted)' }}>{score.toFixed(3)}</span>
              </div>
              <div style={{ height: '5px', background: '#eef1f5', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, (score * 100) / topScore)}%`,
                              background: 'var(--accent)', borderRadius: '3px' }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
