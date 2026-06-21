/**
 * DecisionPanel — guideline-based decision support: target IOP + next ladder step.
 * Explicitly framed as decision support, not a prescription.
 */

import React from 'react'

const card = {
  background: 'var(--panel)',
  border: '1px solid var(--border)',
  borderRadius: '10px',
  padding: '16px',
}

export default function DecisionPanel({ data }) {
  if (!data) {
    return (
      <div style={{ ...card, color: 'var(--text-muted)', fontSize: '13px' }}>
        Run <b>Decision support</b> for a guideline-based target IOP and the next step
        on the standard treatment ladder.
      </div>
    )
  }
  return (
    <div style={card}>
      <div style={{ fontSize: '11px', letterSpacing: '0.06em', textTransform: 'uppercase',
                    color: 'var(--text-muted)', fontWeight: 600, marginBottom: '10px' }}>
        Decision support — {data.stage}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '12px' }}>
        <Metric label="Baseline IOP" value={`${data.baseline_iop} mmHg`} />
        <Metric label="Target IOP"
                value={data.target_iop != null ? `≤ ${data.target_iop} mmHg` : 'n/a'}
                accent={!data.at_target} />
      </div>

      <p style={{ fontSize: '13px', margin: '0 0 8px' }}>{data.target_rationale}</p>

      <div style={{
        background: data.at_target ? '#f0fdf4' : '#fff7ed',
        border: `1px solid ${data.at_target ? '#bbf7d0' : '#fed7aa'}`,
        borderRadius: '8px', padding: '10px 12px', marginBottom: '10px',
      }}>
        <div style={{ fontWeight: 700, fontSize: '13px', marginBottom: '4px' }}>{data.next_step}</div>
        <div style={{ fontSize: '12px', color: 'var(--text)' }}>{data.rationale}</div>
      </div>

      {data.risk_factors?.length > 0 && (
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>
          Risk factors: {data.risk_factors.join(', ')}
        </div>
      )}

      <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
        {data.disclaimer}
      </div>
    </div>
  )
}

function Metric({ label, value, accent }) {
  return (
    <div style={{ background: '#f8fafc', border: '1px solid var(--border)', borderRadius: '8px', padding: '8px 10px' }}>
      <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
        {label}
      </div>
      <div style={{ fontSize: '18px', fontWeight: 700, fontFamily: 'var(--mono)',
                    color: accent ? 'var(--moderate)' : 'var(--text)' }}>
        {value}
      </div>
    </div>
  )
}
