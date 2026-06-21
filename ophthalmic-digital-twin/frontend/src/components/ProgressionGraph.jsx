/**
 * ProgressionGraph — visual-field (Mean Deviation) projection over time.
 * Plots treated vs untreated MD trajectories with an 80% confidence band on the
 * treated curve. Lower (more negative) MD = worse field. Recharts, no animation theatrics.
 */

import React from 'react'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'

const card = {
  background: 'var(--panel)',
  border: '1px solid var(--border)',
  borderRadius: '10px',
  padding: '16px',
}

export default function ProgressionGraph({ data }) {
  if (!data) {
    return (
      <div style={{ ...card, color: 'var(--text-muted)', fontSize: '13px' }}>
        Run <b>Project progression</b> to estimate how the visual field (Mean Deviation)
        may change over time, treated vs untreated.
      </div>
    )
  }

  const rows = data.months.map((m, i) => ({
    month: m,
    treated: data.md_treated[i],
    untreated: data.md_untreated[i],
    band: [data.md_lower[i], data.md_upper[i]],
  }))

  return (
    <div style={card}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '8px' }}>
        <div style={{ fontSize: '11px', letterSpacing: '0.06em', textTransform: 'uppercase',
                      color: 'var(--text-muted)', fontWeight: 600 }}>
          Projected Mean Deviation (dB)
        </div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--text-muted)' }}>
          treated {data.treated_slope_db_yr}/yr · untreated {data.untreated_slope_db_yr}/yr
        </div>
      </div>
      <div style={{ width: '100%', height: 230 }}>
        <ResponsiveContainer>
          <ComposedChart data={rows} margin={{ top: 6, right: 12, bottom: 4, left: -8 }}>
            <CartesianGrid stroke="#eef1f5" />
            <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#6b7280' }}
                   label={{ value: 'months', position: 'insideBottomRight', offset: -2, fontSize: 11, fill: '#9ca3af' }} />
            <YAxis tick={{ fontSize: 11, fill: '#6b7280' }} domain={['dataMin - 1', 1]} />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid var(--border)' }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Area dataKey="band" name="80% band (treated)" stroke="none" fill="#2563eb" fillOpacity={0.12} />
            <Line dataKey="treated" name="Treated" stroke="#2563eb" strokeWidth={2} dot={false} />
            <Line dataKey="untreated" name="Untreated" stroke="#dc2626" strokeWidth={2}
                  strokeDasharray="5 4" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px', lineHeight: 1.4 }}>
        {data.assumptions}
      </div>
    </div>
  )
}
