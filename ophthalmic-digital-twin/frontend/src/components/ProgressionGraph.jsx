/**
 * ProgressionGraph — disease trajectory over simulated horizon.
 * Recharts ComposedChart with historical (solid cyan), future (dashed orange),
 * confidence band area, current-time reference line, and glass tooltip.
 */

import React, { useMemo } from 'react'
import {
  ComposedChart,
  Line,
  Area,
  ReferenceLine,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'

// ── Custom glass tooltip ──────────────────────────────────

function GlassTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'rgba(2,4,8,0.92)',
      border: '1px solid rgba(0,200,255,0.25)',
      borderRadius: '6px',
      padding: '8px 12px',
      backdropFilter: 'blur(12px)',
    }}>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', fontFamily: 'var(--font-data)' }}>
        T = {label}
      </div>
      {payload.map((p) => (
        <div key={p.name} style={{
          fontSize: '11px',
          fontFamily: 'var(--font-data)',
          color: p.color ?? '#00c8ff',
          marginBottom: '2px',
        }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
        </div>
      ))}
    </div>
  )
}

/**
 * @component Disease trajectory graph showing historical and simulated future predictions.
 * @param {Object|null} simulationData - SimulateResponse from /simulate endpoint
 * @param {Object|null} inferenceData  - PredictResponse (for current confidence as baseline)
 * @param {number} currentTimestep
 */
export default function ProgressionGraph({ simulationData, inferenceData, currentTimestep }) {
  const chartData = useMemo(() => {
    if (!inferenceData && !simulationData) return []

    const data = []
    const basePred = typeof inferenceData?.prediction === 'number' ? inferenceData.prediction : 0

    // Historical point (t=0 = current state)
    data.push({
      t: currentTimestep ?? 0,
      historical: basePred,
      confidence: inferenceData?.confidence ?? 0.5,
    })

    if (simulationData?.predictions) {
      simulationData.predictions.forEach((pred, i) => {
        const lower = simulationData.confidence_lower?.[i] ?? 0
        const upper = simulationData.confidence_upper?.[i] ?? 1
        data.push({
          t: (currentTimestep ?? 0) + i + 1,
          future: typeof pred === 'number' ? pred : 0,
          band: [lower, upper],
          bandLow: lower,
          bandHigh: upper,
          uncertainty: simulationData.uncertainties?.[i] ?? 0,
        })
      })
    }

    return data
  }, [inferenceData, simulationData, currentTimestep])

  if (chartData.length === 0) {
    return (
      <div style={{
        width: '100%', height: '100%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--text-muted)', fontSize: '12px', fontFamily: 'var(--font-data)',
        letterSpacing: '0.08em',
      }}>
        RUN SIMULATION TO SEE PROGRESSION
      </div>
    )
  }

  const currentT = currentTimestep ?? 0

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart
        data={chartData}
        margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
      >
        <defs>
          <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ff6b35" stopOpacity={0.25} />
            <stop offset="100%" stopColor="#ff6b35" stopOpacity={0.04} />
          </linearGradient>
        </defs>

        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(255,255,255,0.04)"
          vertical={false}
        />

        <XAxis
          dataKey="t"
          tick={{ fill: 'rgba(226,232,240,0.45)', fontSize: 10, fontFamily: "'Space Mono'" }}
          axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
          tickLine={false}
          label={{ value: 'TIMESTEP', position: 'insideBottomRight', fill: 'rgba(226,232,240,0.3)', fontSize: 9, fontFamily: "'Space Mono'" }}
        />
        <YAxis
          tick={{ fill: 'rgba(226,232,240,0.45)', fontSize: 10, fontFamily: "'Space Mono'" }}
          axisLine={false}
          tickLine={false}
          width={30}
        />

        <Tooltip content={<GlassTooltip />} />

        {/* Confidence band for future */}
        <Area
          type="monotone"
          dataKey="bandHigh"
          stroke="none"
          fill="url(#bandGradient)"
          name="upper bound"
          legendType="none"
          activeDot={false}
          connectNulls
        />
        <Area
          type="monotone"
          dataKey="bandLow"
          stroke="none"
          fill="var(--bg)"
          name="lower bound"
          legendType="none"
          activeDot={false}
          connectNulls
        />

        {/* Historical line */}
        <Line
          type="monotone"
          dataKey="historical"
          stroke="#00c8ff"
          strokeWidth={2}
          dot={{ fill: '#00c8ff', r: 4, strokeWidth: 0 }}
          activeDot={{ r: 6, fill: '#00c8ff', strokeWidth: 0 }}
          name="historical"
          connectNulls
          isAnimationActive
          animationDuration={1000}
          style={{ filter: 'drop-shadow(0 0 4px #00c8ff)' }}
        />

        {/* Future prediction line */}
        <Line
          type="monotone"
          dataKey="future"
          stroke="#ff6b35"
          strokeWidth={2}
          strokeDasharray="6 4"
          dot={{ fill: '#ff6b35', r: 3, strokeWidth: 0 }}
          activeDot={{ r: 5, fill: '#ff6b35', strokeWidth: 0 }}
          name="simulated"
          connectNulls
          isAnimationActive
          animationDuration={1200}
          style={{ filter: 'drop-shadow(0 0 4px #ff6b35)' }}
        />

        {/* Current timestep reference line */}
        <ReferenceLine
          x={currentT}
          stroke="#00c8ff"
          strokeWidth={1.5}
          strokeDasharray="none"
          style={{ filter: 'drop-shadow(0 0 6px #00c8ff)' }}
          label={{
            value: 'NOW',
            position: 'top',
            fill: '#00c8ff',
            fontSize: 9,
            fontFamily: "'Space Mono'",
          }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
