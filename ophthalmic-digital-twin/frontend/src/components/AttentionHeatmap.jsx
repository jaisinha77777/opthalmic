/**
 * AttentionHeatmap — SVG attention weight grid + SHAP waterfall chart.
 * Supports toggling between attention view and SHAP values view.
 */

import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// ── Color scale ──────────────────────────────────────────

function heatColor(v) {
  // 0 → #020408, 0.5 → #00c8ff, 1 → #ffffff
  const t = Math.max(0, Math.min(1, v))
  if (t < 0.5) {
    const r = Math.round(2 + t * 2 * (0 - 2))
    const g = Math.round(4 + t * 2 * (200 - 4))
    const b = Math.round(8 + t * 2 * (255 - 8))
    return `rgb(${r},${g},${b})`
  } else {
    const f = (t - 0.5) * 2
    const r = Math.round(0 + f * 255)
    const g = Math.round(200 + f * (255 - 200))
    const b = 255
    return `rgb(${r},${g},${b})`
  }
}

// ── Attention Heatmap grid ───────────────────────────────

function HeatmapGrid({ heatmap, featureNames }) {
  const [tooltip, setTooltip] = useState(null)
  const svgRef = useRef()

  if (!heatmap || heatmap.length === 0) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '20px', textAlign: 'center' }}>
        No attention data available.
      </div>
    )
  }

  const N = heatmap.length
  const cellSize = Math.max(8, Math.min(24, Math.floor(220 / N)))
  const svgSize = N * cellSize

  const names = featureNames?.length ? featureNames : Array.from({ length: N }, (_, i) => `F${i}`)

  return (
    <div style={{ position: 'relative', overflowX: 'auto' }}>
      <svg
        ref={svgRef}
        width={svgSize}
        height={svgSize}
        style={{ display: 'block', cursor: 'crosshair' }}
      >
        {heatmap.map((row, ri) =>
          row.map((val, ci) => (
            <rect
              key={`${ri}-${ci}`}
              x={ci * cellSize}
              y={ri * cellSize}
              width={cellSize}
              height={cellSize}
              fill={heatColor(val)}
              opacity={0.9}
              onMouseEnter={(e) => {
                setTooltip({
                  x: e.clientX,
                  y: e.clientY,
                  text: `${names[ri] ?? `R${ri}`} → ${names[ci] ?? `C${ci}`}: ${val.toFixed(3)}`,
                })
              }}
              onMouseLeave={() => setTooltip(null)}
            />
          ))
        )}
      </svg>

      {tooltip && (
        <div style={{
          position: 'fixed',
          left: tooltip.x + 12,
          top: tooltip.y - 8,
          background: 'rgba(2,4,8,0.95)',
          border: '1px solid rgba(0,200,255,0.3)',
          borderRadius: '4px',
          padding: '4px 10px',
          fontSize: '11px',
          fontFamily: 'var(--font-data)',
          color: '#00c8ff',
          pointerEvents: 'none',
          zIndex: 1000,
          whiteSpace: 'nowrap',
        }}>
          {tooltip.text}
        </div>
      )}
    </div>
  )
}

// ── Feature importance bars ───────────────────────────────

function ImportanceBars({ importance, featureNames }) {
  if (!importance || importance.length === 0) return null

  const maxVal = Math.max(...importance.map(Math.abs), 1e-9)
  const top10 = importance
    .map((v, i) => ({ val: v, name: featureNames?.[i] ?? `F${i}` }))
    .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
    .slice(0, 10)

  return (
    <div style={{ marginTop: '12px' }}>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.1em', marginBottom: '8px' }}>
        FEATURE IMPORTANCE
      </div>
      {top10.map((item, i) => (
        <div key={item.name} style={{ marginBottom: '6px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
            <span style={{
              fontSize: '10px',
              fontFamily: 'var(--font-data)',
              color: i === 0 ? '#00c8ff' : 'var(--text-muted)',
              maxWidth: '140px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>
              {item.name}
            </span>
            <span style={{ fontSize: '10px', fontFamily: 'var(--font-data)', color: '#00c8ff' }}>
              {item.val.toFixed(3)}
            </span>
          </div>
          <div style={{ height: '4px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px' }}>
            <motion.div
              style={{
                height: '100%',
                borderRadius: '2px',
                background: i === 0
                  ? 'linear-gradient(90deg, #00c8ff, #7c3aed)'
                  : '#00c8ff',
                boxShadow: i === 0 ? '0 0 8px #7c3aed' : 'none',
              }}
              initial={{ width: 0 }}
              animate={{ width: `${(Math.abs(item.val) / maxVal) * 100}%` }}
              transition={{ duration: 0.7, delay: i * 0.06, ease: 'easeOut' }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

// ── SHAP Waterfall chart ──────────────────────────────────

function ShapWaterfall({ shapValues, featureNames, baseValue }) {
  if (!shapValues || shapValues.length === 0) return (
    <div style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '20px', textAlign: 'center' }}>
      No SHAP data available.
    </div>
  )

  const top = shapValues
    .map((v, i) => ({ val: v, name: featureNames?.[i] ?? `F${i}` }))
    .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
    .slice(0, 10)

  const maxAbs = Math.max(...top.map(d => Math.abs(d.val)), 1e-9)

  return (
    <div>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.1em', marginBottom: '8px' }}>
        SHAP WATERFALL
      </div>
      <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginBottom: '10px', fontFamily: 'var(--font-data)' }}>
        base value: {(baseValue ?? 0).toFixed(3)}
      </div>
      {top.map((item, i) => {
        const positive = item.val >= 0
        const color = positive ? '#00c8ff' : '#ff6b35'
        const widthPct = (Math.abs(item.val) / maxAbs) * 100

        return (
          <div key={item.name} style={{ marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{
              width: '120px',
              fontSize: '10px',
              fontFamily: 'var(--font-data)',
              color: 'var(--text-muted)',
              textAlign: 'right',
              flexShrink: 0,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>
              {item.name}
            </span>
            <div style={{ flex: 1, height: '14px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
              <motion.div
                style={{
                  height: '100%',
                  background: color,
                  borderRadius: '3px',
                  float: positive ? 'left' : 'right',
                }}
                initial={{ width: 0 }}
                animate={{ width: `${widthPct}%` }}
                transition={{ duration: 0.6, delay: i * 0.05 }}
              />
            </div>
            <span style={{ fontSize: '10px', fontFamily: 'var(--font-data)', color, minWidth: '44px' }}>
              {item.val > 0 ? '+' : ''}{item.val.toFixed(3)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── Toggle button ─────────────────────────────────────────

function ToggleButton({ active, onClick, label }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '4px 12px',
        fontSize: '10px',
        fontFamily: 'var(--font-data)',
        letterSpacing: '0.1em',
        border: active ? '1px solid #00c8ff' : '1px solid rgba(255,255,255,0.1)',
        background: active ? 'rgba(0,200,255,0.12)' : 'transparent',
        color: active ? '#00c8ff' : 'var(--text-muted)',
        borderRadius: '4px',
        cursor: 'pointer',
        transition: 'all 0.2s',
      }}
    >
      {label}
    </button>
  )
}

/**
 * @component Attention heatmap grid with feature importance and SHAP waterfall toggle.
 * @param {Object|null} data - PredictResponse from API (attention_heatmap, shap_values, etc.)
 * @param {string[]} featureNames
 */
export default function AttentionHeatmap({ data, featureNames }) {
  const [view, setView] = useState('attention')

  return (
    <div style={{ width: '100%', height: '100%', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {/* Toggle */}
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        <ToggleButton active={view === 'attention'} onClick={() => setView('attention')} label="ATTENTION" />
        <ToggleButton active={view === 'shap'} onClick={() => setView('shap')} label="SHAP" />
      </div>

      <AnimatePresence mode="wait">
        {view === 'attention' ? (
          <motion.div
            key="attention"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            style={{
              background: 'var(--glass-bg)',
              border: 'var(--glass-border)',
              borderRadius: '8px',
              padding: '12px',
            }}
          >
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.1em', marginBottom: '8px' }}>
              ATTENTION HEATMAP (aggregated)
            </div>
            <HeatmapGrid
              heatmap={data?.attention_heatmap ?? []}
              featureNames={featureNames}
            />
            <ImportanceBars
              importance={data?.feature_importance ?? []}
              featureNames={featureNames}
            />
          </motion.div>
        ) : (
          <motion.div
            key="shap"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            style={{
              background: 'var(--glass-bg)',
              border: 'var(--glass-border)',
              borderRadius: '8px',
              padding: '12px',
            }}
          >
            <ShapWaterfall
              shapValues={data?.shap_values ?? []}
              featureNames={featureNames}
              baseValue={0}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
