/**
 * InferencePanel — real-time inference display with animated confidence gauge,
 * MC Dropout progress, and animated metric cards.
 */

import React, { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// ── Animated counter hook ────────────────────────────────

function useAnimatedValue(target, duration = 800) {
  const [value, setValue] = useState(0)
  const start = useRef(0)
  const startTime = useRef(null)
  const frame = useRef(null)

  useEffect(() => {
    if (target === null || target === undefined) return
    start.current = value
    startTime.current = null

    const animate = (ts) => {
      if (!startTime.current) startTime.current = ts
      const elapsed = ts - startTime.current
      const progress = Math.min(elapsed / duration, 1)
      // Ease out cubic
      const eased = 1 - (1 - progress) ** 3
      setValue(start.current + (target - start.current) * eased)
      if (progress < 1) frame.current = requestAnimationFrame(animate)
    }
    frame.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame.current)
  }, [target]) // eslint-disable-line

  return value
}

// ── Arc Gauge (SVG) ──────────────────────────────────────

function ConfidenceGauge({ confidence }) {
  const pct = useAnimatedValue(confidence ?? 0, 1000)
  const radius = 50
  const circumference = Math.PI * radius  // half circle
  const strokeDash = (pct * circumference).toFixed(1)

  const color = pct > 0.8 ? '#00c8ff' : pct > 0.5 ? '#ffcc00' : '#ff3300'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
      <svg width="120" height="70" viewBox="-10 0 120 75">
        {/* Track */}
        <path
          d="M 5 65 A 50 50 0 0 1 95 65"
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d="M 5 65 A 50 50 0 0 1 95 65"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${strokeDash} ${circumference}`}
          style={{ filter: `drop-shadow(0 0 6px ${color})`, transition: 'stroke 0.3s' }}
        />
        {/* Percentage text */}
        <text
          x="50" y="58"
          textAnchor="middle"
          fill={color}
          fontSize="16"
          fontFamily="'Space Mono', monospace"
          fontWeight="700"
        >
          {(pct * 100).toFixed(0)}%
        </text>
      </svg>
      <span style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.12em' }}>
        CONFIDENCE
      </span>
    </div>
  )
}

// ── Mini bar ─────────────────────────────────────────────

function MiniBar({ label, value, maxVal = 1, color = '#00c8ff' }) {
  const pct = Math.min((value ?? 0) / maxVal, 1)
  const animPct = useAnimatedValue(pct, 600)

  return (
    <div style={{ marginBottom: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
        <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-data)' }}>
          {label}
        </span>
        <span style={{ fontSize: '10px', color, fontFamily: 'var(--font-data)' }}>
          {(value ?? 0).toFixed(4)}
        </span>
      </div>
      <div style={{
        height: '4px',
        background: 'rgba(255,255,255,0.06)',
        borderRadius: '2px',
        overflow: 'hidden',
      }}>
        <motion.div
          style={{
            height: '100%',
            background: color,
            borderRadius: '2px',
            boxShadow: `0 0 6px ${color}`,
          }}
          initial={{ width: '0%' }}
          animate={{ width: `${animPct * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}

// ── Scanning animation ───────────────────────────────────

function ScanningOverlay({ sampleCount }) {
  return (
    <div style={{ padding: '16px', textAlign: 'center' }}>
      <div style={{ position: 'relative', height: '48px', marginBottom: '16px', overflow: 'hidden', borderRadius: '4px' }}>
        <motion.div
          style={{
            position: 'absolute',
            top: 0, left: 0, right: 0,
            height: '2px',
            background: 'linear-gradient(90deg, transparent, #00c8ff, transparent)',
            boxShadow: '0 0 8px #00c8ff',
          }}
          animate={{ top: ['0%', '100%'] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
        />
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(0,200,255,0.03)',
          border: '1px solid rgba(0,200,255,0.15)',
          borderRadius: '4px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <span style={{
            fontFamily: 'var(--font-data)',
            fontSize: '11px',
            color: '#00c8ff',
            letterSpacing: '0.1em',
          }}>
            RUNNING INFERENCE
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '6px', marginBottom: '12px' }}>
        {[0, 1, 2].map(i => (
          <motion.div
            key={i}
            style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#00c8ff' }}
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.3 }}
          />
        ))}
      </div>

      <div style={{
        fontFamily: 'var(--font-data)',
        fontSize: '11px',
        color: 'var(--text-muted)',
      }}>
        MC Dropout: sample {sampleCount}/50
      </div>
    </div>
  )
}

const cardVariants = {
  hidden: { opacity: 0, y: 12 },
  visible: (i) => ({
    opacity: 1, y: 0,
    transition: { delay: i * 0.1, duration: 0.4, ease: 'easeOut' },
  }),
}

/**
 * @component Right-panel inference results with animated metrics and confidence gauge.
 * @param {Object|null} data - PredictResponse from API
 * @param {boolean} loading - true while inference is running
 * @param {number} mcSampleProgress - animated sample counter during loading
 */
export default function InferencePanel({ data, loading, mcSampleProgress }) {
  const predLabel = data?.prediction_label ?? '—'
  const confidence = data?.confidence ?? null
  const epistemic = data?.epistemic_variance
    ? data.epistemic_variance.reduce((a, b) => a + b, 0) / (data.epistemic_variance.length || 1)
    : null
  const aleatoric = data?.aleatoric_variance ?? null
  const reliable = data?.reliable
  const uncertainty = data?.total_uncertainty ?? null

  const predColor = reliable === false
    ? '#ff6b35'
    : reliable === true
    ? '#00ff88'
    : '#00c8ff'

  return (
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      overflowY: 'auto',
      padding: '0 2px',
    }}>
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              background: 'var(--glass-bg)',
              border: 'var(--glass-border)',
              borderRadius: '8px',
              backdropFilter: 'blur(20px)',
            }}
          >
            <ScanningOverlay sampleCount={mcSampleProgress} />
          </motion.div>
        ) : data ? (
          <motion.div
            key="results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Prediction */}
            <motion.div
              custom={0} variants={cardVariants} initial="hidden" animate="visible"
              style={{
                background: 'var(--glass-bg)',
                border: 'var(--glass-border)',
                borderRadius: '8px',
                padding: '14px',
                marginBottom: '8px',
                backdropFilter: 'blur(20px)',
              }}
            >
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.12em', marginBottom: '6px' }}>
                PREDICTION
              </div>
              <div style={{
                fontSize: '20px',
                fontWeight: 700,
                color: predColor,
                textShadow: `0 0 12px ${predColor}`,
                fontFamily: 'var(--font-data)',
                marginBottom: '8px',
                wordBreak: 'break-word',
              }}>
                {predLabel.toUpperCase()}
              </div>
              <ConfidenceGauge confidence={confidence} />
            </motion.div>

            {/* Reliability badge */}
            <motion.div
              custom={1} variants={cardVariants} initial="hidden" animate="visible"
              style={{
                background: 'var(--glass-bg)',
                border: `1px solid ${reliable ? 'rgba(0,255,136,0.3)' : 'rgba(255,107,53,0.3)'}`,
                borderRadius: '8px',
                padding: '10px 14px',
                marginBottom: '8px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              {reliable ? (
                <>
                  <motion.div
                    style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#00ff88' }}
                    animate={{ boxShadow: ['0 0 4px #00ff88', '0 0 12px #00ff88', '0 0 4px #00ff88'] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                  <span style={{ fontSize: '11px', color: '#00ff88', fontFamily: 'var(--font-data)', letterSpacing: '0.1em' }}>
                    RELIABLE
                  </span>
                </>
              ) : (
                <>
                  <motion.div
                    style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ff6b35' }}
                    animate={{ opacity: [1, 0.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity }}
                  />
                  <span style={{ fontSize: '11px', color: '#ff6b35', fontFamily: 'var(--font-data)', letterSpacing: '0.1em' }}>
                    UNCERTAIN — REVIEW
                  </span>
                </>
              )}
            </motion.div>

            {/* Variance bars */}
            <motion.div
              custom={2} variants={cardVariants} initial="hidden" animate="visible"
              style={{
                background: 'var(--glass-bg)',
                border: 'var(--glass-border)',
                borderRadius: '8px',
                padding: '12px 14px',
              }}
            >
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.12em', marginBottom: '10px' }}>
                UNCERTAINTY DECOMPOSITION
              </div>
              <MiniBar label="EPISTEMIC" value={epistemic} maxVal={0.5} color="#7c3aed" />
              <MiniBar label="ALEATORIC" value={aleatoric} maxVal={0.5} color="#00c8ff" />
              <MiniBar label="TOTAL" value={uncertainty} maxVal={1} color="#ff6b35" />
            </motion.div>
          </motion.div>
        ) : (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              background: 'var(--glass-bg)',
              border: 'var(--glass-border)',
              borderRadius: '8px',
              padding: '24px',
              textAlign: 'center',
              color: 'var(--text-muted)',
              fontSize: '12px',
              fontFamily: 'var(--font-data)',
              letterSpacing: '0.08em',
            }}
          >
            AWAITING PATIENT DATA
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
