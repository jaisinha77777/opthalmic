/**
 * FundusUpload — drag-and-drop fundus image analysis panel.
 *
 * Uploads the image to /api/v1/analyze-fundus, then displays:
 *   - Original image / GradCAM overlay toggle
 *   - Severity prediction with probability bars
 *   - Top anatomical findings
 */

import React, { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const SEVERITY_COLORS = ['#00ff88', '#b8ff00', '#ffcc00', '#ff8c00', '#ff3300']
const SEVERITY_LABELS = ['Normal', 'Suspect', 'Mild', 'Moderate', 'Severe']

// ── Shared micro-styles ────────────────────────────────────

const s = {
  label: {
    fontSize: '9px',
    letterSpacing: '0.15em',
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-data)',
    marginBottom: '4px',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    marginBottom: '4px',
  },
}

// ── Probability bar ────────────────────────────────────────

function ProbBar({ label, prob, color, active }) {
  return (
    <div style={{ marginBottom: '5px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
        <span style={{
          fontSize: '9px',
          fontFamily: 'var(--font-data)',
          color: active ? color : 'var(--text-muted)',
          letterSpacing: '0.08em',
        }}>
          {label}
        </span>
        <span style={{ fontSize: '9px', fontFamily: 'var(--font-data)', color }}>
          {(prob * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{
        height: '3px',
        background: 'rgba(255,255,255,0.05)',
        borderRadius: '2px',
        overflow: 'hidden',
      }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${prob * 100}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          style={{
            height: '100%',
            background: color,
            boxShadow: active ? `0 0 6px ${color}` : 'none',
            borderRadius: '2px',
          }}
        />
      </div>
    </div>
  )
}

// ── Drop zone ─────────────────────────────────────────────

function DropZone({ onFile, loading, hasImage }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const handleDrop = useCallback(e => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) onFile(file)
  }, [onFile])

  const handleDragOver = useCallback(e => {
    e.preventDefault()
    setDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => setDragging(false), [])

  const handleClick = () => inputRef.current?.click()

  const handleInput = e => {
    const file = e.target.files[0]
    if (file) onFile(file)
    e.target.value = ''
  }

  return (
    <motion.div
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      animate={{
        borderColor: dragging
          ? 'rgba(0,200,255,0.8)'
          : hasImage
          ? 'rgba(0,200,255,0.3)'
          : 'rgba(255,255,255,0.1)',
        background: dragging
          ? 'rgba(0,200,255,0.06)'
          : 'rgba(255,255,255,0.02)',
      }}
      style={{
        border: '1.5px dashed rgba(255,255,255,0.1)',
        borderRadius: '8px',
        padding: '16px',
        cursor: 'pointer',
        textAlign: 'center',
        userSelect: 'none',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleInput}
      />
      {loading ? (
        <div style={{ color: '#00c8ff', fontSize: '11px', fontFamily: 'var(--font-data)' }}>
          <motion.span
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          >
            ⟳ ANALYZING FUNDUS…
          </motion.span>
        </div>
      ) : hasImage ? (
        <div style={{ fontSize: '10px', fontFamily: 'var(--font-data)', color: 'var(--text-muted)' }}>
          Click or drop to replace image
        </div>
      ) : (
        <>
          <div style={{ fontSize: '22px', marginBottom: '6px', opacity: 0.4 }}>◎</div>
          <div style={{ fontSize: '10px', fontFamily: 'var(--font-data)', color: 'var(--text-muted)', lineHeight: 1.5 }}>
            Drop fundus photograph here
            <br />
            <span style={{ fontSize: '9px', opacity: 0.6 }}>JPEG · PNG · BMP</span>
          </div>
        </>
      )}
    </motion.div>
  )
}

// ── Main component ─────────────────────────────────────────

export default function FundusUpload({ patientId, onResult }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)    // local object URL
  const [result, setResult] = useState(null)
  const [showGradcam, setShowGradcam] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFile = useCallback(f => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }, [])

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('patient_id', patientId || 'unknown')
      form.append('image', file)

      const res = await fetch('/api/v1/analyze-fundus', {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail ?? `HTTP ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
      if (onResult) onResult(data)
    } catch (e) {
      setError(e.message ?? 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const predColor = result ? SEVERITY_COLORS[result.prediction] : '#00c8ff'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>

      {/* Drop zone */}
      <DropZone onFile={handleFile} loading={loading} hasImage={!!file} />

      {/* Analyze button */}
      {file && !loading && (
        <motion.button
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          onClick={handleAnalyze}
          style={{
            width: '100%',
            padding: '8px 0',
            border: '1px solid rgba(0,200,255,0.35)',
            borderRadius: '5px',
            background: 'rgba(0,200,255,0.1)',
            color: '#00c8ff',
            fontSize: '10px',
            fontFamily: 'var(--font-data)',
            letterSpacing: '0.12em',
            cursor: 'pointer',
          }}
          whileHover={{ scale: 1.02, boxShadow: '0 0 14px rgba(0,200,255,0.15)' }}
          whileTap={{ scale: 0.98 }}
        >
          ◈  ANALYZE FUNDUS IMAGE
        </motion.button>
      )}

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            style={{
              background: 'rgba(255,60,0,0.08)',
              border: '1px solid rgba(255,107,53,0.3)',
              borderRadius: '5px',
              padding: '6px 10px',
              fontSize: '10px',
              fontFamily: 'var(--font-data)',
              color: '#ff6b35',
            }}
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Image display */}
      <AnimatePresence>
        {(preview || result) && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            {/* Toggle */}
            {result && (
              <div style={{ display: 'flex', gap: '4px', marginBottom: '6px' }}>
                {['ORIGINAL', 'GRADCAM'].map(mode => (
                  <button
                    key={mode}
                    onClick={() => setShowGradcam(mode === 'GRADCAM')}
                    style={{
                      flex: 1,
                      padding: '4px 0',
                      border: (mode === 'GRADCAM') === showGradcam
                        ? '1px solid rgba(0,200,255,0.4)'
                        : '1px solid rgba(255,255,255,0.06)',
                      background: (mode === 'GRADCAM') === showGradcam
                        ? 'rgba(0,200,255,0.1)'
                        : 'transparent',
                      color: (mode === 'GRADCAM') === showGradcam ? '#00c8ff' : 'var(--text-muted)',
                      fontSize: '9px',
                      fontFamily: 'var(--font-data)',
                      letterSpacing: '0.1em',
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            )}

            {/* Image */}
            <div style={{
              width: '100%',
              aspectRatio: '1',
              borderRadius: '6px',
              overflow: 'hidden',
              border: '1px solid rgba(255,255,255,0.08)',
              position: 'relative',
            }}>
              <img
                src={
                  result && showGradcam && result.gradcam_overlay
                    ? `data:image/png;base64,${result.gradcam_overlay}`
                    : result?.image_preview
                    ? `data:image/png;base64,${result.image_preview}`
                    : preview
                }
                alt="Fundus"
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
              />
              {result && showGradcam && (
                <div style={{
                  position: 'absolute',
                  bottom: '4px',
                  right: '4px',
                  background: 'rgba(0,0,0,0.7)',
                  borderRadius: '3px',
                  padding: '2px 5px',
                  fontSize: '8px',
                  fontFamily: 'var(--font-data)',
                  color: '#00c8ff',
                  letterSpacing: '0.08em',
                }}>
                  GRAD-CAM
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}
          >
            {/* Prediction badge */}
            <div style={{
              background: 'rgba(0,0,0,0.3)',
              border: `1px solid ${predColor}44`,
              borderRadius: '6px',
              padding: '8px 12px',
            }}>
              <div style={s.label}>DIAGNOSIS</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                <span style={{
                  fontSize: '14px',
                  fontFamily: 'var(--font-ui)',
                  fontWeight: 700,
                  color: predColor,
                  letterSpacing: '0.05em',
                }}>
                  {result.prediction_label}
                </span>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-data)', color: 'var(--text-muted)' }}>
                  {(result.confidence * 100).toFixed(1)}% conf
                </span>
              </div>
            </div>

            {/* Probability bars */}
            <div style={{
              background: 'rgba(0,0,0,0.2)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '6px',
              padding: '8px 10px',
            }}>
              <div style={s.label}>CLASS PROBABILITIES</div>
              {result.probabilities.map((p, i) => (
                <ProbBar
                  key={i}
                  label={SEVERITY_LABELS[i]}
                  prob={p}
                  color={SEVERITY_COLORS[i]}
                  active={i === result.prediction}
                />
              ))}
            </div>

            {/* Anatomical findings */}
            {result.top_findings?.length > 0 && (
              <div style={{
                background: 'rgba(0,0,0,0.2)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: '6px',
                padding: '8px 10px',
              }}>
                <div style={s.label}>ACTIVATED REGIONS</div>
                {result.top_findings.map(([region, score], i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                    <span style={{ fontSize: '9px', fontFamily: 'var(--font-data)', color: 'var(--text-muted)' }}>
                      {region}
                    </span>
                    <span style={{ fontSize: '9px', fontFamily: 'var(--font-data)', color: '#00c8ff' }}>
                      {(score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Disclaimer */}
            <div style={{
              fontSize: '8px',
              fontFamily: 'var(--font-data)',
              color: 'rgba(255,255,255,0.2)',
              lineHeight: 1.4,
              padding: '0 2px',
            }}>
              ⚠ {result.model_note}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
