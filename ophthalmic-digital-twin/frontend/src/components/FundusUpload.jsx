/**
 * FundusUpload — analyze a color fundus photograph.
 * Clinically honest output: referable glaucoma (binary) + estimated vertical CDR,
 * with a GradCAM overlay. Warns clearly when the model is uncalibrated.
 */

import React, { useState, useRef, useCallback } from 'react'

const card = {
  background: 'var(--panel)',
  border: '1px solid var(--border)',
  borderRadius: '10px',
  padding: '16px',
}
const labelStyle = {
  fontSize: '11px', letterSpacing: '0.06em', textTransform: 'uppercase',
  color: 'var(--text-muted)', fontWeight: 600, marginBottom: '8px',
}

export default function FundusUpload({ patientId }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [showGradcam, setShowGradcam] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const pick = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError(null)
  }, [])

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const form = new FormData()
      form.append('patient_id', patientId || 'unknown')
      form.append('image', file)
      const res = await fetch('/api/v1/analyze-fundus', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail ?? `HTTP ${res.status}`)
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message ?? 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const referColor = result?.referable ? 'var(--severe)' : 'var(--normal)'

  return (
    <div style={card}>
      <div style={labelStyle}>Fundus photograph (structural screen)</div>

      <div
        onClick={() => inputRef.current?.click()}
        onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files[0]) }}
        onDragOver={(e) => e.preventDefault()}
        style={{
          border: '1.5px dashed var(--border)', borderRadius: '8px', padding: '14px',
          textAlign: 'center', cursor: 'pointer', color: 'var(--text-muted)', fontSize: '12px',
          background: '#f8fafc',
        }}
      >
        <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }}
               onChange={(e) => { pick(e.target.files[0]); e.target.value = '' }} />
        {file ? 'Click or drop to replace image' : 'Drop or click to upload a fundus image (JPEG / PNG)'}
      </div>

      {file && (
        <button onClick={analyze} disabled={loading}
                style={{ marginTop: '10px', width: '100%', padding: '9px', borderRadius: '8px',
                         border: '1px solid var(--accent)', background: loading ? '#cbd5e1' : 'var(--accent)',
                         color: 'white', fontWeight: 600, cursor: loading ? 'default' : 'pointer' }}>
          {loading ? 'Analyzing…' : 'Analyze fundus'}
        </button>
      )}

      {error && (
        <div style={{ marginTop: '10px', background: 'var(--warn-bg)', border: '1px solid var(--warn-border)',
                      color: 'var(--warn-text)', borderRadius: '8px', padding: '8px 10px', fontSize: '12px' }}>
          {error}
        </div>
      )}

      {(preview || result) && (
        <div style={{ marginTop: '12px' }}>
          {result && (
            <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
              {['Original', 'GradCAM'].map((m) => {
                const on = (m === 'GradCAM') === showGradcam
                return (
                  <button key={m} onClick={() => setShowGradcam(m === 'GradCAM')}
                          style={{ flex: 1, padding: '6px', borderRadius: '6px', fontSize: '12px',
                                   border: `1px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
                                   background: on ? 'var(--accent-soft)' : 'white',
                                   color: on ? 'var(--accent)' : 'var(--text-muted)', cursor: 'pointer' }}>
                    {m}
                  </button>
                )
              })}
            </div>
          )}
          <img
            src={result && showGradcam && result.gradcam_overlay
                  ? `data:image/png;base64,${result.gradcam_overlay}`
                  : result?.image_preview ? `data:image/png;base64,${result.image_preview}` : preview}
            alt="Fundus"
            style={{ width: '100%', borderRadius: '8px', border: '1px solid var(--border)', display: 'block' }}
          />
        </div>
      )}

      {result && (
        <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {!result.calibrated && (
            <div style={{ background: 'var(--warn-bg)', border: '1px solid var(--warn-border)',
                          color: 'var(--warn-text)', borderRadius: '8px', padding: '8px 10px', fontSize: '12px' }}>
              ⚠ Uncalibrated model — outputs are not meaningful until the fundus CNN is trained.
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '16px', fontWeight: 700, color: referColor }}>{result.referral_label}</span>
            <span style={{ fontFamily: 'var(--mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
              p(glaucoma) {(result.glaucoma_probability * 100).toFixed(1)}% · est. vertical CDR {result.estimated_vertical_cdr}
            </span>
          </div>
          {result.top_findings?.length > 0 && (
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              GradCAM focus: {result.top_findings.map(([r]) => r).slice(0, 3).join(', ')}
            </div>
          )}
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' }}>{result.model_note}</div>
        </div>
      )}
    </div>
  )
}
