/**
 * Glaucoma Clinical Support — clean clinical dashboard.
 *
 * Left: patient measurements + fundus upload.
 * Right: model staging, guideline decision support, MD progression, attention map.
 * No 3D, particles, or animation theatrics — just legible clinical information.
 */

import React, { useState, useEffect } from 'react'
import * as api from './api'
import InferencePanel from './components/InferencePanel'
import DecisionPanel from './components/DecisionPanel'
import ProgressionGraph from './components/ProgressionGraph'
import AttentionHeatmap from './components/AttentionHeatmap'
import FundusUpload from './components/FundusUpload'

// Curated measurement inputs with units and a plausible default patient (moderate).
const FIELDS = [
  { key: 'age', label: 'Age', unit: 'yr', def: 68, step: 1 },
  { key: 'iop_od', label: 'IOP (right)', unit: 'mmHg', def: 25, step: 0.1 },
  { key: 'iop_os', label: 'IOP (left)', unit: 'mmHg', def: 24, step: 0.1 },
  { key: 'cup_disc_ratio', label: 'Vertical cup-disc ratio', unit: '', def: 0.72, step: 0.01 },
  { key: 'rnfl_average', label: 'RNFL average', unit: 'µm', def: 78, step: 1 },
  { key: 'rnfl_superior', label: 'RNFL superior', unit: 'µm', def: 86, step: 1 },
  { key: 'rnfl_inferior', label: 'RNFL inferior', unit: 'µm', def: 74, step: 1 },
  { key: 'mean_deviation_od', label: 'Mean deviation (right)', unit: 'dB', def: -8.5, step: 0.1 },
  { key: 'mean_deviation_os', label: 'Mean deviation (left)', unit: 'dB', def: -7.8, step: 0.1 },
  { key: 'pattern_sd', label: 'Pattern SD', unit: 'dB', def: 6.2, step: 0.1 },
  { key: 'va_od', label: 'Visual acuity (right)', unit: 'dec', def: 0.7, step: 0.05 },
  { key: 'va_os', label: 'Visual acuity (left)', unit: 'dec', def: 0.75, step: 0.05 },
  { key: 'bmi', label: 'BMI', unit: '', def: 27, step: 0.1 },
  { key: 'hba1c', label: 'HbA1c', unit: '%', def: 5.6, step: 0.1 },
  { key: 'systolic_bp', label: 'Systolic BP', unit: 'mmHg', def: 134, step: 1 },
  { key: 'diastolic_bp', label: 'Diastolic BP', unit: 'mmHg', def: 84, step: 1 },
]
const BINARY = [
  { key: 'sex', label: 'Male' },
  { key: 'diabetes', label: 'Diabetes' },
  { key: 'hypertension', label: 'Hypertension' },
  { key: 'family_history', label: 'Family history of glaucoma' },
]
// Defaults for categoricals the model expects but the form does not surface.
const CAT_DEFAULTS = { treatment: 'drops', eye_color: 'brown', ethnicity: 'caucasian' }

function initialValues() {
  const v = { ...CAT_DEFAULTS }
  FIELDS.forEach((f) => (v[f.key] = f.def))
  BINARY.forEach((b) => (v[b.key] = b.key === 'sex' ? 1 : 0))
  return v
}

const card = { background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: '10px', padding: '16px' }
const sectionLabel = {
  fontSize: '11px', letterSpacing: '0.06em', textTransform: 'uppercase',
  color: 'var(--text-muted)', fontWeight: 600, marginBottom: '10px',
}

export default function App() {
  const [patientId, setPatientId] = useState('P001')
  const [values, setValues] = useState(initialValues)
  const [inference, setInference] = useState(null)
  const [progression, setProgression] = useState(null)
  const [decision, setDecision] = useState(null)
  const [iopReduction, setIopReduction] = useState(0.3)
  const [featureNames, setFeatureNames] = useState(null)
  const [status, setStatus] = useState('checking')
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.healthCheck().then(() => setStatus('online')).catch(() => setStatus('offline'))
    api.getFeatureNames().then((d) => setFeatureNames(d.feature_names)).catch(() => {})
  }, [])

  const setVal = (k, v) => setValues((p) => ({ ...p, [k]: v }))

  const run = async (kind) => {
    setError(null); setBusy(kind)
    try {
      if (kind === 'stage') {
        setInference(await api.predict(values, patientId, 50))
      } else if (kind === 'progress') {
        setProgression(await api.simulate(values, patientId, 60, iopReduction))
      } else if (kind === 'decision') {
        setDecision(await api.recommendTreatment(values, patientId))
      }
    } catch (e) {
      setError(e?.response?.data?.detail ?? e.message ?? 'Request failed')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div style={{ maxWidth: 1280, margin: '0 auto', padding: '20px' }}>
      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div>
          <h1 style={{ fontSize: '20px', fontWeight: 700 }}>Glaucoma Clinical Support</h1>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Transformer staging · MC-Dropout uncertainty · guideline decision support
          </div>
        </div>
        <div style={{ fontSize: '12px', fontFamily: 'var(--mono)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%',
                         background: status === 'online' ? 'var(--normal)' : status === 'offline' ? 'var(--severe)' : 'var(--suspect)' }} />
          API {status}
        </div>
      </header>

      <div style={{ background: '#fffbeb', border: '1px solid #fde68a', color: '#92400e',
                    borderRadius: '8px', padding: '8px 12px', fontSize: '12px', marginBottom: '16px' }}>
        Research / education demo trained on <b>synthetic</b> data. Not a medical device and not for clinical use.
      </div>

      {error && (
        <div style={{ background: 'var(--warn-bg)', border: '1px solid var(--warn-border)', color: 'var(--warn-text)',
                      borderRadius: '8px', padding: '10px 12px', fontSize: '13px', marginBottom: '16px' }}>
          {error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '16px', alignItems: 'start' }}>
        {/* Left column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={card}>
            <div style={sectionLabel}>Patient</div>
            <input value={patientId} onChange={(e) => setPatientId(e.target.value)}
                   style={inputStyle} placeholder="Patient ID" />

            <div style={{ ...sectionLabel, marginTop: '16px' }}>Measurements</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              {FIELDS.map((f) => (
                <label key={f.key} style={{ fontSize: '12px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{f.label}{f.unit ? ` (${f.unit})` : ''}</span>
                  <input type="number" step={f.step} value={values[f.key]}
                         onChange={(e) => setVal(f.key, e.target.value === '' ? '' : Number(e.target.value))}
                         style={{ ...inputStyle, marginTop: '2px' }} />
                </label>
              ))}
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '12px' }}>
              {BINARY.map((b) => (
                <label key={b.key} style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <input type="checkbox" checked={!!values[b.key]}
                         onChange={(e) => setVal(b.key, e.target.checked ? 1 : 0)} />
                  {b.label}
                </label>
              ))}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '16px' }}>
              <button style={btnPrimary} disabled={busy === 'stage'} onClick={() => run('stage')}>
                {busy === 'stage' ? 'Staging…' : 'Stage patient'}
              </button>
              <button style={btnSecondary} disabled={busy === 'decision'} onClick={() => run('decision')}>
                {busy === 'decision' ? 'Computing…' : 'Decision support'}
              </button>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Projected IOP lowering from treatment: <b>{Math.round(iopReduction * 100)}%</b>
              </div>
              <input type="range" min="0" max="0.6" step="0.05" value={iopReduction}
                     onChange={(e) => setIopReduction(Number(e.target.value))} />
              <button style={btnSecondary} disabled={busy === 'progress'} onClick={() => run('progress')}>
                {busy === 'progress' ? 'Projecting…' : 'Project progression'}
              </button>
            </div>
          </div>

          <FundusUpload patientId={patientId} />
        </div>

        {/* Right column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <InferencePanel data={inference} />
          <DecisionPanel data={decision} />
          <ProgressionGraph data={progression} />
          <AttentionHeatmap data={inference} featureNames={featureNames} />
        </div>
      </div>
    </div>
  )
}

const inputStyle = {
  width: '100%', padding: '7px 9px', borderRadius: '7px', border: '1px solid var(--border)',
  fontSize: '13px', fontFamily: 'var(--mono)', color: 'var(--text)', background: 'white', outline: 'none',
}
const btnPrimary = {
  padding: '10px', borderRadius: '8px', border: '1px solid var(--accent)', background: 'var(--accent)',
  color: 'white', fontWeight: 600, fontSize: '14px', cursor: 'pointer',
}
const btnSecondary = {
  padding: '10px', borderRadius: '8px', border: '1px solid var(--accent)', background: 'white',
  color: 'var(--accent)', fontWeight: 600, fontSize: '14px', cursor: 'pointer',
}
