/**
 * App — main layout for the Ophthalmic Digital Twin UI.
 *
 * Layout:
 *   - Full-screen particle background
 *   - Grid-line overlay (CSS)
 *   - Top navbar
 *   - 3-column layout: left panel | 3D canvas center | right panel (tabbed)
 *   - Bottom: ProgressionGraph
 *
 * All panels slide in on load via Framer Motion.
 */

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ParticleBackground from './components/ParticleBackground'
import TwinCanvas3D from './components/TwinCanvas3D'
import InferencePanel from './components/InferencePanel'
import AttentionHeatmap from './components/AttentionHeatmap'
import ProgressionGraph from './components/ProgressionGraph'
import AgentConsole from './components/AgentConsole'
import FundusUpload from './components/FundusUpload'
import * as api from './api'

// ── Styles ────────────────────────────────────────────────

const styles = {
  root: {
    width: '100vw',
    height: '100vh',
    overflow: 'hidden',
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    background: 'var(--bg)',
  },
  gridOverlay: {
    position: 'fixed',
    inset: 0,
    zIndex: 1,
    pointerEvents: 'none',
    backgroundImage: `
      linear-gradient(rgba(0,200,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,200,255,0.03) 1px, transparent 1px)
    `,
    backgroundSize: '40px 40px',
  },
  navbar: {
    position: 'relative',
    zIndex: 10,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 20px',
    background: 'rgba(2,4,8,0.7)',
    borderBottom: '1px solid rgba(0,200,255,0.1)',
    backdropFilter: 'blur(12px)',
    flexShrink: 0,
  },
  navTitle: {
    fontFamily: 'var(--font-ui)',
    fontSize: '14px',
    fontWeight: 700,
    letterSpacing: '0.2em',
    color: 'var(--text)',
  },
  navDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    background: '#00ff88',
    marginRight: '10px',
    display: 'inline-block',
  },
  body: {
    flex: 1,
    display: 'flex',
    position: 'relative',
    zIndex: 5,
    overflow: 'hidden',
  },
  leftPanel: {
    width: '290px',
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    padding: '12px',
    overflowY: 'auto',
    background: 'rgba(2,4,8,0.4)',
    borderRight: '1px solid rgba(255,255,255,0.05)',
  },
  centerCanvas: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
  },
  rightPanel: {
    width: '300px',
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    padding: '12px',
    background: 'rgba(2,4,8,0.4)',
    borderLeft: '1px solid rgba(255,255,255,0.05)',
    overflow: 'hidden',
  },
  bottomBar: {
    position: 'relative',
    zIndex: 5,
    height: '200px',
    flexShrink: 0,
    background: 'rgba(2,4,8,0.6)',
    borderTop: '1px solid rgba(255,255,255,0.05)',
    padding: '8px 12px 4px',
  },
  glassCard: {
    background: 'var(--glass-bg)',
    border: 'var(--glass-border)',
    borderRadius: '8px',
    backdropFilter: 'blur(16px)',
    transition: 'border-color 0.2s',
  },
  label: {
    fontSize: '9px',
    letterSpacing: '0.15em',
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-data)',
    marginBottom: '6px',
  },
  input: {
    width: '100%',
    padding: '7px 10px',
    background: 'rgba(255,255,255,0.04)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '5px',
    color: 'var(--text)',
    fontSize: '12px',
    fontFamily: 'var(--font-data)',
    outline: 'none',
    boxSizing: 'border-box',
    transition: 'border-color 0.2s',
  },
  button: (variant = 'primary') => ({
    width: '100%',
    padding: '9px 0',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    fontSize: '11px',
    fontFamily: 'var(--font-data)',
    letterSpacing: '0.12em',
    fontWeight: 600,
    transition: 'all 0.2s',
    background: variant === 'primary'
      ? 'linear-gradient(135deg, rgba(0,200,255,0.18), rgba(124,58,237,0.18))'
      : variant === 'success'
      ? 'rgba(0,255,136,0.12)'
      : 'rgba(255,107,53,0.12)',
    color: variant === 'primary' ? '#00c8ff'
          : variant === 'success' ? '#00ff88'
          : '#ff6b35',
    border: `1px solid ${variant === 'primary' ? 'rgba(0,200,255,0.3)'
            : variant === 'success' ? 'rgba(0,255,136,0.3)'
            : 'rgba(255,107,53,0.3)'}`,
    boxShadow: variant === 'primary' ? '0 0 12px rgba(0,200,255,0.08)' : 'none',
  }),
}

// ── Animated dot ──────────────────────────────────────────

function LiveDot({ color = '#00ff88' }) {
  return (
    <motion.span
      style={{
        display: 'inline-block',
        width: '7px', height: '7px',
        borderRadius: '50%',
        background: color,
        marginRight: '10px',
        boxShadow: `0 0 6px ${color}`,
      }}
      animate={{ opacity: [1, 0.3, 1], scale: [1, 0.8, 1] }}
      transition={{ duration: 2, repeat: Infinity }}
    />
  )
}

// ── Tab bar ───────────────────────────────────────────────

function TabBar({ tabs, active, onChange }) {
  return (
    <div style={{ display: 'flex', gap: '4px', marginBottom: '10px', flexShrink: 0 }}>
      {tabs.map(tab => (
        <button
          key={tab}
          onClick={() => onChange(tab)}
          style={{
            flex: 1,
            padding: '6px 0',
            border: active === tab ? '1px solid rgba(0,200,255,0.4)' : '1px solid rgba(255,255,255,0.06)',
            background: active === tab ? 'rgba(0,200,255,0.1)' : 'transparent',
            color: active === tab ? '#00c8ff' : 'var(--text-muted)',
            fontFamily: 'var(--font-data)',
            fontSize: '9px',
            letterSpacing: '0.1em',
            borderRadius: '4px',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {tab}
        </button>
      ))}
    </div>
  )
}

// ── Feature input form ────────────────────────────────────

function FeatureForm({ featureNames, values, onChange }) {
  if (!featureNames?.length) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '11px', fontFamily: 'var(--font-data)', padding: '8px 0' }}>
        Loading features...
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '240px', overflowY: 'auto' }}>
      {featureNames.slice(0, 12).map(name => (
        <div key={name}>
          <div style={styles.label}>{name.toUpperCase()}</div>
          <input
            style={styles.input}
            type="text"
            placeholder="value"
            value={values[name] ?? ''}
            onChange={e => onChange(name, e.target.value)}
            onFocus={e => (e.target.style.borderColor = 'rgba(0,200,255,0.4)')}
            onBlur={e => (e.target.style.borderColor = 'rgba(255,255,255,0.1)')}
          />
        </div>
      ))}
      {featureNames.length > 12 && (
        <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-data)', padding: '4px 0' }}>
          + {featureNames.length - 12} more features (sent as empty if not filled)
        </div>
      )}
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────

export default function App() {
  const [patientId, setPatientId] = useState('P001')
  const [featureValues, setFeatureValues] = useState({})
  const [featureMeta, setFeatureMeta] = useState(null)

  const [inferenceData, setInferenceData] = useState(null)
  const [simulationData, setSimulationData] = useState(null)
  const [treatmentData, setTreatmentData] = useState(null)

  const [inferenceLoading, setInferenceLoading] = useState(false)
  const [simulationLoading, setSimulationLoading] = useState(false)
  const [treatmentLoading, setTreatmentLoading] = useState(false)
  const [mcSampleProgress, setMcSampleProgress] = useState(0)

  const [rightTab, setRightTab] = useState('INFERENCE')
  const [leftTab, setLeftTab] = useState('PARAMETERS')
  const [fundusData, setFundusData] = useState(null)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')
  const [mounted, setMounted] = useState(false)

  // Animate in on mount
  useEffect(() => {
    setTimeout(() => setMounted(true), 100)
    // Check API health
    api.healthCheck()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'))
    // Load feature names
    api.getFeatureNames()
      .then(data => setFeatureMeta(data))
      .catch(() => {})
  }, [])

  // MC sample counter animation during inference
  const mcInterval = useRef(null)
  useEffect(() => {
    if (inferenceLoading) {
      setMcSampleProgress(0)
      let n = 0
      mcInterval.current = setInterval(() => {
        n = Math.min(n + 1, 50)
        setMcSampleProgress(n)
        if (n >= 50) clearInterval(mcInterval.current)
      }, 60)
    } else {
      clearInterval(mcInterval.current)
    }
    return () => clearInterval(mcInterval.current)
  }, [inferenceLoading])

  const handleFundusResult = (data) => {
    setFundusData(data)
    // Mirror into inferenceData shape so InferencePanel/3D canvas react
    setInferenceData(prev => ({
      ...(prev ?? {}),
      prediction: data.prediction,
      prediction_label: data.prediction_label,
      probabilities: data.probabilities,
      confidence: data.confidence,
      top_features: data.top_findings,
      feature_importance: [],
      shap_values: [],
      attention_heatmap: [[0]],
      latent_state_3d: prev?.latent_state_3d ?? [0, 0, 0],
      total_uncertainty: 1 - data.confidence,
    }))
    setRightTab('INFERENCE')
  }

  const handleFeatureChange = (name, value) => {
    setFeatureValues(prev => ({ ...prev, [name]: value === '' ? undefined : isNaN(Number(value)) ? value : Number(value) }))
  }

  const handleAnalyze = async () => {
    if (!patientId.trim()) { setError('Enter a patient ID'); return }
    setError(null)
    setInferenceLoading(true)
    try {
      const data = await api.predict(featureValues, patientId, 50)
      setInferenceData(data)
      setRightTab('INFERENCE')
    } catch (e) {
      setError(e?.response?.data?.detail ?? e.message ?? 'Inference failed')
    } finally {
      setInferenceLoading(false)
    }
  }

  const handleSimulate = async () => {
    if (!inferenceData) { setError('Run analysis first'); return }
    setError(null)
    setSimulationLoading(true)
    try {
      const data = await api.simulate(patientId, 12, 0, 0.8)
      setSimulationData(data)
    } catch (e) {
      setError(e?.response?.data?.detail ?? e.message ?? 'Simulation failed')
    } finally {
      setSimulationLoading(false)
    }
  }

  const handleRecommend = async () => {
    if (!inferenceData) { setError('Run analysis first'); return }
    setError(null)
    setTreatmentLoading(true)
    setRightTab('AGENTS')
    try {
      const data = await api.recommendTreatment(patientId, 50)
      setTreatmentData(data)
    } catch (e) {
      setError(e?.response?.data?.detail ?? e.message ?? 'Recommendation failed')
    } finally {
      setTreatmentLoading(false)
    }
  }

  const panelVariants = {
    hidden: (dir) => ({ opacity: 0, x: dir === 'left' ? -40 : dir === 'right' ? 40 : 0, filter: 'blur(8px)' }),
    visible: { opacity: 1, x: 0, filter: 'blur(0px)', transition: { duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] } },
  }

  return (
    <div style={styles.root}>
      {/* Particle background */}
      <ParticleBackground />

      {/* Grid overlay */}
      <div style={styles.gridOverlay} />

      {/* Navbar */}
      <motion.div
        style={styles.navbar}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: mounted ? 1 : 0, y: mounted ? 0 : -20 }}
        transition={{ duration: 0.5 }}
      >
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <LiveDot color={apiStatus === 'online' ? '#00ff88' : apiStatus === 'offline' ? '#ff3300' : '#ffcc00'} />
          <span style={styles.navTitle}>OPHTHALMIC DIGITAL TWIN</span>
        </div>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <span style={{ fontSize: '10px', fontFamily: 'var(--font-data)', color: 'var(--text-muted)' }}>
            API: <span style={{ color: apiStatus === 'online' ? '#00ff88' : '#ff6b35' }}>{apiStatus.toUpperCase()}</span>
          </span>
          <span style={{ fontSize: '10px', fontFamily: 'var(--font-data)', color: 'var(--text-muted)' }}>v1.0.0</span>
        </div>
      </motion.div>

      {/* Main body */}
      <div style={styles.body}>
        {/* Left panel */}
        <motion.div
          style={styles.leftPanel}
          custom="left"
          variants={panelVariants}
          initial="hidden"
          animate={mounted ? 'visible' : 'hidden'}
          transition={{ delay: 0.15 }}
        >
          {/* Patient ID */}
          <div style={{ ...styles.glassCard, padding: '12px' }}>
            <div style={styles.label}>PATIENT ID</div>
            <input
              style={styles.input}
              type="text"
              value={patientId}
              onChange={e => setPatientId(e.target.value)}
              placeholder="e.g. P001"
              onFocus={e => (e.target.style.borderColor = 'rgba(0,200,255,0.4)')}
              onBlur={e => (e.target.style.borderColor = 'rgba(255,255,255,0.1)')}
            />
          </div>

          {/* Input mode tabs */}
          <div style={{ ...styles.glassCard, padding: '10px 12px' }}>
            <TabBar
              tabs={['PARAMETERS', 'FUNDUS IMAGE']}
              active={leftTab}
              onChange={setLeftTab}
            />

            <AnimatePresence mode="wait">
              {leftTab === 'PARAMETERS' ? (
                <motion.div
                  key="params"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <div style={{ ...styles.label, marginBottom: '6px' }}>PATIENT FEATURES</div>
                  <FeatureForm
                    featureNames={featureMeta?.feature_names}
                    values={featureValues}
                    onChange={handleFeatureChange}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="fundus"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <FundusUpload
                    patientId={patientId}
                    onResult={handleFundusResult}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Action buttons — only shown in PARAMETERS mode */}
          {leftTab === 'PARAMETERS' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <motion.button
                style={styles.button('primary')}
                onClick={handleAnalyze}
                disabled={inferenceLoading}
                whileHover={{ scale: 1.02, boxShadow: '0 0 20px rgba(0,200,255,0.2)' }}
                whileTap={{ scale: 0.98 }}
              >
                {inferenceLoading ? '⟳  ANALYZING...' : '◈  ANALYZE PATIENT'}
              </motion.button>

              <motion.button
                style={styles.button('primary')}
                onClick={handleSimulate}
                disabled={simulationLoading || !inferenceData}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {simulationLoading ? '⟳  SIMULATING...' : '◉  RUN SIMULATION'}
              </motion.button>

              <motion.button
                style={styles.button('success')}
                onClick={handleRecommend}
                disabled={treatmentLoading || !inferenceData}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {treatmentLoading ? '⟳  SOLVING NASH...' : '✦  RECOMMEND TREATMENT'}
              </motion.button>
            </div>
          )}

          {/* Error display */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                style={{
                  background: 'rgba(255,60,0,0.08)',
                  border: '1px solid rgba(255,107,53,0.3)',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  fontSize: '11px',
                  fontFamily: 'var(--font-data)',
                  color: '#ff6b35',
                }}
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Twin state summary */}
          {inferenceData && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ ...styles.glassCard, padding: '10px 12px' }}
            >
              <div style={styles.label}>TWIN STATE</div>
              <div style={{ fontSize: '10px', fontFamily: 'var(--font-data)', color: 'var(--text-muted)', lineHeight: 1.8 }}>
                <div>Pred: <span style={{ color: '#00c8ff' }}>{inferenceData.prediction_label}</span></div>
                <div>Conf: <span style={{ color: '#00c8ff' }}>{(inferenceData.confidence * 100).toFixed(1)}%</span></div>
                <div>3D: <span style={{ color: '#7c3aed' }}>
                  [{inferenceData.latent_state_3d?.map(v => v.toFixed(2)).join(', ')}]
                </span></div>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Center — 3D Canvas */}
        <motion.div
          style={styles.centerCanvas}
          initial={{ opacity: 0, scale: 0.96, filter: 'blur(8px)' }}
          animate={mounted ? { opacity: 1, scale: 1, filter: 'blur(0px)' } : {}}
          transition={{ delay: 0.25, duration: 0.7 }}
        >
          <TwinCanvas3D
            inferenceData={inferenceData}
            simulationData={simulationData}
            patientId={patientId}
          />
        </motion.div>

        {/* Right panel */}
        <motion.div
          style={styles.rightPanel}
          custom="right"
          variants={panelVariants}
          initial="hidden"
          animate={mounted ? 'visible' : 'hidden'}
          transition={{ delay: 0.2 }}
        >
          <TabBar
            tabs={['INFERENCE', 'ATTENTION', 'AGENTS']}
            active={rightTab}
            onChange={setRightTab}
          />

          <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
            <AnimatePresence mode="wait">
              {rightTab === 'INFERENCE' && (
                <motion.div
                  key="inference"
                  style={{ position: 'absolute', inset: 0, overflowY: 'auto' }}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <InferencePanel
                    data={inferenceData}
                    loading={inferenceLoading}
                    mcSampleProgress={mcSampleProgress}
                  />
                </motion.div>
              )}
              {rightTab === 'ATTENTION' && (
                <motion.div
                  key="attention"
                  style={{ position: 'absolute', inset: 0, overflowY: 'auto' }}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <AttentionHeatmap
                    data={inferenceData}
                    featureNames={featureMeta?.feature_names}
                  />
                </motion.div>
              )}
              {rightTab === 'AGENTS' && (
                <motion.div
                  key="agents"
                  style={{ position: 'absolute', inset: 0, overflowY: 'auto' }}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <AgentConsole
                    treatmentData={treatmentData}
                    loading={treatmentLoading}
                    inferenceData={inferenceData}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>

      {/* Bottom progression graph */}
      <motion.div
        style={styles.bottomBar}
        initial={{ opacity: 0, y: 30, filter: 'blur(8px)' }}
        animate={mounted ? { opacity: 1, y: 0, filter: 'blur(0px)' } : {}}
        transition={{ delay: 0.35, duration: 0.5 }}
      >
        <div style={{ ...styles.label, marginBottom: '4px' }}>DISEASE PROGRESSION TRAJECTORY</div>
        <div style={{ height: 'calc(100% - 24px)' }}>
          <ProgressionGraph
            simulationData={simulationData}
            inferenceData={inferenceData}
            currentTimestep={inferenceData?.timestep ?? 0}
          />
        </div>
      </motion.div>
    </div>
  )
}
