/**
 * AgentConsole — terminal-style MARL agent status with typewriter logs,
 * animated progress bars, Nash convergence indicator, and scrolling log window.
 */

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// ── Typewriter hook ───────────────────────────────────────

function useTypewriter(text, speed = 30) {
  const [displayed, setDisplayed] = useState('')
  const timeoutRef = useRef(null)

  useEffect(() => {
    setDisplayed('')
    let i = 0
    const tick = () => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1))
        i++
        timeoutRef.current = setTimeout(tick, speed)
      }
    }
    timeoutRef.current = setTimeout(tick, speed)
    return () => clearTimeout(timeoutRef.current)
  }, [text, speed])

  return displayed
}

// ── Agent row ─────────────────────────────────────────────

function AgentRow({ label, color, actionText, reward, progress }) {
  const displayedText = useTypewriter(actionText, 28)
  const animPct = Math.min(Math.max(progress ?? 0, 0), 1)

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      marginBottom: '10px',
      fontFamily: 'var(--font-data)',
      fontSize: '11px',
    }}>
      {/* Label */}
      <span style={{
        width: '60px',
        color,
        textShadow: `0 0 8px ${color}`,
        letterSpacing: '0.08em',
        flexShrink: 0,
      }}>
        {label}
      </span>

      {/* Progress bar */}
      <div style={{
        width: '80px',
        height: '6px',
        background: 'rgba(255,255,255,0.06)',
        borderRadius: '3px',
        flexShrink: 0,
        overflow: 'hidden',
      }}>
        <motion.div
          style={{ height: '100%', background: color, borderRadius: '3px', boxShadow: `0 0 6px ${color}` }}
          animate={{ width: `${animPct * 100}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        />
      </div>

      {/* Action text */}
      <span style={{ color: 'var(--text-muted)', flex: 1, overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
        {displayedText}
      </span>

      {/* Reward */}
      <span style={{
        color: reward >= 0 ? '#00ff88' : '#ff6b35',
        minWidth: '52px',
        textAlign: 'right',
      }}>
        {reward >= 0 ? '+' : ''}{reward?.toFixed(3) ?? '0.000'}
      </span>
    </div>
  )
}

// ── Convergence indicator ─────────────────────────────────

function ConvergenceIndicator({ step, maxSteps, converged }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '6px 0',
      borderTop: '1px solid rgba(255,255,255,0.05)',
      borderBottom: '1px solid rgba(255,255,255,0.05)',
      marginBottom: '10px',
      fontFamily: 'var(--font-data)',
      fontSize: '10px',
    }}>
      <AnimatePresence>
        {converged ? (
          <motion.span
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            style={{ color: '#00ff88', fontSize: '14px' }}
          >
            ✓
          </motion.span>
        ) : (
          <motion.div
            style={{
              width: '10px', height: '10px',
              border: '2px solid #00c8ff',
              borderTopColor: 'transparent',
              borderRadius: '50%',
            }}
            animate={{ rotate: 360 }}
            transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
          />
        )}
      </AnimatePresence>
      <span style={{ color: converged ? '#00ff88' : '#00c8ff' }}>
        {converged
          ? `Equilibrium reached at step ${step}/${maxSteps}`
          : `Solving Nash... step ${step}/${maxSteps}`}
      </span>
    </div>
  )
}

// ── Scrolling log ─────────────────────────────────────────

function ScrollLog({ lines }) {
  const endRef = useRef()

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines])

  return (
    <div style={{
      background: 'rgba(0,0,0,0.5)',
      border: '1px solid rgba(255,255,255,0.05)',
      borderRadius: '4px',
      padding: '8px',
      height: '90px',
      overflowY: 'auto',
      fontFamily: 'var(--font-data)',
      fontSize: '9px',
      lineHeight: 1.6,
    }}>
      {lines.map((line, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.15 }}
          style={{ color: line.color ?? 'var(--text-muted)' }}
        >
          <span style={{ color: 'rgba(0,200,255,0.4)', marginRight: '6px' }}>{'>'}</span>
          {line.text}
        </motion.div>
      ))}
      <div ref={endRef} />
    </div>
  )
}

// ── Log line generator ────────────────────────────────────

function useNashLogs(treatmentData, isRunning) {
  const [lines, setLines] = useState([])
  const [logStep, setLogStep] = useState(0)

  useEffect(() => {
    if (!isRunning && !treatmentData) {
      setLines([{ text: 'Awaiting Nash solver...', color: 'rgba(226,232,240,0.2)' }])
      setLogStep(0)
      return
    }

    if (isRunning) {
      setLines([{ text: 'Initializing MARL agents...', color: '#00c8ff' }])
      setLogStep(0)
      const msgs = [
        { text: 'Doctor agent: computing action probs...', color: '#00c8ff' },
        { text: 'Disease agent: simulating adversarial state...', color: '#ff6b35' },
        { text: 'Patient agent: sampling compliance level...', color: '#00ff88' },
        { text: 'Running PPO update (Doctor)...', color: '#00c8ff' },
        { text: 'Running PPO update (Disease)...', color: '#ff6b35' },
        { text: 'Running PPO update (Patient)...', color: '#00ff88' },
        { text: 'Checking KL divergence convergence...', color: 'var(--text-muted)' },
        { text: 'Iterating best response...', color: 'var(--text-muted)' },
      ]
      let i = 0
      const interval = setInterval(() => {
        if (i < msgs.length) {
          setLines(prev => [...prev, msgs[i]])
          i++
        } else {
          clearInterval(interval)
        }
      }, 400)
      return () => clearInterval(interval)
    }

    if (treatmentData) {
      const step = treatmentData.nash_convergence_step ?? 20
      setLines([
        { text: `Nash solver converged at step ${step}/20`, color: '#00ff88' },
        { text: `Recommended: ${treatmentData.treatment_name}`, color: '#00c8ff' },
        { text: `Doctor policy max: ${Math.max(...(treatmentData.doctor_policy ?? [0])).toFixed(3)}`, color: '#00c8ff' },
        { text: `Patient compliance: ${(treatmentData.compliance_level * 100).toFixed(0)}%`, color: '#00ff88' },
        { text: `Expected reward: ${treatmentData.expected_outcome?.toFixed(3)}`, color: 'var(--text-muted)' },
      ])
    }
  }, [treatmentData, isRunning])

  return lines
}

/**
 * @component Terminal-style MARL agent console with Nash convergence display.
 * @param {Object|null} treatmentData - TreatmentResponse from /recommend-treatment
 * @param {boolean} loading - true while Nash solver is running
 * @param {Object|null} inferenceData - used to derive agent progress levels
 */
export default function AgentConsole({ treatmentData, loading, inferenceData }) {
  const logs = useNashLogs(treatmentData, loading)

  const doctorProgress = treatmentData
    ? Math.max(...(treatmentData.doctor_policy ?? [0]))
    : loading ? 0.6 : 0.2

  const diseaseProgress = treatmentData
    ? Math.max(...(treatmentData.disease_policy ?? [0]))
    : loading ? 0.4 : 0.15

  const patientProgress = treatmentData
    ? Math.max(...(treatmentData.patient_compliance ?? [0]))
    : loading ? 0.7 : 0.3

  const nashStep = treatmentData?.nash_convergence_step ?? 0
  const converged = !!treatmentData && !loading

  const doctorText = treatmentData
    ? `selecting tx ${treatmentData.recommended_treatment} → "${treatmentData.treatment_name}"`
    : loading ? 'optimizing treatment policy...' : 'idle'

  const diseaseText = loading
    ? 'perturbing latent state...'
    : treatmentData
    ? `adversarial: policy entropy ${(Math.log(treatmentData.disease_policy?.length ?? 1)).toFixed(2)}`
    : 'idle'

  const patientText = treatmentData
    ? `compliance: ${(['LOW', 'MEDIUM', 'HIGH'])[(treatmentData.patient_compliance ?? [0,0,1]).indexOf(Math.max(...(treatmentData.patient_compliance ?? [0])))]}`
    : loading ? 'modeling compliance...' : 'idle'

  const doctorReward = inferenceData ? -(inferenceData.total_uncertainty ?? 0.2) : 0
  const diseaseReward = inferenceData ? (inferenceData.total_uncertainty ?? 0.1) : 0
  const patientReward = treatmentData ? (treatmentData.expected_outcome ?? 0.3) : 0

  return (
    <div style={{
      width: '100%', height: '100%',
      display: 'flex', flexDirection: 'column',
      gap: '8px', overflowY: 'auto',
    }}>
      <div style={{
        background: 'rgba(0,0,0,0.6)',
        border: '1px solid rgba(0,200,255,0.12)',
        borderRadius: '8px',
        padding: '12px',
        flex: '0 0 auto',
      }}>
        <div style={{
          fontSize: '9px',
          color: 'var(--text-muted)',
          letterSpacing: '0.12em',
          marginBottom: '12px',
          fontFamily: 'var(--font-data)',
        }}>
          MARL AGENT STATUS
        </div>

        <AgentRow
          label="DOCTOR"
          color="#00c8ff"
          actionText={doctorText}
          reward={doctorReward}
          progress={doctorProgress}
        />
        <AgentRow
          label="DISEASE"
          color="#ff3300"
          actionText={diseaseText}
          reward={diseaseReward}
          progress={diseaseProgress}
        />
        <AgentRow
          label="PATIENT"
          color="#00ff88"
          actionText={patientText}
          reward={patientReward}
          progress={patientProgress}
        />

        <ConvergenceIndicator
          step={nashStep}
          maxSteps={20}
          converged={converged}
        />

        <ScrollLog lines={logs} />
      </div>
    </div>
  )
}
