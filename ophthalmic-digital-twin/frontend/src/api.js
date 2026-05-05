/**
 * Axios API client for the Ophthalmic Digital Twin FastAPI backend.
 */

import axios from 'axios'

const BASE_URL = '/api/v1'

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
})

/**
 * Run MC Dropout inference on a patient's feature set.
 * @param {Object} patientFeatures - key-value feature dict
 * @param {string} patientId
 * @param {number} mcSamples
 */
export async function predict(patientFeatures, patientId, mcSamples = 50) {
  const res = await client.post('/predict', {
    patient_features: patientFeatures,
    patient_id: patientId,
    mc_samples: mcSamples,
  })
  return res.data
}

/**
 * Get the current digital twin state for a patient.
 * @param {string} patientId
 */
export async function getTwinState(patientId) {
  const res = await client.get(`/twin-state/${patientId}`)
  return res.data
}

/**
 * Simulate future disease trajectory.
 * @param {string} patientId
 * @param {number} horizon
 * @param {number} treatmentAction
 * @param {number} complianceLevel
 */
export async function simulate(patientId, horizon = 12, treatmentAction = 0, complianceLevel = 0.8) {
  const res = await client.post('/simulate', {
    patient_id: patientId,
    horizon,
    treatment_action: treatmentAction,
    compliance_level: complianceLevel,
  })
  return res.data
}

/**
 * Get Nash equilibrium treatment recommendation.
 * @param {string} patientId
 * @param {number} mcSamples
 */
export async function recommendTreatment(patientId, mcSamples = 50) {
  const res = await client.post('/recommend-treatment', {
    patient_id: patientId,
    mc_samples: mcSamples,
  })
  return res.data
}

/**
 * Check API health and model status.
 */
export async function healthCheck() {
  const res = await client.get('/health')
  return res.data
}

/**
 * Get list of feature names for form generation.
 */
export async function getFeatureNames() {
  const res = await client.get('/feature-names')
  return res.data
}
