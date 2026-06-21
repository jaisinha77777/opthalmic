/**
 * Axios API client for the glaucoma clinical-support FastAPI backend.
 */

import axios from 'axios'

const client = axios.create({
  baseURL: '/api/v1',
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
})

/** MC-Dropout staging inference on a patient's measurements. */
export async function predict(patientFeatures, patientId, mcSamples = 50) {
  const res = await client.post('/predict', {
    patient_features: patientFeatures,
    patient_id: patientId,
    mc_samples: mcSamples,
  })
  return res.data
}

/** Transparent visual-field (MD) progression projection. */
export async function simulate(patientFeatures, patientId, horizonMonths = 60, iopReduction = 0.3) {
  const res = await client.post('/simulate', {
    patient_features: patientFeatures,
    patient_id: patientId,
    horizon_months: horizonMonths,
    iop_reduction: iopReduction,
  })
  return res.data
}

/** Guideline-based decision support (target IOP + treatment ladder). */
export async function recommendTreatment(patientFeatures, patientId) {
  const res = await client.post('/recommend-treatment', {
    patient_features: patientFeatures,
    patient_id: patientId,
  })
  return res.data
}

export async function healthCheck() {
  const res = await client.get('/health')
  return res.data
}

export async function getFeatureNames() {
  const res = await client.get('/feature-names')
  return res.data
}
