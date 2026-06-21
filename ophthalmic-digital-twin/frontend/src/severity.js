/** Canonical clinical severity ordering + colors, shared across the UI. */

export const SEVERITY_ORDER = [
  'Normal', 'Suspect', 'Mild Glaucoma', 'Moderate Glaucoma', 'Severe Glaucoma',
]

export const SEVERITY_COLOR = {
  'Normal': 'var(--normal)',
  'Suspect': 'var(--suspect)',
  'Mild Glaucoma': 'var(--mild)',
  'Moderate Glaucoma': 'var(--moderate)',
  'Severe Glaucoma': 'var(--severe)',
}

export function colorForLabel(label) {
  return SEVERITY_COLOR[label] || 'var(--accent)'
}

export function colorForIndex(idx) {
  return colorForLabel(SEVERITY_ORDER[idx] ?? 'Normal')
}
