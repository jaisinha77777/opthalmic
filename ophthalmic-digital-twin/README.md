# Ophthalmic Digital Twin

AI-powered digital twin system for ophthalmic disease modeling using pure Transformer architecture, Monte Carlo Dropout uncertainty, Multi-Agent Reinforcement Learning, and Nash equilibrium treatment optimization.

## Architecture Overview

```
Dataset → FeatureTokenizer → OphthalmicTransformer (6L, 8H, d=256)
       → MC Dropout (50 samples) → Uncertainty + Predictions
       → DigitalTwinEngine (latent state S_t ∈ R^256)
       → MARL (Doctor + Disease + Patient PPO agents)
       → Nash Equilibrium Solver → Optimal Treatment
```

## Quick Start

### 1. Place your dataset
```bash
cp your_dataset.csv data/full_df.csv
```

### 2. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Train the model
```bash
python run.py --mode train
# Uses GPU automatically if available
# Saves: models/best_model.pt, models/final_model.pt, models/training_history.json
# Generates: DATA_REPORT.md with EDA findings
```

### 4. Start the API server
```bash
python run.py --mode serve
# FastAPI on http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### 5. Run the frontend
```bash
cd frontend
npm install
npm run dev
# Vite dev server on http://localhost:5173
```

### 6. Evaluate a trained model
```bash
cd backend
python run.py --mode evaluate
```

## CLI Options

```
python run.py --mode train|serve|evaluate
             [--device cuda|cpu]   # auto-detected if omitted
             [--csv /path/to.csv]  # default: data/full_df.csv
             [--port 8000]         # only for serve mode
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | MC Dropout inference + SHAP + attention |
| GET | `/api/v1/twin-state/{patient_id}` | Digital twin latent state |
| POST | `/api/v1/simulate` | Simulate future disease trajectory |
| POST | `/api/v1/recommend-treatment` | Nash equilibrium treatment recommendation |
| GET | `/api/v1/health` | API health + model status |
| GET | `/api/v1/feature-names` | Feature list for frontend form generation |

### Example: Predict
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "patient_features": {"age": 65, "iop": 22.4},
    "mc_samples": 50
  }'
```

## Model Architecture

### OphthalmicTransformer
- **FeatureTokenizer**: per-feature Linear/Embedding projection → d=256 tokens
- **CLS token** (learned) prepended to feature sequence
- **6× Pre-LN TransformerEncoderLayer** (nhead=8, d_ff=512, dropout=0.15, GELU)
- **Prediction head**: CLS → LayerNorm → Linear(256→128) → GELU → Linear(128→n_classes)
- **Uncertainty head**: CLS → Linear(256→128) → GELU → Softplus → σ² (aleatoric)
- **Attention capture hooks** on every layer for explainability

### TemporalTransformer (if timestamp data detected)
- FeatureTokenizer per timestep
- 2D positional encoding: time_pos + feat_pos (learned)
- 4× cross-feature encoder + 2× causal temporal encoder
- Mean pool over features → causal autoregressive output

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| Loss (classification) | LabelSmoothingCrossEntropy (ε=0.1) + 0.1×HeteroscedasticNLL |
| Loss (regression) | Huber + 0.1×HeteroscedasticNLL |
| Gradient clipping | max_norm=1.0 |
| Mixed precision | AMP (CUDA only) |
| Early stopping | patience=15 on val_loss |
| Max epochs | 100 |

## Uncertainty Quantification

- **MC Dropout**: 50 stochastic forward passes at inference
- **Epistemic variance**: variance across MC samples
- **Aleatoric variance**: mean of model's predicted σ² head
- **Total uncertainty**: epistemic + aleatoric
- **Rejection threshold**: predictions with uncertainty > 0.15 flagged unreliable
- **ECE calibration**: 15-bin Expected Calibration Error

## MARL Agents

| Agent | Role | Action Space |
|-------|------|-------------|
| DoctorAgent | Minimize disease progression | N_treatments (from dataset) |
| DiseaseAgent | Adversarial: maximize progression | 16 perturbation directions |
| PatientAgent | Model compliance variability | 3 levels: {0.3, 0.7, 1.0} |

All agents use **PPO** (ε=0.2, entropy_coeff=0.01, GAE-λ=0.95).

**Nash Equilibrium**: iterated best-response over 20 iterations, convergence criterion KL < 1e-3.

## Digital Twin

Each patient has a stateful `DigitalTwinEngine` with:
- Latent state **S_t ∈ R^256** evolving via `StateTransitionMLP`
- Treatment action embedding scaled by compliance level
- Residual skip connection: S_{t+1} = f(S_t, a) + S_t
- Full state trajectory for 3D visualization
- **simulate_horizon(H)**: non-destructive future projection with confidence bands

## Frontend Components

| Component | Description |
|-----------|-------------|
| `ParticleBackground` | 2000-particle Three.js sphere, mouse parallax |
| `TwinCanvas3D` | Glowing icosahedron + trajectory tubes + bloom |
| `InferencePanel` | Arc confidence gauge, variance bars, reliability badge |
| `AttentionHeatmap` | SVG heatmap + SHAP waterfall toggle |
| `ProgressionGraph` | Recharts trajectory with confidence bands |
| `AgentConsole` | Terminal MARL display with Nash convergence |

## Data Pipeline

1. **EDA**: shape, nulls, column classification → `DATA_REPORT.md`
2. **Missing values**: learned mask tokens (no mean imputation)
3. **Preprocessing**: StandardScaler (3σ clip) for numerical; LabelEncoder for categorical
4. **Sequences**: KMeans(k=6) pseudo-time if no timestamp; sliding windows if timestamps exist
5. **Split**: 70/15/15 patient-level stratified (no leakage)

## Constraints

- No LSTM/GRU/RNN
- No ensembles (XGBoost, LightGBM, RandomForest)
- No mean imputation
- No sinusoidal positional encoding (all learned)
- No pre-trained models
- No CSS framework on frontend (pure CSS custom properties)
