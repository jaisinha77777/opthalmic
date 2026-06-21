# Glaucoma Clinical Support

A glaucoma decision-support demo built around **clinically grounded** methods:
measurement-driven severity staging, calibrated uncertainty, guideline-based target-IOP
recommendations, a transparent visual-field progression projection, and a fundus CNN
that does only what a photograph can honestly support.

> ⚠️ **Research / education only.** The tabular model is trained on *synthetic* data and
> the fundus model on public research datasets. This is **not a medical device** and must
> **not** be used for real patient care.

## What this is (and what it deliberately is not)

This project was rebuilt to remove components that looked sophisticated but had no clinical
validity. Removed: KMeans "pseudo-time" sequences (glaucoma progression is longitudinal and
cannot be recovered from one cross-sectional row), an adversarial "disease agent" + Nash-
equilibrium "treatment" engine, and an opaque 256-dim "digital twin." In their place:

| Concern | Clinically grounded approach |
|---|---|
| Severity staging | **Hodapp-Parrish-Anderson** on visual-field Mean Deviation (mild > −6 dB, moderate −6…−12 dB, severe < −12 dB) plus structural gates (vertical CDR, RNFL). |
| Model | Per-row tabular **Transformer** classifier (`OphthalmicTransformer`), no fabricated time axis. |
| Uncertainty | **MC-Dropout** (epistemic + aleatoric) with a reliability flag for low-confidence cases. |
| Progression | **Transparent closed-form MD projection** (EMGT/CIGTS-consistent slopes), treated vs untreated, with a confidence band. |
| Treatment | **Guideline target-IOP + treatment ladder** (EMGT/CIGTS/AGIS, AAO POAG PPP). Decision support, *not* a prescription. |
| Fundus image | **Referable glaucoma (binary) + vertical CDR** — what a photo can support — with real GradCAM. Not the VF-based stage. |

## Quick start

### 1. Generate synthetic training data (labels derived from clinical rules)
```bash
python scripts/generate_synthetic_data.py --rows 4000 --seed 42
# -> data/synthetic_train.csv
```

### 2. Train the tabular model
```bash
cd backend
pip install -r requirements.txt
python run.py --mode train --csv ../data/synthetic_train.csv
# -> models/best_model.pt, data/feature_metadata.json, data/preprocessor.pkl
```

### 3. (Optional) Train the fundus model on a real dataset
```bash
# Option A: Kaggle (needs a free account + ~/.kaggle/kaggle.json)
python scripts/fetch_fundus_dataset.py --source kaggle --dataset arnavjain1/glaucoma-datasets
# Option B: a manually downloaded archive (REFUGE / G1020 / RIM-ONE DL / ACRIMA)
python scripts/fetch_fundus_dataset.py --zip /path/to/dataset.zip

python scripts/train_fundus.py --epochs 15
# -> backend/models/fundus_model.pt  (reports referable-glaucoma AUC + CDR MAE)
```
Without trained fundus weights the API still runs but clearly reports the fundus model as
**uncalibrated** rather than emitting confident noise.

### 4. Serve the API
```bash
cd backend
python run.py --mode serve          # http://localhost:8000  (docs at /docs)
```

### 5. Run the dashboard
```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173
```

## Deployment

The app ships as a **single container**: the frontend is built and served by the
FastAPI process, with the API mounted under `/api/v1` (no separate web server or
reverse proxy required).

```bash
# From the project root (ophthalmic-digital-twin/)
docker compose up --build        # -> http://localhost:8000  (UI + API + /docs)
```

Or with plain Docker:
```bash
docker build -t glaucoma-clinical-support .
docker run -p 8000:8000 -e ALLOWED_ORIGINS="https://your-domain" glaucoma-clinical-support
```

Without Docker (single host):
```bash
cd frontend && npm ci && npm run build      # produces frontend/dist
cd ../backend && pip install -r requirements.txt
python run.py --mode serve                  # serves UI + API on :8000
```

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Port the API/UI listens on (PaaS platforms inject this). |
| `HOST` | `0.0.0.0` | Bind address. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. **Set this to your real frontend origin in production** — with `*`, credentials are automatically disabled. |
| `FRONTEND_DIST` | `<repo>/frontend/dist` | Location of the built frontend to serve. |

The trained tabular model (`backend/models/best_model.pt`), preprocessor
(`data/preprocessor.pkl`), and fundus weights (`backend/models/fundus_model.pt`)
are tracked in the repo, so a fresh clone runs without retraining. If they are
missing the API still starts: the preprocessor is refit from `data/full_df.csv`
and the fundus model reports as **uncalibrated**.

> In dev, `npm run dev` (port 5173) proxies `/api` to the backend on `:8000`
> (see `frontend/vite.config.js`); in production the same `/api/v1` calls are
> served same-origin by FastAPI, so no frontend rebuild is needed per environment.

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | MC-Dropout staging + attention/SHAP attribution |
| POST | `/api/v1/simulate` | Transparent visual-field (MD) progression projection |
| POST | `/api/v1/recommend-treatment` | Guideline target-IOP + treatment-ladder decision support |
| POST | `/api/v1/analyze-fundus` | Referable glaucoma + vertical CDR + GradCAM |
| GET  | `/api/v1/feature-names` | Feature list for the form |
| GET  | `/api/v1/health` | Model + fundus-calibration status |

## Data model

25 columns: `patient_id`, demographics (age, sex, bmi, ethnicity, eye_color), IOP (od/os),
structural (cup_disc_ratio, RNFL superior/inferior/average), functional (mean_deviation od/os,
pattern_sd, va od/os), systemic (hba1c, BP, diabetes, hypertension, family_history), `treatment`,
and the target `disease_severity` ∈ {Normal, Suspect, Mild/Moderate/Severe Glaucoma}.

The generator samples a plausible clinic population (a latent optic-nerve-damage factor drives
correlated structure and function) and then **derives the label from the measurements** via the
staging rules above — the direction real diagnosis works. Realistic missingness is injected on the
same columns as real clinic data.

## Frontend

Plain, legible clinical dashboard (no 3D, particles, or animation theatrics): a measurement form +
fundus upload on the left; staged result with reliability flag, guideline decision support, MD
progression chart, and a feature-attention map on the right. Dependencies: React, Recharts, Axios.

## Clinical references
- Hodapp E, Parrish RK, Anderson DR. *Clinical Decisions in Glaucoma.* (MD-based staging)
- Early Manifest Glaucoma Trial (EMGT); CIGTS; AGIS — IOP lowering and progression.
- AAO Primary Open-Angle Glaucoma Preferred Practice Pattern — target IOP & treatment ladder.
- Public fundus datasets: REFUGE, ORIGA, G1020, RIM-ONE DL, ACRIMA.
