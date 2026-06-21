# Glaucoma Clinical Support

A full-stack, **clinically grounded** glaucoma decision-support demo: measurement-driven
severity staging, calibrated uncertainty, guideline-based target-IOP recommendations, a
transparent visual-field progression projection, and a fundus CNN that does only what a
photograph can honestly support.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/api-FastAPI-009688">
  <img alt="PyTorch" src="https://img.shields.io/badge/ml-PyTorch-ee4c2c">
  <img alt="React" src="https://img.shields.io/badge/ui-React%2018%20%2B%20Vite-61dafb">
  <img alt="Docker" src="https://img.shields.io/badge/deploy-Docker-2496ed">
</p>

> ⚠️ **Research / education only.** The tabular model is trained on *synthetic* data and the
> fundus model on public research datasets. This is **not a medical device** and must **not**
> be used for real patient care.

---

## Why this exists

The project was deliberately rebuilt to strip out components that *looked* sophisticated but
had no clinical validity (KMeans "pseudo-time" progression, an adversarial Nash-equilibrium
"treatment" engine, an opaque 256-dim "digital twin"). What replaced them maps to how
glaucoma is actually staged and managed:

| Concern | Clinically grounded approach |
|---|---|
| **Severity staging** | Hodapp-Parrish-Anderson on visual-field Mean Deviation (mild > −6 dB, moderate −6…−12 dB, severe < −12 dB) plus structural gates (vertical CDR, RNFL). |
| **Model** | Per-row tabular **Transformer** (`OphthalmicTransformer`) — no fabricated time axis. |
| **Uncertainty** | **MC-Dropout** (epistemic + aleatoric) with a reliability flag for low-confidence cases. |
| **Progression** | Transparent closed-form **MD projection** (EMGT/CIGTS-consistent slopes), treated vs untreated, with a confidence band. |
| **Treatment** | Guideline **target-IOP + treatment ladder** (EMGT/CIGTS/AGIS, AAO POAG PPP). Decision support, *not* a prescription. |
| **Fundus image** | **Referable glaucoma (binary) + vertical CDR** with real GradCAM — what a photo can support, not the VF-based stage. |

---

## Architecture

```
┌─────────────────────────┐        ┌──────────────────────────────────────┐
│  React + Vite frontend  │  /api  │  FastAPI backend (PyTorch)            │
│  measurement form,      │ ─────► │  • OphthalmicTransformer + MC-Dropout │
│  fundus upload, charts  │        │  • SHAP / attention explainability    │
└─────────────────────────┘        │  • progression + decision support     │
                                    │  • ResNet-18 fundus encoder + GradCAM │
   In production the built          └──────────────────────────────────────┘
   frontend is served by FastAPI → the whole app ships as ONE container.
```

The codebase lives under [`ophthalmic-digital-twin/`](ophthalmic-digital-twin/):

```
ophthalmic-digital-twin/
├── backend/
│   ├── api/            FastAPI app, routes, schemas (serves API + built UI)
│   ├── core/           model, training, uncertainty, explainability,
│   │                   progression, decision support, fundus encoder
│   ├── models/         trained weights (best_model.pt, fundus_model.pt)
│   └── run.py          entry point: --mode train | serve | evaluate
├── frontend/           React + Vite dashboard (Recharts, Axios)
├── data/               feature metadata, preprocessor, synthetic CSV
├── scripts/            synthetic-data + fundus-dataset + training scripts
├── Dockerfile          multi-stage build (node → python, single artifact)
└── docker-compose.yml
```

---

## Quick start (local dev)

**Backend** — the repo ships with trained weights, so it runs without retraining:

```bash
cd ophthalmic-digital-twin/backend
pip install -r requirements.txt
python run.py --mode serve            # http://localhost:8000  (docs at /docs)
```

**Frontend** (separate terminal):

```bash
cd ophthalmic-digital-twin/frontend
npm install
npm run dev                           # http://localhost:5173 (proxies /api → :8000)
```

---

## Deployment (single container)

The frontend is built and served by FastAPI under the same origin, so there's no separate
web server or reverse proxy to run.

```bash
cd ophthalmic-digital-twin
docker compose up --build             # http://localhost:8000  (UI + API + /docs)
```

Or with plain Docker:

```bash
cd ophthalmic-digital-twin
docker build -t glaucoma-clinical-support .
docker run -p 8000:8000 -e ALLOWED_ORIGINS="https://your-domain" glaucoma-clinical-support
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Port the API/UI listens on (PaaS platforms inject this). |
| `HOST` | `0.0.0.0` | Bind address. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. **Set to your real frontend origin in production** (with `*`, credentials are auto-disabled). |
| `FRONTEND_DIST` | `<repo>/frontend/dist` | Location of the built frontend to serve. |

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | MC-Dropout staging + attention/SHAP attribution |
| `POST` | `/api/v1/simulate` | Transparent visual-field (MD) progression projection |
| `POST` | `/api/v1/recommend-treatment` | Guideline target-IOP + treatment-ladder decision support |
| `POST` | `/api/v1/analyze-fundus` | Referable glaucoma + vertical CDR + GradCAM |
| `GET`  | `/api/v1/feature-names` | Feature list for the form |
| `GET`  | `/api/v1/health` | Model + fundus-calibration status |

Interactive docs at `/docs` when the server is running.

---

## Retraining (optional)

```bash
# 1. Generate synthetic training data (labels derived from clinical rules)
python ophthalmic-digital-twin/scripts/generate_synthetic_data.py --rows 4000 --seed 42

# 2. Train the tabular model
cd ophthalmic-digital-twin/backend
python run.py --mode train --csv ../data/synthetic_train.csv

# 3. (Optional) Train the fundus model on a real dataset (REFUGE / G1020 / RIM-ONE / ACRIMA)
python ../scripts/fetch_fundus_dataset.py --zip /path/to/dataset.zip
python ../scripts/train_fundus.py --epochs 15
```

Without trained fundus weights the API still runs but clearly reports the fundus model as
**uncalibrated** rather than emitting confident noise.

---

## Data model

25 columns: `patient_id`, demographics (age, sex, bmi, ethnicity, eye_color), IOP (od/os),
structural (cup_disc_ratio, RNFL superior/inferior/average), functional (mean_deviation od/os,
pattern_sd, va od/os), systemic (hba1c, BP, diabetes, hypertension, family_history),
`treatment`, and the target `disease_severity` ∈ {Normal, Suspect, Mild/Moderate/Severe
Glaucoma}. The generator samples a plausible clinic population (a latent optic-nerve-damage
factor drives correlated structure and function) and **derives the label from the
measurements** — the direction real diagnosis works. See
[`DATA_REPORT.md`](ophthalmic-digital-twin/DATA_REPORT.md).

---

## Clinical references

- Hodapp E, Parrish RK, Anderson DR. *Clinical Decisions in Glaucoma.* (MD-based staging)
- Early Manifest Glaucoma Trial (EMGT); CIGTS; AGIS — IOP lowering and progression.
- AAO Primary Open-Angle Glaucoma Preferred Practice Pattern — target IOP & treatment ladder.
- Public fundus datasets: REFUGE, ORIGA, G1020, RIM-ONE DL, ACRIMA.

---

## License

See [LICENSE](LICENSE).
