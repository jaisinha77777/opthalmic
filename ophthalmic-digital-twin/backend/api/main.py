"""
FastAPI application with lifespan model loading and CORS.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)

# Global application state shared across requests
app_state: Dict[str, Any] = {}

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "backend" / "models"
DATA_DIR = BASE_DIR / "data"


def _load_feature_metadata() -> Dict[str, Any]:
    meta_path = DATA_DIR / "feature_metadata.json"
    if not meta_path.exists():
        log.warning("feature_metadata.json not found — using defaults")
        return {
            "col_names": ["feature_0"],
            "col_types": ["numerical"],
            "cat_vocab_sizes": {},
            "n_features": 1,
            "n_classes": 2,
            "task": "classification",
            "has_sequences": False,
            "seq_len": 1,
            "numerical_cols": ["feature_0"],
            "categorical_cols": [],
            "binary_cols": [],
        }
    with open(meta_path) as f:
        return json.load(f)


def _rebuild_preprocessor(metadata: Dict[str, Any]):
    """Reconstruct preprocessor from saved metadata (best-effort)."""
    import sys, os
    sys.path.insert(0, str(BASE_DIR / "backend"))
    from core.dataset import FeaturePreprocessor
    import pandas as pd

    # Try to reload from CSV for fitting
    csv_path = DATA_DIR / "full_df.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        numerical_cols = metadata.get("numerical_cols", [])
        categorical_cols = metadata.get("categorical_cols", [])
        binary_cols = metadata.get("binary_cols", [])
        # col_types drives the full column ordering — honour it exactly
        prep = FeaturePreprocessor(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            binary_cols=binary_cols,
            high_cardinality_cols=[],
        )
        prep.fit(df)
        return prep
    except Exception as e:
        log.warning("Preprocessor reconstruction failed: %s", e)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and supporting components on startup."""
    import sys
    sys.path.insert(0, str(BASE_DIR / "backend"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app_state["device"] = device
    log.info("Starting Ophthalmic Digital Twin API on device=%s", device)

    # Load feature metadata
    metadata = _load_feature_metadata()
    app_state["feature_metadata"] = metadata

    # Reconstruct preprocessor
    preprocessor = _rebuild_preprocessor(metadata)
    app_state["preprocessor"] = preprocessor

    # Build and load model
    model = None
    model_type = "none"
    ckpt_path = MODELS_DIR / "best_model.pt"

    if ckpt_path.exists():
        try:
            from core.model import build_model
            model = build_model(metadata)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            model.eval()
            model_type = type(model).__name__
            log.info("Loaded model: %s from epoch %d", model_type, ckpt.get("epoch", 0))
        except Exception as e:
            log.error("Model load failed: %s", e)
    else:
        log.warning("No checkpoint found at %s — running without model", ckpt_path)

    app_state["model"] = model
    app_state["model_type"] = model_type

    # Build background data for SHAP (100 random samples from training distribution)
    # Always use flat [100, F] — TemporalTransformer input is handled by passing [1, T, F]
    # but the SHAP background wrapper flattens T*F, so use flat here for stability
    background_data = torch.randn(100, metadata["n_features"])

    # Explainability engine
    if model is not None:
        try:
            from core.explainability import ExplainabilityEngine
            explainability = ExplainabilityEngine(
                model=model,
                feature_names=metadata["col_names"],
                background_data=background_data.to(device),
                device=device,
            )
            app_state["explainability_engine"] = explainability
            log.info("ExplainabilityEngine initialized")
        except Exception as e:
            log.error("ExplainabilityEngine init failed: %s", e)
            app_state["explainability_engine"] = None
    else:
        app_state["explainability_engine"] = None

    # Fundus image encoder (ResNet-18 backbone)
    try:
        from core.fundus import FundusEncoder
        fundus_encoder = FundusEncoder(n_classes=metadata["n_classes"]).to(device)
        fundus_encoder.eval()
        app_state["fundus_encoder"] = fundus_encoder
        log.info("FundusEncoder initialized (ResNet-18, %d classes)", metadata["n_classes"])
    except Exception as e:
        log.error("FundusEncoder init failed: %s", e)
        app_state["fundus_encoder"] = None

    # Build MARL agents
    n_treatments = max(metadata.get("n_classes", 4), 4)
    try:
        from core.agents import build_agents
        doctor, disease, patient_agent = build_agents(
            n_treatments=n_treatments, device=device
        )
        app_state["doctor_agent"] = doctor
        app_state["disease_agent"] = disease
        app_state["patient_agent"] = patient_agent
        log.info("MARL agents initialized (n_treatments=%d)", n_treatments)
    except Exception as e:
        log.error("MARL agent init failed: %s", e)
        app_state["doctor_agent"] = None
        app_state["disease_agent"] = None
        app_state["patient_agent"] = None

    # Treatment names
    app_state["treatment_names"] = [f"Treatment_{i}" for i in range(n_treatments)]

    # Twin engine factory
    if model is not None:
        from core.twin_engine import ActionEmbedder, StateTransitionMLP

        action_embedder = ActionEmbedder(n_treatments, action_dim=64).to(device)
        state_mlp = StateTransitionMLP(d_model=256, action_dim=64).to(device)
        app_state["action_embedder"] = action_embedder
        app_state["state_mlp"] = state_mlp

        def twin_engine_factory(patient_id: str, features: torch.Tensor, latent: torch.Tensor):
            from core.twin_engine import DigitalTwinEngine, _REGISTRY
            if patient_id not in _REGISTRY:
                twin = DigitalTwinEngine(
                    patient_id=patient_id,
                    initial_features=features.to(device),
                    model=model,
                    action_embedder=action_embedder,
                    state_mlp=state_mlp,
                    device=device,
                )
                # Override initial state with already-computed latent
                twin.S_t = latent.detach().to(device)
                twin.state_trajectory = [twin.S_t.clone()]

        app_state["twin_engine_factory"] = twin_engine_factory
        log.info("DigitalTwinEngine factory ready")

    log.info("API startup complete.")
    yield

    # Shutdown
    log.info("API shutting down.")
    app_state.clear()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Ophthalmic Digital Twin API",
        description="AI-powered digital twin system for ophthalmic disease modeling.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .routes import router
    app.include_router(router)

    return app


app = create_app()
