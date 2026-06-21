"""
FastAPI application with lifespan model loading and CORS.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger(__name__)

# Global application state shared across requests
app_state: Dict[str, Any] = {}

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "backend" / "models"
DATA_DIR = BASE_DIR / "data"
# Built frontend (vite output). Served by the API in production so the whole
# app ships as a single artifact. Override with FRONTEND_DIST if needed.
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", str(BASE_DIR / "frontend" / "dist")))


def _cors_origins() -> list[str]:
    """Allowed CORS origins from env (comma-separated). Defaults to '*'.

    Note: when origins is '*', credentials must be disabled — the browser
    rejects 'Access-Control-Allow-Origin: *' together with credentials.
    """
    raw = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if raw == "*" or not raw:
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


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
    """
    Load the preprocessor fitted at training time (preferred), so the API uses the
    exact scalers/encoders the model was trained with. Falls back to re-fitting on
    the recorded training CSV only if the pickle is missing.
    """
    import sys
    import pickle
    sys.path.insert(0, str(BASE_DIR / "backend"))
    from core.dataset import FeaturePreprocessor
    import pandas as pd

    pkl_path = DATA_DIR / "preprocessor.pkl"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                prep = pickle.load(f)
            log.info("Loaded fitted preprocessor from %s", pkl_path)
            return prep
        except Exception as e:
            log.warning("Preprocessor unpickle failed (%s); re-fitting.", e)

    csv_path = Path(metadata.get("trained_csv", DATA_DIR / "full_df.csv"))
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        prep = FeaturePreprocessor(
            numerical_cols=metadata.get("numerical_cols", []),
            categorical_cols=metadata.get("categorical_cols", []),
            binary_cols=metadata.get("binary_cols", []),
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
    log.info("Starting Glaucoma Clinical Support API on device=%s", device)

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

    # Fundus image encoder (ResNet-18 backbone) + trained weights if present
    try:
        from core.fundus import FundusEncoder
        fundus_encoder = FundusEncoder(pretrained=True).to(device)
        fundus_encoder.load_weights(MODELS_DIR / "fundus_model.pt", device)
        fundus_encoder.eval()
        app_state["fundus_encoder"] = fundus_encoder
        log.info("FundusEncoder initialized (calibrated=%s)", fundus_encoder.is_calibrated)
    except Exception as e:
        log.error("FundusEncoder init failed: %s", e)
        app_state["fundus_encoder"] = None

    log.info("API startup complete.")
    yield

    # Shutdown
    log.info("API shutting down.")
    app_state.clear()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Glaucoma Clinical Support API",
        description="Glaucoma staging, uncertainty, progression projection, and guideline decision support.",
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = _cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Credentials cannot be combined with the '*' wildcard (browser rejects it).
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .routes import router
    app.include_router(router)

    # Serve the built frontend (if present) so the app deploys as one artifact.
    # The API router (prefix /api/v1) is matched first; everything else falls
    # through to the SPA, with index.html as the client-side-routing fallback.
    if FRONTEND_DIST.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(FRONTEND_DIST / "assets")),
            name="assets",
        )

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            candidate = (FRONTEND_DIST / full_path).resolve()
            if (
                full_path
                and candidate.is_file()
                and str(candidate).startswith(str(FRONTEND_DIST.resolve()))
            ):
                return FileResponse(str(candidate))
            return FileResponse(str(FRONTEND_DIST / "index.html"))

        log.info("Serving frontend from %s", FRONTEND_DIST)
    else:
        log.info("No frontend build at %s — API only.", FRONTEND_DIST)

    return app


app = create_app()
