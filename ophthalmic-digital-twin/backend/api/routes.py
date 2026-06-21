"""
FastAPI route handlers for the Ophthalmic clinical model API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .schemas import (
    DecisionRequest,
    DecisionResponse,
    FeatureNamesResponse,
    FundusAnalysisResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ProgressionRequest,
    ProgressionResponse,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

# Canonical clinical severity order (model class indices are alphabetical, not clinical).
CANON_SEVERITY = {
    "Normal": 0, "Suspect": 1, "Mild Glaucoma": 2,
    "Moderate Glaucoma": 3, "Severe Glaucoma": 4,
}


def _preprocess_features(
    patient_features: Dict[str, Any], app_state: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a raw feature dict to model input tensors [1, F]."""
    import pandas as pd
    metadata = app_state["feature_metadata"]
    preprocessor = app_state["preprocessor"]
    col_names = metadata["col_names"]

    row = {col: patient_features.get(col, None) for col in col_names}
    feature_matrix, miss_matrix, _ = preprocessor.transform(pd.DataFrame([row]))
    return torch.from_numpy(feature_matrix), torch.from_numpy(miss_matrix)


def _run_model(patient_features: Dict[str, Any], app_state: Dict[str, Any], mc_samples: int = 50):
    """Shared MC-Dropout inference. Returns (mc_out, label, severity_index, x, m)."""
    from core.uncertainty import mc_predict

    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    device = app_state["device"]
    metadata = app_state["feature_metadata"]

    try:
        x, m = _preprocess_features(patient_features, app_state)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature preprocessing failed: {e}")
    x, m = x.to(device), m.to(device)

    mc_out = mc_predict(model, x, m, n_samples=mc_samples)
    pred_idx = int(mc_out["prediction"].item())
    class_labels = metadata.get("class_labels", [str(i) for i in range(metadata["n_classes"])])
    label = class_labels[pred_idx] if pred_idx < len(class_labels) else str(pred_idx)
    severity_index = CANON_SEVERITY.get(label, 0)
    return mc_out, label, severity_index, x, m


def _baseline_iop(features: Dict[str, Any]) -> float:
    vals = [features.get(k) for k in ("iop_od", "iop_os") if features.get(k) is not None]
    vals = [float(v) for v in vals]
    return float(np.mean(vals)) if vals else 18.0


def _baseline_md(features: Dict[str, Any]) -> float:
    vals = [features.get(k) for k in ("mean_deviation_od", "mean_deviation_os")
            if features.get(k) is not None]
    vals = [float(v) for v in vals]
    return float(min(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────
# GET /api/v1/health
# ─────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    from .main import app_state
    fundus = app_state.get("fundus_encoder")
    return HealthResponse(
        status="ok",
        model_loaded=app_state.get("model") is not None,
        device=str(app_state.get("device", "cpu")),
        model_type=app_state.get("model_type", "unknown"),
        fundus_calibrated=bool(getattr(fundus, "is_calibrated", False)),
    )


# ─────────────────────────────────────────────────────────
# GET /api/v1/feature-names
# ─────────────────────────────────────────────────────────

@router.get("/feature-names", response_model=FeatureNamesResponse)
async def get_feature_names() -> FeatureNamesResponse:
    from .main import app_state
    metadata = app_state.get("feature_metadata")
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return FeatureNamesResponse(
        feature_names=metadata["col_names"],
        col_types=metadata["col_types"],
        n_features=metadata["n_features"],
        task=metadata["task"],
        n_classes=metadata["n_classes"],
    )


# ─────────────────────────────────────────────────────────
# POST /api/v1/predict  (MC Dropout + SHAP + attention)
# ─────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    from .main import app_state

    mc_out, label, severity_index, x, m = _run_model(
        request.patient_features, app_state, request.mc_samples
    )
    metadata = app_state["feature_metadata"]
    class_labels = metadata.get("class_labels", [])

    attention_result: Dict[str, Any] = {"aggregated": [], "feature_importance": [], "top_features": []}
    shap_result: Dict[str, Any] = {"shap_values": []}
    explainability = app_state.get("explainability_engine")
    if explainability is not None:
        try:
            attention_result = explainability.get_attention_heatmap(x, m)
            shap_result = explainability.get_shap_values(x)
        except Exception as e:
            log.warning("Explainability failed: %s", e)

    probs = mc_out["probabilities"].squeeze(0)
    epistemic = mc_out["epistemic_variance"].squeeze(0)
    aleatoric = mc_out["aleatoric_variance"].squeeze(0)

    agg = attention_result.get("aggregated", [])
    try:
        if agg and isinstance(agg[0], list):
            heatmap = [[float(v) for v in row] for row in agg]
        elif agg:
            heatmap = [[float(v) for v in agg]]
        else:
            heatmap = [[0.0]]
    except Exception:
        heatmap = [[0.0]]

    return PredictResponse(
        patient_id=request.patient_id,
        prediction=int(mc_out["prediction"].item()),
        prediction_label=label,
        severity_index=severity_index,
        class_labels=class_labels,
        probabilities=probs.tolist(),
        confidence=float(mc_out["confidence"].item()),
        epistemic_variance=epistemic.tolist(),
        aleatoric_variance=float(aleatoric.mean().item()),
        reliable=bool(mc_out["reliable"].item()),
        shap_values=shap_result.get("shap_values", []),
        top_features=[[n, s] for n, s in attention_result.get("top_features", [])],
        attention_heatmap=heatmap,
        feature_importance=attention_result.get("feature_importance", []),
        total_uncertainty=float(mc_out["total_uncertainty"].item()),
    )


# ─────────────────────────────────────────────────────────
# POST /api/v1/simulate  (transparent MD progression projection)
# ─────────────────────────────────────────────────────────

@router.post("/simulate", response_model=ProgressionResponse)
async def simulate(request: ProgressionRequest) -> ProgressionResponse:
    from .main import app_state
    from core.progression import project

    _, _, severity_index, _, _ = _run_model(request.patient_features, app_state, mc_samples=20)
    proj = project(
        baseline_md=_baseline_md(request.patient_features),
        baseline_iop=_baseline_iop(request.patient_features),
        stage_idx=severity_index,
        horizon_months=request.horizon_months,
        iop_reduction=request.iop_reduction,
    )
    return ProgressionResponse(patient_id=request.patient_id, **proj)


# ─────────────────────────────────────────────────────────
# POST /api/v1/recommend-treatment  (guideline decision support)
# ─────────────────────────────────────────────────────────

@router.post("/recommend-treatment", response_model=DecisionResponse)
async def recommend_treatment(request: DecisionRequest) -> DecisionResponse:
    from .main import app_state
    from core.decision_support import recommend

    _, _, severity_index, _, _ = _run_model(request.patient_features, app_state, mc_samples=20)
    f = request.patient_features
    rec = recommend(
        stage_idx=severity_index,
        baseline_iop=_baseline_iop(f),
        risk_factors={
            "family_history": bool(f.get("family_history")),
            "diabetes": bool(f.get("diabetes")),
            "hypertension": bool(f.get("hypertension")),
        },
    )
    return DecisionResponse(patient_id=request.patient_id, **rec)


# ─────────────────────────────────────────────────────────
# POST /api/v1/analyze-fundus
# ─────────────────────────────────────────────────────────

@router.post("/analyze-fundus", response_model=FundusAnalysisResponse)
async def analyze_fundus(
    patient_id: str = Form(default="unknown"),
    image: UploadFile = File(...),
) -> FundusAnalysisResponse:
    """Analyze a fundus photograph: referable glaucoma + vertical CDR + GradCAM."""
    from .main import app_state
    from core.fundus import (
        REFERRAL_LABELS, anatomical_findings, image_preview_b64,
        load_image_tensor, overlay_gradcam,
    )

    fundus_encoder = app_state.get("fundus_encoder")
    if fundus_encoder is None:
        raise HTTPException(status_code=503, detail="Fundus encoder not initialized.")
    device = app_state["device"]

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Empty image file.")
    try:
        x = load_image_tensor(image_bytes, device)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image decode failed: {e}")

    fundus_encoder.eval()
    with torch.no_grad():
        out = fundus_encoder(x)
    probs = torch.softmax(out["referral_logits"], dim=-1).squeeze(0)
    glaucoma_prob = float(probs[1].item())
    referable = glaucoma_prob >= 0.5
    est_cdr = float(out["cdr"].squeeze(0).item())

    try:
        cam = fundus_encoder.gradcam(x, target="referral")
        gradcam_b64 = overlay_gradcam(cam, image_bytes)
    except Exception as e:
        log.warning("GradCAM failed: %s", e)
        cam = np.zeros((7, 7), dtype=np.float32)
        gradcam_b64 = ""

    calibrated = bool(getattr(fundus_encoder, "is_calibrated", False))
    note = ("Calibrated on a real fundus dataset." if calibrated else
            "UNCALIBRATED: ImageNet backbone with untrained heads. Run "
            "scripts/fetch_fundus_dataset.py + scripts/train_fundus.py for valid outputs.")

    return FundusAnalysisResponse(
        patient_id=patient_id,
        referable=referable,
        referral_label=REFERRAL_LABELS[int(referable)],
        glaucoma_probability=glaucoma_prob,
        estimated_vertical_cdr=round(est_cdr, 3),
        calibrated=calibrated,
        gradcam_overlay=gradcam_b64,
        image_preview=image_preview_b64(image_bytes),
        top_findings=[[r, s] for r, s in anatomical_findings(cam)],
        model_note=note,
    )
