"""
FastAPI route handlers for the Ophthalmic Digital Twin API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .schemas import (
    FeatureNamesResponse,
    FundusAnalysisResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SimulateRequest,
    SimulateResponse,
    TreatmentRequest,
    TreatmentResponse,
    TwinStateResponse,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def _get_app_state(request=None):
    """Import app state lazily to avoid circular imports."""
    from .main import app_state
    return app_state


def _preprocess_features(
    patient_features: Dict[str, Any],
    app_state: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert raw feature dict to model input tensors.
    Returns (x_tensor, missingness_tensor) each shaped [1, N_features] or [1, T, N_features].
    """
    metadata = app_state["feature_metadata"]
    preprocessor = app_state["preprocessor"]
    seq_len = metadata.get("seq_len", 1)
    has_sequences = metadata.get("has_sequences", False)

    import pandas as pd
    col_names = metadata["col_names"]

    # Build a single-row DataFrame aligned with training columns
    row = {}
    for col in col_names:
        row[col] = patient_features.get(col, None)

    df_row = pd.DataFrame([row])
    feature_matrix, miss_matrix, _ = preprocessor.transform(df_row)

    x = torch.from_numpy(feature_matrix)   # [1, F]
    m = torch.from_numpy(miss_matrix)      # [1, F]

    if has_sequences and seq_len > 1:
        # Match training distribution: real data at t=0, zeros+missing at t=1..T-1.
        # Training pseudo-sequences are [real@t0, zeros@t1..T-1] with M=1 for padding.
        # expand() was wrong — it made all timesteps identical, diverging from training.
        x_seq = torch.zeros(1, seq_len, x.shape[-1], dtype=x.dtype)
        m_seq = torch.ones(1, seq_len, x.shape[-1], dtype=m.dtype)   # padding = missing
        x_seq[:, 0, :] = x          # real observation at t=0
        m_seq[:, 0, :] = m          # real missingness flags at t=0
        x, m = x_seq, m_seq

    return x, m


def _project_latent_3d(latent: torch.Tensor, app_state: Dict[str, Any]) -> List[float]:
    """Project latent state to 3D via fitted PCA."""
    from core.twin_engine import get_global_pca
    pca = get_global_pca()
    state_np = latent.detach().cpu().numpy().flatten()
    if pca is not None:
        try:
            proj = pca.transform(state_np.reshape(1, -1))[0]
            while len(proj) < 3:
                proj = np.append(proj, 0.0)
            return proj[:3].tolist()
        except Exception:
            pass
    # Fallback
    norm = np.linalg.norm(state_np[:3]) + 1e-8
    return (state_np[:3] / norm).tolist()


# ─────────────────────────────────────────────────────────
# GET /api/v1/health
# ─────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    from .main import app_state
    from core.twin_engine import _REGISTRY

    return HealthResponse(
        status="ok",
        model_loaded=app_state.get("model") is not None,
        device=str(app_state.get("device", "cpu")),
        n_active_twins=len(_REGISTRY),
        model_type=app_state.get("model_type", "unknown"),
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
# POST /api/v1/predict
# ─────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    from .main import app_state
    from core.uncertainty import mc_predict
    from core.twin_engine import DigitalTwinEngine

    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    device: torch.device = app_state["device"]
    metadata = app_state["feature_metadata"]
    explainability = app_state.get("explainability_engine")

    try:
        x, m = _preprocess_features(request.patient_features, app_state)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature preprocessing failed: {e}")

    x = x.to(device)
    m = m.to(device)

    # MC Dropout inference
    mc_out = mc_predict(model, x, m, n_samples=request.mc_samples)

    # Explainability
    attention_result: Dict[str, Any] = {"aggregated": [], "feature_importance": [], "top_features": []}
    shap_result: Dict[str, Any] = {"shap_values": [], "normalized": []}

    if explainability is not None:
        try:
            attention_result = explainability.get_attention_heatmap(x, m)
            shap_result = explainability.get_shap_values(x)
        except Exception as e:
            log.warning("Explainability failed: %s", e)

    # Latent state 3D projection
    with torch.no_grad():
        out = model(x, m)
    latent = out["latent_state"].squeeze(0)
    latent_3d = _project_latent_3d(latent, app_state)

    # Register/update digital twin
    try:
        twin_engine_module = app_state.get("twin_engine_factory")
        if twin_engine_module is not None:
            twin_engine_module(request.patient_id, x.squeeze(0).cpu(), latent)
    except Exception as e:
        log.debug("Twin registration skipped: %s", e)

    # Map prediction to label
    n_classes = metadata["n_classes"]
    pred_idx = int(mc_out["prediction"].item())
    class_labels = metadata.get("class_labels", [str(i) for i in range(n_classes)])
    pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else str(pred_idx)

    probs = mc_out["probabilities"].squeeze(0)
    epistemic = mc_out["epistemic_variance"].squeeze(0)
    aleatoric = mc_out["aleatoric_variance"].squeeze(0)

    # Ensure aggregated heatmap is a 2D list of floats
    agg = attention_result.get("aggregated", [])
    try:
        if agg and isinstance(agg[0], list):
            heatmap = [[float(v) for v in row] for row in agg]
        elif agg and isinstance(agg, list):
            heatmap = [[float(v) for v in agg]]
        else:
            heatmap = [[0.0]]
    except Exception:
        heatmap = [[0.0]]

    return PredictResponse(
        patient_id=request.patient_id,
        prediction=pred_idx,
        prediction_label=pred_label,
        probabilities=probs.tolist(),
        confidence=float(mc_out["confidence"].item()),
        epistemic_variance=epistemic.tolist(),
        aleatoric_variance=float(aleatoric.mean().item()),
        reliable=bool(mc_out["reliable"].item()),
        shap_values=shap_result.get("shap_values", []),
        top_features=[[name, score] for name, score in attention_result.get("top_features", [])],
        attention_heatmap=heatmap,
        feature_importance=attention_result.get("feature_importance", []),
        latent_state_3d=latent_3d,
        total_uncertainty=float(mc_out["total_uncertainty"].item()),
    )


# ─────────────────────────────────────────────────────────
# GET /api/v1/twin-state/{patient_id}
# ─────────────────────────────────────────────────────────

@router.get("/twin-state/{patient_id}", response_model=TwinStateResponse)
async def get_twin_state(patient_id: str) -> TwinStateResponse:
    from core.twin_engine import _REGISTRY

    twin = _REGISTRY.get(patient_id)
    if twin is None:
        raise HTTPException(status_code=404, detail=f"No twin found for patient_id='{patient_id}'. Run /predict first.")

    state_dict = twin.get_state_dict()
    return TwinStateResponse(
        patient_id=state_dict["patient_id"],
        current_state_3d=state_dict["current_state_pca"],
        state_trajectory_3d=state_dict["state_trajectory_3d"],
        timestep=state_dict["timestep"],
        uncertainty_log=state_dict["uncertainty_log"],
        action_history=state_dict["action_history"],
        latent_norm=state_dict["latent_norm"],
    )


# ─────────────────────────────────────────────────────────
# POST /api/v1/simulate
# ─────────────────────────────────────────────────────────

@router.post("/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest) -> SimulateResponse:
    from .main import app_state
    from core.twin_engine import _REGISTRY, DigitalTwinEngine

    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    twin = _REGISTRY.get(request.patient_id)
    if twin is None:
        raise HTTPException(
            status_code=404,
            detail=f"No digital twin for patient '{request.patient_id}'. Call /predict first.",
        )

    def policy_fn(S_t: torch.Tensor) -> int:
        return request.treatment_action

    sim_result = twin.simulate_horizon(
        horizon=request.horizon,
        policy_fn=policy_fn,
        n_mc=30,
    )

    return SimulateResponse(
        patient_id=request.patient_id,
        states_3d=sim_result["states"],
        predictions=sim_result["predictions"],
        uncertainties=sim_result["uncertainties"],
        confidence_lower=sim_result["confidence_bands"]["lower"],
        confidence_upper=sim_result["confidence_bands"]["upper"],
        horizon=request.horizon,
    )


# ─────────────────────────────────────────────────────────
# POST /api/v1/recommend-treatment
# ─────────────────────────────────────────────────────────

@router.post("/recommend-treatment", response_model=TreatmentResponse)
async def recommend_treatment(request: TreatmentRequest) -> TreatmentResponse:
    from .main import app_state
    from core.twin_engine import _REGISTRY
    from core.nash_solver import NashEquilibriumSolver

    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    twin = _REGISTRY.get(request.patient_id)
    if twin is None:
        raise HTTPException(
            status_code=404,
            detail=f"No twin for patient '{request.patient_id}'. Call /predict first.",
        )

    doctor = app_state.get("doctor_agent")
    disease = app_state.get("disease_agent")
    patient_agent = app_state.get("patient_agent")
    device = app_state["device"]

    if doctor is None or disease is None or patient_agent is None:
        raise HTTPException(status_code=503, detail="MARL agents not initialized.")

    solver = NashEquilibriumSolver(
        doctor=doctor,
        disease=disease,
        patient=patient_agent,
        twin_engine=twin,
        n_iters=20,
        convergence_eps=1e-3,
        n_steps=64,
        device=device,
    )

    S_t = twin.S_t.to(device)
    nash_result = solver.solve(S_t, request.patient_id)

    # Map action to treatment name
    treatment_names = app_state.get("treatment_names", [])
    action_idx = nash_result["nash_treatment"]
    treatment_name = (
        treatment_names[action_idx]
        if action_idx < len(treatment_names)
        else f"Treatment_{action_idx}"
    )

    # Confidence derived from Nash expected reward (clamped to [0, 1])
    confidence = float(np.clip(nash_result["expected_reward"], 0.0, 1.0))

    return TreatmentResponse(
        patient_id=request.patient_id,
        recommended_treatment=action_idx,
        treatment_name=treatment_name,
        doctor_policy=nash_result["doctor_policy"],
        disease_policy=nash_result["disease_policy"],
        patient_compliance=nash_result["patient_compliance"],
        compliance_level=nash_result["compliance_level"],
        expected_outcome=nash_result["expected_reward"],
        nash_convergence_step=nash_result["convergence_step"],
        confidence=confidence,
    )


# ─────────────────────────────────────────────────────────
# POST /api/v1/analyze-fundus
# ─────────────────────────────────────────────────────────

@router.post("/analyze-fundus", response_model=FundusAnalysisResponse)
async def analyze_fundus(
    patient_id: str = Form(default="unknown"),
    image: UploadFile = File(...),
) -> FundusAnalysisResponse:
    """
    Analyze a fundus photograph.
    Accepts multipart/form-data with fields:
      - patient_id  (string, optional)
      - image       (file: JPEG or PNG)

    Returns ResNet-18 classification + GradCAM overlay + anatomical findings.
    """
    from .main import app_state
    from core.fundus import (
        SEVERITY_LABELS,
        anatomical_findings,
        image_preview_b64,
        load_image_tensor,
        overlay_gradcam,
    )

    fundus_encoder = app_state.get("fundus_encoder")
    if fundus_encoder is None:
        raise HTTPException(status_code=503, detail="Fundus encoder not initialized.")

    device: torch.device = app_state["device"]

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Empty image file.")

    try:
        x = load_image_tensor(image_bytes, device)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image decode failed: {e}")

    # Forward pass (no grad needed for probabilities)
    fundus_encoder.eval()
    with torch.no_grad():
        logits = fundus_encoder(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    pred_idx = int(probs.argmax().item())
    confidence = float(probs.max().item())

    # GradCAM requires grad, so computed separately
    try:
        cam = fundus_encoder.gradcam(x, class_idx=pred_idx)
        gradcam_b64 = overlay_gradcam(cam, image_bytes)
    except Exception as e:
        log.warning("GradCAM failed: %s", e)
        cam = np.zeros((7, 7), dtype=np.float32)
        gradcam_b64 = ""

    findings = anatomical_findings(cam)
    preview = image_preview_b64(image_bytes)

    return FundusAnalysisResponse(
        patient_id=patient_id,
        prediction=pred_idx,
        prediction_label=SEVERITY_LABELS[pred_idx],
        probabilities=probs.tolist(),
        confidence=confidence,
        gradcam_overlay=gradcam_b64,
        image_preview=preview,
        top_findings=[[region, score] for region, score in findings],
        model_note="ResNet-18 pretrained on ImageNet. Fine-tune on labelled fundus data for clinical use.",
    )
