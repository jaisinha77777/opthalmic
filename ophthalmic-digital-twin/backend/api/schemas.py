"""
Pydantic request/response schemas for the FastAPI application.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    patient_features: Dict[str, Any] = Field(
        ...,
        description="Raw feature values keyed by feature name.",
        example={"age": 65, "iop": 22.4, "diagnosis": "glaucoma"},
    )
    patient_id: str = Field(..., description="Unique patient identifier.")
    mc_samples: int = Field(50, ge=1, le=200, description="MC Dropout samples.")


class SimulateRequest(BaseModel):
    patient_id: str
    horizon: int = Field(12, ge=1, le=52, description="Future timesteps to simulate.")
    treatment_action: int = Field(0, ge=0, description="Treatment action index to apply each step.")
    compliance_level: float = Field(0.8, ge=0.0, le=1.0)


class TreatmentRequest(BaseModel):
    patient_id: str
    mc_samples: int = Field(50, ge=1, le=200)


# ─────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    patient_id: str
    prediction: int
    prediction_label: str
    probabilities: List[float]
    confidence: float
    epistemic_variance: List[float]
    aleatoric_variance: float
    reliable: bool
    shap_values: List[float]
    top_features: List[List[Any]]           # [(name, score), ...]
    attention_heatmap: List[List[float]]    # aggregated N×N heatmap
    feature_importance: List[float]
    latent_state_3d: List[float]            # [x, y, z]
    total_uncertainty: float


class TwinStateResponse(BaseModel):
    patient_id: str
    current_state_3d: List[float]
    state_trajectory_3d: List[List[float]]
    timestep: int
    uncertainty_log: List[Dict[str, float]]
    action_history: List[Dict[str, Any]]
    latent_norm: float


class SimulateResponse(BaseModel):
    patient_id: str
    states_3d: List[List[float]]
    predictions: List[Any]
    uncertainties: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    horizon: int


class TreatmentResponse(BaseModel):
    patient_id: str
    recommended_treatment: int
    treatment_name: str
    doctor_policy: List[float]
    disease_policy: List[float]
    patient_compliance: List[float]
    compliance_level: float
    expected_outcome: float
    nash_convergence_step: int
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    n_active_twins: int
    model_type: str


class FeatureNamesResponse(BaseModel):
    feature_names: List[str]
    col_types: List[str]
    n_features: int
    task: str
    n_classes: int


class FundusAnalysisResponse(BaseModel):
    patient_id: str
    prediction: int
    prediction_label: str
    probabilities: List[float]
    confidence: float
    gradcam_overlay: str          # base64 PNG (224×224 image + heatmap blend)
    image_preview: str            # base64 PNG of resized original
    top_findings: List[List[Any]] # [[region, score], ...]
    model_note: str               # disclaimer about pretrained weights
