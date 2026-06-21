"""
Pydantic request/response schemas for the FastAPI application.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# Requests
# ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    patient_features: Dict[str, Any] = Field(
        ..., description="Raw feature values keyed by feature name.",
        example={"age": 65, "iop_od": 24.0, "cup_disc_ratio": 0.7, "mean_deviation_od": -8.0},
    )
    patient_id: str = Field(..., description="Unique patient identifier.")
    mc_samples: int = Field(50, ge=1, le=200, description="MC Dropout samples.")


class ProgressionRequest(BaseModel):
    patient_features: Dict[str, Any]
    patient_id: str = "unknown"
    horizon_months: int = Field(60, ge=6, le=240)
    iop_reduction: float = Field(0.30, ge=0.0, le=0.7,
                                 description="Fraction of IOP removed by treatment (0=untreated).")


class DecisionRequest(BaseModel):
    patient_features: Dict[str, Any]
    patient_id: str = "unknown"


# ─────────────────────────────────────────────────────────
# Responses
# ─────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    patient_id: str
    prediction: int
    prediction_label: str
    severity_index: int                     # canonical 0=Normal .. 4=Severe
    class_labels: List[str]                 # model class order (matches probabilities)
    probabilities: List[float]
    confidence: float
    epistemic_variance: List[float]
    aleatoric_variance: float
    reliable: bool
    shap_values: List[float]
    top_features: List[List[Any]]           # [(name, score), ...]
    attention_heatmap: List[List[float]]
    feature_importance: List[float]
    total_uncertainty: float


class ProgressionResponse(BaseModel):
    patient_id: str
    months: List[int]
    md_treated: List[float]
    md_untreated: List[float]
    md_lower: List[float]
    md_upper: List[float]
    projected_stage: List[str]
    untreated_slope_db_yr: float
    treated_slope_db_yr: float
    assumptions: str


class DecisionResponse(BaseModel):
    patient_id: str
    stage: str
    baseline_iop: float
    target_iop: Optional[float]
    target_reduction: float
    target_rationale: str
    at_target: bool
    next_step: str
    rationale: str
    risk_factors: List[str]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_type: str
    fundus_calibrated: bool


class FeatureNamesResponse(BaseModel):
    feature_names: List[str]
    col_types: List[str]
    n_features: int
    task: str
    n_classes: int


class FundusAnalysisResponse(BaseModel):
    patient_id: str
    referable: bool
    referral_label: str
    glaucoma_probability: float
    estimated_vertical_cdr: float
    calibrated: bool                # False until trained weights are loaded
    gradcam_overlay: str            # base64 PNG
    image_preview: str              # base64 PNG
    top_findings: List[List[Any]]   # [[region, score], ...]
    model_note: str
