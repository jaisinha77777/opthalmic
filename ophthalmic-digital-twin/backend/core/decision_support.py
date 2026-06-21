"""
Guideline-based glaucoma decision-support.

Replaces the previous "Nash equilibrium between a doctor and an adversarial disease
agent" (which had no clinical basis) with a transparent rule engine that mirrors how
target IOP and the treatment ladder are actually chosen in practice (EMGT, CIGTS,
AGIS; AAO Primary Open-Angle Glaucoma Preferred Practice Pattern).

Every recommendation comes with an explicit rationale string. This is DECISION SUPPORT,
not a prescription: the final plan is always the treating ophthalmologist's.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Stage index -> (target % IOP reduction from baseline, absolute target ceiling mmHg).
# Lower, stricter targets for more advanced disease (CIGTS/AGIS philosophy).
_STAGE_TARGETS = {
    0: (0.0,  None),   # Normal           - no IOP target unless ocular hypertension
    1: (0.20, 24.0),   # Suspect          - modest target if risk factors present
    2: (0.25, 21.0),   # Mild Glaucoma
    3: (0.30, 18.0),   # Moderate Glaucoma
    4: (0.40, 15.0),   # Severe Glaucoma  - aggressive, protect remaining field
}

STAGE_NAMES = ["Normal", "Suspect", "Mild Glaucoma", "Moderate Glaucoma", "Severe Glaucoma"]


def _treatment_ladder(stage_idx: int, at_target: bool, risk_factors: List[str]) -> Dict[str, Any]:
    """
    Suggest the next rung of the standard glaucoma treatment ladder.

        observation -> prostaglandin analogue drops -> add 2nd agent / SLT laser
        -> incisional surgery (trabeculectomy / MIGS / tube)
    """
    if stage_idx == 0:
        step = "Observation"
        detail = ("No glaucomatous damage. Routine screening; treat only if ocular "
                  "hypertension with high conversion risk.")
    elif stage_idx == 1:
        if risk_factors:
            step = "First-line topical therapy (prostaglandin analogue)"
            detail = ("Glaucoma suspect with risk factors ({}). Consider starting a "
                      "once-daily prostaglandin analogue and monitor.").format(", ".join(risk_factors))
        else:
            step = "Observation with monitoring"
            detail = ("Glaucoma suspect, no strong risk factors. Baseline OCT/visual "
                      "field and re-evaluate in 6-12 months.")
    elif stage_idx == 2:
        step = "First-line topical therapy (prostaglandin analogue) or SLT"
        detail = ("Mild glaucoma. Start a prostaglandin analogue or offer selective "
                  "laser trabeculoplasty (SLT) as primary therapy (LiGHT trial).")
    elif stage_idx == 3:
        step = "Escalate therapy: add second agent and/or SLT"
        detail = ("Moderate glaucoma. If not at target on monotherapy, add a second "
                  "class (beta-blocker, alpha-agonist, or CAI) and/or perform SLT.")
    else:
        step = "Consider incisional surgery (trabeculectomy / tube / MIGS)"
        detail = ("Severe glaucoma. If progressing or above target on maximal medical "
                  "therapy, refer for surgical IOP lowering to protect remaining field.")

    if not at_target and stage_idx >= 2:
        detail += " Current IOP is above the target below -- escalation is warranted."
    return {"next_step": step, "detail": detail}


def recommend(
    stage_idx: int,
    baseline_iop: float,
    risk_factors: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    """
    Produce target IOP and a ladder suggestion for a patient at a given stage.

    Args:
        stage_idx    : severity index 0..4
        baseline_iop : current mean IOP (mmHg)
        risk_factors : optional dict, e.g. {"family_history": True, "diabetes": False}
    """
    stage_idx = int(max(0, min(stage_idx, 4)))
    pct, ceiling = _STAGE_TARGETS[stage_idx]

    if pct <= 0.0 and ceiling is None:
        target_iop = None
        target_reduction = 0.0
        target_str = "No IOP target (no glaucomatous damage)."
    else:
        by_pct = baseline_iop * (1.0 - pct)
        target_iop = round(min(by_pct, ceiling) if ceiling else by_pct, 1)
        target_reduction = round(pct, 2)
        target_str = (
            f"Target IOP <= {target_iop} mmHg "
            f"(>= {int(pct*100)}% reduction from {round(baseline_iop,1)} mmHg"
            + (f", capped at {ceiling} mmHg" if ceiling else "") + ")."
        )

    active_rf = [k.replace("_", " ") for k, v in (risk_factors or {}).items() if v]
    at_target = (target_iop is None) or (baseline_iop <= target_iop)
    ladder = _treatment_ladder(stage_idx, at_target, active_rf)

    return {
        "stage": STAGE_NAMES[stage_idx],
        "baseline_iop": round(baseline_iop, 1),
        "target_iop": target_iop,
        "target_reduction": target_reduction,
        "target_rationale": target_str,
        "at_target": at_target,
        "next_step": ladder["next_step"],
        "rationale": ladder["detail"],
        "risk_factors": active_rf,
        "disclaimer": (
            "Guideline-based decision support (EMGT/CIGTS/AGIS, AAO POAG PPP). "
            "Not a prescription -- the treating ophthalmologist decides the plan."
        ),
    }
