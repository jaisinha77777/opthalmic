"""
Transparent glaucoma progression projection.

Replaces the previous opaque 256-dim "digital twin" latent-state evolution with a
simple, interpretable model of how the visual field (Mean Deviation, dB) is expected
to change over time, and how lowering intraocular pressure (IOP) slows that change.

It is deliberately a closed-form curve, not a black box, so every number can be
explained to a clinician. The constants are order-of-magnitude consistent with
landmark trials (EMGT, CIGTS, AGIS):
  - Untreated glaucoma worsens MD on the order of ~0.5-1.5 dB/year, faster at higher
    IOP and in more advanced disease (EMGT).
  - Each ~1 mmHg of IOP lowering reduces progression risk by roughly 10% (EMGT), so a
    treatment that lowers IOP by a fraction f slows the MD slope by a similar factor.

NOT for clinical use -- educational projection only.
"""

from __future__ import annotations

from typing import Dict, List

# HPA Mean-Deviation cut-points for staging the projected field.
_HPA_BANDS = [(-6.0, "Mild Glaucoma"), (-12.0, "Moderate Glaucoma")]


def _stage_from_md(md: float) -> str:
    if md > -6.0:
        return "Mild / no field loss"
    if md >= -12.0:
        return "Moderate Glaucoma"
    return "Severe Glaucoma"


def _annual_md_slope(iop: float, stage_idx: int) -> float:
    """
    Expected untreated MD loss in dB/year (negative = worsening).

    base 0.5 dB/yr, increased for IOP above a 15 mmHg reference (~0.07 dB/yr per mmHg)
    and for more advanced stage. Clamped to a clinically plausible 0.3-3.0 dB/yr.
    """
    base = 0.5
    iop_term = max(0.0, iop - 15.0) * 0.07
    stage_term = 0.20 * max(0, stage_idx)  # 0..4 -> up to +0.8 dB/yr
    slope = base + iop_term + stage_term
    return -min(max(slope, 0.3), 3.0)


def project(
    baseline_md: float,
    baseline_iop: float,
    stage_idx: int,
    horizon_months: int = 60,
    iop_reduction: float = 0.0,
    step_months: int = 6,
) -> Dict[str, object]:
    """
    Project the worse-eye Mean Deviation forward.

    Args:
        baseline_md   : current worse-eye MD (dB)
        baseline_iop  : current mean IOP (mmHg)
        stage_idx     : current severity index 0..4
        horizon_months: projection length
        iop_reduction : fraction of IOP removed by treatment (0 = untreated, 0.3 = -30%)
        step_months   : sampling interval

    Returns dict with parallel arrays: months, md (treated), md_untreated,
    md_lower/md_upper (80% band on the treated curve), and projected_stage.
    """
    untreated_slope = _annual_md_slope(baseline_iop, stage_idx)  # dB/yr
    # Treatment lowers IOP, which slows the slope. Residual slope is proportional to
    # the residual IOP burden above the 15 mmHg reference plus a floor (disease still
    # creeps even at low IOP).
    treated_iop = baseline_iop * (1.0 - iop_reduction)
    treated_slope = _annual_md_slope(treated_iop, stage_idx)
    # Never let "treatment" look better than a 70% slowing of intrinsic progression.
    treated_slope = max(treated_slope, untreated_slope * 0.3)

    months: List[int] = list(range(0, horizon_months + 1, step_months))
    md_treated, md_untreated, lower, upper, stages = [], [], [], [], []
    for m in months:
        yrs = m / 12.0
        mt = baseline_md + treated_slope * yrs
        mu = baseline_md + untreated_slope * yrs
        # Uncertainty grows with time (process noise ~0.6 dB/yr 1-sigma -> 80% ~1.28sigma).
        band = 1.28 * 0.6 * yrs
        md_treated.append(round(mt, 2))
        md_untreated.append(round(mu, 2))
        lower.append(round(mt - band, 2))
        upper.append(round(min(mt + band, 3.0), 2))
        stages.append(_stage_from_md(mt))

    return {
        "months": months,
        "md_treated": md_treated,
        "md_untreated": md_untreated,
        "md_lower": lower,
        "md_upper": upper,
        "projected_stage": stages,
        "untreated_slope_db_yr": round(untreated_slope, 3),
        "treated_slope_db_yr": round(treated_slope, 3),
        "assumptions": (
            "Closed-form EMGT/CIGTS-consistent projection. Untreated MD slope scales "
            "with IOP above 15 mmHg and disease stage; treatment slows it in proportion "
            "to IOP lowering. Educational only -- not a validated progression forecast."
        ),
    }
