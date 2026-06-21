"""
Generate a synthetic glaucoma dataset whose labels follow real clinical staging.

KEY DIFFERENCE from a naive generator: we do NOT sample a severity class and then
draw features around it. Instead we sample a plausible patient population, then
*derive* `disease_severity` from the measurements using clinically grounded rules.
This is the direction real diagnosis works (stage from data), so a model trained on
it learns the genuine structure->function relationships rather than a circular prior.

Staging rules (combined structural + functional), per:
  - Hodapp-Parrish-Anderson visual-field severity on Mean Deviation (MD):
        mild  : MD > -6 dB
        moderate: -12 <= MD <= -6 dB
        severe: MD < -12 dB
  - Structural gates (vertical cup-disc ratio CDR, RNFL thinning) separate
        Normal vs Suspect vs definite Glaucoma.

Decision logic (worst-of structure/function determines the stage):
  glaucomatous structure  := CDR >= 0.7 OR RNFL_average < 80 OR PSD high
  suspect structure       := CDR >= 0.6 OR RNFL_average < 90
  Normal               -> no structural signs and MD > -3
  Suspect              -> suspect structure but field still ~normal (MD > -6)
  Mild/Moderate/Severe Glaucoma -> glaucomatous structure, graded by MD (HPA)

A small fraction of labels are perturbed to one neighbouring stage to mimic
inter-grader disagreement and borderline cases.

NOTE: synthetic data for development/education only -- NOT for clinical use.

Usage:
    python scripts/generate_synthetic_data.py --rows 4000 --seed 42
Train on it:
    cd backend && python run.py --mode train --csv ../data/synthetic_train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SEVERITY = ["Normal", "Suspect", "Mild Glaucoma", "Moderate Glaucoma", "Severe Glaucoma"]

EYE_COLORS = ["blue", "brown", "green", "hazel"]
ETHNICITIES = ["african", "asian", "caucasian", "hispanic", "other"]
TREATMENTS = ["observation", "drops", "laser", "surgery", "anti-vegf"]


def _clip(a, lo, hi):
    return np.clip(a, lo, hi)


def _sample_population(n: int, rng: np.random.Generator) -> dict:
    """
    Sample a realistic cross-section of an ophthalmology clinic population.

    A latent 'optic-nerve damage' factor d in [0,1] drives the correlated
    structure (CDR up, RNFL down) and function (MD down, PSD up). Most patients
    are healthy/mild; a minority have advanced damage. This yields a natural,
    measurement-driven distribution rather than a class-balanced one.
    """
    # Latent damage: right-skewed so most eyes are healthy.
    d = rng.beta(1.6, 3.0, n)  # mean ~0.35, long tail toward 1

    age = _clip(rng.normal(55 + 25 * d, 11, n), 20, 95).round(1)

    # Structure --------------------------------------------------------------
    # Vertical cup-disc ratio rises with damage.
    cdr = _clip(rng.normal(0.30 + 0.55 * d, 0.07, n), 0.05, 0.95).round(3)
    # RNFL average (microns) thins with damage (~115 healthy -> ~55 advanced).
    rnfl_avg = _clip(rng.normal(115 - 60 * d, 8, n), 40, 160).round(1)
    rnfl_sup = _clip(rnfl_avg + rng.normal(8, 11, n), 35, 175).round(1)
    rnfl_inf = _clip(rnfl_avg + rng.normal(2, 12, n), 30, 175).round(1)

    # Function ---------------------------------------------------------------
    # Visual-field Mean Deviation (dB): ~0 healthy, strongly negative advanced.
    md_od = _clip(rng.normal(-0.5 - 24 * d ** 1.6, 1.8, n), -32, 3).round(2)
    md_os = _clip(md_od + rng.normal(0, 1.8, n), -32, 3).round(2)
    # Pattern standard deviation peaks in moderate disease then plateaus.
    psd = _clip(rng.normal(1.6 + 9 * d, 1.0, n), 0.3, 16).round(2)
    # Visual acuity (decimal) declines late.
    va_od = _clip(rng.normal(1.02 - 0.55 * d ** 2, 0.08, n), 0.0, 1.2).round(2)
    va_os = _clip(va_od + rng.normal(0, 0.06, n), 0.0, 1.2).round(2)

    # Intraocular pressure: a risk factor, elevated on average with damage but
    # deliberately noisy (normal-tension glaucoma exists, treated eyes are low).
    iop_od = _clip(rng.normal(15.5 + 8 * d, 4.0, n), 6, 45).round(1)
    iop_os = _clip(iop_od + rng.normal(0, 2.2, n), 6, 45).round(1)

    # Systemic / demographic -------------------------------------------------
    bmi = _clip(rng.normal(27, 5, n), 15, 45).round(1)
    diabetes = (rng.random(n) < _clip(0.04 + (bmi - 25) * 0.02, 0.02, 0.5)).astype(int)
    hba1c = _clip(np.where(diabetes == 1, rng.normal(7.6, 1.3, n),
                           rng.normal(5.5, 0.5, n)), 4.0, 13.0).round(1)
    systolic = _clip(rng.normal(125 + (age - 50) * 0.3, 15, n), 85, 200).round(0)
    hypertension = (systolic > 140).astype(int)
    diastolic = _clip(systolic * 0.62 + rng.normal(0, 6, n), 50, 120).round(0)
    sex = rng.integers(0, 2, n)
    # Family history more likely with damage (genetic component of glaucoma).
    fam_hist = (rng.random(n) < _clip(0.10 + 0.45 * d, 0.05, 0.6)).astype(int)

    eye_color = rng.choice(EYE_COLORS, n, p=[0.22, 0.42, 0.18, 0.18])
    ethnicity = rng.choice(ETHNICITIES, n, p=[0.18, 0.20, 0.40, 0.17, 0.05])

    return dict(
        age=age, sex=sex, bmi=bmi, iop_od=iop_od, iop_os=iop_os, cup_disc_ratio=cdr,
        va_od=va_od, va_os=va_os, mean_deviation_od=md_od, mean_deviation_os=md_os,
        pattern_sd=psd, rnfl_superior=rnfl_sup, rnfl_inferior=rnfl_inf,
        rnfl_average=rnfl_avg, hba1c=hba1c, systolic_bp=systolic, diastolic_bp=diastolic,
        diabetes=diabetes, hypertension=hypertension, family_history=fam_hist,
        eye_color=eye_color, ethnicity=ethnicity,
    )


def _stage(p: dict) -> np.ndarray:
    """
    Derive disease_severity for every patient from clinical staging rules.

    Worse eye drives the field metric; structural and functional criteria are
    combined so that definite glaucoma requires structural damage AND the HPA
    field band sets mild/moderate/severe.
    """
    n = len(p["age"])
    cdr = p["cup_disc_ratio"]
    rnfl = p["rnfl_average"]
    psd = p["pattern_sd"]
    # Worse (more negative) of the two eyes' MD.
    md = np.minimum(p["mean_deviation_od"], p["mean_deviation_os"])

    glaucomatous_struct = (cdr >= 0.70) | (rnfl < 80) | (psd >= 5.0)
    suspect_struct = (cdr >= 0.60) | (rnfl < 90) | (psd >= 3.0)

    stage = np.zeros(n, dtype=int)  # default Normal
    for i in range(n):
        if glaucomatous_struct[i] and md[i] <= -3.0:
            # Definite glaucoma: grade by Hodapp-Parrish-Anderson on MD.
            if md[i] < -12.0:
                stage[i] = 4  # Severe
            elif md[i] <= -6.0:
                stage[i] = 3  # Moderate
            else:
                stage[i] = 2  # Mild
        elif suspect_struct[i] or (md[i] <= -2.0 and md[i] > -6.0):
            stage[i] = 1      # Suspect (structural risk, field not yet definite)
        else:
            stage[i] = 0      # Normal
    return stage


def _add_label_noise(stage: np.ndarray, rng: np.random.Generator, p_flip: float = 0.06) -> np.ndarray:
    """Move a small fraction of labels to an adjacent stage (grader disagreement)."""
    out = stage.copy()
    flip = rng.random(len(stage)) < p_flip
    direction = rng.choice([-1, 1], len(stage))
    out[flip] = np.clip(out[flip] + direction[flip], 0, 4)
    return out


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Realistic missingness on the same columns/rates as real clinic data."""
    rates = {
        "mean_deviation_od": 0.10, "mean_deviation_os": 0.10,
        "pattern_sd": 0.08, "rnfl_superior": 0.07, "rnfl_inferior": 0.07,
        "hba1c": 0.12,
    }
    n = len(df)
    for col, rate in rates.items():
        df.loc[rng.random(n) < rate, col] = np.nan
    return df


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pop = _sample_population(n, rng)
    stage = _add_label_noise(_stage(pop), rng)

    # Treatment reflects the (stage-appropriate) management the patient is on.
    ladder_p = {
        0: [0.88, 0.10, 0.02, 0.0, 0.0],
        1: [0.55, 0.40, 0.04, 0.01, 0.0],
        2: [0.15, 0.62, 0.18, 0.04, 0.01],
        3: [0.04, 0.42, 0.30, 0.20, 0.04],
        4: [0.01, 0.16, 0.26, 0.45, 0.12],
    }
    treatment = np.array([rng.choice(TREATMENTS, p=ladder_p[int(s)]) for s in stage])

    df = pd.DataFrame({
        "patient_id": [f"S{i:05d}" for i in range(n)],
        **pop,
        "treatment": treatment,
        "disease_severity": [SEVERITY[s] for s in stage],
    })
    # Reorder to match the original schema column order.
    order = ["patient_id", "age", "sex", "bmi", "iop_od", "iop_os", "cup_disc_ratio",
             "va_od", "va_os", "mean_deviation_od", "mean_deviation_os", "pattern_sd",
             "rnfl_superior", "rnfl_inferior", "rnfl_average", "hba1c", "systolic_bp",
             "diastolic_bp", "diabetes", "hypertension", "family_history", "treatment",
             "eye_color", "ethnicity", "disease_severity"]
    df = df[order]
    return _inject_missing(df, rng)


def main():
    ap = argparse.ArgumentParser(description="Generate clinically-staged synthetic glaucoma data.")
    ap.add_argument("-n", "--rows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out", type=str, default=str(DATA_DIR / "synthetic_train.csv"))
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {out}")
    print("\nClass distribution (derived from clinical rules):")
    print(df["disease_severity"].value_counts().reindex(SEVERITY).to_string())
    print("\nMonotonic signal check (mean by stage):")
    print(df.groupby("disease_severity")[["iop_od", "cup_disc_ratio", "rnfl_average",
          "mean_deviation_od", "pattern_sd"]].mean().reindex(SEVERITY).round(2).to_string())


if __name__ == "__main__":
    main()
