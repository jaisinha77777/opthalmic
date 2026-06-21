"""
Augment a binary glaucoma fundus manifest with realistic vertical cup-disc ratios.

Some public datasets (e.g. ACRIMA) ship only a binary glaucoma/normal label and no
cup-disc-ratio (CDR) ground truth, so the FundusEncoder's CDR regression head has
nothing to learn from. This script fills in a clinically realistic CDR for each image,
conditioned on its diagnosis:

    normal eyes    : vertical CDR ~ N(0.40, 0.10), clipped [0.10, 0.70]
    glaucoma eyes  : vertical CDR ~ N(0.72, 0.10), clipped [0.45, 0.95]

These ranges reflect real optic-nerve morphology (glaucomatous discs have larger
cups) with realistic overlap in the 0.5-0.6 borderline zone.

IMPORTANT — honesty caveat: these CDR values are *synthetic*, derived from the
diagnosis label, NOT measured from the images by an expert. The CDR head therefore
learns to estimate a quantity that is correlated with (and partly defined by) the
glaucoma label. Treat the model's CDR output as an indicative structural estimate
only. The binary referable-glaucoma output remains trained on real expert labels.

    python scripts/augment_fundus_cdr.py [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "fundus" / "labels.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing CDR values if already present")
    args = ap.parse_args()

    path = Path(args.manifest)
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}\nRun scripts/fetch_fundus_dataset.py first.")

    df = pd.read_csv(path)
    df["glaucoma"] = pd.to_numeric(df["glaucoma"], errors="coerce")
    if "cdr" not in df.columns:
        df["cdr"] = np.nan
    df["cdr"] = pd.to_numeric(df["cdr"], errors="coerce")

    rng = np.random.default_rng(args.seed)
    n = len(df)

    glaucoma = df["glaucoma"].values.astype(float)
    mean = np.where(glaucoma > 0.5, 0.72, 0.40)
    cdr = rng.normal(mean, 0.10, n)
    # Clip per class to clinically plausible ranges.
    lo = np.where(glaucoma > 0.5, 0.45, 0.10)
    hi = np.where(glaucoma > 0.5, 0.95, 0.70)
    cdr = np.clip(cdr, lo, hi).round(3)

    fill = df["cdr"].isna().values | args.force
    df.loc[fill, "cdr"] = cdr[fill]
    df.to_csv(path, index=False)

    print(f"Wrote synthetic CDR for {int(fill.sum())}/{n} images -> {path}")
    print(df.groupby("glaucoma")["cdr"].describe()[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
