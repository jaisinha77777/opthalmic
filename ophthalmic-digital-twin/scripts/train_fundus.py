"""
Fine-tune the FundusEncoder on a real glaucoma dataset.

Reads the uniform manifest produced by scripts/fetch_fundus_dataset.py:
    data/fundus/labels.csv   columns: filepath, glaucoma[0/1], cdr[optional]

Trains two heads jointly:
    - referable glaucoma : binary cross-entropy
    - vertical CDR       : MSE, only on rows that have a CDR ground truth

Saves backend/models/fundus_model.pt (+ metrics) which the API loads at startup.
Runs on GPU if available, otherwise CPU with conservative defaults.

    python scripts/train_fundus.py --epochs 15 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.fundus import FundusEncoder, _build_transform  # noqa: E402

MANIFEST = ROOT / "data" / "fundus" / "labels.csv"
OUT_PATH = ROOT / "backend" / "models" / "fundus_model.pt"


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.tf = _build_transform()
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(ROOT / r["filepath"]).convert("RGB")
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = self.tf(img)
        y = torch.tensor(int(r["glaucoma"]), dtype=torch.long)
        cdr = r.get("cdr", "")
        has_cdr = isinstance(cdr, (int, float)) and not (isinstance(cdr, float) and np.isnan(cdr))
        cdr_val = torch.tensor(float(cdr) if has_cdr else 0.0, dtype=torch.float32)
        cdr_mask = torch.tensor(1.0 if has_cdr else 0.0, dtype=torch.float32)
        return x, y, cdr_val, cdr_mask


def _load_manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        sys.exit(f"Manifest not found: {MANIFEST}\nRun scripts/fetch_fundus_dataset.py first.")
    df = pd.read_csv(MANIFEST)
    df["glaucoma"] = pd.to_numeric(df["glaucoma"], errors="coerce")
    df = df.dropna(subset=["glaucoma"])
    df["cdr"] = pd.to_numeric(df.get("cdr"), errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cdr-weight", type=float, default=2.0, help="weight on the CDR MSE loss")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = _load_manifest()
    print(f"Loaded {len(df)} images | glaucoma+={int(df['glaucoma'].sum())} "
          f"| with CDR={int(df['cdr'].notna().sum())} | device={device}")

    strat = df["glaucoma"] if df["glaucoma"].nunique() > 1 else None
    tr_df, va_df = train_test_split(df, test_size=0.2, random_state=42, stratify=strat)

    tr = DataLoader(FundusDataset(tr_df, True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(FundusDataset(va_df, False), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FundusEncoder(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss(reduction="none")

    best_auc = -1.0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for x, y, cdr, cmask in tr:
            x, y, cdr, cmask = x.to(device), y.to(device), cdr.to(device), cmask.to(device)
            out = model(x)
            loss = ce(out["referral_logits"], y)
            if cmask.sum() > 0:
                cdr_loss = (mse(out["cdr"], cdr) * cmask).sum() / cmask.sum()
                loss = loss + args.cdr_weight * cdr_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validation
        model.eval()
        probs, ys, cdr_err = [], [], []
        with torch.no_grad():
            for x, y, cdr, cmask in va:
                x = x.to(device)
                out = model(x)
                probs.extend(torch.softmax(out["referral_logits"], -1)[:, 1].cpu().tolist())
                ys.extend(y.tolist())
                if cmask.sum() > 0:
                    m = cmask.bool()
                    cdr_err.extend((out["cdr"].cpu()[m] - cdr[m]).abs().tolist())
        try:
            auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        cdr_mae = float(np.mean(cdr_err)) if cdr_err else float("nan")
        print(f"epoch {epoch+1:2d}/{args.epochs} | val AUC={auc:.3f} | CDR MAE={cdr_mae:.3f}")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_auc": auc, "val_cdr_mae": cdr_mae, "epoch": epoch + 1,
            }, OUT_PATH)

    metrics = {"best_val_auc": best_auc, "n_images": len(df)}
    (OUT_PATH.parent / "fundus_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {OUT_PATH} (best val AUC={best_auc:.3f})")


if __name__ == "__main__":
    main()
