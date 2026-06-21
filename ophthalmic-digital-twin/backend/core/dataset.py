"""
Data loading, tokenization, and pseudo-sequence construction for ophthalmic disease modeling.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORT_PATH = Path(__file__).resolve().parents[2] / "DATA_REPORT.md"


# ─────────────────────────────────────────────────────────
# EDA helpers
# ─────────────────────────────────────────────────────────

def _run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse the raw dataframe and return column classification."""
    report_lines: List[str] = ["# Data Report\n"]

    report_lines.append(f"## Shape\n{df.shape[0]} rows × {df.shape[1]} columns\n")

    null_pct = (df.isnull().mean() * 100).round(2)
    report_lines.append("## Null Percentage per Column\n```")
    report_lines.append(null_pct.to_string())
    report_lines.append("```\n")

    report_lines.append("## dtypes\n```")
    report_lines.append(df.dtypes.to_string())
    report_lines.append("```\n")

    numerical_cols: List[str] = []
    categorical_cols: List[str] = []
    binary_cols: List[str] = []
    high_cardinality_cols: List[str] = []
    id_cols: List[str] = []

    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 2:
            binary_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if n_unique > 50:
                numerical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            if n_unique > 50:
                high_cardinality_cols.append(col)
            else:
                categorical_cols.append(col)

    # Detect potential id columns (100% unique or name contains 'id'/'ID')
    for col in list(numerical_cols) + list(categorical_cols) + list(high_cardinality_cols):
        if "id" in col.lower() and df[col].nunique() == len(df):
            id_cols.append(col)

    report_lines.append(f"## Column Classification\n"
                        f"- Numerical: {numerical_cols}\n"
                        f"- Categorical: {categorical_cols}\n"
                        f"- Binary: {binary_cols}\n"
                        f"- High Cardinality (>50 unique): {high_cardinality_cols}\n"
                        f"- ID columns: {id_cols}\n")

    # Target detection
    target_keywords = ["disease", "severity", "grade", "label", "target", "class",
                       "diagnosis", "stage", "outcome", "score", "category"]
    target_col: Optional[str] = None
    for kw in target_keywords:
        matches = [c for c in df.columns if kw in c.lower()]
        if matches:
            target_col = matches[0]
            break
    if target_col is None:
        target_col = df.columns[-1]

    report_lines.append(f"## Target Column\n`{target_col}`\n")

    vc = df[target_col].value_counts()
    if len(vc) <= 20:
        imbalance = (vc.max() / vc.min()) if vc.min() > 0 else float("inf")
        report_lines.append(f"## Class Distribution\n```\n{vc.to_string()}\n```\n"
                            f"Imbalance ratio (max/min): {imbalance:.2f}\n")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("DATA_REPORT.md written to %s", REPORT_PATH)

    print("\n".join(report_lines))

    return {
        "target_col": target_col,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "binary_cols": binary_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "id_cols": id_cols,
    }


# ─────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────

class FeaturePreprocessor:
    """
    Fits StandardScaler on numericals, LabelEncoder on categoricals.
    Tracks missingness and produces aligned numpy arrays.
    """

    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        binary_cols: List[str],
        high_cardinality_cols: List[str],
    ) -> None:
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols + high_cardinality_cols
        self.binary_cols = binary_cols
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.cat_vocab_sizes: Dict[str, int] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        for col in self.numerical_cols:
            scaler = StandardScaler()
            valid = df[col].dropna().values.reshape(-1, 1)
            scaler.fit(valid)
            self.scalers[col] = scaler

        for col in self.categorical_cols:
            enc = LabelEncoder()
            valid = df[col].dropna().astype(str).values
            enc.fit(valid)
            self.encoders[col] = enc
            self.cat_vocab_sizes[col] = len(enc.classes_)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns:
            feature_matrix: [N, total_features] float32 — categorical values are integer indices
            missingness_matrix: [N, total_features] binary float32
            meta: feature ordering info
        """
        N = len(df)
        all_cols_ordered: List[str] = (
            self.numerical_cols + self.categorical_cols + self.binary_cols
        )
        feature_matrix = np.zeros((N, len(all_cols_ordered)), dtype=np.float32)
        miss_matrix = np.zeros((N, len(all_cols_ordered)), dtype=np.float32)

        col_types: List[str] = []

        idx = 0
        for col in self.numerical_cols:
            mask = df[col].isnull().values
            miss_matrix[:, idx] = mask.astype(np.float32)
            vals = df[col].values.copy()
            non_missing = ~mask
            if non_missing.any():
                scaled = self.scalers[col].transform(
                    vals[non_missing].reshape(-1, 1)
                ).flatten()
                # clip outliers at 3σ
                scaled = np.clip(scaled, -3.0, 3.0)
                vals[non_missing] = scaled
            vals[mask] = 0.0  # will be replaced by learned mask token in model
            feature_matrix[:, idx] = vals.astype(np.float32)
            col_types.append("numerical")
            idx += 1

        for col in self.categorical_cols:
            mask = df[col].isnull().values
            miss_matrix[:, idx] = mask.astype(np.float32)
            vocab_size = self.cat_vocab_sizes[col]
            encoded = np.full(N, vocab_size, dtype=np.float32)  # vocab_size = MASK idx
            non_missing = ~mask
            if non_missing.any():
                str_vals = df[col][non_missing].astype(str).values
                # Handle unseen labels gracefully
                known_classes = set(self.encoders[col].classes_)
                safe_vals = np.where(
                    np.isin(str_vals, list(known_classes)),
                    str_vals,
                    self.encoders[col].classes_[0],
                )
                encoded[non_missing] = self.encoders[col].transform(safe_vals).astype(np.float32)
            feature_matrix[:, idx] = encoded
            col_types.append("categorical")
            idx += 1

        for col in self.binary_cols:
            mask = df[col].isnull().values
            miss_matrix[:, idx] = mask.astype(np.float32)
            vals = pd.to_numeric(df[col], errors="coerce").fillna(2.0).values.astype(np.float32)
            vals[mask] = 2.0  # MASK index
            feature_matrix[:, idx] = vals
            col_types.append("binary")
            idx += 1

        meta = {
            "col_names": all_cols_ordered,
            "col_types": col_types,
            "cat_vocab_sizes": self.cat_vocab_sizes,
        }
        return feature_matrix, miss_matrix, meta


# ─────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────

class OphthalmicDataset(Dataset):
    """Wraps pre-processed arrays into a PyTorch Dataset."""

    def __init__(
        self,
        X: np.ndarray,
        M: np.ndarray,
        y: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.X = torch.from_numpy(X)
        self.M = torch.from_numpy(M)
        self.y = torch.from_numpy(y)
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"X": self.X[idx], "M": self.M[idx], "y": self.y[idx]}


# ─────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────

def load_data(
    csv_path: Optional[str] = None,
    batch_size: int = 64,
    patient_id_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    target_col_override: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Full data pipeline.

    Returns: train_loader, val_loader, test_loader, feature_metadata
    """
    if csv_path is None:
        csv_path = str(DATA_DIR / "full_df.csv")

    log.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    log.info("Raw shape: %s", df.shape)

    col_info = _run_eda(df)
    target_col = target_col_override or col_info["target_col"]

    # Auto-detect patient id column
    if patient_id_col is None:
        id_candidates = [c for c in df.columns if "id" in c.lower() or "patient" in c.lower()]
        patient_id_col = id_candidates[0] if id_candidates else None

    # Detect timestamp column
    ts_col = timestamp_col
    if ts_col is None:
        ts_candidates = [c for c in df.columns
                         if any(kw in c.lower() for kw in ["date", "time", "visit", "timestamp"])]
        ts_col = ts_candidates[0] if ts_candidates else None

    # Strip non-feature columns from matrix construction
    skip_cols = {target_col}
    if patient_id_col and patient_id_col in df.columns:
        skip_cols.add(patient_id_col)
    if ts_col and ts_col in df.columns:
        skip_cols.add(ts_col)
    skip_cols.update(col_info["id_cols"])

    numerical_cols = [c for c in col_info["numerical_cols"] if c not in skip_cols]
    categorical_cols = [c for c in col_info["categorical_cols"] if c not in skip_cols]
    binary_cols = [c for c in col_info["binary_cols"] if c not in skip_cols]
    high_cardinality_cols = [c for c in col_info["high_cardinality_cols"] if c not in skip_cols]

    # Build preprocessor
    prep = FeaturePreprocessor(numerical_cols, categorical_cols, binary_cols, high_cardinality_cols)
    prep.fit(df)
    feature_matrix, miss_matrix, col_meta = prep.transform(df)

    # Target encoding
    y_raw = df[target_col].values
    task = "regression"
    n_classes = 1
    if pd.api.types.is_object_dtype(df[target_col]) or df[target_col].nunique() <= 20:
        task = "classification"
        tgt_enc = LabelEncoder()
        y_encoded = tgt_enc.fit_transform(y_raw.astype(str)).astype(np.int64)
        n_classes = len(tgt_enc.classes_)
    else:
        y_encoded = y_raw.astype(np.float32)

    # Patient ids
    if patient_id_col and patient_id_col in df.columns:
        pids = df[patient_id_col].values
    else:
        pids = np.arange(len(df))

    # Cross-sectional design: each row is one patient encounter. We do NOT
    # fabricate a temporal axis (the previous KMeans "pseudo-time" had no clinical
    # meaning — glaucoma progression is longitudinal and cannot be recovered from a
    # single cross-sectional row). The model is a per-row tabular classifier.
    has_sequences = False
    seq_len = 1
    X_final, M_final, y_final, pids_final = feature_matrix, miss_matrix, y_encoded, pids

    # y dtype
    if task == "classification":
        y_final = y_final.astype(np.int64)
    else:
        y_final = y_final.astype(np.float32)

    # Patient-level stratified split
    unique_pids = np.unique(pids_final)
    strat_labels = np.array([y_final[pids_final == p][0] for p in unique_pids])

    # collapse strat to finite buckets for stratification
    if task == "regression":
        strat_labels = np.digitize(strat_labels, bins=np.percentile(strat_labels, [33, 66]))

    def _safe_stratify(labels: np.ndarray, min_per_class: int = 2) -> Optional[np.ndarray]:
        """Return stratify array only if every class has >= min_per_class members."""
        if len(np.unique(labels)) <= 1:
            return None
        counts = np.bincount(labels.astype(int)) if labels.dtype.kind in "iu" else np.array(
            [np.sum(labels == v) for v in np.unique(labels)]
        )
        return labels if counts.min() >= min_per_class else None

    pid_train, pid_temp = train_test_split(
        unique_pids, test_size=0.30, random_state=42,
        stratify=_safe_stratify(strat_labels),
    )
    strat_temp = strat_labels[np.isin(unique_pids, pid_temp)]
    pid_val, pid_test = train_test_split(
        pid_temp, test_size=0.50, random_state=42,
        stratify=_safe_stratify(strat_temp),
    )

    def _subset(pid_set: np.ndarray):
        mask = np.isin(pids_final, pid_set)
        return X_final[mask], M_final[mask], y_final[mask], pids_final[mask]

    X_tr, M_tr, y_tr, pids_tr = _subset(pid_train)
    X_va, M_va, y_va, pids_va = _subset(pid_val)
    X_te, M_te, y_te, pids_te = _subset(pid_test)

    train_ds = OphthalmicDataset(X_tr, M_tr, y_tr, pids_tr)
    val_ds   = OphthalmicDataset(X_va, M_va, y_va, pids_va)
    test_ds  = OphthalmicDataset(X_te, M_te, y_te, pids_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    n_features = feature_matrix.shape[1]
    feature_metadata: Dict[str, Any] = {
        "numerical_cols": col_meta["col_names"][:len(numerical_cols)],
        "categorical_cols": col_meta["col_names"][len(numerical_cols):len(numerical_cols) + len(categorical_cols + high_cardinality_cols)],
        "binary_cols": col_meta["col_names"][len(numerical_cols) + len(categorical_cols + high_cardinality_cols):],
        "col_types": col_meta["col_types"],
        "col_names": col_meta["col_names"],
        "cat_vocab_sizes": col_meta["cat_vocab_sizes"],
        "n_features": n_features,
        "n_classes": n_classes,
        "task": task,
        "has_sequences": has_sequences,
        "seq_len": seq_len,
        "preprocessor": prep,
        "target_col": target_col,
        "class_labels": tgt_enc.classes_.tolist() if task == "classification" else [],
    }

    log.info(
        "Data split: train=%d val=%d test=%d | task=%s n_classes=%d n_features=%d seq_len=%d",
        len(train_ds), len(val_ds), len(test_ds), task, n_classes, n_features, seq_len,
    )

    # Persist metadata (minus non-serializable objects)
    meta_path = DATA_DIR / "feature_metadata.json"
    serializable_meta = {k: v for k, v in feature_metadata.items()
                         if k not in ("preprocessor",)}
    serializable_meta["trained_csv"] = str(csv_path)
    with open(meta_path, "w") as f:
        json.dump(serializable_meta, f, indent=2)

    # Persist the fitted preprocessor so the API uses the EXACT scalers/encoders
    # fit on the training CSV (not a re-fit on a different file).
    import pickle
    with open(DATA_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(prep, f)

    return train_loader, val_loader, test_loader, feature_metadata
