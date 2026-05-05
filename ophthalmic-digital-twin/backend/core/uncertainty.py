"""
Monte Carlo Dropout inference and calibration metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────
# Monte Carlo Dropout
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def mc_predict(
    model: nn.Module,
    x: torch.Tensor,
    missingness: Optional[torch.Tensor] = None,
    n_samples: int = 50,
    rejection_threshold: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    Monte Carlo Dropout inference.

    Activates dropout at inference by calling model.train() then manually
    enabling eval mode only for BatchNorm layers (so BN statistics are stable
    while Dropout remains stochastic).

    Args:
        model            : nn.Module with dropout layers
        x                : input tensor [B, ...] or [B, T, F]
        missingness      : optional binary mask [B, ...] same shape prefix as x
        n_samples        : number of stochastic forward passes (default 50)
        rejection_threshold : total_uncertainty threshold above which prediction
                              is flagged as unreliable

    Returns dict with keys:
        mean_logits       : [B, n_classes]
        probabilities     : [B, n_classes]  softmax(mean_logits)
        prediction        : [B]             argmax class
        epistemic_variance: [B, n_classes]
        aleatoric_variance: [B, 1]
        total_uncertainty : [B]
        confidence        : [B]
        reliable          : [B]  bool mask
    """
    # Activate dropout while freezing BN
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            m.eval()

    device = next(model.parameters()).device
    x = x.to(device)
    if missingness is not None:
        missingness = missingness.to(device)

    sample_logits: List[torch.Tensor] = []
    sample_variances: List[torch.Tensor] = []

    for _ in range(n_samples):
        with torch.no_grad():
            if missingness is not None:
                out = model(x, missingness)
            else:
                out = model(x)
            sample_logits.append(out["logits"])        # [B, n_classes]
            sample_variances.append(out["pred_variance"])  # [B, 1]

    # [n_samples, B, n_classes]
    stacked = torch.stack(sample_logits, dim=0)
    # [n_samples, B, 1]
    var_stacked = torch.stack(sample_variances, dim=0)

    mu = stacked.mean(dim=0)                             # [B, n_classes]
    epistemic_var = stacked.var(dim=0)                   # [B, n_classes]
    aleatoric_var = var_stacked.mean(dim=0)              # [B, 1]
    total_uncertainty = (
        epistemic_var.mean(dim=-1) + aleatoric_var.squeeze(-1)
    ).clamp(0.0)                                         # [B]

    confidence = (1.0 - total_uncertainty.clamp(0.0, 1.0))
    reliable_mask = total_uncertainty < rejection_threshold

    model.eval()

    return {
        "mean_logits": mu,
        "probabilities": F.softmax(mu, dim=-1),
        "prediction": mu.argmax(dim=-1),
        "epistemic_variance": epistemic_var,
        "aleatoric_variance": aleatoric_var,
        "total_uncertainty": total_uncertainty,
        "confidence": confidence,
        "reliable": reliable_mask,
    }


# ─────────────────────────────────────────────────────────
# Calibration metrics
# ─────────────────────────────────────────────────────────

def calibration_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    n_bins: int = 15,
    n_mc_samples: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Compute Expected Calibration Error (ECE) with `n_bins` bins.
    Also returns reliability diagram data.

    Args:
        model      : trained model
        val_loader : validation DataLoader
        n_bins     : number of calibration bins
        n_mc_samples: MC dropout samples per batch
        device     : torch device

    Returns:
        {
          "ece": float,
          "reliability_diagram": [(conf_bin_center, acc_bin), ...]
        }
    """
    if device is None:
        device = next(model.parameters()).device

    all_confidences: List[float] = []
    all_correct: List[float] = []

    model.eval()
    for batch in val_loader:
        x = batch["X"].to(device)
        m = batch["M"].to(device)
        y = batch["y"].to(device)

        mc_out = mc_predict(model, x, m, n_samples=n_mc_samples)
        probs = mc_out["probabilities"]          # [B, C]
        preds = mc_out["prediction"]             # [B]
        conf, _ = probs.max(dim=-1)             # [B]

        correct = (preds == y.long()).float()

        all_confidences.extend(conf.cpu().tolist())
        all_correct.extend(correct.cpu().tolist())

    confidences = np.array(all_confidences)
    corrects = np.array(all_correct)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability_diagram: List[Tuple[float, float]] = []
    n_total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = corrects[mask].mean()
        ece += (n_bin / n_total) * abs(avg_conf - avg_acc)
        reliability_diagram.append((float(avg_conf), float(avg_acc)))

    return {
        "ece": float(ece),
        "reliability_diagram": reliability_diagram,
    }


# ─────────────────────────────────────────────────────────
# Regression uncertainty helpers
# ─────────────────────────────────────────────────────────

def mc_predict_regression(
    model: nn.Module,
    x: torch.Tensor,
    missingness: Optional[torch.Tensor] = None,
    n_samples: int = 50,
    rejection_threshold: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    MC Dropout for regression tasks.
    Returns mean prediction, std, and confidence estimate.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            m.eval()

    device = next(model.parameters()).device
    x = x.to(device)
    if missingness is not None:
        missingness = missingness.to(device)

    sample_preds: List[torch.Tensor] = []
    sample_variances: List[torch.Tensor] = []

    for _ in range(n_samples):
        with torch.no_grad():
            out = model(x, missingness) if missingness is not None else model(x)
            sample_preds.append(out["logits"].squeeze(-1))
            sample_variances.append(out["pred_variance"].squeeze(-1))

    stacked = torch.stack(sample_preds, dim=0)     # [S, B]
    var_stacked = torch.stack(sample_variances, dim=0)

    mu = stacked.mean(dim=0)
    epistemic_var = stacked.var(dim=0)
    aleatoric_var = var_stacked.mean(dim=0)
    total_uncertainty = (epistemic_var + aleatoric_var).clamp(0.0)
    confidence = (1.0 - (total_uncertainty / (total_uncertainty.max() + 1e-8)).clamp(0.0, 1.0))
    reliable_mask = total_uncertainty < rejection_threshold

    model.eval()

    return {
        "mean_prediction": mu,
        "epistemic_variance": epistemic_var,
        "aleatoric_variance": aleatoric_var,
        "total_uncertainty": total_uncertainty,
        "confidence": confidence,
        "reliable": reliable_mask,
        # For API compatibility
        "mean_logits": mu.unsqueeze(-1),
        "probabilities": mu.unsqueeze(-1),
        "prediction": mu,
    }
