"""
Training loop, evaluation, and calibration for the OphthalmicTransformer.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast as _amp_autocast
    def autocast(enabled=True):
        return _amp_autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# Custom losses
# ─────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing. Returns per-sample losses (shape [B])."""

    def __init__(self, smoothing: float = 0.1, n_classes: int = 2) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / max(self.n_classes - 1, 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        # Return per-sample loss [B] so caller can apply class weights before mean()
        return -(smooth_targets * log_probs).sum(dim=-1)


def heteroscedastic_nll(
    logits: torch.Tensor,
    pred_variance: torch.Tensor,
    targets: torch.Tensor,
    task: str = "classification",
) -> torch.Tensor:
    """
    Variance regularization term.

    For classification: Brier-score-style auxiliary loss on predicted variance.
      Penalises high variance when the model is already correct (overconfident
      uncertainty) and low variance when the model is wrong.
      L = mean(σ² * correct  +  (1 - σ²) * (1 - correct))
      where correct = 1 if argmax == target else 0.

    For regression: standard heteroscedastic NLL.
      L = mean(log σ² + (y − μ)² / σ²)
    """
    sigma2 = pred_variance.clamp(min=1e-6).squeeze(-1)   # [B]

    if task == "classification":
        preds = logits.argmax(dim=-1)                          # [B]
        correct = (preds == targets).float()                   # [B]  1=right 0=wrong
        # Encourage σ² → 0 when correct, σ² → 1 when wrong
        loss = (sigma2 * correct + (1.0 - sigma2) * (1.0 - correct)).mean()
        return loss
    else:
        mu = logits.squeeze(-1)
        y = targets.float()
        return (sigma2.log() + (y - mu) ** 2 / sigma2).mean()


# ─────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────

def _eval_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            m = batch["M"].to(device)
            y = batch["y"].to(device)
            out = model(x, m)
            probs = F.softmax(out["logits"], dim=-1)
            preds = probs.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_probs)

    acc = accuracy_score(targets_np, preds_np)
    f1 = f1_score(targets_np, preds_np, average="macro", zero_division=0)

    try:
        present_classes = np.unique(targets_np)
        if n_classes == 2:
            auc = roc_auc_score(targets_np, probs_np[:, 1])
        elif len(present_classes) >= 2:
            # Filter probs to only classes present in targets to avoid OvR crash
            auc = roc_auc_score(
                targets_np, probs_np[:, present_classes],
                multi_class="ovr", average="macro",
                labels=present_classes,
            )
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    # Brier score (binary; multi-class approximation via one-vs-rest)
    try:
        if n_classes == 2:
            brier = brier_score_loss(targets_np, probs_np[:, 1])
        else:
            # Average OvR Brier scores across present classes
            brier_vals = []
            for c in np.unique(targets_np):
                brier_vals.append(brier_score_loss((targets_np == c).astype(int), probs_np[:, c]))
            brier = float(np.mean(brier_vals))
    except Exception:
        brier = float("nan")

    # ECE with 15 bins
    confidences = probs_np.max(axis=1)
    correct = (preds_np == targets_np).astype(float)
    ece = 0.0
    bins = np.linspace(0, 1, 16)
    for i in range(15):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / len(targets_np)) * abs(bin_conf - bin_acc)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "roc_auc": float(auc),
        "brier_score": float(brier),
        "ece": float(ece),
    }


def _eval_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            m = batch["M"].to(device)
            y = batch["y"].to(device)
            out = model(x, m)
            preds = out["logits"].squeeze(-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)

    mae = mean_absolute_error(targets_np, preds_np)
    rmse = np.sqrt(mean_squared_error(targets_np, preds_np))
    try:
        r2 = r2_score(targets_np, preds_np)
    except Exception:
        r2 = float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# ─────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_metadata: Dict[str, Any],
    n_epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,   # increased from 1e-4 — tabular data overfits easily
    patience: int = 15,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    warmup_epochs: int = 5,        # linear LR warmup to stabilise early gradient steps
) -> Dict[str, Any]:
    """
    Full training pipeline.

    Args:
        model           : untrained nn.Module
        train_loader    : training DataLoader
        val_loader      : validation DataLoader
        feature_metadata: dataset metadata dict
        n_epochs        : max training epochs (default 100)
        lr              : AdamW learning rate
        weight_decay    : L2 regularization
        patience        : early stopping patience
        device          : torch device
        use_amp         : enable mixed-precision (only on CUDA)

    Returns history dict and saves checkpoints.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    task = feature_metadata["task"]
    n_classes = feature_metadata["n_classes"]
    use_amp = use_amp and device.type == "cuda"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # CosineAnnealingLR over full budget — avoids premature restarts that conflict
    # with early stopping (old T_0=10 reset mid-training and wasted epochs).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.05)
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else GradScaler("cpu", enabled=False)

    if task == "classification":
        # Class-frequency inverse weights to handle imbalanced severity distribution.
        all_labels = torch.cat([b["y"] for b in train_loader]).numpy()
        counts = np.bincount(all_labels.astype(int), minlength=n_classes).astype(np.float32)
        counts = np.maximum(counts, 1)                       # avoid div-by-zero
        raw_weights = 1.0 / counts
        class_weights = torch.tensor(
            raw_weights / raw_weights.sum() * n_classes,     # normalise so mean weight ≈ 1
            dtype=torch.float32, device=device,
        )
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1, n_classes=n_classes)
    else:
        class_weights = None
        criterion = nn.HuberLoss()

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "epoch_time": [], "lr": [],
    }
    if task == "classification":
        history.update({"val_accuracy": [], "val_f1": [], "val_auc": [], "val_ece": []})
    else:
        history.update({"val_mae": [], "val_rmse": [], "val_r2": []})

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    log.info("Training on %s | task=%s | n_classes=%d | epochs=%d",
             device, task, n_classes, n_epochs)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Linear LR warmup: scale lr from lr/warmup_epochs → lr over first warmup_epochs
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * warmup_factor

        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["X"].to(device)
            m = batch["M"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = model(x, m)
                logits = out["logits"]
                pred_var = out["pred_variance"]

                if task == "classification":
                    base_loss = criterion(logits, y.long())
                    # Apply per-sample class weights to amplify rare-class gradients
                    if class_weights is not None:
                        sample_w = class_weights[y.long()]        # [B]
                        base_loss = (base_loss * sample_w).mean() if base_loss.dim() > 0 else base_loss
                else:
                    base_loss = criterion(logits.squeeze(-1), y.float())

                var_reg = 0.05 * heteroscedastic_nll(logits, pred_var, y, task)
                loss = base_loss + var_reg

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            n_batches += 1

        if epoch > warmup_epochs:
            scheduler.step()
        avg_train_loss = total_train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["X"].to(device)
                m = batch["M"].to(device)
                y = batch["y"].to(device)
                with autocast(enabled=use_amp):
                    out = model(x, m)
                    logits = out["logits"]
                    pred_var = out["pred_variance"]
                    if task == "classification":
                        vloss = criterion(logits, y.long()).mean()
                    else:
                        vloss = criterion(logits.squeeze(-1), y.float())
                    var_reg = 0.1 * heteroscedastic_nll(logits, pred_var, y, task)
                    vloss = vloss + var_reg
                total_val_loss += vloss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / max(n_val_batches, 1)
        epoch_time = time.time() - t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epoch_time"].append(epoch_time)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Full eval metrics every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            if task == "classification":
                metrics = _eval_classification(model, val_loader, device, n_classes)
                history["val_accuracy"].append(metrics["accuracy"])
                history["val_f1"].append(metrics["f1_macro"])
                history["val_auc"].append(metrics["roc_auc"])
                history["val_ece"].append(metrics["ece"])
                log.info(
                    "Epoch %3d/%d | train=%.4f val=%.4f | acc=%.3f f1=%.3f auc=%.3f ece=%.3f | %.1fs",
                    epoch, n_epochs, avg_train_loss, avg_val_loss,
                    metrics["accuracy"], metrics["f1_macro"], metrics["roc_auc"], metrics["ece"],
                    epoch_time,
                )
            else:
                metrics = _eval_regression(model, val_loader, device)
                history["val_mae"].append(metrics["mae"])
                history["val_rmse"].append(metrics["rmse"])
                history["val_r2"].append(metrics["r2"])
                log.info(
                    "Epoch %3d/%d | train=%.4f val=%.4f | mae=%.4f rmse=%.4f r2=%.4f | %.1fs",
                    epoch, n_epochs, avg_train_loss, avg_val_loss,
                    metrics["mae"], metrics["rmse"], metrics["r2"], epoch_time,
                )
        else:
            log.info(
                "Epoch %3d/%d | train=%.4f val=%.4f | %.1fs",
                epoch, n_epochs, avg_train_loss, avg_val_loss, epoch_time,
            )

        # Checkpoint best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "feature_metadata": {k: v for k, v in feature_metadata.items()
                                     if k != "preprocessor"},
            }, MODELS_DIR / "best_model.pt")
            log.info("  ✓ Saved best model (epoch %d, val_loss=%.4f)", epoch, avg_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping at epoch %d (best was epoch %d)", epoch, best_epoch)
                break

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_loss,
        "feature_metadata": {k: v for k, v in feature_metadata.items()
                             if k != "preprocessor"},
    }, MODELS_DIR / "final_model.pt")

    # Save history
    hist_path = MODELS_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    log.info("Training complete. Best epoch: %d, Best val loss: %.4f", best_epoch, best_val_loss)
    return history


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    feature_metadata: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Run full evaluation on test set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    task = feature_metadata["task"]
    n_classes = feature_metadata["n_classes"]

    if task == "classification":
        metrics = _eval_classification(model, test_loader, device, n_classes)
    else:
        metrics = _eval_regression(model, test_loader, device)

    log.info("Test metrics: %s", metrics)
    return metrics


def load_best_model(model: nn.Module, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict]:
    """Load the best saved checkpoint into model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = MODELS_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run training first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    log.info("Loaded best model from epoch %d (val_loss=%.4f)", ckpt["epoch"], ckpt["val_loss"])
    return model, ckpt.get("feature_metadata", {})
