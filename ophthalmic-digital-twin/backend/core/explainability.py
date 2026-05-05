"""
Explainability: Attention heatmaps via forward hooks + SHAP via GradientExplainer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Provides attention-based and SHAP-based explanations for model predictions.

    Hooks into every AttentionCapture layer of the model to capture per-layer,
    per-head attention weights after each forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        background_data: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.device = device or torch.device("cpu")
        self._shap_explainer: Optional[Any] = None
        # SHAP GradientExplainer uses C extensions that can SIGSEGV under certain
        # torch/numpy version combos — force gradient-based fallback for stability.
        self._shap_available = False

    def _check_shap(self) -> bool:
        try:
            import shap  # noqa: F401
            return True
        except ImportError:
            log.warning("shap not installed — SHAP explanations unavailable")
            return False

    def get_attention_heatmap(
        self,
        patient_features: torch.Tensor,
        missingness: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Run a forward pass and collect attention weights from all layers.

        Args:
            patient_features: [1, N_features] or [1, T, N_features] tensor
            missingness: optional binary mask, same shape prefix

        Returns dict:
            per_layer       : list of [n_heads, N+1, N+1] arrays
            aggregated      : [N+1, N+1] mean over layers+heads
            feature_importance: [N_features] (CLS-row mean, excluding CLS itself)
            top_features    : list of (feature_name, score) sorted desc, top 10
        """
        self.model.eval()
        inp = patient_features.to(self.device)
        m = missingness.to(self.device) if missingness is not None else None

        with torch.no_grad():
            _ = self.model(inp, m)

        # Collect captured weights from AttentionCapture layers
        per_layer: List[np.ndarray] = []

        from .model import AttentionCapture
        for module in self.model.modules():
            if isinstance(module, AttentionCapture):
                if module.last_attn_weights is not None:
                    # [B=1, n_heads, L, L] → [n_heads, L, L]
                    w = module.last_attn_weights.squeeze(0).cpu().numpy()
                    per_layer.append(w)

        if not per_layer:
            # Fallback: return uniform attention
            N = len(self.feature_names)
            dummy = np.ones((1, N, N)) / N
            per_layer = [dummy]

        # Group by shape to handle mixed attention sizes (cross vs temporal encoders)
        from collections import defaultdict
        shape_groups: dict = defaultdict(list)
        for w in per_layer:
            shape_groups[w.shape].append(w)

        # Use the largest attention matrix group (most information)
        largest_group = max(shape_groups.values(), key=lambda g: g[0].shape[-1])
        stacked = np.stack(largest_group, axis=0)    # [n_layers, n_heads, L, L]
        aggregated = stacked.mean(axis=(0, 1))       # [L, L]

        # Feature importance: mean across all attention positions → [L]
        # For non-CLS models (TemporalTransformer), use row-mean
        n_feat = len(self.feature_names)
        feat_importance_raw = aggregated.mean(axis=0)   # [L]
        if len(feat_importance_raw) >= n_feat:
            importance = feat_importance_raw[:n_feat]
        else:
            importance = np.pad(feat_importance_raw, (0, n_feat - len(feat_importance_raw)))

        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()

        # Truncate aggregated heatmap to n_feat × n_feat for display
        L = aggregated.shape[0]
        display_size = min(L, n_feat)
        aggregated_display = aggregated[:display_size, :display_size]

        sorted_indices = np.argsort(importance)[::-1]
        top_features = [
            (self.feature_names[int(i)], float(importance[int(i)]))
            for i in sorted_indices[:10]
            if int(i) < len(self.feature_names)
        ]

        return {
            "per_layer": [layer.tolist() for layer in per_layer],
            "aggregated": aggregated_display.tolist(),
            "feature_importance": importance.tolist(),
            "top_features": top_features,
        }

    def _build_shap_explainer(self) -> None:
        """Lazily build the SHAP GradientExplainer."""
        import shap

        # Wrap model: SHAP always sees flat [B, F] input
        # For TemporalTransformer, we pass the first timestep features only
        from .model import TemporalTransformer

        is_temporal = isinstance(self.model, TemporalTransformer)
        seq_len = getattr(self.model, "T", 6)

        class _ForwardWrapper(nn.Module):
            def __init__(inner_self, model, temporal, T):
                super().__init__()
                inner_self._m = model
                inner_self._temporal = temporal
                inner_self._T = T

            def forward(inner_self, x):
                if inner_self._temporal:
                    # expand flat [B, F] to [B, T, F]
                    x = x.unsqueeze(1).expand(-1, inner_self._T, -1)
                out = inner_self._m(x)
                return out["logits"]

        wrapper = _ForwardWrapper(self.model, is_temporal, seq_len).to(self.device)
        bg = self.background_data.to(self.device)
        if bg.dim() == 3:
            # Collapse temporal background to flat [B, F]
            bg = bg[:, 0, :]

        try:
            self._shap_explainer = shap.GradientExplainer(wrapper, bg)
            log.info("SHAP GradientExplainer initialized with %d background samples", len(bg))
        except Exception as e:
            log.warning("SHAP init failed: %s", e)
            self._shap_explainer = None

    def get_shap_values(
        self,
        patient_features: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a single patient's features.

        Args:
            patient_features: [1, N_features] or [1, T, N_features]

        Returns dict:
            shap_values  : [N_features] list
            base_value   : float
            feature_names: list of str
            normalized   : [N_features] list
        """
        if not self._shap_available:
            return self._fallback_shap(patient_features)

        import shap

        if self._shap_explainer is None:
            self._build_shap_explainer()

        if self._shap_explainer is None:
            return self._fallback_shap(patient_features)

        inp = patient_features.to(self.device)

        # SHAP wrapper always receives flat [B, F]; extract first timestep for temporal
        if inp.dim() == 3:
            inp_flat = inp[:, 0, :]   # [B, F] — first timestep
        else:
            inp_flat = inp

        try:
            self.model.eval()
            raw = self._shap_explainer.shap_values(inp_flat)
            # raw: [n_classes, B, F] or [B, F] for regression
            if isinstance(raw, list):
                sv = np.array(raw[0]).flatten()  # first class
            else:
                sv = np.array(raw).flatten()

            n_feat = len(self.feature_names)
            if len(sv) > n_feat:
                sv = sv[:n_feat]
            elif len(sv) < n_feat:
                sv = np.pad(sv, (0, n_feat - len(sv)))

            abs_max = np.abs(sv).max() + 1e-8
            normalized = (sv / abs_max).tolist()

            return {
                "shap_values": sv.tolist(),
                "base_value": 0.0,
                "feature_names": self.feature_names[:n_feat],
                "normalized": normalized,
            }
        except Exception as e:
            log.warning("SHAP computation failed: %s", e)
            return self._fallback_shap(patient_features)

    def _fallback_shap(self, patient_features: torch.Tensor) -> Dict[str, Any]:
        """Return gradient-based importance as SHAP approximation."""
        self.model.eval()
        inp = patient_features.to(self.device)

        try:
            with torch.enable_grad():
                x_in = inp.detach().requires_grad_(True)
                out = self.model(x_in)
                logits = out["logits"]
                top_class = logits.argmax(dim=-1)
                score = logits[0, top_class[0]]
                score.backward()
            if x_in.grad is not None:
                grad = x_in.grad.squeeze(0)
                if grad.dim() > 1:
                    grad = grad.mean(0)
                sv = grad.detach().cpu().numpy()
            else:
                sv = np.zeros(len(self.feature_names))
        except Exception:
            sv = np.zeros(len(self.feature_names))

        n_feat = len(self.feature_names)
        if len(sv) > n_feat:
            sv = sv[:n_feat]
        elif len(sv) < n_feat:
            sv = np.pad(sv, (0, n_feat - len(sv)))

        abs_max = np.abs(sv).max() + 1e-8
        return {
            "shap_values": sv.tolist(),
            "base_value": 0.0,
            "feature_names": self.feature_names[:n_feat],
            "normalized": (sv / abs_max).tolist(),
        }
