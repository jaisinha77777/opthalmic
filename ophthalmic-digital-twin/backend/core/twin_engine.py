"""
DigitalTwinEngine: stateful per-patient digital twin with latent state evolution.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .uncertainty import mc_predict

log = logging.getLogger(__name__)

# Module-level registry: {patient_id: DigitalTwinEngine}
_REGISTRY: Dict[str, "DigitalTwinEngine"] = {}

# Shared PCA fitted on accumulated latent states
_pca_model: Optional[PCA] = None
_pca_basis: List[np.ndarray] = []  # accumulate latent states to fit PCA


def _fit_or_update_pca(new_state: np.ndarray) -> None:
    """Incrementally collect states and (re)fit PCA when enough samples exist."""
    global _pca_model, _pca_basis
    _pca_basis.append(new_state.flatten())
    if len(_pca_basis) >= 4:
        data = np.stack(_pca_basis, axis=0)
        pca = PCA(n_components=min(3, data.shape[1]))
        pca.fit(data)
        _pca_model = pca


def _project_to_3d(state: np.ndarray) -> List[float]:
    """Project a 256-dim latent state to 3D using PCA or random projection fallback."""
    global _pca_model
    flat = state.flatten()
    if _pca_model is not None:
        try:
            proj = _pca_model.transform(flat.reshape(1, -1))[0]
            # Pad to 3D if PCA returned fewer components
            while len(proj) < 3:
                proj = np.append(proj, 0.0)
            return proj[:3].tolist()
        except Exception:
            pass
    # Fallback: first 3 dimensions scaled
    return (flat[:3] / (np.linalg.norm(flat[:3]) + 1e-8)).tolist()


# ─────────────────────────────────────────────────────────
# State transition MLP
# ─────────────────────────────────────────────────────────

class StateTransitionMLP(nn.Module):
    """Linear(320,256) → GELU → LayerNorm → Linear(256,256)"""

    def __init__(self, d_model: int = 256, action_dim: int = 64) -> None:
        super().__init__()
        in_dim = d_model + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx)


# ─────────────────────────────────────────────────────────
# Action embedder
# ─────────────────────────────────────────────────────────

class ActionEmbedder(nn.Module):
    """Maps discrete treatment action index to a continuous embedding."""

    def __init__(self, n_actions: int, action_dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_actions, action_dim)
        self.action_dim = action_dim

    def forward(self, action_idx: torch.Tensor) -> torch.Tensor:
        return self.emb(action_idx.long())


# ─────────────────────────────────────────────────────────
# Projection head (S_t → logits for MC inference on state)
# ─────────────────────────────────────────────────────────

class StateProjectionHead(nn.Module):
    """Wraps a fixed prediction head to run mc_predict on a latent state."""

    def __init__(self, pred_head: nn.Module, unc_head: nn.Module, n_classes: int) -> None:
        super().__init__()
        self.pred_head = pred_head
        self.unc_head = unc_head
        self.n_classes = n_classes

    def forward(self, latent: torch.Tensor, **kwargs) -> Dict[str, Any]:
        logits = self.pred_head(latent)
        pred_variance = self.unc_head(latent)
        return {
            "logits": logits,
            "latent_state": latent,
            "pred_variance": pred_variance,
            "attention_weights": [],
        }


# ─────────────────────────────────────────────────────────
# Digital Twin Engine
# ─────────────────────────────────────────────────────────

class DigitalTwinEngine:
    """
    Stateful per-patient digital twin.

    Maintains a latent state S_t ∈ R^256 that evolves as treatment actions
    are applied, enabling counterfactual simulation and horizon planning.
    """

    def __init__(
        self,
        patient_id: str,
        initial_features: torch.Tensor,
        model: nn.Module,
        action_embedder: ActionEmbedder,
        state_mlp: StateTransitionMLP,
        device: torch.device,
    ) -> None:
        self.patient_id = patient_id
        self.model = model
        self.action_embedder = action_embedder
        self.state_mlp = state_mlp
        self.device = device

        self.S_t = self._encode_initial(initial_features)
        self.state_trajectory: List[torch.Tensor] = [self.S_t.clone()]
        self.feature_trajectory: List[torch.Tensor] = [initial_features.clone()]
        self.action_history: List[Dict[str, Any]] = []
        self.uncertainty_log: List[Dict[str, float]] = []
        self.timestep: int = 0

        _fit_or_update_pca(self.S_t.detach().cpu().numpy())
        _REGISTRY[patient_id] = self

    def _encode_initial(self, features: torch.Tensor) -> torch.Tensor:
        """Run model encoder to get initial latent state."""
        self.model.eval()
        with torch.no_grad():
            inp = features.unsqueeze(0).to(self.device)
            out = self.model(inp)
            return out["latent_state"].squeeze(0).detach()

    def step(
        self,
        treatment_action_idx: int,
        compliance_level: float = 1.0,
        n_mc_samples: int = 30,
    ) -> Dict[str, Any]:
        """
        Advance the digital twin by one timestep with a given treatment.

        Args:
            treatment_action_idx: discrete action index
            compliance_level: float [0,1] scaling the action effect
            n_mc_samples: MC Dropout samples for uncertainty estimation

        Returns dict with keys: new_state, prediction, uncertainty
        """
        # 1. Embed action and scale by compliance
        act_t = torch.tensor([treatment_action_idx], dtype=torch.long, device=self.device)
        a_emb = self.action_embedder(act_t).squeeze(0) * compliance_level  # [action_dim]

        # 2. Context: concat S_t + a_emb
        ctx = torch.cat([self.S_t, a_emb], dim=0).unsqueeze(0)  # [1, d_model+action_dim]

        # 3. State transition
        S_next_raw = self.state_mlp(ctx).squeeze(0)              # [d_model]

        # 4. Residual skip connection
        S_next = S_next_raw + self.S_t                           # [d_model]

        # 5. MC inference on new state via projection head
        with torch.no_grad():
            state_input = S_next.unsqueeze(0)

            # Wrap model for state-based prediction
            class _LatentWrapper(nn.Module):
                def __init__(inner_self, model):
                    super().__init__()
                    inner_self._m = model

                def forward(inner_self, x, *a, **kw):
                    logits = inner_self._m.pred_head(x)
                    pred_var = inner_self._m.unc_head(x)
                    return {
                        "logits": logits,
                        "latent_state": x,
                        "pred_variance": pred_var,
                        "attention_weights": [],
                    }

            wrapper = _LatentWrapper(self.model).to(self.device)
            mc_out = mc_predict(wrapper, state_input, n_samples=n_mc_samples)

        # 6. Update state
        self.S_t = S_next.detach()
        self.state_trajectory.append(self.S_t.clone())
        self.action_history.append({
            "timestep": self.timestep,
            "action_idx": treatment_action_idx,
            "compliance": compliance_level,
        })

        unc_entry = {
            "total_uncertainty": float(mc_out["total_uncertainty"].item()),
            "confidence": float(mc_out["confidence"].item()),
        }
        self.uncertainty_log.append(unc_entry)
        self.timestep += 1

        _fit_or_update_pca(self.S_t.detach().cpu().numpy())

        return {
            "new_state": self.S_t,
            "prediction": mc_out["prediction"].item(),
            "uncertainty": unc_entry,
            "probabilities": mc_out["probabilities"].squeeze(0).tolist(),
            "confidence": float(mc_out["confidence"].item()),
        }

    def simulate_horizon(
        self,
        horizon: int = 12,
        policy_fn: Optional[Callable[[torch.Tensor], int]] = None,
        n_mc: int = 30,
    ) -> Dict[str, Any]:
        """
        Simulate H future timesteps.

        Args:
            horizon  : number of future steps
            policy_fn: S_t → action_idx. If None, uses action 0.
            n_mc     : MC samples per step

        Returns:
            states, predictions, uncertainties, confidence_bands
        """
        # Save current state to restore after simulation
        saved_state = self.S_t.clone()
        saved_trajectory = [s.clone() for s in self.state_trajectory]
        saved_actions = deepcopy(self.action_history)
        saved_unc = deepcopy(self.uncertainty_log)
        saved_t = self.timestep

        sim_states: List[List[float]] = []
        sim_preds: List[Any] = []
        sim_uncs: List[float] = []
        conf_lower: List[float] = []
        conf_upper: List[float] = []

        for h in range(horizon):
            action_idx = policy_fn(self.S_t) if policy_fn else 0
            result = self.step(action_idx, compliance_level=0.8, n_mc_samples=n_mc)
            sim_states.append(_project_to_3d(result["new_state"].cpu().numpy()))
            sim_preds.append(result["prediction"])
            unc = result["uncertainty"]["total_uncertainty"]
            conf = result["uncertainty"]["confidence"]
            sim_uncs.append(unc)
            conf_lower.append(max(0.0, conf - unc))
            conf_upper.append(min(1.0, conf + unc))

        # Restore
        self.S_t = saved_state
        self.state_trajectory = saved_trajectory
        self.action_history = saved_actions
        self.uncertainty_log = saved_unc
        self.timestep = saved_t

        return {
            "states": sim_states,
            "predictions": sim_preds,
            "uncertainties": sim_uncs,
            "confidence_bands": {
                "lower": conf_lower,
                "upper": conf_upper,
            },
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """JSON-serializable twin state summary."""
        state_np = self.S_t.detach().cpu().numpy()
        state_3d = _project_to_3d(state_np)

        trajectory_3d = [
            _project_to_3d(s.detach().cpu().numpy())
            for s in self.state_trajectory
        ]

        return {
            "patient_id": self.patient_id,
            "timestep": self.timestep,
            "current_state_pca": state_3d,
            "state_trajectory_3d": trajectory_3d,
            "action_history": self.action_history,
            "uncertainty_log": self.uncertainty_log,
            "latent_norm": float(torch.norm(self.S_t).item()),
        }

    @classmethod
    def get_registry(cls) -> Dict[str, "DigitalTwinEngine"]:
        return _REGISTRY

    @classmethod
    def get_or_create(
        cls,
        patient_id: str,
        initial_features: torch.Tensor,
        model: nn.Module,
        action_embedder: ActionEmbedder,
        state_mlp: StateTransitionMLP,
        device: torch.device,
    ) -> "DigitalTwinEngine":
        if patient_id in _REGISTRY:
            return _REGISTRY[patient_id]
        return cls(patient_id, initial_features, model, action_embedder, state_mlp, device)


def get_global_pca() -> Optional[PCA]:
    return _pca_model
