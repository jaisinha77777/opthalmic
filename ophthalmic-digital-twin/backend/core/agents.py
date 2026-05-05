"""
Multi-Agent Reinforcement Learning: DoctorAgent, DiseaseAgent, PatientAgent.
All use PPO with shared latent state S_t as observation.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Base PPO Agent
# ─────────────────────────────────────────────────────────

class PPOAgent(nn.Module):
    """
    Generic PPO agent for an agent with obs_dim=256 (latent state).

    Actor  : Linear(256,128) → GELU → Linear(128, action_dim) → softmax
    Critic : Linear(256,128) → GELU → Linear(128, 1)

    PPO hyperparameters:
        clip_ratio  = 0.2
        value_coeff = 0.5
        entropy_coeff = 0.01
        max_grad_norm = 0.5
    """

    CLIP_RATIO = 0.2
    VALUE_COEFF = 0.5
    ENTROPY_COEFF = 0.01
    MAX_GRAD_NORM = 0.5

    def __init__(self, obs_dim: int = 256, action_dim: int = 4) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Return softmax action probabilities. obs: [B, 256] or [256]"""
        logits = self.actor(obs)
        return F.softmax(logits, dim=-1)

    def get_best_action(self, obs: torch.Tensor) -> int:
        """Greedy action selection."""
        with torch.no_grad():
            probs = self.action_probs(obs.unsqueeze(0) if obs.dim() == 1 else obs)
            return probs.argmax(dim=-1).item()

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic value estimate."""
        return self.critic(obs).squeeze(-1)

    def select_action(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Stochastic action selection.
        Returns: (action_idx, log_prob, value)
        """
        probs = self.action_probs(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.get_value(obs)
        return action.item(), log_prob, value

    def ppo_update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        n_epochs: int = 4,
    ) -> Dict[str, float]:
        """
        PPO clipped objective update.

        Args:
            observations  : [N, obs_dim]
            actions       : [N] int64
            old_log_probs : [N]
            advantages    : [N]  (normalized externally or here)
            returns       : [N]  target values for critic

        Returns dict of loss components.
        """
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(n_epochs):
            probs = self.action_probs(observations)                   # [N, A]
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)                    # [N]
            entropy = dist.entropy().mean()                           # scalar

            ratio = (new_log_probs - old_log_probs).exp()             # [N]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.CLIP_RATIO, 1 + self.CLIP_RATIO) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            values = self.get_value(observations)                     # [N]
            value_loss = F.mse_loss(values, returns)

            loss = policy_loss + self.VALUE_COEFF * value_loss - self.ENTROPY_COEFF * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.MAX_GRAD_NORM)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        return {
            "policy_loss": total_policy_loss / n_epochs,
            "value_loss": total_value_loss / n_epochs,
            "entropy": total_entropy / n_epochs,
        }


# ─────────────────────────────────────────────────────────
# Side-effect and discomfort scorers
# ─────────────────────────────────────────────────────────

class SideEffectMLP(nn.Module):
    """Learned side-effect score from action embedding → [0,1]."""

    def __init__(self, action_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, a_emb: torch.Tensor) -> torch.Tensor:
        return self.net(a_emb).squeeze(-1)


class DiscomfortMLP(nn.Module):
    """Discomfort score from treatment index → [0,1]."""

    def __init__(self, n_treatments: int, action_dim: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_treatments, action_dim)
        self.net = nn.Sequential(
            nn.Linear(action_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, action_idx: torch.Tensor) -> torch.Tensor:
        e = self.emb(action_idx.long())
        return self.net(e).squeeze(-1)


# ─────────────────────────────────────────────────────────
# Doctor Agent
# ─────────────────────────────────────────────────────────

class DoctorAgent(PPOAgent):
    """
    Models optimal treatment selection.

    action_dim = N_treatments
    reward(S_t, a, S_next) = -progression_delta(S_t, S_next)
                             - 0.3 * side_effect_score(a)
                             + 0.2 * patient_compliance_bonus
    """

    def __init__(self, n_treatments: int, obs_dim: int = 256, action_dim: int = 64) -> None:
        super().__init__(obs_dim=obs_dim, action_dim=n_treatments)
        self.side_effect_scorer = SideEffectMLP(action_dim)
        self.action_embedder = nn.Embedding(n_treatments, action_dim)

    def compute_reward(
        self,
        S_t: torch.Tensor,
        action_idx: int,
        S_next: torch.Tensor,
        compliance_bonus: float = 0.0,
    ) -> float:
        """
        Compute doctor reward.

        progression_delta: L2 distance moved in latent space
        (assumes smaller norm = better health in learned space)
        """
        with torch.no_grad():
            # Movement in latent space (proxy for progression)
            progression_delta = float(torch.norm(S_next - S_t).item())

            a_t = torch.tensor([action_idx], dtype=torch.long)
            a_emb = self.action_embedder(a_t)
            side_effect = float(self.side_effect_scorer(a_emb).item())

            reward = -progression_delta - 0.3 * side_effect + 0.2 * compliance_bonus
        return reward


# ─────────────────────────────────────────────────────────
# Disease Agent
# ─────────────────────────────────────────────────────────

class DiseaseAgent(PPOAgent):
    """
    Adversarial agent modeling worst-case disease evolution.

    action_dim = 16 (perturbation directions quantized to 16)
    reward = +progression_delta (opposite of DoctorAgent)
    """

    N_DIRECTIONS = 16

    def __init__(self, obs_dim: int = 256) -> None:
        super().__init__(obs_dim=obs_dim, action_dim=self.N_DIRECTIONS)
        # Fixed perturbation directions: [16, 256] unit vectors
        directions = torch.randn(self.N_DIRECTIONS, obs_dim)
        directions = F.normalize(directions, dim=-1)
        self.register_buffer("perturbation_directions", directions)
        self.perturbation_scale = nn.Parameter(torch.tensor(0.05))

    def apply_perturbation(self, S_t: torch.Tensor, action_idx: int) -> torch.Tensor:
        """Apply disease perturbation to latent state."""
        direction = self.perturbation_directions[action_idx]
        scale = torch.sigmoid(self.perturbation_scale) * 0.2  # max 0.2
        return S_t + scale * direction

    def compute_reward(
        self,
        S_t: torch.Tensor,
        S_next: torch.Tensor,
    ) -> float:
        """Reward is progression delta (disease wants to maximize progression)."""
        with torch.no_grad():
            return float(torch.norm(S_next - S_t).item())


# ─────────────────────────────────────────────────────────
# Patient Agent
# ─────────────────────────────────────────────────────────

class PatientAgent(PPOAgent):
    """
    Models patient compliance variability.

    action_dim = 3: {low=0.3, medium=0.7, high=1.0} compliance levels
    reward = -discomfort_score(treatment) + health_improvement_score(S_t, S_next)
    """

    COMPLIANCE_LEVELS = [0.3, 0.7, 1.0]

    def __init__(self, n_treatments: int, obs_dim: int = 256) -> None:
        super().__init__(obs_dim=obs_dim, action_dim=3)
        self.discomfort_scorer = DiscomfortMLP(n_treatments)

    def get_compliance_level(self, action_idx: int) -> float:
        return self.COMPLIANCE_LEVELS[int(action_idx) % 3]

    def compute_reward(
        self,
        S_t: torch.Tensor,
        action_idx: int,
        treatment_idx: int,
        S_next: torch.Tensor,
    ) -> float:
        """
        Patient reward balances compliance effort with perceived health improvement.
        """
        with torch.no_grad():
            t = torch.tensor([treatment_idx], dtype=torch.long)
            discomfort = float(self.discomfort_scorer(t).item())

            # Health improvement: reduction in latent state norm (proxy)
            improvement = float((torch.norm(S_t) - torch.norm(S_next)).item())
            improvement = max(0.0, improvement)  # only credit positive changes

            reward = -discomfort + improvement
        return reward


# ─────────────────────────────────────────────────────────
# Experience collector for PPO updates
# ─────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores rollout transitions for one PPO update cycle."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.observations: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool = False,
    ) -> None:
        self.observations.append(obs.detach())
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE-lambda advantage estimation."""
        n = len(self.rewards)
        advantages = torch.zeros(n, device=self.device)
        returns = torch.zeros(n, device=self.device)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1].item()
            delta = self.rewards[t] + gamma * next_value * (1 - float(self.dones[t])) - self.values[t].item()
            gae = delta + gamma * gae_lambda * (1 - float(self.dones[t])) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]

        return returns, advantages

    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.stack(self.observations).to(self.device)
        acts = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        lps = torch.stack(self.log_probs).to(self.device)
        returns, advantages = self.compute_returns_and_advantages()
        return obs, acts, lps, returns, advantages

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


def build_agents(
    n_treatments: int,
    obs_dim: int = 256,
    action_dim: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[DoctorAgent, DiseaseAgent, PatientAgent]:
    """Convenience factory for all three MARL agents."""
    if device is None:
        device = torch.device("cpu")
    doctor = DoctorAgent(n_treatments=n_treatments, obs_dim=obs_dim, action_dim=action_dim).to(device)
    disease = DiseaseAgent(obs_dim=obs_dim).to(device)
    patient = PatientAgent(n_treatments=n_treatments, obs_dim=obs_dim).to(device)
    return doctor, disease, patient
