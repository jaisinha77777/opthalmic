"""
Nash Equilibrium Solver using Iterated Best Response with PPO agents.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agents import DoctorAgent, DiseaseAgent, PatientAgent, RolloutBuffer

log = logging.getLogger(__name__)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """KL(p || q) for discrete distributions."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return float((p * (p / q).log()).sum().item())


def _collect_rollout(
    agent: nn.Module,
    S_t: torch.Tensor,
    twin_engine: Any,
    other_actions: Tuple[int, int],
    role: str,
    n_steps: int,
    device: torch.device,
) -> RolloutBuffer:
    """
    Collect n_steps of experience for one agent while others play fixed actions.

    other_actions: (disease_action, patient_action) for Doctor's turn, etc.
    role: 'doctor', 'disease', or 'patient'
    """
    buffer = RolloutBuffer(device)
    obs = S_t.detach().clone().to(device)

    for _ in range(n_steps):
        action, log_prob, value = agent.select_action(obs)

        # Compute reward and next state based on role
        if role == "doctor":
            disease_action, patient_action = other_actions
            compliance = [0.3, 0.7, 1.0][int(patient_action) % 3]
            # Simulate state transition
            try:
                result = twin_engine.step(
                    treatment_action_idx=action,
                    compliance_level=compliance,
                    n_mc_samples=5,  # fast during training
                )
                S_next = result["new_state"].to(device)
            except Exception:
                S_next = obs + 0.01 * torch.randn_like(obs)

            reward = agent.compute_reward(obs, action, S_next, compliance_bonus=compliance - 0.5)

        elif role == "disease":
            disease_agent = agent
            S_next = disease_agent.apply_perturbation(obs, action)
            reward = disease_agent.compute_reward(obs, S_next)

        elif role == "patient":
            doctor_action, _ = other_actions
            compliance = agent.get_compliance_level(action)
            try:
                result = twin_engine.step(
                    treatment_action_idx=doctor_action,
                    compliance_level=compliance,
                    n_mc_samples=5,
                )
                S_next = result["new_state"].to(device)
            except Exception:
                S_next = obs + 0.01 * torch.randn_like(obs)
            reward = agent.compute_reward(obs, action, doctor_action, S_next)
        else:
            S_next = obs
            reward = 0.0

        buffer.add(obs, action, log_prob, reward, value, done=False)
        obs = S_next.detach().clone()

    return buffer


class NashEquilibriumSolver:
    """
    Iterated Best Response approximation of Nash Equilibrium.

    In each iteration:
      Step 1: Fix Disease + Patient → Optimize Doctor   (64 rollout steps)
      Step 2: Fix Doctor + Patient → Optimize Disease   (64 rollout steps)
      Step 3: Fix Doctor + Disease → Optimize Patient   (64 rollout steps)

    Convergence criterion: KL(new_doctor_policy || old_doctor_policy) < eps
    """

    def __init__(
        self,
        doctor: DoctorAgent,
        disease: DiseaseAgent,
        patient: PatientAgent,
        twin_engine: Any,
        n_iters: int = 20,
        convergence_eps: float = 1e-3,
        n_steps: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        self.doctor = doctor
        self.disease = disease
        self.patient = patient
        self.twin_engine = twin_engine
        self.n_iters = n_iters
        self.convergence_eps = convergence_eps
        self.n_steps = n_steps
        self.device = device or torch.device("cpu")

    def solve(
        self, S_t: torch.Tensor, patient_id: str
    ) -> Dict[str, Any]:
        """
        Run iterated best response starting from state S_t.

        Returns dict with:
            nash_treatment, convergence_step, doctor_policy,
            disease_policy, patient_compliance, expected_reward,
            nash_convergence_step
        """
        S_t = S_t.detach().clone().to(self.device)
        prev_doctor_policy: Optional[torch.Tensor] = None
        convergence_step = self.n_iters

        # Initial actions (greedy start)
        with torch.no_grad():
            obs = S_t.unsqueeze(0) if S_t.dim() == 1 else S_t
            doctor_action = self.doctor.get_best_action(obs)
            disease_action = self.disease.get_best_action(obs)
            patient_action = self.patient.get_best_action(obs)

        log.info("Nash solver starting: patient=%s n_iters=%d", patient_id, self.n_iters)

        for k in range(self.n_iters):
            # ── Step 1: Optimize Doctor ──────────────────────────────
            doc_buffer = _collect_rollout(
                self.doctor, S_t, self.twin_engine,
                other_actions=(disease_action, patient_action),
                role="doctor", n_steps=self.n_steps, device=self.device,
            )
            if len(doc_buffer.observations) > 0:
                obs_t, acts_t, lps_t, rets_t, advs_t = doc_buffer.as_tensors()
                self.doctor.ppo_update(obs_t, acts_t, lps_t, advs_t, rets_t)
                with torch.no_grad():
                    doctor_action = self.doctor.get_best_action(S_t)
            doc_buffer.clear()

            # ── Step 2: Optimize Disease ─────────────────────────────
            dis_buffer = _collect_rollout(
                self.disease, S_t, self.twin_engine,
                other_actions=(doctor_action, patient_action),
                role="disease", n_steps=self.n_steps, device=self.device,
            )
            if len(dis_buffer.observations) > 0:
                obs_t, acts_t, lps_t, rets_t, advs_t = dis_buffer.as_tensors()
                self.disease.ppo_update(obs_t, acts_t, lps_t, advs_t, rets_t)
                with torch.no_grad():
                    disease_action = self.disease.get_best_action(S_t)
            dis_buffer.clear()

            # ── Step 3: Optimize Patient ─────────────────────────────
            pat_buffer = _collect_rollout(
                self.patient, S_t, self.twin_engine,
                other_actions=(doctor_action, disease_action),
                role="patient", n_steps=self.n_steps, device=self.device,
            )
            if len(pat_buffer.observations) > 0:
                obs_t, acts_t, lps_t, rets_t, advs_t = pat_buffer.as_tensors()
                self.patient.ppo_update(obs_t, acts_t, lps_t, advs_t, rets_t)
                with torch.no_grad():
                    patient_action = self.patient.get_best_action(S_t)
            pat_buffer.clear()

            # ── Convergence check ────────────────────────────────────
            with torch.no_grad():
                obs_1d = S_t if S_t.dim() == 1 else S_t.squeeze(0)
                new_doctor_policy = self.doctor.action_probs(obs_1d.unsqueeze(0)).squeeze(0)

            if prev_doctor_policy is not None:
                kl = _kl_divergence(new_doctor_policy, prev_doctor_policy)
                log.debug("Nash iter %d/%d KL=%.5f", k + 1, self.n_iters, kl)
                if kl < self.convergence_eps:
                    convergence_step = k + 1
                    log.info("Nash converged at step %d (KL=%.5f)", convergence_step, kl)
                    break

            prev_doctor_policy = new_doctor_policy.clone()

        # ── Collect final policies ───────────────────────────────────
        with torch.no_grad():
            obs_unsq = S_t.unsqueeze(0)
            doc_probs = self.doctor.action_probs(obs_unsq).squeeze(0)
            dis_probs = self.disease.action_probs(obs_unsq).squeeze(0)
            pat_probs = self.patient.action_probs(obs_unsq).squeeze(0)
            expected_reward = float(self.doctor.get_value(obs_unsq).item())

        nash_treatment = int(doc_probs.argmax().item())
        patient_compliance_action = int(pat_probs.argmax().item())
        compliance_level = self.patient.COMPLIANCE_LEVELS[patient_compliance_action]

        return {
            "nash_treatment": nash_treatment,
            "convergence_step": convergence_step,
            "doctor_policy": doc_probs.tolist(),
            "disease_policy": dis_probs.tolist(),
            "patient_compliance": pat_probs.tolist(),
            "compliance_level": compliance_level,
            "expected_reward": expected_reward,
        }
