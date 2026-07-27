"""ST-4-φ: Delta shaping with φ-subtracted critic (killer falsification).

Same reward as ST-4: r_t = φ(t) - φ(t-1), γ_s = 1.0.
But the critic is parameterized as V(s) = V_θ(s) - φ(s), so V_θ only
needs to learn V^Dense(s) — the dense value function that ST-6's
critic learns easily.

Theoretical prediction: if the "accounting deadlock" theory is correct,
this experiment should converge at ≈U437 (same as ST-6).  If it still
fails, the theory is falsified and there is another root cause.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.humanoid21.curriculum.experiments.exp_st4_delta import ST4DeltaConfig
from baseline.humanoid21.curriculum.experiments.phi_critic import PhiSubtractedCritic
from baseline.framework.ppo_trainer import _extract_per_step_field
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


class ST4DeltaPhiCriticConfig(ST4DeltaConfig):
    """ST-4 with φ-subtracted critic.  Inherits everything from ST-4
    except the critic construction."""

    name = "st4_delta_phi_critic"
    reward_keys = ("r_fall",)
    gammas = {"r_fall": 0.99}

    BLUEPRINT = "basic_balance_phi_env.yaml"
    shaping_gamma: float = 1.0

    _survival_rate: float = 0.0

    def build_v_critic(self, reward_key: str, device: torch.device) -> nn.Module:
        """Override: return PhiSubtractedCritic instead of CriticMLP."""
        return PhiSubtractedCritic(
            obs_dim=self.obs_dim,
            hidden_dim=self.critic_hidden_dim,
            standing_height=1.28,
        ).to(device)


EXPERIMENT = ST4DeltaPhiCriticConfig()
