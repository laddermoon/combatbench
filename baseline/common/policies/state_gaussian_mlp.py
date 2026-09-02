"""State-dependent diagonal Gaussian policy (Stage 1).

Same trunk as :class:`TanhGaussianMLPPolicy` but the head outputs
``2 * action_dim`` values: mean and a state-dependent log-std.  The
log-std is bounded via a smooth tanh squash (not a hard clamp) to
avoid permanent gradient dead zones in regions of state space that
push σ to the boundary.

See ``DESIGN_state_dependent_gaussian.md`` for the full design rationale.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from baseline.common.policies.tanh_squashed_base import TanhSquashedPolicyBase

DEFAULT_LOG_STD_MIN = -4.0
DEFAULT_LOG_STD_MAX = 0.0  # Matches baseline's default (not 1.0)


class StateGaussianMLPPolicy(TanhSquashedPolicyBase):
    """Tanh-squashed diagonal Gaussian with state-dependent log_std.

    Architecture:
        trunk: Linear(obs_dim, hidden) -> Tanh -> Linear(hidden, hidden) -> Tanh
        head:  Linear(hidden, 2 * action_dim)  →  split into mean, raw_log_std

    The raw_log_std is squashed into [log_std_min, log_std_max] via:
        bounded = log_std_min + 0.5 * (log_std_max - log_std_min) * (tanh(raw) + 1)

    This is a smooth function with non-zero gradient everywhere (tanh
    saturates but never reaches its asymptote for finite input), so no
    region of state space is permanently gradient-dead.

    Initialization: the log-std head's weights are zeroed and its bias
    is set so that the initial effective log_std ≈ -1.0 everywhere,
    matching the baseline's initial σ ≈ 0.368.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
        entropy_coef: float = 0.0,
        temperature: float = 1.0,
        noise_tau_steps: float = 0.0,
        noise_scale: float = 0.0,
        model_path: Optional[str] = None,
    ):
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim,
            device=device, deterministic=deterministic,
            entropy_coef=entropy_coef, temperature=temperature,
            noise_tau_steps=noise_tau_steps, noise_scale=noise_scale,
        )
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Trunk (same as baseline).
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        # Head: outputs 2 * action_dim (mean | raw_log_std).
        self.head = nn.Linear(hidden_dim, 2 * action_dim)

        self._init_head()

        if model_path is not None:
            payload = torch.load(model_path, map_location="cpu")
            state_dict = payload.get("state_dict", payload)
            self.load_state_dict(state_dict, strict=True)
            self.to(self.device)

    def _init_head(self):
        """Initialize the head so initial σ ≈ 0.368 (log_std ≈ -1.0).

        Mean half: default PyTorch Linear init (matches baseline's net).
        Log-std half: zero weight, bias set so tanh(bias) maps to -1.0
        after the squash.
        """
        with torch.no_grad():
            # Mean half: leave at default init (already done by nn.Linear).
            # Log-std half: zero the weights, set bias for target log_std = -1.0.
            target_log_std = -1.0
            # Solve: log_std_min + 0.5*(max-min)*(tanh(b)+1) = target
            # tanh(b) = 2*(target - min)/(max - min) - 1
            tanh_b = 2.0 * (target_log_std - self.log_std_min) / (self.log_std_max - self.log_std_min) - 1.0
            # Clamp to (-0.999, 0.999) to avoid atanh domain error.
            tanh_b = max(-0.999, min(0.999, tanh_b))
            bias_val = float(np.arctanh(tanh_b))

            action_dim = self.action_dim
            # Zero the log-std half of the weight matrix.
            self.head.weight.data[action_dim:, :] = 0.0
            # Set the log-std half of the bias.
            self.head.bias.data[action_dim:] = bias_val

    # ------------------------------------------------------------------
    # Bounded log-std (smooth squash, not hard clamp)
    # ------------------------------------------------------------------

    def _bounded_log_std(self, raw_log_std: torch.Tensor) -> torch.Tensor:
        """Squash raw log-std into [log_std_min, log_std_max] smoothly.

        Uses tanh:  bounded = min + 0.5*(max-min)*(tanh(raw + offset) + 1)
        where offset = log(temperature) so temperature scales σ before
        bounding (matching baseline's semantics: high temperature
        saturates against log_std_max rather than exceeding it).
        """
        offset = float(np.log(self._temperature)) if self._temperature > 0 else 0.0
        t = torch.tanh(raw_log_std + offset)
        return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (t + 1.0)

    def _forward_head(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through trunk + head, return (mean, bounded_log_std)."""
        h = self.trunk(obs)
        out = self.head(h)
        mean, raw_log_std = out.split(self.action_dim, dim=-1)
        log_std = self._bounded_log_std(raw_log_std)
        return mean, log_std

    # ------------------------------------------------------------------
    # Raw-space hooks
    # ------------------------------------------------------------------

    def _raw_sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, None]:
        mean, log_std = self._forward_head(obs)
        std = log_std.exp()
        raw = mean + std * torch.randn_like(mean)
        return raw, None

    def _raw_log_prob(
        self, obs: torch.Tensor, raw_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        mean, log_std = self._forward_head(obs)
        dist = Normal(mean, log_std.exp())
        return dist.log_prob(raw_action).sum(-1), None

    def _raw_log_prob_per_dim(
        self, obs: torch.Tensor, raw_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """Per-dimension log_prob for bit-identical baseline matching."""
        mean, log_std = self._forward_head(obs)
        dist = Normal(mean, log_std.exp())
        return dist.log_prob(raw_action), None

    def _raw_mode(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._forward_head(obs)
        return mean

    def _regularizer_and_stats(
        self, obs, raw_action, raw_log_prob, want_stats,
        sample_extras, score_extras,
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
        mean, log_std = self._forward_head(obs)
        entropy = Normal(mean, log_std.exp()).entropy().sum(-1)

        regularizer = None
        if self._entropy_coef != 0.0:
            regularizer = -self._entropy_coef * entropy.mean()

        stats = None
        if want_stats:
            with torch.no_grad():
                eff_std = log_std.exp()
                raw_log_std_for_sat = torch.log(eff_std + 1e-8)
                # Saturation: fraction of batch with |raw_log_std| > 3
                # (operating in the flat part of tanh).
                # We need the pre-squash raw value; approximate by
                # checking if bounded log_std is near the bounds.
                near_min = (log_std - self.log_std_min).abs() < 0.05
                near_max = (log_std - self.log_std_max).abs() < 0.05
                sat_frac = (near_min | near_max).float().mean()
                stats = {
                    "std_mean_batch": float(eff_std.mean().item()),
                    "std_min_batch": float(eff_std.min().item()),
                    "std_max_batch": float(eff_std.max().item()),
                    "std_squash_sat_frac": float(sat_frac.item()),
                }
        return regularizer, stats

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_config(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
        }

    @property
    def export_class_path(self) -> str:
        return "baseline.common.policies.state_gaussian_mlp:StateGaussianMLPPolicy"
