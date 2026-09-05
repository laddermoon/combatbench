"""State-independent Tanh-Gaussian policy with OU exploration support.

This is a drop-in replacement for :class:`TanhGaussianMLPPolicy` that
inherits from :class:`TanhSquashedPolicyBase` to gain temporally
correlated (OU) exploration noise support.  The parameter names
(``net.*``, ``log_std``) are identical to the baseline so existing
checkpoints load without remapping.

Differences from the baseline:

1. **OU exploration.  The baseline has no temporal correlation; this
   policy does (via the base class).  When ``noise_scale=0`` (default),
   behavior is bit-identical to the baseline.**

2. **Architecture via base class hooks.  The baseline implements
   ``sample_action`` / ``evaluate_actions`` directly; this policy
   implements the raw-space hooks (``_raw_sample``, ``_raw_log_prob``,
   ...) and lets the base class own the tanh Jacobian and the OU
   shift.  This means the tanh correction and the shift logic are
   maintained in exactly one place.**

3. **Closed-form entropy regularizer.  Same as the baseline
   (``Normal.entropy()``), avoiding the score-function gradient
   vanishing as σ → 0.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from baseline.framework.ppo.policies.tanh_squashed_base import TanhSquashedPolicyBase

DEFAULT_LOG_STD_MIN = -4.0
DEFAULT_LOG_STD_MAX = 0.0


class FixedSigmaGaussianMLPPolicy(TanhSquashedPolicyBase):
    """Tanh-squashed diagonal Gaussian with state-independent log_std.

    Architecture (identical to :class:`TanhGaussianMLPPolicy`)::

        net = Linear(obs, hidden) → Tanh → Linear(hidden, hidden) → Tanh → Linear(hidden, action)
        log_std = Parameter((action_dim,))   # state-independent

    Effective log_std = clamp(log_std + offset(explore_intensity), log_std_min, log_std_max)

    The ``net`` and ``log_std`` parameter names match the baseline
    exactly, so a baseline checkpoint can be loaded with ``strict=True``
    into this policy.  This enables resuming a baseline run with OU
    exploration enabled.
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
        noise_tau_steps: float = 0.0,
        noise_scale: float = 0.0,
        model_path: Optional[str] = None,
    ):
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim,
            device=device, deterministic=deterministic,
            entropy_coef=entropy_coef,
            noise_tau_steps=noise_tau_steps, noise_scale=noise_scale,
        )
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Same structure and parameter names as TanhGaussianMLPPolicy
        # so baseline checkpoints load with strict=True.
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(
            torch.full((action_dim,), -1.0, dtype=torch.float32)
        )

        if model_path is not None:
            payload = torch.load(model_path, map_location="cpu")
            state_dict = payload.get("state_dict", payload)
            self.load_state_dict(state_dict, strict=True)
            self.to(self.device)

    # ------------------------------------------------------------------
    # Effective log-std (explore_intensity offset + hard clamp, same as baseline)
    # ------------------------------------------------------------------

    def _effective_log_std(self, explore_intensity: Any = 0.5) -> torch.Tensor:
        """Return the ``(action_dim,)`` log-sigma used for sampling.

        ``explore_intensity`` is mapped to an additive offset before
        clamping to ``[log_std_min, log_std_max]`` — identical in spirit
        to :meth:`TanhGaussianMLPPolicy.effective_log_std`.  This ensures
        training-time scoring and rollout-time sampling can never disagree.

        ``explore_intensity=0.5`` (neutral) yields a zero offset, matching
        the baseline ``temperature=1.0`` behavior.
        """
        if isinstance(explore_intensity, torch.Tensor):
            offset = (explore_intensity - 0.5) * 2.0
            offset = offset.unsqueeze(-1)  # (B, 1) for broadcasting
        else:
            offset = float(explore_intensity - 0.5) * 2.0
        return torch.clamp(
            self.log_std + offset,
            self.log_std_min,
            self.log_std_max,
        )

    def _forward(
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, effective_log_std_expanded)."""
        mean = self.net(obs)
        log_std = self._effective_log_std(explore_intensity)
        return mean, log_std.expand_as(mean)

    # ------------------------------------------------------------------
    # Raw-space hooks
    # ------------------------------------------------------------------

    def _raw_sample(
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, None]:
        mean, log_std = self._forward(obs, explore_intensity=explore_intensity)
        std = log_std.exp()
        raw = mean + std * torch.randn_like(mean)
        return raw, None

    def _raw_log_prob(
        self, obs: torch.Tensor, raw_action: torch.Tensor,
        *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, None]:
        mean, log_std = self._forward(obs, explore_intensity=explore_intensity)
        dist = Normal(mean, log_std.exp())
        return dist.log_prob(raw_action).sum(-1), None

    def _raw_log_prob_per_dim(
        self, obs: torch.Tensor, raw_action: torch.Tensor,
        *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, None]:
        """Per-dimension log_prob for bit-identical baseline matching."""
        mean, log_std = self._forward(obs, explore_intensity=explore_intensity)
        dist = Normal(mean, log_std.exp())
        return dist.log_prob(raw_action), None

    def _raw_mode(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._forward(obs, explore_intensity=0.5)
        return mean

    def _regularizer_and_stats(
        self,
        obs: torch.Tensor,
        raw_action: torch.Tensor,
        raw_log_prob: torch.Tensor,
        want_stats: bool,
        sample_extras: Optional[Dict[str, Any]],
        score_extras: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
        mean, log_std = self._forward(obs, explore_intensity=0.5)
        entropy = Normal(mean, log_std.exp()).entropy().sum(-1)

        regularizer = None
        if self._entropy_coef != 0.0:
            regularizer = -self._entropy_coef * entropy.mean()

        stats: Optional[Dict[str, float]] = None
        if want_stats:
            with torch.no_grad():
                eff_std = self._effective_log_std(0.5).exp()
                stats = {
                    "entropy": float(entropy.mean().item()),
                    "std_mean": float(eff_std.mean().item()),
                    "std_min": float(eff_std.min().item()),
                    "std_max": float(eff_std.max().item()),
                    "tanh_sat_frac": float(
                        (mean.abs() > 2.0).float().mean().item()
                    ),
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
        return "baseline.framework.ppo.policies.fixed_sigma_gaussian_mlp:FixedSigmaGaussianMLPPolicy"
