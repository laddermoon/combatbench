"""RealNVP normalizing flow policy (Stage 4).

The most expressive family: a normalizing flow can represent arbitrary
continuous distributions over the action space — not just mixtures of
Gaussians, but distributions with skewed modes, curved manifolds, holes,
rings, etc.

Architecture:
  * Base distribution: diagonal Gaussian conditioned on state (like ①).
  * Flow: L coupling layers (RealNVP), each transforming half of the
    dimensions conditioned on the other half + the observation.

Sampling:  z ~ base_dist → forward flow → raw_action → tanh → action
Scoring:   action → atanh → raw_action → inverse flow → z → base_log_prob + flow_log_det + tanh_jac

See ``DESIGN_normalizing_flow.md`` for the full design rationale.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from baseline.framework.ppo.policies.tanh_squashed_base import TanhSquashedPolicyBase

DEFAULT_LOG_STD_MIN = -4.0
DEFAULT_LOG_STD_MAX = 0.0
DEFAULT_NUM_LAYERS = 4
DEFAULT_SCALE_MAX = 1.0
DEFAULT_CONDITIONER_HIDDEN = 64


class _CouplingLayer(nn.Module):
    """One RealNVP coupling layer.

    Transforms the unmasked half of the input conditioned on the masked
    half and the observation trunk features.  The masked half passes
    through unchanged.

    Forward:  y_masked = m * x
              y_unmasked = (1-m) * (x * exp(s) + t)
              y = y_masked + y_unmasked

    Inverse:  x_masked = m * y
              x_unmasked = (1-m) * ((y - m*y) * exp(-s) - t)
              x = x_masked + x_unmasked

    The conditioner takes (trunk_obs, masked_x) as input and outputs
    (s, t) for the unmasked dimensions.  s is bounded by
    ``tanh(s_raw) * scale_max`` to prevent scale explosion.
    """

    def __init__(
        self,
        obs_trunk_dim: int,
        action_dim: int,
        mask: torch.Tensor,
        hidden_dim: int = DEFAULT_CONDITIONER_HIDDEN,
        scale_max: float = DEFAULT_SCALE_MAX,
    ):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_max = float(scale_max)
        self.action_dim = int(action_dim)

        # Conditioner: (trunk_obs + masked_x) → (s, t)
        # masked_x has action_dim dims (masked half is zeroed, but we
        # pass the full vector for simplicity; the conditioner learns
        # to ignore the unmasked half).
        conditioner_input = obs_trunk_dim + action_dim
        self.conditioner = nn.Sequential(
            nn.Linear(conditioner_input, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )

    def _compute_s_t(
        self, trunk_obs: torch.Tensor, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (s, t) conditioned on the masked half of x."""
        masked_x = x * self.mask  # (B, action_dim), masked half zeroed
        inp = torch.cat([trunk_obs, masked_x], dim=-1)
        out = self.conditioner(inp)
        s_raw, t = out.split(self.action_dim, dim=-1)
        s = torch.tanh(s_raw) * self.scale_max
        # Zero out s, t for the masked dimensions (they pass through).
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        return s, t

    def forward(
        self, x: torch.Tensor, trunk_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transform (base → action direction).

        Returns:
            (y, log_det) where log_det = sum(s * (1 - mask)).
        """
        s, t = self._compute_s_t(trunk_obs, x)
        y = x * self.mask + (1.0 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)  # sum over unmasked dims only (s is 0 for masked)
        return y, log_det

    def inverse(
        self, y: torch.Tensor, trunk_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse transform (action → base direction).

        Returns:
            (x, log_det) where log_det = -sum(s * (1 - mask)).

        The conditioner uses the masked half of y, which equals the
        masked half of x (unchanged by forward), so s, t are the same
        as in the forward pass.

        Forward:  y = x * mask + (1-mask) * (x * exp(s) + t)
        Inverse:  x = y * mask + (1-mask) * (y - t) * exp(-s)

        Note the order: subtract t *before* scaling by exp(-s).
        """
        s, t = self._compute_s_t(trunk_obs, y)
        x = y * self.mask + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        log_det = -s.sum(dim=-1)
        return x, log_det


class RealNVPTanhMLPPolicy(TanhSquashedPolicyBase):
    """Tanh-squashed RealNVP normalizing flow policy.

    Architecture:
        trunk: Linear(obs_dim, hidden) -> Tanh -> Linear(hidden, hidden) -> Tanh
        base_head: Linear(hidden, 2 * action_dim)  →  base_mean, base_log_std
        flow: L coupling layers, each with its own conditioner

    The base distribution is a diagonal Gaussian (like ①).  The flow
    transforms the base sample into a more expressive raw distribution.
    Tanh is applied after the flow.

    Temperature scales the base distribution's σ only (not the flow).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = DEFAULT_NUM_LAYERS,
        scale_max: float = DEFAULT_SCALE_MAX,
        conditioner_hidden: int = DEFAULT_CONDITIONER_HIDDEN,
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
            entropy_coef=entropy_coef,
            noise_tau_steps=noise_tau_steps, noise_scale=noise_scale,
        )
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.scale_max = float(scale_max)
        self.conditioner_hidden = int(conditioner_hidden)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Trunk (shared by base head and flow conditioners).
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        # Base distribution head: mean + raw_log_std.
        self.base_head = nn.Linear(hidden_dim, 2 * action_dim)

        # Coupling layers with alternating masks.
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i % 2 == 0:
                mask = self._make_even_mask(action_dim)
            else:
                mask = self._make_odd_mask(action_dim)
            self.layers.append(_CouplingLayer(
                obs_trunk_dim=hidden_dim,
                action_dim=action_dim,
                mask=mask,
                hidden_dim=conditioner_hidden,
                scale_max=scale_max,
            ))

        self._init_heads()

        if model_path is not None:
            payload = torch.load(model_path, map_location="cpu")
            state_dict = payload.get("state_dict", payload)
            self.load_state_dict(state_dict, strict=True)
            self.to(self.device)

    @staticmethod
    def _make_even_mask(action_dim: int) -> torch.Tensor:
        """Mask even-indexed dimensions (0, 2, 4, ...)."""
        mask = torch.zeros(action_dim)
        mask[0::2] = 1.0
        return mask

    @staticmethod
    def _make_odd_mask(action_dim: int) -> torch.Tensor:
        """Mask odd-indexed dimensions (1, 3, 5, ...)."""
        mask = torch.zeros(action_dim)
        mask[1::2] = 1.0
        return mask

    def _init_heads(self):
        """Initialize base head so initial σ ≈ 0.368, and flow layers
        so the initial flow is approximately identity (s≈0, t≈0)."""
        with torch.no_grad():
            ad = self.action_dim
            target_log_std = -1.0
            tanh_b = 2.0 * (target_log_std - self.log_std_min) / (self.log_std_max - self.log_std_min) - 1.0
            tanh_b = max(-0.999, min(0.999, tanh_b))
            bias_val = float(np.arctanh(tanh_b))

            # Base head: mean = default init, log_std = zero weight + target bias.
            self.base_head.weight.data[ad:, :] = 0.0
            self.base_head.bias.data[ad:] = bias_val

            # Flow layers: zero the last layer's weights and biases
            # so s_raw ≈ 0 → s ≈ 0, t ≈ 0 → identity transform.
            for layer in self.layers:
                layer.conditioner[-1].weight.data.zero_()
                layer.conditioner[-1].bias.data.zero_()

    # ------------------------------------------------------------------
    # Bounded log-std (same smooth squash as ①)
    # ------------------------------------------------------------------

    def _bounded_log_std(
        self, raw_log_std: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> torch.Tensor:
        if isinstance(explore_intensity, torch.Tensor):
            offset = (explore_intensity - 0.5) * 2.0
            offset = offset.unsqueeze(-1)  # (B, 1) for broadcasting
        else:
            offset = float(explore_intensity - 0.5) * 2.0
        t = torch.tanh(raw_log_std + offset)
        return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (t + 1.0)

    def _base_dist(
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, Normal]:
        """Return (trunk_obs, base_normal_dist)."""
        h = self.trunk(obs)
        out = self.base_head(h)
        ad = self.action_dim
        mean, raw_log_std = out.split(ad, dim=-1)
        log_std = self._bounded_log_std(raw_log_std, explore_intensity=explore_intensity)
        dist = Normal(mean, log_std.exp())
        return h, dist

    # ------------------------------------------------------------------
    # Flow forward / inverse
    # ------------------------------------------------------------------

    def _flow_forward(
        self, z: torch.Tensor, trunk_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward flow: base → raw_action. Returns (raw_action, total_log_det)."""
        x = z
        total_log_det = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            x, ld = layer.forward(x, trunk_obs)
            total_log_det = total_log_det + ld
        return x, total_log_det

    def _flow_inverse(
        self, raw_action: torch.Tensor, trunk_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse flow: raw_action → base. Returns (z, total_log_det)."""
        x = raw_action
        total_log_det = torch.zeros(raw_action.shape[0], device=raw_action.device)
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x, trunk_obs)
            total_log_det = total_log_det + ld
        return x, total_log_det

    # ------------------------------------------------------------------
    # Raw-space hooks
    # ------------------------------------------------------------------

    def _raw_sample(
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, None]:
        trunk_obs, base_dist = self._base_dist(obs, explore_intensity=explore_intensity)
        z = base_dist.rsample()
        raw, _ = self._flow_forward(z, trunk_obs)
        return raw, None

    def _raw_log_prob(
        self, obs: torch.Tensor, raw_action: torch.Tensor,
        *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, None]:
        trunk_obs, base_dist = self._base_dist(obs, explore_intensity=explore_intensity)
        z, flow_log_det = self._flow_inverse(raw_action, trunk_obs)
        base_lp = base_dist.log_prob(z).sum(-1)
        raw_log_prob = base_lp + flow_log_det
        return raw_log_prob, None

    def _raw_mode(self, obs: torch.Tensor) -> torch.Tensor:
        """Return forward(base_mean) — the image of the base mode.

        Not the true mode of the flow distribution, but a reasonable
        deterministic action.
        """
        trunk_obs, base_dist = self._base_dist(obs, explore_intensity=0.5)
        z = base_dist.mean
        raw, _ = self._flow_forward(z, trunk_obs)
        return raw

    def _regularizer_and_stats(
        self, obs, raw_action, raw_log_prob, want_stats,
        sample_extras, score_extras,
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
        regularizer = None  # Base class handles the sampled entropy regularizer.

        stats = None
        if want_stats:
            with torch.no_grad():
                trunk_obs, base_dist = self._base_dist(obs, explore_intensity=0.5)
                base_std = base_dist.stddev  # (B, action_dim)

                # Compute flow log_det for the scored actions.
                z, flow_log_det = self._flow_inverse(raw_action, trunk_obs)

                # Scale saturation: fraction of s_raw values near ±scale_max.
                sat_count = 0
                total_count = 0
                for layer in self.layers:
                    # Recompute s_raw for this batch.
                    masked_x = raw_action * layer.mask
                    inp = torch.cat([trunk_obs, masked_x], dim=-1)
                    out = layer.conditioner(inp)
                    s_raw, _ = out.split(self.action_dim, dim=-1)
                    sat_count += (s_raw.abs() > 2.5).sum().item()
                    total_count += s_raw.numel()
                sat_frac = sat_count / max(total_count, 1)

                # Inverse reconstruction error (gated behind want_stats).
                z_recon, _ = self._flow_inverse(raw_action, trunk_obs)
                raw_recon, _ = self._flow_forward(z_recon, trunk_obs)
                recon_err = (raw_action - raw_recon).abs().max(dim=-1).values

                stats = {
                    "base_std_mean_batch": float(base_std.mean().item()),
                    "base_std_min_batch": float(base_std.min().item()),
                    "base_std_max_batch": float(base_std.max().item()),
                    "flow_logdet_mean_batch": float(flow_log_det.mean().item()),
                    "flow_logdet_std_batch": float(flow_log_det.std().item()),
                    "scale_sat_frac": float(sat_frac),
                    "inverse_recon_err_mean_batch": float(recon_err.mean().item()),
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
            "num_layers": self.num_layers,
            "scale_max": self.scale_max,
            "conditioner_hidden": self.conditioner_hidden,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
        }

    @property
    def export_class_path(self) -> str:
        return "baseline.framework.ppo.policies.realnvp_tanh_mlp:RealNVPTanhMLPPolicy"
