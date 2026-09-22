"""StateTruncatedNormalPolicy — truncated normal with state-dependent σ.

Identical to :class:`TruncatedNormalPolicy` in every respect except one:
σ is a function of the observation, not a global ``nn.Parameter``.

    mean        = tanh(head_mean(trunk(obs)))            ∈ (-1, 1)
    σ(obs)      = exp(clamp(head_logstd(trunk(obs))))    > 0, per-state

The distribution is ``Normal(mean, σ(obs))`` truncated to [-1, 1] and
renormalized — same math, same sampling, same log_prob, same
explore_factor scaling, same uncertainty definition U = 1/(2×peak) as
the global-σ baseline.  Only the parameterization of σ differs, so A/B
against ``TruncatedNormalPolicy`` isolates exactly one variable:
state-dependence of exploration width.

σ deliberately carries no business bounds — the head output is only
clamped to the same numerical safety range (±20) the baseline applies
to its global ``log_std``.  This keeps U → 1 (uniform) reachable and
keeps the comparison clean.  See DESIGN_state_truncated_normal.md.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from envs.framework.policy import Policy, PolicyBlueprint

from baseline.framework.ppo import ActorEval, TrainablePolicy
from baseline.framework.ppo.stochastic_policy import StochasticPolicy

from baseline.framework.ppo.policies.truncated_normal_mlp import (
    _ACTION_HIGH,
    _ACTION_LOW,
    _ACTION_WIDTH,
    _EXPLORE_K,
    _LOG_STD_SAFE_MAX,
    _LOG_STD_SAFE_MIN,
    _SQRT_2PI,
    _std_normal_cdf,
    _std_normal_icdf,
)

__all__ = [
    "StateTruncatedNormalPolicy",
]


def _build_export_policy_code() -> str:
    """Return the source of the ``policy.py`` embedded in export dirs.

    The export is self-contained — it reads from a real template file
    (``_export_template_state_truncnorm.py``) that has no imports from
    ``baseline.*`` or ``envs.*``.  See TruncatedNormalPolicy.to_blueprint
    for the full rationale (P0-6).
    """
    template_path = (
        Path(__file__).resolve().parent
        / "_export_template_state_truncnorm.py"
    )
    return template_path.read_text(encoding="utf-8")


class StateTruncatedNormalPolicy(nn.Module, TrainablePolicy, Policy):
    """Truncated normal policy on [-1, 1] with state-dependent σ.

    mean = tanh(head_mean(trunk(obs)))  ∈ (-1, 1)
    σ    = exp(clamp(raw_log_std(obs)))  > 0  (per-state, per-dim)

    The distribution is Normal(mean, σ) truncated to [-1, 1] and
    renormalized.  Sampling uses inverse-CDF reparameterization;
    log_prob includes the truncation normalization term.

    Uncertainty U = 1 / (2 × peak) is a geometric area ratio in [0, 1]:
    0 = deterministic, 1 = uniform.  See DESIGN_truncated_normal.md §3.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        # Trunk shape matches TruncatedNormalPolicy's first two layers so
        # the parameter delta vs baseline is attributable only to the
        # wider head (hidden → 2·action_dim instead of → action_dim).
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Head outputs raw_mean | raw_log_std (each action_dim wide).
        self.head = nn.Linear(hidden_dim, 2 * action_dim)
        self._init_head()

        # Device is derived from parameters via the @property below, so
        # .to(device) / .cuda() / DataParallel all keep it in sync (P0-4).
        self.to(torch.device(device))

    def _init_head(self) -> None:
        """Init so σ(obs) = e⁻¹ ≈ 0.368 everywhere at step 0.

        σ half: weights zeroed, bias = -1.0 → raw_log_std ≡ -1.0,
        matching TruncatedNormalPolicy's ``log_std`` init exactly.  This
        makes the degenerate-equivalence test (copy baseline weights,
        get bit-identical outputs) possible.

        Mean half keeps default PyTorch Linear init — same distribution
        as the baseline's final layer init.
        """
        d = self.action_dim
        with torch.no_grad():
            self.head.weight[d:, :].zero_()
            self.head.bias[d:].fill_(-1.0)

    @property
    def device(self) -> torch.device:
        """Runtime device of this policy's parameters (P0-4)."""
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    def _head_forward(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One trunk+head pass → (mean, policy_sigma), both (B, action_dim).

        ``policy_sigma`` is the *unscaled* σ — explore_factor is applied
        on top by :meth:`forward` / :meth:`evaluate_actions`.
        """
        out = self.head(self.trunk(obs))
        raw_mean, raw_log_std = out.split(self.action_dim, dim=-1)
        mean = torch.tanh(raw_mean)
        log_std = torch.clamp(
            raw_log_std, _LOG_STD_SAFE_MIN, _LOG_STD_SAFE_MAX,
        )
        return mean, log_std.exp()

    def _explore_scale(self, explore_factor: Any = 0.0) -> Any:
        """Exponential σ scaling factor from explore_factor.

        ei=0 → 1.0 (neutral), ei=+1 → 3.0 (max explore), ei=-1 → 1/3 (max
        suppress).  scale = exp(ei * ln(3)).  Accepts scalar float or
        (B,) tensor.  Same mapping as TruncatedNormalPolicy.
        """
        if isinstance(explore_factor, torch.Tensor):
            return torch.exp(explore_factor * _EXPLORE_K)
        return math.exp(float(explore_factor) * _EXPLORE_K)

    def effective_sigma(
        self, policy_sigma: torch.Tensor, explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        """σ used for sampling / log_prob (includes explore scale).

        Unlike the baseline, σ is already (B, action_dim) — the scale is
        applied multiplicatively on top of the state-dependent σ, with
        the same semantics: ei=±1 always means σ×3 / σ÷3 regardless of
        where the state-dependent σ sits.
        """
        scale = self._explore_scale(explore_factor)
        if isinstance(scale, torch.Tensor):
            return policy_sigma * scale.unsqueeze(-1)  # (B,D) * (B,1)
        return policy_sigma * scale

    def forward(
        self, obs: torch.Tensor, *, explore_factor: Any = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, effective_sigma), both (B, action_dim)."""
        mean, policy_sigma = self._head_forward(obs)
        return mean, self.effective_sigma(policy_sigma, explore_factor)

    @staticmethod
    def _trunc_params(
        mean: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute standardized truncation bounds and normalization Z.

        Returns (a, b, log_Z) where:
            a = (low  - mean) / sigma   (standardized lower bound)
            b = (high - mean) / sigma   (standardized upper bound)
            Z  = Φ(b) - Φ(a)            (truncation normalization)
        """
        a = (_ACTION_LOW - mean) / sigma
        b = (_ACTION_HIGH - mean) / sigma
        cdf_b = _std_normal_cdf(b)
        cdf_a = _std_normal_cdf(a)
        Z = cdf_b - cdf_a
        # Clamp Z away from 0 for numerical stability.
        Z = torch.clamp(Z, min=1e-8)
        log_Z = torch.log(Z)
        return a, b, log_Z

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_action(
        self, obs: torch.Tensor, *, explore_factor: Any = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action ∈ [-1, 1] via inverse-CDF reparameterization.

        Returns (action, log_prob) where log_prob is summed over dims.
        """
        mean, sigma = self.forward(obs, explore_factor=explore_factor)
        a, b, log_Z = self._trunc_params(mean, sigma)

        # Inverse-CDF sampling:
        #   u ~ Uniform(Φ(a), Φ(b))
        #   ε = Φ⁻¹(u)
        #   action = mean + σ × ε
        cdf_a = _std_normal_cdf(a)
        cdf_b = _std_normal_cdf(b)
        u = torch.rand_like(mean) * (cdf_b - cdf_a) + cdf_a
        eps = _std_normal_icdf(u)
        action = mean + sigma * eps
        # Numerical safety: clamp to [-1, 1]
        action = torch.clamp(action, _ACTION_LOW + 1e-6, _ACTION_HIGH - 1e-6)

        # log_prob = Normal.log_prob(action) - log(Z)
        z = (action - mean) / sigma
        log_prob = (-0.5 * z * z - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
                    - log_Z)
        return action, log_prob.sum(dim=-1)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return mean action (no sampling)."""
        mean, _ = self.forward(obs)
        return mean

    # ------------------------------------------------------------------
    # Evaluation (training-side)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        explore_factor: torch.Tensor,
        *,
        want_stats: bool = False,
    ) -> ActorEval:
        """Score actions and compute uncertainty for PPO.

        ``explore_factor`` is a ``(B,)`` tensor recording the per-frame
        exploration intensity used at rollout time.  log_prob uses
        effective σ (with explore scale) so the PPO importance ratio is
        correct.  uncertainty (U) uses policy σ (without explore
        scale) so it reflects the policy's own certainty.
        """
        # Single trunk+head pass yields mean and policy σ; effective σ is
        # the same tensor scaled per-frame by explore_factor.  This
        # satisfies P1-8 (no redundant forward for the U computation) by
        # construction rather than by reuse.
        mean, policy_sigma = self._head_forward(obs)
        eff_sigma = self.effective_sigma(policy_sigma, explore_factor)
        a, b, log_Z = self._trunc_params(mean, eff_sigma)

        # log_prob: effective σ
        actions_clamped = torch.clamp(
            actions, _ACTION_LOW + 1e-6, _ACTION_HIGH - 1e-6
        )
        z = (actions_clamped - mean) / eff_sigma
        log_prob = (-0.5 * z * z - torch.log(eff_sigma)
                    - 0.5 * math.log(2 * math.pi) - log_Z)
        log_prob = log_prob.sum(dim=-1)

        # Uncertainty U = 1 / (2 × peak), using policy σ (no explore scale)
        # mean ∈ (-1, 1) so peak is at x = mean
        # peak = 1 / (σ × √(2π) × Z)
        # U = σ × √(2π) × Z / 2
        _, _, log_Z_policy = self._trunc_params(mean, policy_sigma)
        Z_policy = torch.exp(log_Z_policy)
        U_per_dim = policy_sigma * _SQRT_2PI * Z_policy / _ACTION_WIDTH
        # Arithmetic mean over dims → (B,)
        uncertainty = U_per_dim.mean(dim=-1)
        # Contract guard: U ∈ [0, 1].  At very large σ the formula
        # slightly overshoots 1 (~4e-4 at σ≈2e4) because Z = Φ(b)−Φ(a)
        # subtracts two numbers both ≈1 in float32.  U > 1 is only
        # ever precision noise (super-uniform regime, where the floor
        # hinge is inactive anyway), so the clamp is identity in every
        # meaningful operating range.
        uncertainty = uncertainty.clamp(0.0, 1.0)

        stats: Optional[Dict[str, float]] = None
        if want_stats:
            with torch.no_grad():
                stats = {
                    "uncertainty": float(uncertainty.mean().item()),
                    "std_mean": float(policy_sigma.mean().item()),
                    "eff_std_mean": float(eff_sigma.mean().item()),
                    "std_min": float(policy_sigma.min().item()),
                    "std_max": float(policy_sigma.max().item()),
                    # Spread of σ across the batch — measures how much
                    # state-dependence is actually being used.  ~0 means
                    # the σ head is (near-)constant and the policy is
                    # behaving like the global-σ baseline.
                    "std_std": float(policy_sigma.std().item()),
                    "mean_abs": float(mean.abs().mean().item()),
                }

        return ActorEval(
            log_prob=log_prob,
            uncertainty=uncertainty,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Policy contract (deterministic default behaviour)
    # ------------------------------------------------------------------

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """Deterministic action — returns the mean of the truncated normal."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.deterministic_action(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    # ------------------------------------------------------------------
    # StochasticPolicy contract (sampling with exploration control)
    # ------------------------------------------------------------------

    def sample(
        self,
        observation: Any,
        *,
        explore_factor: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Stochastic action — sample from truncated normal with explore_factor."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.sample_action(
                obs_tensor, explore_factor=explore_factor,
            )
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if not want_extra or log_prob is None:
            return action_np, None
        return action_np, {
            "log_prob": float(log_prob.item()),
        }

    def to_blueprint(
        self, dest_path: Optional[str] = None,
    ) -> "PolicyBlueprint":
        """Export to a deployable, self-contained PolicyBlueprint.

        Writes ``model.pt`` + ``policy.py`` (self-contained, no imports
        from ``baseline.*``) + ``MANIFEST.json`` into ``dest_path`` and
        returns a blueprint that rebuilds the policy via the
        ``ExportedStateTruncNormPolicy`` class.  Same export contract as
        TruncatedNormalPolicy (P0-5/P0-6).
        """
        if dest_path is None:
            dest_path = tempfile.mkdtemp(prefix="policy_export_")
        policy_dir = Path(dest_path)
        policy_dir.mkdir(parents=True, exist_ok=True)

        # Save model payload
        state_dict = {
            k: v.detach().cpu() for k, v in self.state_dict().items()
        }
        payload = {
            "format_version": 1,
            "policy_class": "StateTruncatedNormalPolicy",
            "arch": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
            },
            # Legacy flat fields (kept for backward compat with old
            # loaders that read payload["obs_dim"] directly).
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "state_dict": state_dict,
            "state_dict_keys": sorted(state_dict.keys()),
        }
        torch.save(payload, policy_dir / "model.pt")

        # Self-contained policy.py from the template file — no
        # string-literal codegen, the template is a real .py file.
        policy_code = _build_export_policy_code()
        (policy_dir / "policy.py").write_text(policy_code, encoding="utf-8")

        manifest = {
            "format_version": 1,
            "policy_class": "StateTruncatedNormalPolicy",
            "arch": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
            },
            "files": ["model.pt", "policy.py", "MANIFEST.json"],
            "exported_class": "ExportedStateTruncNormPolicy",
        }
        (policy_dir / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )

        policy_py_path = policy_dir / "policy.py"
        return PolicyBlueprint(
            cls=f"file:{policy_py_path}:ExportedStateTruncNormPolicy",
        )
