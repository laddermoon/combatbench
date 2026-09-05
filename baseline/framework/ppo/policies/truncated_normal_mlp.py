"""TruncatedNormalPolicy — truncated normal distribution on action space.

Distribution is defined directly on [-1, 1] via a truncated normal,
eliminating the pre-tanh / tanh-transform indirection of
TanhGaussianMLPPolicy.

See DESIGN_truncated_normal.md for the full design rationale.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from envs.framework.policy import Policy, PolicyBlueprint

from baseline.framework.ppo import ActorEval

# explore_intensity ∈ [-1, 1]: 0 = neutral, +1 = max explore, -1 = max suppress.
# Mapping: scale = exp(ei * ln(3)), so ei=0→1, ei=+1→3, ei=-1→1/3.
_EXPLORE_K = math.log(3.0)

__all__ = [
    "TruncatedNormalPolicy",
]


def _build_export_policy_code() -> str:
    """Return the source of the ``policy.py`` embedded in export dirs.

    The produced module defines ``ExportedTruncNormPolicy`` that reuses
    :class:`TruncatedNormalPolicy` from the repo.  Requires the repo
    to be on ``sys.path`` (e.g., via PYTHONPATH=. when running).
    """
    return '''"""Policy module - imports from repo to reuse TruncatedNormalPolicy."""
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch

# Import from repo - requires baseline/ to be on sys.path
from baseline.framework.ppo.policies.truncated_normal_mlp import TruncatedNormalPolicy
from envs.framework.policy import Policy


class ExportedTruncNormPolicy(Policy):
    """Runtime-loadable policy backed by a ``model.pt`` checkpoint.

    Uses :class:`TruncatedNormalPolicy` from the training repo for
    consistent architecture and behavior.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        stochastic: bool = False,
        **_ignored: Any,
    ):
        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "model.pt"
        payload = torch.load(payload_path, map_location="cpu")

        self._policy = TruncatedNormalPolicy(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
        )
        self._policy.load_state_dict(payload["state_dict"], strict=False)
        self._policy.eval()
        self.stochastic = bool(stochastic)

    def act(
        self,
        observation: Any,
        explore_intensity: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """Return action for given observation."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.stochastic:
                action, _ = self._policy.sample_action(
                    obs_tensor, explore_intensity=explore_intensity,
                )
            else:
                action = self._policy.deterministic_action(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def reset(self, seed: Optional[int] = None) -> None:
        """Optional: reseed RNG for reproducible rollouts."""
        if seed is not None:
            torch.manual_seed(seed)
        return None


# Backward compatibility alias
Policy = ExportedTruncNormPolicy
'''

# Numerical safety bounds for log_std.
_LOG_STD_SAFE_MIN = -20.0  # exp(-20) ≈ 2e-9
_LOG_STD_SAFE_MAX = 20.0   # exp(20) ≈ 5e8

# Constants for standard normal CDF / PDF.
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_2PI = 1.0 / _SQRT_2PI

# Action space bounds (hardcoded for humanoid21: [-1, 1]).
_ACTION_LOW = -1.0
_ACTION_HIGH = 1.0
_ACTION_WIDTH = _ACTION_HIGH - _ACTION_LOW  # = 2.0


def _std_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, differentiable via erf."""
    return 0.5 * (1.0 + torch.erf(x / _SQRT_2))


def _std_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF, differentiable."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _std_normal_icdf(u: torch.Tensor) -> torch.Tensor:
    """Standard normal inverse CDF, differentiable via erfinv.

    u must be in (0, 1).  Clamped for numerical safety.
    """
    u_clamped = torch.clamp(u, 1e-6, 1.0 - 1e-6)
    return _SQRT_2 * torch.erfinv(2.0 * u_clamped - 1.0)


class TruncatedNormalPolicy(nn.Module, Policy):
    """Truncated normal policy on [-1, 1].

    mean = tanh(net(obs))  ∈ (-1, 1)
    σ    = exp(log_std)    > 0  (global parameter, per-dim)

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
        deterministic: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.device = torch.device(device)
        self._deterministic = bool(deterministic)

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

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    def effective_log_std(self) -> torch.Tensor:
        """log_std with numerical safety clamp (no business bounds)."""
        return torch.clamp(
            self.log_std, _LOG_STD_SAFE_MIN, _LOG_STD_SAFE_MAX
        )

    def _explore_scale(self, explore_intensity: Any = 0.0) -> Any:
        """Exponential σ scaling factor from explore_intensity.

        ei=0 → 1.0 (neutral), ei=+1 → 3.0 (max explore), ei=-1 → 1/3 (max suppress).
        scale = exp(ei * ln(3)).  Accepts scalar float or (B,) tensor.
        """
        if isinstance(explore_intensity, torch.Tensor):
            return torch.exp(explore_intensity * _EXPLORE_K)
        return math.exp(float(explore_intensity) * _EXPLORE_K)

    def effective_sigma(self, explore_intensity: Any = 0.0) -> torch.Tensor:
        """σ used for sampling / log_prob (includes explore scale)."""
        scale = self._explore_scale(explore_intensity)
        sigma = self.effective_log_std().exp()
        if isinstance(scale, torch.Tensor):
            return sigma * scale.unsqueeze(-1)  # (B, 1) * (action_dim,) → (B, action_dim)
        return sigma * scale

    def policy_sigma(self) -> torch.Tensor:
        """σ without explore scale — for uncertainty U."""
        return self.effective_log_std().exp()

    def forward(self, obs: torch.Tensor, *, explore_intensity: Any = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, effective_sigma), both (B, action_dim) or broadcastable."""
        raw_mean = self.net(obs)
        mean = torch.tanh(raw_mean)  # ensure mean ∈ (-1, 1)
        sigma = self.effective_sigma(explore_intensity)
        return mean, sigma.expand_as(mean)

    def _trunc_params(
        self, mean: torch.Tensor, sigma: torch.Tensor
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
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action ∈ [-1, 1] via inverse-CDF reparameterization.

        Returns (action, log_prob) where log_prob is summed over dims.
        """
        mean, sigma = self.forward(obs, explore_intensity=explore_intensity)
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
        explore_intensity: torch.Tensor,
        *,
        want_stats: bool = False,
    ) -> ActorEval:
        """Score actions and compute uncertainty for PPO.

        ``explore_intensity`` is a ``(B,)`` tensor recording the per-frame
        exploration intensity used at rollout time.  log_prob uses
        effective σ (with explore scale) so the PPO importance ratio is
        correct.  entropy (uncertainty U) uses policy σ (without explore
        scale) so it reflects the policy's own certainty.
        """
        mean, eff_sigma = self.forward(obs, explore_intensity=explore_intensity)
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
        policy_sigma = self.policy_sigma()
        policy_mean = torch.tanh(self.net(obs))  # recompute without explore
        # mean ∈ (-1, 1) so peak is at x = mean
        # peak = 1 / (σ × √(2π) × Z)
        # U = σ × √(2π) × Z / 2
        _, _, log_Z_policy = self._trunc_params(policy_mean, policy_sigma)
        Z_policy = torch.exp(log_Z_policy)
        U_per_dim = policy_sigma * _SQRT_2PI * Z_policy / _ACTION_WIDTH
        # Arithmetic mean over dims → (B,)
        uncertainty = U_per_dim.mean(dim=-1)

        stats: Optional[Dict[str, float]] = None
        if want_stats:
            with torch.no_grad():
                stats = {
                    "uncertainty": float(uncertainty.mean().item()),
                    "std_mean": float(policy_sigma.mean().item()),
                    "eff_std_mean": float(eff_sigma.mean().item()),
                    "std_min": float(policy_sigma.min().item()),
                    "std_max": float(policy_sigma.max().item()),
                    "mean_abs": float(policy_mean.abs().mean().item()),
                }

        return ActorEval(
            log_prob=log_prob,
            entropy=uncertainty,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Policy contract
    # ------------------------------------------------------------------

    def act(
        self,
        observation: Any,
        explore_intensity: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        action_np, log_prob = self.act_numpy(
            observation, device=self.device, deterministic=self._deterministic,
            explore_intensity=explore_intensity,
        )
        if not want_extra or log_prob is None:
            return action_np, None
        return action_np, {
            "log_prob": float(log_prob),
            "explore_intensity": float(explore_intensity),
        }

    def set_deterministic(self, deterministic: bool) -> None:
        self._deterministic = bool(deterministic)

    def to_blueprint(
        self, dest_path: Optional[str] = None, *, stochastic: bool = False,
    ) -> "PolicyBlueprint":
        """Export to a deployable PolicyBlueprint.

        Writes ``model.pt`` + ``policy.py`` (standalone, imports
        TruncatedNormalPolicy from repo) into ``dest_path`` and returns
        a blueprint that rebuilds the policy via the generated
        ``ExportedTruncNormPolicy`` class.
        """
        import tempfile

        if dest_path is None:
            dest_path = tempfile.mkdtemp(prefix="policy_export_")
        policy_dir = Path(dest_path)
        policy_dir.mkdir(parents=True, exist_ok=True)

        # Save model payload
        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "state_dict": {
                k: v.detach().cpu() for k, v in self.state_dict().items()
            },
        }
        torch.save(payload, policy_dir / "model.pt")

        # Generate standalone policy.py that imports from repo
        policy_code = _build_export_policy_code()
        (policy_dir / "policy.py").write_text(policy_code, encoding="utf-8")

        policy_py_path = policy_dir / "policy.py"
        return PolicyBlueprint(
            cls=f"file:{policy_py_path}:ExportedTruncNormPolicy",
            config={"stochastic": stochastic},
        )

    # ------------------------------------------------------------------
    # Numpy inference (for rollout workers)
    # ------------------------------------------------------------------

    def act_numpy(
        self, obs: np.ndarray, device: torch.device, deterministic: bool,
        *, explore_intensity: Any = 0.0,
    ) -> tuple[np.ndarray, Optional[float]]:
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.deterministic_action(obs_tensor)
                log_prob = None
            else:
                action, log_prob = self.sample_action(
                    obs_tensor, explore_intensity=explore_intensity,
                )
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if log_prob is None:
            return action_np, None
        return action_np, float(log_prob.item())
