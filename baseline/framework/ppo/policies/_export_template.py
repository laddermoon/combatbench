"""Self-contained exported policy template for TruncatedNormalPolicy.

P0-6: This file is the source of the ``policy.py`` that ``to_blueprint``
writes into every export directory.  It is **self-contained** — it does
NOT import from ``baseline.*`` or any other repo module.  The only
dependencies are ``torch``, ``numpy``, ``math``, and the Python standard
library.  This means:

- Exported policies work without the repo on ``sys.path``.
- Internal refactoring (renaming modules, moving files) does not break
  historical artifacts.
- The export is suitable for benchmark/competition submission where the
  user may not have the repo at all.

The inference logic (truncated normal sampling, log_prob, uncertainty)
is inlined from ``truncated_normal_mlp.py``.  A parity test in
``test_truncated_normal.py`` verifies that the training-side
``TruncatedNormalPolicy`` and this exported version produce
bit-identical outputs on the same inputs.

This file is lint-able, type-checkable, and testable on its own.  Do not
add imports from ``baseline.*`` or ``envs.*`` — that would re-introduce
the P0-6 bug.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Environment protocol (minimal inline stubs — no repo import needed)
# ---------------------------------------------------------------------------


class Policy:
    """Minimal Policy ABC stub for the exported policy.

    The real ``envs.framework.policy.Policy`` is an ABC with ``act()``.
    We inline a no-op base here so the export does not depend on the
    repo.  At runtime, the framework's ``PolicyBlueprint`` loader
    instantiates this class directly — it does not need to be the same
    object as the repo's ``Policy``.
    """

    def act(self, observation: Any, *, want_extra: bool = False) -> Tuple[np.ndarray, Any]:
        raise NotImplementedError


class SamplingContext:
    """Self-contained sampling context — mirrors the framework's
    ``baseline.framework.ppo.sampling_context.SamplingContext`` so this
    file needs no repo import.  Fields are per-frame values (scalars or
    arrays); the exported policies consume ``explore_factor`` plus the
    ``reference_action``/``delta_factor``/``delta_mix`` fields.

    A plain class (not a dataclass): this file is exec-loaded without a
    real module entry, so ``from __future__ import annotations`` +
    ``@dataclass`` would fail to resolve string annotations.
    """

    __slots__ = (
        "explore_factor", "reference_action", "delta_factor", "delta_mix",
    )

    def __init__(self, explore_factor=None, reference_action=None,
                 delta_factor=None, delta_mix=None):
        self.explore_factor = explore_factor
        self.reference_action = reference_action
        self.delta_factor = delta_factor
        self.delta_mix = delta_mix

    def has_delta(self) -> bool:
        """True iff the reference-delta scale is active — mirrors the
        upstream ``SamplingContext.has_delta``."""
        if self.reference_action is None:
            return False
        dm = self.delta_mix
        if dm is None:
            return False
        if hasattr(dm, "any"):  # ndarray / torch.Tensor
            return bool((dm != 0).any())
        return dm != 0


#: σ-domain floor for the reference-delta scale (σ_delta² ≥ ε²).
_DELTA_EPS = 1e-2


def _ctx_bcast(x: Any, target: torch.Tensor) -> Any:
    """ctx scalar field → float, or a (B,) tensor → broadcastable."""
    if not torch.is_tensor(x):
        return float(x)
    while x.ndim < target.ndim:
        x = x.unsqueeze(-1)
    return x


def delta_mix_sigma(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    ctx: Optional["SamplingContext"],
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
) -> torch.Tensor:
    """σ²-domain mix of the policy σ with the reference-delta scale —
    inlined copy of ``truncated_normal_mlp.delta_mix_sigma``."""
    if ctx is None or not ctx.has_delta():
        return sigma
    ref = torch.as_tensor(
        ctx.reference_action, dtype=sigma.dtype, device=sigma.device,
    )
    while ref.ndim < mean.ndim:
        ref = ref.unsqueeze(-2)
    c = _ctx_bcast(ctx.delta_factor, sigma)
    lam = _ctx_bcast(ctx.delta_mix, sigma)
    delta2 = (c * (mean - ref)).pow(2).clamp_min(_DELTA_EPS ** 2)
    mixed = torch.sqrt((1.0 - lam) * sigma.pow(2) + lam * delta2)
    if sigma_min is not None:
        mixed = mixed.clamp(sigma_min, sigma_max)
    return mixed


class StochasticPolicy:
    """Minimal StochasticPolicy stub for the exported policy."""

    def sample(
        self,
        observation: Any,
        *,
        ctx: Optional["SamplingContext"] = None,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Truncated normal math (inlined from truncated_normal_mlp.py)
# ---------------------------------------------------------------------------

_EXPLORE_K = math.log(3.0)

_LOG_STD_SAFE_MIN = -20.0
_LOG_STD_SAFE_MAX = 20.0

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_2PI = 1.0 / _SQRT_2PI

_ACTION_LOW = -1.0
_ACTION_HIGH = 1.0
_ACTION_WIDTH = _ACTION_HIGH - _ACTION_LOW


def _std_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, differentiable via erf."""
    return 0.5 * (1.0 + torch.erf(x / _SQRT_2))


def _std_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF, differentiable."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _std_normal_icdf(u: torch.Tensor) -> torch.Tensor:
    """Standard normal inverse CDF, differentiable via erfinv."""
    u_clamped = torch.clamp(u, 1e-6, 1.0 - 1e-6)
    return _SQRT_2 * torch.erfinv(2.0 * u_clamped - 1.0)


# ---------------------------------------------------------------------------
# Self-contained inference network
# ---------------------------------------------------------------------------


class _TruncNormalInferenceNet(nn.Module):
    """Inference-only truncated normal policy on [-1, 1].

    This is a self-contained copy of the inference subset of
    ``TruncatedNormalPolicy``.  It excludes ``evaluate_actions`` (which
    is training-only) and ``to_blueprint`` (which is export-only).

    The parity test guarantees that for the same ``state_dict`` and
    inputs, this net produces bit-identical outputs to the training-side
    ``TruncatedNormalPolicy``.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

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
        self._gen = torch.Generator()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed this net's private RNG (per-episode reproducibility)."""
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def effective_log_std(self) -> torch.Tensor:
        return torch.clamp(self.log_std, _LOG_STD_SAFE_MIN, _LOG_STD_SAFE_MAX)

    def _explore_scale(self, explore_factor: Any = 0.0) -> Any:
        if isinstance(explore_factor, torch.Tensor):
            return torch.exp(explore_factor * _EXPLORE_K)
        return math.exp(float(explore_factor) * _EXPLORE_K)

    def effective_sigma(self, explore_factor: Any = 0.0) -> torch.Tensor:
        scale = self._explore_scale(explore_factor)
        sigma = self.effective_log_std().exp()
        if isinstance(scale, torch.Tensor):
            return sigma * scale.unsqueeze(-1)
        return sigma * scale

    def forward(self, obs: torch.Tensor, *, ctx: Optional["SamplingContext"] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ef = ctx.explore_factor if ctx is not None else 0.0
        raw_mean = self.net(obs)
        mean = torch.tanh(raw_mean)
        sigma = self.effective_sigma(ef)
        sigma = delta_mix_sigma(mean, sigma, ctx)
        return mean, sigma.expand_as(mean)

    def _trunc_params(
        self, mean: torch.Tensor, sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = (_ACTION_LOW - mean) / sigma
        b = (_ACTION_HIGH - mean) / sigma
        cdf_b = _std_normal_cdf(b)
        cdf_a = _std_normal_cdf(a)
        Z = cdf_b - cdf_a
        Z = torch.clamp(Z, min=1e-8)
        log_Z = torch.log(Z)
        return a, b, log_Z

    def sample_action(
        self, obs: torch.Tensor, *, ctx: Optional["SamplingContext"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, sigma = self.forward(obs, ctx=ctx)
        a, b, log_Z = self._trunc_params(mean, sigma)
        cdf_a = _std_normal_cdf(a)
        cdf_b = _std_normal_cdf(b)
        u = (
            torch.rand(
                mean.shape,
                generator=self._gen,
                device=mean.device,
                dtype=mean.dtype,
            )
            * (cdf_b - cdf_a)
            + cdf_a
        )
        eps = _std_normal_icdf(u)
        action = mean + sigma * eps
        action = torch.clamp(action, _ACTION_LOW + 1e-6, _ACTION_HIGH - 1e-6)
        z = (action - mean) / sigma
        log_prob = (-0.5 * z * z - torch.log(sigma) - 0.5 * math.log(2 * math.pi) - log_Z)
        return action, log_prob.sum(dim=-1)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return mean


# ---------------------------------------------------------------------------
# Exported policy class (loaded by PolicyBlueprint at runtime)
# ---------------------------------------------------------------------------

_EXPORT_FORMAT_VERSION = 1
_DISTRIBUTION_KIND = "diagonal_truncated_normal_v1"
_STD_PARAMETERIZATION = "log_std_v1"
_EXPLORATION_KIND = "log_std_multiplicative_v1"
_STD_SOURCE = "shared"


class ExportedTruncNormPolicy(Policy, StochasticPolicy):
    """Runtime-loadable policy backed by a ``model.pt`` checkpoint.

    Implements both ``Policy`` (deterministic ``act()``) and
    ``StochasticPolicy`` (sampling ``sample()``) so it can be used:
    - As a ``Policy`` for deployment / competition / eval (``act()`` -> mean).
    - As a ``StochasticPolicy`` for training rollouts (``sample()``).

    Whether the policy is used stochastically or deterministically is
    controlled by the ``Job.stochastic`` flag at rollout time, not by
    this class.

    P0-5: Loading is strict (``strict=True``) and validates
    ``format_version`` + ``policy_class`` before attempting to load.
    P0-6: This module is self-contained — no imports from ``baseline.*``.
    """

    def __init__(self, model_path: Optional[str] = None):
        payload_path = (
            Path(model_path) if model_path is not None
            else Path(__file__).resolve().parent / "model.pt"
        )
        payload = torch.load(payload_path, map_location="cpu")

        # P0-5: Validate format version and policy class before loading.
        fv = payload.get("format_version", 0)
        if fv != _EXPORT_FORMAT_VERSION:
            raise RuntimeError(
                f"Policy export format version mismatch: "
                f"file has {fv}, loader expects {_EXPORT_FORMAT_VERSION}. "
                f"This export was created by a different version of "
                f"the framework. Re-export the policy with the current code."
            )
        pcls = payload.get("policy_class", "unknown")
        if pcls != "TruncatedNormalPolicy":
            raise RuntimeError(
                f"Policy class mismatch: file says {pcls!r}, "
                f"loader expects 'TruncatedNormalPolicy'."
            )
        for key, expected in (
            ("distribution_kind", _DISTRIBUTION_KIND),
            ("std_parameterization", _STD_PARAMETERIZATION),
            ("exploration_kind", _EXPLORATION_KIND),
            ("std_source", _STD_SOURCE),
        ):
            got = payload.get(key, "missing")
            if got != expected:
                raise RuntimeError(
                    f"{key} mismatch: file says {got!r}, "
                    f"loader expects {expected!r}."
                )
        arch = payload.get("arch", {})
        obs_dim = int(arch.get("obs_dim", payload.get("obs_dim", 0)))
        action_dim = int(arch.get("action_dim", payload.get("action_dim", 0)))
        hidden_dim = int(arch.get("hidden_dim", payload.get("hidden_dim", 0)))

        self._policy = _TruncNormalInferenceNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        # P0-5: strict=True so missing or unexpected keys raise
        # RuntimeError instead of silently producing a partially-random
        # policy.
        self._policy.load_state_dict(payload["state_dict"], strict=True)
        self._policy.eval()

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """Deterministic action — returns the mean (Policy interface)."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self._policy.deterministic_action(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def sample(
        self,
        observation: Any,
        *,
        ctx: Optional["SamplingContext"] = None,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Stochastic action — sample from truncated normal (StochasticPolicy interface)."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self._policy.sample_action(
                obs_tensor, ctx=ctx,
            )
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if not want_extra or log_prob is None:
            return action_np, None
        return action_np, {"log_prob": float(log_prob.item())}

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed the inference net's private RNG for reproducible
        rollouts (per-policy stream, independent of other agents)."""
        self._policy.reset(seed)
        return None
