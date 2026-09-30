"""Self-contained exported policy for StateBoundedStdTruncatedNormalPolicy.

This file is the source of the ``policy.py`` that ``to_blueprint``
writes into every export directory.  It is **self-contained** — it does
NOT import from ``baseline.*`` or any other repo module.  The only
dependencies are ``torch``, ``numpy``, ``math``, and the Python standard
library.  This means:

- Exported policies work without the repo on ``sys.path``.
- Internal refactoring (renaming modules, moving files) does not break
  historical artifacts.
- The export is suitable for benchmark/competition submission where the
  user may not have the repo at all.

The inference logic (state-dependent bounded-σ truncated normal
sampling, log_prob) is inlined from
``state_bounded_std_truncated_normal_mlp.py``:

    out = head(trunk(obs));  μ = tanh(out[:D]);  v = out[D:]
    σ(v) = exp(r_min + Δr·sigmoid(v)),   σ_e = σ(v + alpha·e)

A parity test verifies that the training-side
``StateBoundedStdTruncatedNormalPolicy`` and this exported version
produce bit-identical outputs on the same inputs.

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
    # Attribute-based activation check — NOT ctx.has_delta(): ``ctx`` may
    # be an instance of an older SamplingContext class held by a spawned
    # rollout worker; its fields exist but newer methods may not.
    if ctx is None:
        return sigma
    ref = getattr(ctx, "reference_action", None)
    dm = getattr(ctx, "delta_mix", None)
    if ref is None or dm is None:
        return sigma
    if not (bool((dm != 0).any()) if hasattr(dm, "any") else dm != 0):
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

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_2PI = 1.0 / _SQRT_2PI

_ACTION_LOW = -1.0
_ACTION_HIGH = 1.0
_ACTION_WIDTH = _ACTION_HIGH - _ACTION_LOW

_EI_TOLERANCE = 1e-6


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


def _check_ei(explore_factor: Any) -> None:
    """explore_factor must stay within [-1, 1] — no silent clamp."""
    if isinstance(explore_factor, torch.Tensor):
        if not bool(torch.isfinite(explore_factor).all()):
            raise ValueError("explore_factor contains non-finite values")
        bad = (explore_factor < -1.0 - _EI_TOLERANCE) | (
            explore_factor > 1.0 + _EI_TOLERANCE
        )
        if bool(bad.any()):
            raise ValueError(
                f"explore_factor out of [-1, 1]: "
                f"min={float(explore_factor.min())}, "
                f"max={float(explore_factor.max())}"
            )
    else:
        e = float(explore_factor)
        if not math.isfinite(e) or abs(e) > 1.0 + _EI_TOLERANCE:
            raise ValueError(
                f"explore_factor out of [-1, 1]: {explore_factor}"
            )


# ---------------------------------------------------------------------------
# Self-contained inference network
# ---------------------------------------------------------------------------


class _StateBoundedStdInferenceNet(nn.Module):
    """Inference-only state-dependent bounded-σ truncated normal policy.

    Self-contained copy of the inference subset of
    ``StateBoundedStdTruncatedNormalPolicy``.  Excludes
    ``evaluate_actions`` (training-only) and ``to_blueprint``
    (export-only).

    Head layout: ``[raw_mean | v]`` — (B, 2D) split into (B, D) each.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        sigma_min: float,
        sigma_max: float,
        explore_alpha: float,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, 2 * action_dim)

        self._r_min = math.log(float(sigma_min))
        self._r_max = math.log(float(sigma_max))
        self._delta_r = self._r_max - self._r_min
        self._sigma_min = float(sigma_min)
        self._sigma_max = float(sigma_max)
        self.explore_alpha = float(explore_alpha)
        self._gen = torch.Generator()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed this net's private RNG (per-episode reproducibility)."""
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def _bounded_sigma(self, v_e: torch.Tensor) -> torch.Tensor:
        """σ = exp(r_min + Δr · sigmoid(v_e))."""
        return torch.exp(
            self._r_min + self._delta_r * torch.sigmoid(v_e)
        )

    def _head_forward(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.head(self.trunk(obs))
        raw_mean, v = out.split(self.action_dim, dim=-1)
        return torch.tanh(raw_mean), v

    def policy_sigma(self, obs: torch.Tensor) -> torch.Tensor:
        """σ without explore shift — for uncertainty U."""
        _, v = self._head_forward(obs)
        return self._bounded_sigma(v)

    def effective_sigma(
        self, v: torch.Tensor, explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        """σ(v + alpha·e) — additive shift on the raw sigmoid input."""
        if isinstance(explore_factor, torch.Tensor):
            v_e = v + self.explore_alpha * explore_factor.unsqueeze(-1)
        else:
            v_e = v + self.explore_alpha * float(explore_factor)
        return self._bounded_sigma(v_e)

    def forward(self, obs: torch.Tensor, *, ctx: Optional["SamplingContext"] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ef = ctx.explore_factor if ctx is not None else 0.0
        _check_ei(ef)
        mean, v = self._head_forward(obs)
        sigma = self.effective_sigma(v, ef)
        return mean, delta_mix_sigma(
            mean, sigma, ctx, self._sigma_min, self._sigma_max,
        )

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
        uniform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ``uniform`` (B, D+1): column 0 is reserved for mixture-component
        # selection (unused here); columns 1: are the per-dim action draws
        # that replace the internal RNG — used by the batched inference
        # server so noise stays per-request regardless of batch
        # composition.  ``None`` (default) keeps the internal ``_gen``.
        mean, sigma = self.forward(obs, ctx=ctx)
        a, b, log_Z = self._trunc_params(mean, sigma)
        cdf_a = _std_normal_cdf(a)
        cdf_b = _std_normal_cdf(b)
        u_raw = (
            uniform[:, 1:]
            if uniform is not None
            else torch.rand(
                mean.shape,
                generator=self._gen,
                device=mean.device,
                dtype=mean.dtype,
            )
        )
        u = u_raw * (cdf_b - cdf_a) + cdf_a
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
_DISTRIBUTION_KIND = "bounded_std_diagonal_truncated_normal_v1"
_STD_PARAMETERIZATION = "sigmoid_log_std_v1"
_EXPLORATION_KIND = "raw_std_additive_shift_v1"
_STD_SOURCE = "state"


class ExportedStateBoundedStdTruncNormPolicy(Policy, StochasticPolicy):
    """Runtime-loadable policy backed by a ``model.pt`` checkpoint.

    Implements both ``Policy`` (deterministic ``act()``) and
    ``StochasticPolicy`` (sampling ``sample()``).

    Loading is strict (``strict=True``) and validates
    ``format_version``, ``policy_class``, the distribution/kind
    metadata, and the bounded-σ config before attempting to load —
    missing config fails loud, no silent defaults.
    """
    # Capability flag read by SamplingPolicy at wrap time — this cell
    # implements the reference-delta σ mix (delta_mix_sigma above).
    SUPPORTS_REFERENCE_DELTA = True


    def __init__(self, model_path: Optional[str] = None):
        payload_path = (
            Path(model_path) if model_path is not None
            else Path(__file__).resolve().parent / "model.pt"
        )
        payload = torch.load(payload_path, map_location="cpu")

        fv = payload.get("format_version", 0)
        if fv != _EXPORT_FORMAT_VERSION:
            raise RuntimeError(
                f"Policy export format version mismatch: "
                f"file has {fv}, loader expects {_EXPORT_FORMAT_VERSION}. "
                f"This export was created by a different version of "
                f"the framework. Re-export the policy with the current code."
            )
        pcls = payload.get("policy_class", "unknown")
        if pcls != "StateBoundedStdTruncatedNormalPolicy":
            raise RuntimeError(
                f"Policy class mismatch: file says {pcls!r}, "
                f"loader expects 'StateBoundedStdTruncatedNormalPolicy'."
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
        for key in ("sigma_min", "sigma_max", "explore_alpha"):
            val = payload.get(key)
            if val is None or not math.isfinite(float(val)):
                raise RuntimeError(
                    f"missing or non-finite {key} in export payload: {val!r}"
                )
        sigma_min = float(payload["sigma_min"])
        sigma_max = float(payload["sigma_max"])
        explore_alpha = float(payload["explore_alpha"])
        if not (0.0 < sigma_min < sigma_max):
            raise RuntimeError(
                f"invalid σ bounds: sigma_min={sigma_min} "
                f"sigma_max={sigma_max}"
            )
        if explore_alpha <= 0.0:
            raise RuntimeError(
                f"invalid explore_alpha: {explore_alpha}"
            )
        arch = payload.get("arch", {})
        obs_dim = int(arch.get("obs_dim", payload.get("obs_dim", 0)))
        action_dim = int(arch.get("action_dim", payload.get("action_dim", 0)))
        hidden_dim = int(arch.get("hidden_dim", payload.get("hidden_dim", 0)))

        self._policy = _StateBoundedStdInferenceNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            explore_alpha=explore_alpha,
        )
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
