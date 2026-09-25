"""Self-contained exported policy template for
StateMixtureBoundedStdTruncatedNormalPolicy.

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

The inference logic (K-component truncated-normal mixture, shared
component sampling, erf-space inverse-CDF, mixture log_prob, bounded
sigmoid σ map with v+αe explore) is inlined from
``state_mixture_bounded_std_truncated_normal_mlp.py``.  A parity test
in ``test_state_mixture_bounded_std_truncated_normal.py`` verifies
that the training-side policy and this exported version produce
identical outputs on the same inputs.

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
    """Minimal Policy ABC stub for the exported policy."""

    def act(self, observation: Any, *, want_extra: bool = False) -> Tuple[np.ndarray, Any]:
        raise NotImplementedError


class StochasticPolicy:
    """Minimal StochasticPolicy stub for the exported policy."""

    def sample(
        self,
        observation: Any,
        *,
        explore_factor: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Truncated-normal mixture math (inlined from the mixture policies)
# ---------------------------------------------------------------------------

_SQRT_2 = math.sqrt(2.0)
_LOG_2PI_HALF = 0.5 * math.log(2.0 * math.pi)

_ACTION_LOW = -1.0
_ACTION_HIGH = 1.0

_ERFINV_EPS = 1e-7
_ACTION_EPS = 1e-6

_EI_TOLERANCE = 1e-6            # float32 slack on the [-1, 1] ei range


# ---------------------------------------------------------------------------
# Self-contained inference network
# ---------------------------------------------------------------------------


class _StateMixtureBoundedStdInferenceNet(nn.Module):
    """Inference-only K-component mixture with state-dependent bounded σ.

    Self-contained copy of the inference subset of
    ``StateMixtureBoundedStdTruncatedNormalPolicy`` — excludes
    ``evaluate_actions`` (training-only) and ``to_blueprint``
    (export-only).  State-dict keys are identical to the training side
    (``trunk.*``, ``head.*``).

    Head layout: ``logits (K) | raw_mean (K·D) | raw_v (K·D)``,
    component-major; σ is the bounded map of the per-state ``v`` block.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_components: int,
        sigma_min: float,
        sigma_max: float,
        explore_alpha: float,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_components = int(num_components)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        K, D = self.num_components, self.action_dim
        self.head = nn.Linear(hidden_dim, K + 2 * K * D)

        self._r_min = math.log(float(sigma_min))
        self._delta_r = math.log(float(sigma_max)) - self._r_min
        self._explore_alpha = float(explore_alpha)
        self._gen = torch.Generator()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed this net's private RNG (per-episode reproducibility)."""
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def _bounded_sigma(self, v_e: torch.Tensor) -> torch.Tensor:
        """σ = exp(r_min + Δr · sigmoid(v_e))  ∈ (σ_min, σ_max)."""
        return torch.exp(
            self._r_min + self._delta_r * torch.sigmoid(v_e)
        )

    @staticmethod
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

    def _head_forward(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """→ (log_pi, mean, v) — v is the head's raw σ-control block."""
        K, D = self.num_components, self.action_dim
        out = self.head(self.trunk(obs))
        logits = out[..., :K]
        raw_mean = out[..., K:K + K * D].reshape(-1, K, D)
        v = out[..., K + K * D:].reshape(-1, K, D)
        log_pi = torch.log_softmax(logits, dim=-1)
        mean = torch.tanh(raw_mean)
        return log_pi, mean, v

    def _effective_sigma(
        self, obs_v: torch.Tensor, explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        """σ(v + αe) — additive shift on the raw sigmoid input."""
        v = obs_v
        if isinstance(explore_factor, torch.Tensor):
            v_e = v + self._explore_alpha * explore_factor.view(-1, 1, 1)
        else:
            v_e = v + self._explore_alpha * float(explore_factor)
        return self._bounded_sigma(v_e)

    @staticmethod
    def _log_trunc_Z(
        mean: torch.Tensor, sigma: torch.Tensor,
    ) -> torch.Tensor:
        erf_hi = torch.erf((_ACTION_HIGH - mean) / (_SQRT_2 * sigma))
        erf_lo = torch.erf((_ACTION_HIGH + mean) / (_SQRT_2 * sigma))
        Z = 0.5 * (erf_hi + erf_lo)
        Z = torch.clamp_min(Z, torch.finfo(Z.dtype).tiny)
        return torch.log(Z)

    def _mixture_log_prob(
        self,
        actions: torch.Tensor,
        mean: torch.Tensor,
        sigma: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        a = actions.unsqueeze(1)
        log_Z = self._log_trunc_Z(mean, sigma)
        z = (a - mean) / sigma
        comp_lp = (
            -0.5 * z * z - torch.log(sigma) - _LOG_2PI_HALF - log_Z
        ).sum(dim=-1)
        return torch.logsumexp(comp_lp + log_pi, dim=-1)

    def sample_action(
        self, obs: torch.Tensor, *, explore_factor: Any = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_ei(explore_factor)
        log_pi, mean, v = self._head_forward(obs)
        sigma = self._effective_sigma(v, explore_factor)
        B, K, D = mean.shape

        if K == 1:
            idx = torch.zeros(B, dtype=torch.long, device=mean.device)
        else:
            idx = torch.multinomial(
                log_pi.exp(), 1, generator=self._gen,
            ).squeeze(-1)
        sel = idx.view(-1, 1, 1).expand(-1, 1, D)
        mu_k = mean.gather(1, sel).squeeze(1)
        sg_k = sigma.gather(1, sel).squeeze(1)

        erf_a = torch.erf((_ACTION_LOW - mu_k) / (_SQRT_2 * sg_k))
        erf_b = torch.erf((_ACTION_HIGH - mu_k) / (_SQRT_2 * sg_k))
        u = torch.rand(
            mu_k.shape,
            generator=self._gen,
            device=mu_k.device,
            dtype=mu_k.dtype,
        )
        q = (1.0 - u) * erf_a + u * erf_b
        q = q.clamp(-1.0 + _ERFINV_EPS, 1.0 - _ERFINV_EPS)
        action = mu_k + _SQRT_2 * sg_k * torch.erfinv(q)
        action = torch.clamp(
            action, _ACTION_LOW + _ACTION_EPS, _ACTION_HIGH - _ACTION_EPS,
        )

        log_prob = self._mixture_log_prob(action, mean, sigma, log_pi)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        log_pi, mean, _ = self._head_forward(obs)
        idx = log_pi.argmax(dim=-1)
        sel = idx.view(-1, 1, 1).expand(-1, 1, self.action_dim)
        return mean.gather(1, sel).squeeze(1)


# ---------------------------------------------------------------------------
# Exported policy class (loaded by PolicyBlueprint at runtime)
# ---------------------------------------------------------------------------

_EXPORT_FORMAT_VERSION = 1
_DISTRIBUTION_KIND = "bounded_std_mixture_truncated_normal_v1"
_STD_PARAMETERIZATION = "sigmoid_log_std_v1"
_EXPLORATION_KIND = "raw_std_additive_shift_v1"
_STD_SOURCE = "state"


class ExportedStateMixtureBoundedStdTruncNormPolicy(Policy, StochasticPolicy):
    """Runtime-loadable policy backed by a ``model.pt`` checkpoint.

    Implements both ``Policy`` (deterministic ``act()`` → highest-weight
    component mean) and ``StochasticPolicy`` (mixture ``sample()``).

    Loading is strict (``strict=True``) and validates
    ``format_version``, ``policy_class``, the distribution-identity
    metadata fields, the bounded-σ config (``sigma_min``/``sigma_max``/
    ``explore_alpha``), and ``num_components`` before attempting to load.
    """

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
        if pcls != "StateMixtureBoundedStdTruncatedNormalPolicy":
            raise RuntimeError(
                f"Policy class mismatch: file says {pcls!r}, "
                f"loader expects 'StateMixtureBoundedStdTruncatedNormalPolicy'."
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
        num_components = int(
            arch.get("num_components", payload.get("num_components", 0))
        )
        if num_components < 1:
            raise RuntimeError(
                f"Invalid num_components in payload: {num_components}"
            )

        self._policy = _StateMixtureBoundedStdInferenceNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_components=num_components,
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
        """Deterministic action — highest-weight component mean."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self._policy.deterministic_action(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def sample(
        self,
        observation: Any,
        *,
        explore_factor: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Stochastic action — mixture sample with explore_factor."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self._policy.sample_action(
                obs_tensor, explore_factor=explore_factor,
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
