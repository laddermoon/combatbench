"""Self-contained exported policy template for MixtureTruncatedNormalPolicy.

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
component sampling, erf-space inverse-CDF, mixture log_prob) is inlined
from ``mixture_truncated_normal_mlp.py``.  A parity test in
``test_mixture_truncated_normal.py`` verifies that the training-side
``MixtureTruncatedNormalPolicy`` and this exported version produce
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
# Truncated-normal mixture math (inlined from mixture_truncated_normal_mlp.py)
# ---------------------------------------------------------------------------

_EXPLORE_K = math.log(3.0)

_LOG_STD_SAFE_MIN = -20.0
_LOG_STD_SAFE_MAX = 20.0

_SQRT_2 = math.sqrt(2.0)
_LOG_2PI_HALF = 0.5 * math.log(2.0 * math.pi)

_ACTION_LOW = -1.0
_ACTION_HIGH = 1.0

_ERFINV_EPS = 1e-7
_ACTION_EPS = 1e-6


# ---------------------------------------------------------------------------
# Self-contained inference network
# ---------------------------------------------------------------------------


class _MixtureTruncNormInferenceNet(nn.Module):
    """Inference-only K-component truncated-normal mixture on [-1, 1].

    Self-contained copy of the inference subset of
    ``MixtureTruncatedNormalPolicy`` — excludes ``evaluate_actions``
    (training-only) and ``to_blueprint`` (export-only).  State-dict keys
    are identical to the training side (``trunk.*``, ``head.*``).

    Head layout: ``logits (K) | raw_mean (K·D) | raw_log_std (K·D)``,
    component-major.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_components: int,
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
        self._gen = torch.Generator()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed this net's private RNG (per-episode reproducibility)."""
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def _head_forward(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K, D = self.num_components, self.action_dim
        out = self.head(self.trunk(obs))
        logits = out[..., :K]
        raw_mean = out[..., K:K + K * D].reshape(-1, K, D)
        raw_log_std = out[..., K + K * D:].reshape(-1, K, D)
        log_pi = torch.log_softmax(logits, dim=-1)
        mean = torch.tanh(raw_mean)
        sigma = torch.clamp(
            raw_log_std, _LOG_STD_SAFE_MIN, _LOG_STD_SAFE_MAX,
        ).exp()
        return log_pi, mean, sigma

    def _explore_scale(self, explore_factor: Any = 0.0) -> Any:
        if isinstance(explore_factor, torch.Tensor):
            return torch.exp(explore_factor * _EXPLORE_K)
        return math.exp(float(explore_factor) * _EXPLORE_K)

    def _effective_sigma(
        self, policy_sigma: torch.Tensor, explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        scale = self._explore_scale(explore_factor)
        if isinstance(scale, torch.Tensor):
            return policy_sigma * scale.view(-1, 1, 1)
        return policy_sigma * scale

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
        log_pi, mean, policy_sigma = self._head_forward(obs)
        sigma = self._effective_sigma(policy_sigma, explore_factor)
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


class ExportedMixtureTruncNormPolicy(Policy, StochasticPolicy):
    """Runtime-loadable policy backed by a ``model.pt`` checkpoint.

    Implements both ``Policy`` (deterministic ``act()`` → highest-weight
    component mean) and ``StochasticPolicy`` (mixture ``sample()``).

    Loading is strict (``strict=True``) and validates
    ``format_version``, ``policy_class``, and ``num_components`` before
    attempting to load.  A wrong K changes the head output width and is
    caught as a shape error at ``strict=True`` load.
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
        if pcls != "MixtureTruncatedNormalPolicy":
            raise RuntimeError(
                f"Policy class mismatch: file says {pcls!r}, "
                f"loader expects 'MixtureTruncatedNormalPolicy'."
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

        self._policy = _MixtureTruncNormInferenceNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_components=num_components,
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
