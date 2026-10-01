"""Self-contained exported policy template for
SharedMixtureBoundedStdTruncatedNormalPolicy.

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
``shared_mixture_bounded_std_truncated_normal_mlp.py``.  A parity test
in ``test_shared_mixture_bounded_std_truncated_normal.py`` verifies
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


class SamplingContext:
    """Self-contained sampling context — mirrors the framework's
    ``baseline.framework.ppo.sampling_context.SamplingContext`` so this
    file needs no repo import.  Fields are per-frame values (scalars or
    arrays); the exported policies consume ``explore_factor`` plus the
    ``reference_action``/``delta_factor``/``delta`` fields.

    A plain class (not a dataclass): this file is exec-loaded without a
    real module entry, so ``from __future__ import annotations`` +
    ``@dataclass`` would fail to resolve string annotations.
    """

    __slots__ = (
        "explore_factor", "reference_action", "delta_factor",
        "delta",
    )

    def __init__(self, explore_factor=None, reference_action=None,
                 delta_factor=None,
                 delta=None):
        self.explore_factor = explore_factor
        self.reference_action = reference_action
        self.delta_factor = delta_factor
        self.delta = delta

    def has_delta(self) -> bool:
        """True iff the reference-delta scale is active — mirrors the
        upstream ``SamplingContext.has_delta``."""
        if self.reference_action is None and self.delta is None:
            return False
        c = self.delta_factor
        if c is None:
            return False
        if hasattr(c, "any"):  # ndarray / torch.Tensor
            return bool((c != 0).any())
        return c != 0



def _ctx_bcast(x: Any, target: torch.Tensor) -> Any:
    """ctx scalar field → float, or a (B,) tensor → broadcastable."""
    if not torch.is_tensor(x):
        return float(x)
    while x.ndim < target.ndim:
        x = x.unsqueeze(-1)
    return x


def _ctx_nonzero(x):
    """True iff a ctx scalar field is nonzero (scalar or any-element)."""
    if x is None:
        return False
    return bool((x != 0).any()) if hasattr(x, "any") else bool(x != 0)


def _delta_of(mean, ctx):
    """The Δ payload the σ floor consumes this call, or ``None``.

    An explicit ``ctx.delta`` payload wins over the fresh
    ``mean − a_ref`` path — ``ctx.delta`` is the action-level Δ
    (``det_action − a_ref``) supplied by the sampling layer / replay,
    never a function of current θ.  Payload presence is the contract;
    ``delta_max_sigma`` decides what a missing payload means.
    """
    if ctx is None:
        return None
    def _bcast(v):
        t = torch.as_tensor(v, dtype=mean.dtype, device=mean.device)
        while t.ndim < mean.ndim:
            t = t.unsqueeze(-2)
        return t

    frozen = getattr(ctx, "delta", None)
    if frozen is not None:
        return _bcast(frozen)
    ref = getattr(ctx, "reference_action", None)
    if ref is None:
        return None
    return mean - _bcast(ref)


def delta_max_sigma(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    ctx: Optional["SamplingContext"],
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
) -> torch.Tensor:
    """Element-wise σ floor from the reference delta —
    inlined copy of ``truncated_normal_mlp.delta_max_sigma``.

        σ_eff = max(σ, c·|Δ|)

    ``mean`` is detached in the Δ term (σ_eff exogenous w.r.t. m_θ —
    see the source docstring).  The max keeps σ_eff ≥ σ in every dim:
    the delta can only widen exploration, never narrow it."""
    # Attribute-based activation check — NOT ctx.has_delta(): ``ctx`` may
    # be an instance of an older SamplingContext class held by a spawned
    # rollout worker; its fields exist but methods may not.
    c = getattr(ctx, "delta_factor", None) if ctx is not None else None
    if not _ctx_nonzero(c):
        return sigma
    delta = _delta_of(mean, ctx)
    if delta is None:
        # c ≠ 0 but no Δ source — the rollout-side record lost its
        # payload (stale buffer or pipeline bug); fail loud.
        raise ValueError(
            "delta ctx carries delta_factor != 0 but neither `delta` "
            "nor `reference_action` — the rollout-side sctx__ "
            "payload record is missing (stale buffer or pipeline "
            "bug)"
        )
    out = torch.maximum(
        sigma, (_ctx_bcast(c, sigma) * delta.detach()).abs(),
    )
    if sigma_min is not None:
        out = out.clamp(sigma_min, sigma_max)
    return out


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


class _SharedMixtureBoundedStdInferenceNet(nn.Module):
    """Inference-only K-component mixture with shared bounded σ.

    Self-contained copy of the inference subset of
    ``SharedMixtureBoundedStdTruncatedNormalPolicy`` — excludes
    ``evaluate_actions`` (training-only) and ``to_blueprint``
    (export-only).  State-dict keys are identical to the training side
    (``trunk.*``, ``head.*``, ``raw_std``).

    Head layout: ``logits (K) | raw_mean (K·D)``, component-major; σ is
    the broadcast bounded map of ``raw_std`` (K, D).
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
        self.head = nn.Linear(hidden_dim, K + K * D)
        self.raw_std = nn.Parameter(torch.zeros(K, D))

        self._r_min = math.log(float(sigma_min))
        self._delta_r = math.log(float(sigma_max)) - self._r_min
        self._sigma_min = float(sigma_min)
        self._sigma_max = float(sigma_max)
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
        if (
            isinstance(explore_factor, torch.Tensor)
            and explore_factor.is_cuda
            and torch.cuda.is_current_stream_capturing()
        ):
            # GPU→CPU validation syncs are illegal while a CUDA graph is
            # being captured; eager calls keep the full check.
            return
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
        """→ (log_pi, mean, policy_sigma) with σ from the bounded map."""
        K, D = self.num_components, self.action_dim
        out = self.head(self.trunk(obs))
        logits = out[..., :K]
        raw_mean = out[..., K:K + K * D].reshape(-1, K, D)
        v = self.raw_std.view(1, K, D).expand(obs.shape[0], -1, -1)
        log_pi = torch.log_softmax(logits, dim=-1)
        mean = torch.tanh(raw_mean)
        sigma = self._bounded_sigma(v)
        return log_pi, mean, sigma

    def _effective_sigma(
        self, obs_batch: int, explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        """σ(v + αe) — additive shift on the raw sigmoid input."""
        K, D = self.num_components, self.action_dim
        v = self.raw_std.view(1, K, D).expand(obs_batch, -1, -1)
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
        self, obs: torch.Tensor, *, ctx: Optional["SamplingContext"] = None,
        uniform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ``uniform`` (B, D+1): column 0 selects the mixture component via
        # inverse-CDF on the cumulated weights (replacing multinomial);
        # columns 1: are the per-dim action draws (replacing the internal
        # RNG).  The batched inference server injects per-request noise
        # here so results are independent of batch composition.
        ef = ctx.explore_factor if ctx is not None else 0.0
        self._check_ei(ef)
        log_pi, mean, _ = self._head_forward(obs)
        sigma = self._effective_sigma(obs.shape[0], ef)
        sigma = delta_max_sigma(
            mean, sigma, ctx, self._sigma_min, self._sigma_max,
        )
        B, K, D = mean.shape

        if K == 1:
            idx = torch.zeros(B, dtype=torch.long, device=mean.device)
        elif uniform is not None:
            cum = torch.cumsum(log_pi.exp(), dim=-1)
            idx = (
                torch.searchsorted(cum, uniform[:, 0].unsqueeze(-1).contiguous())
                .squeeze(-1)
                .clamp_(max=K - 1)
            )
        else:
            idx = torch.multinomial(
                log_pi.exp(), 1, generator=self._gen,
            ).squeeze(-1)
        sel = idx.view(-1, 1, 1).expand(-1, 1, D)
        mu_k = mean.gather(1, sel).squeeze(1)
        sg_k = sigma.gather(1, sel).squeeze(1)

        erf_a = torch.erf((_ACTION_LOW - mu_k) / (_SQRT_2 * sg_k))
        erf_b = torch.erf((_ACTION_HIGH - mu_k) / (_SQRT_2 * sg_k))
        u = (
            uniform[:, 1:]
            if uniform is not None
            else torch.rand(
                mu_k.shape,
                generator=self._gen,
                device=mu_k.device,
                dtype=mu_k.dtype,
            )
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
_STD_SOURCE = "shared"


class ExportedSharedMixtureBoundedStdTruncNormPolicy(Policy, StochasticPolicy):
    """Runtime-loadable policy backed by a ``model.pt`` checkpoint.

    Implements both ``Policy`` (deterministic ``act()`` → highest-weight
    component mean) and ``StochasticPolicy`` (mixture ``sample()``).

    Loading is strict (``strict=True``) and validates
    ``format_version``, ``policy_class``, the distribution-identity
    metadata fields, the bounded-σ config (``sigma_min``/``sigma_max``/
    ``explore_alpha``), and ``num_components`` before attempting to load.
    """
    # Capability flag read by SamplingPolicy at wrap time — this cell
    # implements the reference-delta σ floor (delta_max_sigma above).
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
        if pcls != "SharedMixtureBoundedStdTruncatedNormalPolicy":
            raise RuntimeError(
                f"Policy class mismatch: file says {pcls!r}, "
                f"loader expects 'SharedMixtureBoundedStdTruncatedNormalPolicy'."
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

        self._policy = _SharedMixtureBoundedStdInferenceNet(
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
        ctx: Optional["SamplingContext"] = None,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Stochastic action — mixture sample with explore_factor."""
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
