"""SamplingPolicy — wraps a StochasticPolicy with a per-agent SamplingSpec.

This wrapper is the bridge between the training layer (which knows
about :class:`SamplingSpec`) and the core framework (which does not).
It implements the plain :class:`Policy` interface so that
:class:`EpisodeRunner` never sees sampling internals — the runner
just calls ``policy.act(obs, want_extra=...)``.

Per frame the wrapper:

1. Resolves ``explore_factor`` (constant float or callable
   ``(obs, step) -> float``).
2. If the spec carries a :class:`ReferenceSpec`, evaluates every
   reference policy's deterministic ``act()`` on the same observation
   and forms the weighted ``reference_action``.
3. Builds a :class:`SamplingContext` and calls
   ``inner.sample(obs, ctx=ctx, want_extra=...)``.
4. Records the ctx fields into ``extra["sampling_ctx"]`` so they travel
   through ``action_extras`` to recorders and trainers — the recorded
   values are exactly what was passed to ``sample()``.

:class:`ExploratoryPolicy` is kept as a backward-compatible alias —
``ExploratoryPolicy(inner, ef)`` behaves identically to
``SamplingPolicy(inner, SamplingSpec(explore_factor=ef))``.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from envs.framework.policy import Policy

from baseline.framework.ppo.sampling_context import SamplingContext
from baseline.framework.rollout.job import SamplingSpec

if TYPE_CHECKING:
    from baseline.framework.ppo.stochastic_policy import StochasticPolicy

#: Per-frame explore_factor: a constant float, or a callable
#: ``(obs, step) -> float``.  Callables must be top-level functions to
#: be picklable across multiprocessing workers.
EfSpec = Union[float, Callable[[np.ndarray, int], float]]


def _try_build_ref_ensemble(ref_pairs) -> Optional[Callable]:
    """Fast weighted reference forward over all reference nets, or ``None``.

    Reference policies are ``file:`` exports whose runtime objects carry
    the inner inference net at ``policy._policy`` with a
    ``deterministic_action`` method.  The serial path pays one full
    ``act()`` call (obs→tensor→forward→numpy) per reference per frame;
    the fast path instead converts the observation once, invokes each
    net's ``deterministic_action`` directly, and accumulates the
    weighted sum in torch — roughly a 3× reduction of the per-frame
    reference cost at a full ``reference_horizon`` window.  Returns
    ``None`` on any deviation from the convention; the caller then
    falls back to the serial loop, which is always correct.  Internal
    convention only — no interface changes.
    """
    try:
        import torch
    except ImportError:
        return None

    nets_w = []
    for w, pol in ref_pairs:
        net = getattr(pol, "_policy", None)
        if net is None or not callable(
            getattr(net, "deterministic_action", None)
        ):
            return None
        nets_w.append((w, net))

    @torch.no_grad()
    def batched(observation: Any) -> np.ndarray:
        x = torch.as_tensor(
            np.asarray(observation, dtype=np.float32)
        ).unsqueeze(0)
        acc: Optional[torch.Tensor] = None
        for w, net in nets_w:
            a = net.deterministic_action(x).squeeze(0)
            acc = w * a if acc is None else acc + w * a
        assert acc is not None
        return acc.cpu().numpy().astype(np.float32)

    return batched


class SamplingPolicy(Policy):
    """Wrap a :class:`StochasticPolicy` with a :class:`SamplingSpec`.

    Implements :class:`Policy` by delegating to ``inner.sample()``.
    The ``EpisodeRunner`` sees a plain ``Policy`` and is unaware of
    the sampling context.

    Parameters
    ----------
    inner:
        The stochastic policy to wrap (must implement ``sample()``).
    spec:
        Per-agent sampling directive.  Reference policies (if any) are
        built once at wrap time and reused across episodes; each frame
        calls their deterministic ``act()`` on the current observation
        and combines them with ``weights`` — action-space ensemble,
        NOT a parameter EMA.
    """

    def __init__(
        self,
        inner: "StochasticPolicy",
        spec: SamplingSpec,
    ) -> None:
        if not hasattr(inner, "sample"):
            raise TypeError(
                f"inner must implement sample(); got {type(inner).__name__}"
            )
        if not isinstance(spec, SamplingSpec):
            raise TypeError(
                f"spec must be a SamplingSpec; got {type(spec).__name__}"
            )
        # Intent-vs-capability handshake: a spec that demands the
        # reference-delta σ mix must not wrap a policy that lacks it —
        # otherwise the mechanism silently degrades to plain ef sampling
        # and the run reports misleading metrics.  Pre-ctx exports and
        # pre-tanh cells fail here at wrap time, not mid-rollout.
        delta_demanded = (
            spec.reference is not None and float(spec.delta_mix) != 0.0
        )
        if delta_demanded and not getattr(
            inner, "SUPPORTS_REFERENCE_DELTA", False
        ):
            raise TypeError(
                f"SamplingSpec demands the reference-delta σ mix "
                f"(reference set, delta_mix={spec.delta_mix}) but "
                f"{type(inner).__name__} does not declare "
                f"SUPPORTS_REFERENCE_DELTA — export the policy with "
                f"current code or pick a supported policy cell"
            )
        self.inner = inner
        self._spec = spec
        self._step: int = 0
        # Build reference policies once — they are frozen blueprints for
        # this whole rollout batch, independent of update bookkeeping.
        if spec.reference is not None:
            self._ref_pairs = tuple(
                (float(w), bp.build())
                for w, bp in zip(
                    spec.reference.weights, spec.reference.policies,
                )
            )
        else:
            self._ref_pairs = ()
        # Fast path: one obs→tensor conversion + direct
        # ``deterministic_action`` calls + weighted sum in torch,
        # instead of K full ``act()`` round-trips.  First use is
        # verified against the serial loop once — any drift disables
        # the fast path permanently.
        self._ensemble = (
            _try_build_ref_ensemble(self._ref_pairs)
            if self._ref_pairs else None
        )
        self._ensemble_validated = False

    def _serial_reference_action(self, observation: Any) -> np.ndarray:
        acc: Optional[np.ndarray] = None
        for w, ref_policy in self._ref_pairs:
            action, _ = ref_policy.act(observation)
            a = np.asarray(action, dtype=np.float32)
            acc = w * a if acc is None else acc + w * a
        # _ref_pairs is non-empty whenever this is called (enforced by
        # ReferenceSpec validation: policies must be non-empty).
        assert acc is not None
        return acc.astype(np.float32)

    def _reference_action(self, observation: Any) -> np.ndarray:
        """Weighted deterministic action of the reference ensemble."""
        if self._ensemble is None:
            return self._serial_reference_action(observation)
        batched = self._ensemble(observation)
        if not self._ensemble_validated:
            self._ensemble_validated = True
            ref = self._serial_reference_action(observation)
            if not np.allclose(batched, ref, atol=1e-5):
                self._ensemble = None
                return ref
        return batched

    def _build_ctx(self, observation: Any, ef: float) -> SamplingContext:
        ref = (
            self._reference_action(observation)
            if self._ref_pairs
            else None
        )
        # Frozen mode: Δ = det_action(current policy) − a_ref is an
        # action-level quantity computed HERE — the reference ensemble
        # supplies a_ref, the inner policy supplies μ₀ via its
        # deterministic act().  It enters the ctx as an input field
        # (`delta`, mutually exclusive with `reference_action`), is
        # consumed verbatim by the σ-mix, and flows back out through
        # record_fields() → sctx__delta for frozen replay at train.
        if self._spec.delta_mode == "frozen" and ref is not None:
            mu0, _ = self.inner.act(observation)
            delta = (
                np.asarray(mu0, dtype=np.float32) - ref
            ).astype(np.float32)
            return SamplingContext(
                explore_factor=ef,
                reference_action=None,
                delta_factor=float(self._spec.delta_factor),
                delta_mix=float(self._spec.delta_mix),
                delta=delta,
            )
        return SamplingContext(
            explore_factor=ef,
            reference_action=ref,
            delta_factor=float(self._spec.delta_factor),
            delta_mix=float(self._spec.delta_mix),
        )

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[Any, Optional[dict]]:
        ef = (
            float(self._spec.explore_factor(observation, self._step))
            if callable(self._spec.explore_factor)
            else float(self._spec.explore_factor)
        )
        self._step += 1
        ctx = self._build_ctx(observation, ef)
        action, extra = self.inner.sample(
            observation, ctx=ctx, want_extra=want_extra,
        )
        extra = dict(extra) if extra is not None else {}
        # Record the ctx fields flat under the sctx__ prefix so they
        # travel through action_extras / Episode stacking / dump npz
        # serialization as plain arrays (a nested dict would become a
        # pickle-requiring object array and break Episode.load()).
        for key, value in ctx.record_fields().items():
            if key == "explore_factor":
                continue  # recorded via the legacy key below
            extra[f"sctx__{key}"] = value
        # Legacy key kept for existing consumers (dump viewers, older
        # analysis code that reads extras["explore_factor"]).
        extra["explore_factor"] = ef
        return action, extra

    def reset(self, seed: Optional[int] = None) -> None:
        self._step = 0
        reset_fn = getattr(self.inner, "reset", None)
        if callable(reset_fn):
            reset_fn(seed)


class ExploratoryPolicy(SamplingPolicy):
    """Backward-compatible alias for the pre-SamplingSpec API.

    ``ExploratoryPolicy(inner, ef)`` is equivalent to
    ``SamplingPolicy(inner, SamplingSpec(explore_factor=ef))``.
    """

    def __init__(
        self,
        inner: "StochasticPolicy",
        explore_factor: EfSpec = 0.0,
    ) -> None:
        super().__init__(inner, SamplingSpec(explore_factor=explore_factor))
