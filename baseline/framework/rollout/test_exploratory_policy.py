"""SamplingPolicy reference-ensemble fast path — A1 CPU batching.

``SamplingPolicy`` prefers an internal fast path over the serial
per-reference ``act()`` loop: expose ``policy._policy`` with a
``deterministic_action`` method and the wrapper calls it once per net
on a shared obs tensor, accumulating the weighted sum in torch.  The
first use is cross-checked against the serial loop; any deviation —
missing convention, divergent output — permanently falls back to the
serial path, which is always correct.  These tests pin that contract:

- activation when the export convention holds;
- bit-identical output vs the serial path (same det code path);
- fallback when the convention is absent or the outputs disagree;
- no-reference and non-delta specs untouched.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from envs.framework.policy import Policy, PolicyBlueprint

from baseline.framework.rollout.exploratory_policy import SamplingPolicy
from baseline.framework.rollout.job import ReferenceSpec, SamplingSpec

OBS_DIM, ACT_DIM = 8, 3
_MODULE = "baseline.framework.rollout.test_exploratory_policy"


class _FixtureNet(nn.Module):
    """Tiny deterministic net mirroring the export ``_policy`` shape."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBS_DIM, 16), nn.Tanh())
        self.head = nn.Linear(16, ACT_DIM)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))


class _FixtureRefPolicy(Policy):
    """Reference policy honouring the ``_policy.deterministic_action``
    export convention — the ensemble fast path must engage."""

    def __init__(self, seed: int = 0, **_: object) -> None:
        torch.manual_seed(seed)
        self._policy = _FixtureNet()

    def act(self, observation, *, want_extra=False):
        x = torch.as_tensor(
            np.asarray(observation, dtype=np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            a = self._policy.deterministic_action(x)
        return a.squeeze(0).numpy().copy(), {}


class _MismatchedRefPolicy(_FixtureRefPolicy):
    """``_policy`` exists but its det output diverges from ``act()`` —
    first-use validation must detect this and disable the fast path."""

    def act(self, observation, *, want_extra=False):
        return np.full(ACT_DIM, 7.0, dtype=np.float32), {}


class _PlainRefPolicy(Policy):
    """No ``_policy``/``deterministic_action`` — convention absent,
    so ``_ensemble`` must be ``None`` and the serial loop used."""

    def __init__(self, seed: int = 0, **_: object) -> None:
        self._v = np.full(ACT_DIM, float(seed) + 1.0, dtype=np.float32)

    def act(self, observation, *, want_extra=False):
        return self._v.copy(), {}


class _FixtureInnerPolicy(Policy):
    """Minimal StochasticPolicy-shaped inner: declares delta support,
    records the ctx it was sampled under."""

    SUPPORTS_REFERENCE_DELTA = True

    def __init__(self, **_: object) -> None:
        self.last_ctx = None

    def act(self, observation, *, want_extra=False):
        return np.zeros(ACT_DIM, dtype=np.float32), {}

    def sample(self, observation, *, ctx=None, want_extra=False):
        self.last_ctx = ctx
        return np.zeros(ACT_DIM, dtype=np.float32), {"logp": 0.0}


def _bps(kind: str, n: int) -> list:
    return [
        PolicyBlueprint(cls=f"{_MODULE}:{kind}", config={"seed": i})
        for i in range(n)
    ]


def _spec(bps, lam: float = 1.0, ef: float = 0.0, c: float = 5.0):
    w = [1.0 / len(bps)] * len(bps)
    return SamplingSpec(
        explore_factor=ef,
        reference=ReferenceSpec(
            policies=tuple(bps), weights=tuple(w),
        ),
        delta_factor=c,
        delta_mix=lam,
    )


def _wrap(bps=None, spec=None):
    inner = _FixtureInnerPolicy()
    sp = SamplingPolicy(
        inner, spec if spec is not None else _spec(bps or _bps("_FixtureRefPolicy", 3))
    )
    return inner, sp


_OBS = np.linspace(-0.5, 0.5, OBS_DIM, dtype=np.float32)


class TestRefEnsembleFastPath(unittest.TestCase):
    def test_ensemble_activates_on_export_convention(self):
        _, sp = _wrap()
        self.assertIsNotNone(sp._ensemble)

    def test_batched_output_bit_identical_to_serial(self):
        _, sp = _wrap()
        got = sp._reference_action(_OBS)
        self.assertTrue(sp._ensemble_validated)
        serial = sp._serial_reference_action(_OBS)
        np.testing.assert_array_equal(got, serial)

    def test_validated_once_then_stays(self):
        _, sp = _wrap()
        sp._reference_action(_OBS)
        self.assertIsNotNone(sp._ensemble)
        out = sp._reference_action(_OBS * 2.0)
        np.testing.assert_allclose(
            out, sp._serial_reference_action(_OBS * 2.0), atol=0,
        )

    def test_fallback_when_convention_absent(self):
        _, sp = _wrap(bps=_bps("_PlainRefPolicy", 2))
        self.assertIsNone(sp._ensemble)
        out = sp._reference_action(_OBS)
        np.testing.assert_allclose(out, np.full(ACT_DIM, 1.5), atol=1e-6)

    def test_mismatched_output_disables_ensemble(self):
        _, sp = _wrap(bps=_bps("_MismatchedRefPolicy", 2))
        self.assertIsNotNone(sp._ensemble)
        out = sp._reference_action(_OBS)
        self.assertIsNone(sp._ensemble)
        np.testing.assert_allclose(out, np.full(ACT_DIM, 7.0), atol=0)

    def test_reference_action_flows_into_ctx(self):
        inner, sp = _wrap()
        sp.act(_OBS)
        self.assertIsNotNone(inner.last_ctx)
        np.testing.assert_allclose(
            inner.last_ctx.reference_action,
            sp._reference_action(_OBS),
            atol=1e-6,
        )


class TestNoReferenceSpecs(unittest.TestCase):
    def test_no_reference_no_ensemble(self):
        inner, sp = _wrap(spec=SamplingSpec(explore_factor=0.5))
        self.assertIsNone(sp._ensemble)
        sp.act(_OBS)
        self.assertIsNone(inner.last_ctx.reference_action)
        self.assertEqual(float(inner.last_ctx.explore_factor), 0.5)

    def test_ef_callable_consumed(self):
        ef_calls = []

        def sched(obs, step):
            ef_calls.append(step)
            return 0.25

        inner, sp = _wrap(spec=SamplingSpec(explore_factor=sched))
        sp.act(_OBS)
        sp.act(_OBS)
        self.assertEqual(ef_calls, [0, 1])
        self.assertEqual(float(inner.last_ctx.explore_factor), 0.25)


if __name__ == "__main__":
    unittest.main()
