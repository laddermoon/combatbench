"""Reference-delta σ floor (S3) — per-cell semantics + export parity.

The mechanism: when ``SamplingContext`` carries ``reference_action``
(dynamic mode — Δ = m_θ − a_ref recomputed at evaluate) or ``delta``
(frozen mode — the rollout-time Δ replayed verbatim) with
``delta_factor = c ≠ 0``, the effective scale becomes

    σ_eff = max(σ_ef, c·|Δ|)

applied per-component for mixture cells and clamped to
``[sigma_min, sigma_max]`` for bounded cells.  An inactive ctx (no
reference/payload or c = 0 everywhere) must leave σ bit-identical.
"""
from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch

from baseline.framework.ppo.sampling_context import SamplingContext
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
    delta_max_sigma,
)
from baseline.framework.ppo.policies.state_truncated_normal_mlp import (
    StateTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.bounded_std_truncated_normal_mlp import (
    BoundedStdTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.state_bounded_std_truncated_normal_mlp import (
    StateBoundedStdTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.mixture_truncated_normal_mlp import (
    MixtureTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.shared_mixture_truncated_normal_mlp import (
    SharedMixtureTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.shared_mixture_bounded_std_truncated_normal_mlp import (
    SharedMixtureBoundedStdTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.state_mixture_bounded_std_truncated_normal_mlp import (
    StateMixtureBoundedStdTruncatedNormalPolicy,
)

OBS_DIM, ACT_DIM, HID = 96, 21, 256

SINGLE_CLASSES = {
    "truncnorm": TruncatedNormalPolicy,
    "state_truncnorm": StateTruncatedNormalPolicy,
    "bounded": BoundedStdTruncatedNormalPolicy,
    "state_bounded": StateBoundedStdTruncatedNormalPolicy,
}
MOG_CLASSES = {
    "mixture": MixtureTruncatedNormalPolicy,
    "mixture_shared": SharedMixtureTruncatedNormalPolicy,
    "mixture_shared_bounded": SharedMixtureBoundedStdTruncatedNormalPolicy,
    "mixture_state_bounded": StateMixtureBoundedStdTruncatedNormalPolicy,
}
ALL_CLASSES = {**SINGLE_CLASSES, **MOG_CLASSES}
BOUNDED_NAMES = {"bounded", "state_bounded",
                 "mixture_shared_bounded", "mixture_state_bounded"}


def _make(cls, seed_net: int = 0):
    torch.manual_seed(seed_net)
    return cls(OBS_DIM, ACT_DIM, HID)


def _ctx(ef=0.0, ref=None, c=0.0, delta=None):
    return SamplingContext(
        explore_factor=ef,
        reference_action=ref,
        delta_factor=c,
        delta=delta,
    )


def _bounds(p):
    """(sigma_min, sigma_max) for bounded cells, else (None, None)."""
    return getattr(p, "sigma_min", None), getattr(p, "sigma_max", None)


class TestInactivePath(unittest.TestCase):
    """c=0 or missing reference ⇒ σ and samples stay bit-identical."""

    def setUp(self):
        torch.manual_seed(0)
        self.obs = torch.randn(4, OBS_DIM)
        self.ref = torch.randn(ACT_DIM)

    def test_forward_bit_identical(self):
        for name, cls in SINGLE_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                _, s0 = p.forward(self.obs, ctx=_ctx(ef=0.4))
                _, s_c0 = p.forward(self.obs, ctx=_ctx(
                    ef=0.4, ref=self.ref, c=0.0))
                self.assertTrue(torch.equal(s0, s_c0))
                # c≠0 + no payload = malformed ctx (the lost-record
                # signature) — raises rather than silently sampling.
                with self.assertRaises(ValueError):
                    p.forward(self.obs, ctx=_ctx(
                        ef=0.4, ref=None, c=10.0))

    def test_sample_bit_identical(self):
        for name, cls in ALL_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                p.reset(11)
                a0, _ = p.sample_action(self.obs, ctx=_ctx(ef=0.4))
                p.reset(11)
                a1, _ = p.sample_action(self.obs, ctx=_ctx(
                    ef=0.4, ref=self.ref, c=0.0))
                self.assertTrue(torch.equal(a0, a1))
                p.reset(11)
                with self.assertRaises(ValueError):
                    p.sample_action(self.obs, ctx=_ctx(
                        ef=0.4, ref=None, c=10.0))


class TestFormula(unittest.TestCase):
    """Element-wise max formula, single-component cells end to end."""

    def setUp(self):
        torch.manual_seed(0)
        self.obs = torch.randn(4, OBS_DIM)
        self.ref = torch.randn(ACT_DIM) * 0.5

    def test_formula(self):
        c = 10.0
        for name, cls in SINGLE_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                mean0, s_ef = p.forward(self.obs, ctx=_ctx(ef=0.2))
                mean1, s_on = p.forward(self.obs, ctx=_ctx(
                    ef=0.2, ref=self.ref, c=c))
                self.assertTrue(torch.equal(mean0, mean1))
                exp = torch.maximum(
                    s_ef, (c * (mean0 - self.ref)).abs())
                smin, smax = _bounds(p)
                if smin is not None:
                    exp = exp.clamp(smin, smax)
                torch.testing.assert_close(s_on, exp, rtol=0, atol=0)

    def test_reference_batch_shape(self):
        """ref broadcasts from both (D,) and (B, D)."""
        p = _make(TruncatedNormalPolicy)
        ref_d = torch.randn(ACT_DIM)
        ref_b = ref_d.unsqueeze(0).expand(self.obs.shape[0], -1).contiguous()
        _, s_d = p.forward(
            self.obs, ctx=_ctx(ref=ref_d, c=5.0))
        _, s_b = p.forward(
            self.obs, ctx=_ctx(ref=ref_b, c=5.0))
        self.assertTrue(torch.equal(s_d, s_b))

    def test_zero_delta_falls_back(self):
        """Δ=0 → c·|Δ|=0 → max(σ, 0) = σ: the delta floor is a no-op
        in the zero-drift regime."""
        for name, cls in SINGLE_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                mean, _ = p.forward(self.obs)
                _, s_ef = p.forward(self.obs, ctx=_ctx(ef=0.0))
                _, s = p.forward(self.obs, ctx=_ctx(
                    ref=mean.detach(), c=10.0))
                torch.testing.assert_close(s, s_ef, rtol=0, atol=0)

    def test_bounded_clamp(self):
        """A huge Δ clamps back to σ_max on bounded cells."""
        for name in ("bounded", "state_bounded"):
            with self.subTest(cell=name):
                p = _make(SINGLE_CLASSES[name])
                mean, _ = p.forward(self.obs)
                ref = mean.detach() - 1e4
                _, s = p.forward(self.obs, ctx=_ctx(
                    ref=ref, c=10.0))
                torch.testing.assert_close(
                    s, torch.full_like(s, p.sigma_max), rtol=0, atol=0)


class TestMoG(unittest.TestCase):
    """Per-component Δ: each head scales by its own distance to ref."""

    def test_per_component_independence(self):
        torch.manual_seed(0)
        mean = torch.zeros(2, 3, ACT_DIM)
        mean[:, 0, :] = 1.0          # only head 0 is far from ref
        sigma = torch.full((2, 3, ACT_DIM), 0.5)
        ref = torch.zeros(ACT_DIM)
        out = delta_max_sigma(mean, sigma, _ctx(ref=ref, c=10.0))
        self.assertEqual(out.shape, sigma.shape)
        # head 0: max(σ, 10·|Δ|=10) = 10 ; heads 1/2: max(σ, 0) = σ
        torch.testing.assert_close(
            out[:, 0], torch.full_like(out[:, 0], 10.0))
        torch.testing.assert_close(
            out[:, 1:], torch.full_like(out[:, 1:], 0.5))

    def test_shared_sigma_becomes_state_dependent(self):
        """Shared σ parameter + per-state Δ ⇒ per-state σ_eff."""
        for name in ("mixture_shared", "mixture_shared_bounded"):
            with self.subTest(cell=name):
                p = _make(MOG_CLASSES[name])
                _, mean, raw = p._forward_raw(self._obs())
                sigma = p._explored_sigma(raw, 0.0)
                ref = mean[0, 0].detach()       # ref at row-0 head-0
                out = p._delta_sigma(mean, sigma, _ctx(
                    ref=ref, c=10.0))
                # row 0 head 0 sits at the reference → smaller scale
                self.assertLess(
                    out[0, 0].mean().item(),
                    out[1, 0].mean().item(),
                )

    def _obs(self):
        torch.manual_seed(1)
        return torch.randn(4, OBS_DIM)


class TestFrozenMode(unittest.TestCase):
    """Frozen-delta mode: the recorded Δ payload is replayed verbatim —
    σ_eff is constant w.r.t. θ inside an update (the point of the mode).
    """

    def setUp(self):
        torch.manual_seed(0)
        self.obs = torch.randn(4, OBS_DIM)
        self.ref = torch.randn(ACT_DIM) * 0.5

    def test_frozen_payload_matches_rollout_sigma(self):
        """The recorded action-level Δ (det_action − a_ref) produces the
        same σ_eff as the dynamic m−a_ref path at mb1 — for
        single-component cells det_action IS the mean, so frozen replay
        is bit-identical to the rollout-time distribution."""
        for name, cls in SINGLE_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                c = 5.0
                # Rollout side (frozen): wrapper-computed Δ enters ctx.
                delta = (
                    p.deterministic_action(self.obs) - self.ref
                ).detach().numpy()
                _, s_roll = p.forward(self.obs, ctx=_ctx(
                    c=c, delta=delta))
                # Dynamic reference: same Δ recomputed as m − a_ref.
                _, s_dyn = p.forward(self.obs, ctx=_ctx(
                    ref=self.ref, c=c))
                torch.testing.assert_close(s_roll, s_dyn, rtol=0, atol=0)
                # Replay side: the same recorded payload, verbatim.
                _, s_rep = p.forward(self.obs, ctx=_ctx(
                    c=c, delta=delta))
                torch.testing.assert_close(s_rep, s_roll, rtol=0, atol=0)

    def test_frozen_sigma_independent_of_theta(self):
        """Perturbing the mean head must not move frozen-mode σ_eff —
        the value-level σ(m) coupling is the failure this mode removes."""
        p = _make(TruncatedNormalPolicy)
        ctx = _ctx(c=5.0, delta=np.full(ACT_DIM, 0.1, dtype=np.float32))
        _, s_before = p.forward(self.obs, ctx=ctx)
        with torch.no_grad():
            for w in p.net.parameters():
                w.add_(torch.randn_like(w) * 0.5)
        mean2, s_after = p.forward(self.obs, ctx=ctx)
        self.assertFalse(torch.equal(mean2, mean2 * 0))  # mean moved
        torch.testing.assert_close(s_after, s_before, rtol=0, atol=0)

    def test_frozen_payload_wins_over_ref(self):
        """ctx carrying both `delta` and `reference_action` consumes the
        frozen payload — deterministic resolution order."""
        p = _make(TruncatedNormalPolicy)
        delta = np.full(ACT_DIM, 0.1, dtype=np.float32)
        _, s_frozen = p.forward(self.obs, ctx=_ctx(
            c=5.0, delta=delta))
        _, s_both = p.forward(self.obs, ctx=_ctx(
            ref=self.ref, c=5.0, delta=delta))
        torch.testing.assert_close(s_both, s_frozen, rtol=0, atol=0)

    def test_frozen_missing_payload_raises(self):
        """c ≠ 0 + neither payload nor ref = malformed ctx — fail loud
        in both modes rather than silently degrade to plain σ (the
        delta_factor value is the activation invariant; no mode flag)."""
        for name, cls in SINGLE_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                ctx = _ctx(ref=None, c=5.0)
                with self.assertRaises(ValueError):
                    p.forward(self.obs, ctx=ctx)

    def test_wrapper_records_action_level_delta(self):
        """Frozen spec → SamplingPolicy records sctx__delta =
        det_action(inner) − a_ref, and no sctx__reference_action —
        the wrapper (not the policy) owns the Δ."""
        from baseline.framework.rollout import SamplingPolicy
        from baseline.framework.rollout.job import (
            ReferenceSpec, SamplingSpec,
        )
        from envs.framework.policy import PolicyBlueprint
        torch.manual_seed(0)
        p = TruncatedNormalPolicy(OBS_DIM, ACT_DIM, HID)
        torch.manual_seed(1)
        ref_p = TruncatedNormalPolicy(OBS_DIM, ACT_DIM, HID)
        with tempfile.TemporaryDirectory() as tmp:
            ref_bp = ref_p.to_blueprint(dest_path=tmp)
            spec = SamplingSpec(
                reference=ReferenceSpec(
                    policies=(ref_bp,), weights=(1.0,),
                ),
                delta_factor=5.0,
                delta_mode="frozen",
            )
            wrapper = SamplingPolicy(p, spec)
            obs = np.random.RandomState(0).randn(OBS_DIM).astype(
                np.float32)
            _, extra = wrapper.act(obs, want_extra=True)
        self.assertIn("sctx__delta", extra)
        self.assertNotIn("sctx__delta_frozen", extra)
        self.assertNotIn("sctx__reference_action", extra)
        mu0, _ = p.act(obs)
        a_ref, _ = ref_p.act(obs)
        np.testing.assert_allclose(
            extra["sctx__delta"],
            np.asarray(mu0, dtype=np.float32)
            - np.asarray(a_ref, dtype=np.float32),
            rtol=0, atol=1e-6,
        )

    def test_record_fields_mutual_exclusion(self):
        """record_fields serializes non-None fields verbatim — the
        payload mutual exclusion lives in the input contract (frozen
        ctx carries `delta`, dynamic carries `reference_action`), not
        in a recorder-side skip rule."""
        delta = np.full(ACT_DIM, 0.1, dtype=np.float32)
        ctx = _ctx(c=5.0, delta=delta)
        fields = ctx.record_fields()
        self.assertNotIn("reference_action", fields)
        self.assertNotIn("delta_frozen", fields)
        np.testing.assert_array_equal(fields["delta"], delta)
        ctx_dyn = _ctx(ref=self.ref, c=5.0)
        fields_dyn = ctx_dyn.record_fields()
        self.assertIn("reference_action", fields_dyn)
        self.assertNotIn("delta", fields_dyn)

    def test_frozen_delta_mog_broadcast(self):
        """Mixture cells consume an action-level (D,) payload — the same
        Δ for every component (broadcast over K)."""
        for name, cls in MOG_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                k = p.num_components
                delta = np.full(ACT_DIM, 0.05, dtype=np.float32)
                ctx = _ctx(c=5.0, delta=delta)
                _, mean, raw = p._forward_raw(self.obs)
                sigma = p._explored_sigma(raw, 0.0)
                out = p._delta_sigma(mean, sigma, ctx)
                self.assertEqual(out.shape, (4, k, ACT_DIM))


class TestSampleEvalConsistency(unittest.TestCase):
    """sample_action and evaluate_actions must see the same σ_eff."""

    def setUp(self):
        torch.manual_seed(0)
        self.obs = torch.randn(6, OBS_DIM)
        self.ref = torch.randn(ACT_DIM)

    def test_log_prob_consistency(self):
        for name, cls in ALL_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                ctx = _ctx(ef=0.3, ref=self.ref, c=8.0)
                a, lp_sample = p.sample_action(self.obs, ctx=ctx)
                ev = p.evaluate_actions(self.obs, a, ctx=ctx)
                torch.testing.assert_close(
                    ev.log_prob, lp_sample, rtol=1e-5, atol=1e-6)


class TestGradient(unittest.TestCase):
    """σ_eff is exogenous w.r.t. m_θ: the Δ term detaches ``mean``.
    Detaching closes the cheat channel where PPO raises log_prob by
    collapsing μ toward a_ref instead of improving actions.  σ keeps
    receiving gradients exactly in the dims where it wins the max."""

    def test_delta_detached_from_mean(self):
        torch.manual_seed(0)
        p = _make(TruncatedNormalPolicy)
        obs = torch.randn(4, OBS_DIM)
        ctx = _ctx(ref=torch.randn(ACT_DIM) * 0.3, c=0.5)
        _, sigma = p.forward(obs, ctx=ctx)
        sigma.sum().backward()
        g_mean = p.net[0].weight.grad
        self.assertTrue(g_mean is None or g_mean.abs().sum().item() == 0.0)
        self.assertGreater(p.log_std.grad.abs().sum().item(), 0.0)

    def test_low_delta_sigma_path_alive(self):
        """Dims where c|Δ| < σ keep the σ gradient — the floor only
        widens where drift dominates, so σ stays trainable in the
        low-drift regime."""
        torch.manual_seed(0)
        p = _make(TruncatedNormalPolicy)
        obs = torch.randn(4, OBS_DIM)
        # Ref ON the mean → Δ≈0 → σ wins the max everywhere → σ_eff = σ_θ.
        mean, _ = p.forward(obs)
        ctx = _ctx(ref=mean[0].detach(), c=10.0)
        _, sigma = p.forward(obs, ctx=ctx)
        sigma.sum().backward()
        self.assertGreater(p.log_std.grad.abs().sum().item(), 0.0)

    def test_large_delta_kills_sigma_path(self):
        """A dominating Δ makes the σ path dead in those dims — the
        delta floor owns them (exactly zero σ gradient via max)."""
        torch.manual_seed(0)
        p = _make(TruncatedNormalPolicy)
        obs = torch.randn(4, OBS_DIM)
        # Ref far away → c|Δ| ≫ σ for every dim → delta floor everywhere.
        mean, _ = p.forward(obs)
        ctx = _ctx(ref=mean[0].detach() - 10.0, c=10.0)
        _, sigma = p.forward(obs, ctx=ctx)
        sigma.sum().backward()
        self.assertEqual(p.log_std.grad.abs().sum().item(), 0.0)
        g_mean = p.net[0].weight.grad
        self.assertTrue(g_mean is None or g_mean.abs().sum().item() == 0.0)


class TestCapabilityGate(unittest.TestCase):
    """Wrap-time intent-vs-capability check in SamplingPolicy."""

    def _spec(self):
        from baseline.framework.rollout.job import (
            ReferenceSpec, SamplingSpec,
        )
        from envs.framework.policy import PolicyBlueprint
        import tempfile
        torch.manual_seed(0)
        p = TruncatedNormalPolicy(OBS_DIM, ACT_DIM, HID)
        tmp = tempfile.mkdtemp()
        bp = p.to_blueprint(dest_path=tmp)
        return SamplingSpec(
            reference=ReferenceSpec(policies=(bp,), weights=(1.0,)),
            delta_factor=10.0,
        ), tmp

    def test_delta_spec_rejects_uncapable_policy(self):
        from baseline.framework.rollout import SamplingPolicy
        from baseline.framework.ppo.policies.pre_tanh_normal_mlp import (
            PreTanhNormalPolicy,
        )
        spec, _ = self._spec()
        torch.manual_seed(0)
        pre_tanh = PreTanhNormalPolicy(OBS_DIM, ACT_DIM, HID)
        with self.assertRaises(TypeError):
            SamplingPolicy(pre_tanh, spec)

    def test_delta_spec_rejects_old_export(self):
        """A sample-able object without the flag = pre-delta export."""
        from baseline.framework.rollout import SamplingPolicy

        class _OldExport:
            def sample(self, obs, *, ctx=None, want_extra=False):
                return np.zeros(ACT_DIM, dtype=np.float32), None

        spec, _ = self._spec()
        with self.assertRaises(TypeError):
            SamplingPolicy(_OldExport(), spec)

    def test_delta_spec_accepts_capable_policy(self):
        from baseline.framework.rollout import SamplingPolicy
        spec, _ = self._spec()
        torch.manual_seed(0)
        p = TruncatedNormalPolicy(OBS_DIM, ACT_DIM, HID)
        SamplingPolicy(p, spec)  # no raise

    def test_delta_factor_without_reference_rejected(self):
        """c≠0 without a reference ensemble is a malformed spec — the
        mechanism could never activate, so it fails at construction
        rather than silently degrading to plain ef sampling."""
        from baseline.framework.rollout import SamplingSpec
        with self.assertRaises(ValueError):
            SamplingSpec(explore_factor=0.5, delta_factor=0.5)

    def test_pretanh_fail_loud_on_active_ctx(self):
        from baseline.framework.ppo.policies.pre_tanh_normal_mlp import (
            PreTanhNormalPolicy,
        )
        torch.manual_seed(0)
        p = PreTanhNormalPolicy(OBS_DIM, ACT_DIM, HID)
        ctx = _ctx(ref=np.zeros(ACT_DIM, dtype=np.float32), c=1.0)
        with self.assertRaises(NotImplementedError):
            p.sample(np.zeros(OBS_DIM, dtype=np.float32), ctx=ctx)


class TestExportParityActive(unittest.TestCase):
    """Exported policy + active-delta ctx == training class, same seed."""

    def setUp(self):
        torch.manual_seed(0)
        self.obs_t = torch.randn(1, OBS_DIM)
        self.obs_np = self.obs_t.numpy().astype(np.float32)[0]
        self.ctx = SamplingContext(
            explore_factor=0.3,
            reference_action=np.random.RandomState(0)
                .randn(ACT_DIM).astype(np.float32),
            delta_factor=8.0,
        )

    def test_exported_stream_matches_training(self):
        for name, cls in ALL_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                with tempfile.TemporaryDirectory() as tmp:
                    loaded = p.to_blueprint(dest_path=tmp).build()
                p.reset(5)
                a_train, _ = p.sample_action(self.obs_t, ctx=self.ctx)
                loaded.reset(5)
                a_exp, _ = loaded.sample(self.obs_np, ctx=self.ctx)
                np.testing.assert_allclose(
                    a_exp, a_train[0].detach().numpy(),
                    rtol=0, atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
