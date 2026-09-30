"""Tests for the UDS remote-inference path (RemoteSamplingPolicy +
InferenceServerHandle).

Covers the contract that matters for GPU rollout:

- Extras round-trip: keys/shapes/dtypes identical to SamplingPolicy's
  ``sctx__*`` recording (record_fields() contract).
- Per-episode reset: same seed → same noise stream → bit-identical
  actions regardless of server batching.
- Batch-composition independence: the same (obs, noise) request sent
  through different connection counts / arrival orders must return
  bit-identical results — this is the GPU↔GPU determinism property.
- CPU↔GPU numerical equivalence (small tolerance — different math
  libraries, never bitwise).

The server device defaults to ``cpu`` so the suite runs anywhere; the
cross-instance GPU test is skipped without CUDA.
"""
from __future__ import annotations

import socket
import tempfile
import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from baseline.framework.ppo.policies.shared_mixture_truncated_normal_mlp import (
    SharedMixtureTruncatedNormalPolicy,
)
from baseline.framework.rollout.inference_server import (
    InferenceServerHandle,
    send_act,
    send_register,
)
from baseline.framework.rollout.job import ReferenceSpec, SamplingSpec
from baseline.framework.rollout.remote_policy import RemoteSamplingPolicy
from envs.framework.policy import PolicyBlueprint

OBS_DIM = 32
ACTION_DIM = 8
HIDDEN_DIM = 32
NUM_COMPONENTS = 3
EF = 0.3


def _make_export(seed: int, tmp: str) -> "PolicyBlueprint":
    torch.manual_seed(seed)
    p = SharedMixtureTruncatedNormalPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_components=NUM_COMPONENTS,
        device="cpu",
    )
    return p.to_blueprint(dest_path=tmp)


class _ServerFixture(unittest.TestCase):
    """Spawns a real UDS inference server once per test class."""

    device = "cpu"
    capacity = 64
    N_REQ = 48

    @classmethod
    def setUpClass(cls):
        cls._tmp_inner = tempfile.mkdtemp(prefix="cb_ri_inner_")
        cls._tmp_ref = tempfile.mkdtemp(prefix="cb_ri_ref_")
        cls._bp_inner = _make_export(1000, cls._tmp_inner)
        cls._bp_ref = _make_export(2000, cls._tmp_ref)
        cls._spec_dict = SamplingSpec(
            explore_factor=EF,
            reference=ReferenceSpec(policies=(cls._bp_ref,), weights=(1.0,)),
            delta_factor=5.0,
            delta_mix=1.0,
        ).to_dict()
        cls._srv = InferenceServerHandle(
            device=cls.device, capacity=cls.capacity,
        )
        cls._srv.wait_ready(timeout=120.0)

    @classmethod
    def tearDownClass(cls):
        cls._srv.close()

    def _remote(self, seed: int = 0) -> RemoteSamplingPolicy:
        spec = dict(self._spec_dict)
        spec["_remote_addr"] = self._srv.address
        rp = RemoteSamplingPolicy(self._bp_inner.to_dict(), spec)
        rp.reset(seed)
        return rp

    def _raw_conn(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(60.0)
        s.connect(self._srv.address)
        sid = int(
            send_register(s, self._bp_inner.to_dict(), self._spec_dict)
            ["spec_id"]
        )
        return s, sid

    # --- shared helpers for batch-independence checks -----------------
    def _requests(self):
        rng = np.random.default_rng(7)
        return [
            (
                rng.random(OBS_DIM).astype(np.float32),
                rng.random(ACTION_DIM + 1).astype(np.float32),
                float(rng.uniform(-0.5, 0.5)),
            )
            for _ in range(self.N_REQ)
        ]

    def _round(self, reqs, n_conns):
        conns, sids = zip(*(self._raw_conn() for _ in range(n_conns)))
        sid = sids[0]
        outs = [None] * len(reqs)

        def worker(ci, idxs):
            for i in idxs:
                obs, noise, ef = reqs[i]
                a, lp, ref, delta = send_act(
                    conns[ci], sid, ef, True, obs, noise)
                outs[i] = (
                    a.copy(), lp,
                    None if ref is None else ref.copy(),
                    None if delta is None else delta.copy(),
                )

        threads = [
            threading.Thread(
                target=worker,
                args=(ci, list(range(ci, len(reqs), n_conns))),
            )
            for ci in range(n_conns)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for c in conns:
            c.close()
        return outs


class TestFunctional(_ServerFixture):
    def test_register_returns_dims(self):
        conn, sid = self._raw_conn()
        self.assertIsInstance(sid, int)
        conn.close()

    def test_act_shapes_and_extras(self):
        rp = self._remote(seed=42)
        obs = np.random.default_rng(0).random(OBS_DIM).astype(np.float32)
        action, extra = rp.act(obs, want_extra=True)
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))
        # Full extras contract (mirrors SamplingPolicy.record_fields).
        self.assertIn("log_prob", extra)
        self.assertIn("explore_factor", extra)
        self.assertIn("sctx__reference_action", extra)
        self.assertIn("sctx__delta_factor", extra)
        self.assertIn("sctx__delta_mix", extra)
        self.assertEqual(
            np.asarray(extra["sctx__reference_action"]).shape,
            (ACTION_DIM,),
        )
        rp.close()

    def test_reset_reproducibility(self):
        obs = np.random.default_rng(1).random(OBS_DIM).astype(np.float32)
        rp_a, rp_b = self._remote(seed=7), self._remote(seed=7)
        a1, _ = rp_a.act(obs, want_extra=True)
        a2, _ = rp_b.act(obs, want_extra=True)
        np.testing.assert_array_equal(a1, a2)

    def test_seed_changes_stream(self):
        obs = np.random.default_rng(2).random(OBS_DIM).astype(np.float32)
        rp1, rp2 = self._remote(seed=1), self._remote(seed=2)
        a1, _ = rp1.act(obs, want_extra=True)
        a2, _ = rp2.act(obs, want_extra=True)
        self.assertFalse(np.array_equal(a1, a2))

    def test_want_extra_false_omits_log_prob(self):
        rp = self._remote(seed=3)
        obs = np.random.default_rng(3).random(OBS_DIM).astype(np.float32)
        _, extra = rp.act(obs, want_extra=False)
        self.assertNotIn("log_prob", extra)
        self.assertIn("sctx__delta_factor", extra)


class TestBatchIndependence(_ServerFixture):
    """The determinism contract: identical (obs, noise, ef) must produce
    identical output regardless of how requests co-batch."""

    def test_arrival_order_bitwise(self):
        reqs = self._requests()
        serial = self._round(reqs, n_conns=1)
        fanned = self._round(reqs, n_conns=8)
        for i in range(len(reqs)):
            np.testing.assert_array_equal(serial[i][0], fanned[i][0])
            self.assertEqual(serial[i][1], fanned[i][1])
            np.testing.assert_array_equal(serial[i][2], fanned[i][2])
            np.testing.assert_array_equal(serial[i][3], fanned[i][3])


class TestFrozenDeltaWire(_ServerFixture):
    """delta_mode='frozen' over the remote path: the reply carries the
    Δ payload (sctx__delta) instead of reference_action."""

    def _frozen_remote(self, seed: int = 0) -> RemoteSamplingPolicy:
        spec = dict(self._spec_dict)
        spec["delta_mode"] = "frozen"
        spec["_remote_addr"] = self._srv.address
        rp = RemoteSamplingPolicy(self._bp_inner.to_dict(), spec)
        rp.reset(seed)
        return rp

    def test_frozen_extras_contract(self):
        rp = self._frozen_remote(seed=42)
        obs = np.random.default_rng(0).random(OBS_DIM).astype(np.float32)
        action, extra = rp.act(obs, want_extra=True)
        self.assertEqual(action.shape, (ACTION_DIM,))
        # Mutual exclusion: delta payload present, reference absent
        # (the payload itself is the mode marker — no flag field).
        self.assertIn("sctx__delta", extra)
        self.assertNotIn("sctx__delta_frozen", extra)
        self.assertNotIn("sctx__reference_action", extra)
        # Action-level payload (D,) — det_action(Gen0) − a_ref.
        self.assertEqual(
            np.asarray(extra["sctx__delta"]).shape, (ACTION_DIM,),
        )
        rp.close()

    def test_frozen_delta_matches_local(self):
        """Server-emitted Δ equals det_action(inner) − a_ref — the
        current policy supplies μ₀ via its deterministic action."""
        rp = self._frozen_remote(seed=99)
        obs = np.random.default_rng(5).random(OBS_DIM).astype(np.float32)
        _, extra = rp.act(obs, want_extra=True)
        inner = PolicyBlueprint.from_dict(self._bp_inner.to_dict()).build()
        ref = PolicyBlueprint.from_dict(self._bp_ref.to_dict()).build()
        with torch.no_grad():
            ref_act = ref._policy.deterministic_action(
                torch.as_tensor(obs).unsqueeze(0))
            mu0 = inner._policy.deterministic_action(
                torch.as_tensor(obs).unsqueeze(0))
        delta_local = (mu0.squeeze(0) - ref_act.squeeze(0)).numpy()
        np.testing.assert_allclose(
            extra["sctx__delta"], delta_local, atol=1e-5,
        )
        rp.close()


class TestNumericParity(_ServerFixture):
    """Remote reply vs. locally-computed sampling with the *same* injected
    uniform — isolates transport+server math from RNG differences."""

    def test_same_noise_close(self):
        obs = np.random.default_rng(5).random(OBS_DIM).astype(np.float32)
        rp = self._remote(seed=99)
        # Peek the noise the policy will use: replicate its stream.
        noise = np.random.default_rng(99).random(ACTION_DIM + 1).astype(
            np.float32,
        )
        action_r, extra_r = rp.act(obs, want_extra=True)

        # Local replica: inner net + ref net, same uniform, same ctx.
        inner = PolicyBlueprint.from_dict(self._bp_inner.to_dict()).build()
        ref = PolicyBlueprint.from_dict(self._bp_ref.to_dict()).build()
        with torch.no_grad():
            ref_act = ref._policy.deterministic_action(
                torch.as_tensor(obs).unsqueeze(0))
        ctx = SimpleNamespace(
            explore_factor=EF,
            reference_action=ref_act,
            delta_factor=5.0,
            delta_mix=1.0,
        )
        with torch.no_grad():
            a_l, lp_l = inner._policy.sample_action(
                torch.as_tensor(obs).unsqueeze(0), ctx=ctx,
                uniform=torch.as_tensor(noise).unsqueeze(0),
            )
        np.testing.assert_allclose(
            a_l.squeeze(0).numpy(), action_r, atol=1e-4,
        )
        self.assertAlmostEqual(float(lp_l.item()), extra_r["log_prob"],
                               places=3)


class TestGPUBitwise(_ServerFixture):
    """GPU↔GPU cross-instance bitwise check — separate server processes,
    concurrent connections.  Skipped without CUDA."""

    device = "cuda"

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        super().setUpClass()

    def test_cross_instance_bitwise(self):
        reqs = self._requests()
        out_a = self._round(reqs, n_conns=8)
        # Second independent server process on the same GPU.
        srv2 = InferenceServerHandle(device=self.device,
                                     capacity=self.capacity)
        srv2.wait_ready(timeout=120.0)
        try:
            saved = self._srv
            self._srv = srv2
            out_b = self._round(reqs, n_conns=8)
        finally:
            self._srv = saved
            srv2.close()
        for i in range(len(reqs)):
            np.testing.assert_array_equal(out_a[i][0], out_b[i][0])
            self.assertEqual(out_a[i][1], out_b[i][1])


if __name__ == "__main__":
    unittest.main()
