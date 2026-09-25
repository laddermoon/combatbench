"""Per-policy RNG replay tests — the uniform generator contract.

Every truncated-normal cell owns a private ``torch.Generator`` seeded
per episode by ``Policy.reset(seed)``:

- ``reset(s)`` then sampling, then ``reset(s)`` then sampling must be
  bit-identical (episode-level replay).
- Two instances reset with different seeds produce independent streams
  — interleaved sampling never overwrites the other stream (this was
  impossible under the old ``torch.manual_seed`` global-RNG scheme).
- A shared instance (rollout self-play case) follows last-reset-wins
  semantics — deterministic, documented.
- Exported policies implement the same contract, and the exported
  stream is bit-identical to the training class's stream given the
  same weights and seed (same generator algorithm, same draw order).
"""
from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch

from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
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

POLICY_CLASSES = {
    "truncnorm": TruncatedNormalPolicy,
    "state_truncnorm": StateTruncatedNormalPolicy,
    "bounded": BoundedStdTruncatedNormalPolicy,
    "state_bounded": StateBoundedStdTruncatedNormalPolicy,
    "mixture": MixtureTruncatedNormalPolicy,
    "mixture_shared": SharedMixtureTruncatedNormalPolicy,
    "mixture_shared_bounded": SharedMixtureBoundedStdTruncatedNormalPolicy,
    "mixture_state_bounded": StateMixtureBoundedStdTruncatedNormalPolicy,
}


def _make(cls, seed_net: int = 0):
    torch.manual_seed(seed_net)
    return cls(OBS_DIM, ACT_DIM, HID)


class TestTrainingPolicyReplay(unittest.TestCase):
    """reset(seed) replay + per-instance stream independence."""

    def setUp(self):
        torch.manual_seed(0)
        self.obs = torch.randn(8, OBS_DIM)

    def _replay_actions(self, p, seed, obs=None):
        p.reset(seed)
        a, _ = p.sample_action(self.obs if obs is None else obs)
        return a

    def test_reset_replay(self):
        for name, cls in POLICY_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                a1 = self._replay_actions(p, 7)
                a2 = self._replay_actions(p, 7)
                self.assertTrue(torch.equal(a1, a2))

    def test_replay_with_explore_factor(self):
        for name in ("bounded", "state_bounded", "mixture"):
            cls = POLICY_CLASSES[name]
            with self.subTest(cell=name):
                p = _make(cls)
                p.reset(9)
                a1, _ = p.sample_action(self.obs, explore_factor=0.5)
                p.reset(9)
                a2, _ = p.sample_action(self.obs, explore_factor=0.5)
                self.assertTrue(torch.equal(a1, a2))
                self.assertFalse(torch.equal(a1, torch.zeros_like(a1)))

    def test_independent_streams(self):
        """Two instances seeded differently never disturb each other."""
        for name, cls in POLICY_CLASSES.items():
            with self.subTest(cell=name):
                p, q = _make(cls, 0), _make(cls, 0)
                q.load_state_dict(p.state_dict())
                p.reset(111)
                q.reset(222)
                # Interleaved sampling from both streams.
                a_p, _ = p.sample_action(self.obs)
                a_q, _ = q.sample_action(self.obs)
                # Solo replays must reproduce the same draws.
                p.reset(111)
                a_p2, _ = p.sample_action(self.obs)
                q.reset(222)
                a_q2, _ = q.sample_action(self.obs)
                self.assertTrue(torch.equal(a_p, a_p2))
                self.assertTrue(torch.equal(a_q, a_q2))
                # The two seeds really do give different streams.
                self.assertFalse(torch.equal(a_p, a_q))

    def test_shared_instance_last_reset_wins(self):
        """Documented degenerate semantics: one instance, two resets."""
        for name, cls in POLICY_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                p.reset(1)
                p.reset(2)
                a1, _ = p.sample_action(self.obs)
                p.reset(2)
                a2, _ = p.sample_action(self.obs)
                self.assertTrue(torch.equal(a1, a2))


class TestExportedPolicyReplay(unittest.TestCase):
    """Exported policies honor the same reset contract and the same
    sampling stream as their training classes."""

    def setUp(self):
        torch.manual_seed(0)
        self.obs_t = torch.randn(4, OBS_DIM)
        self.obs_np = self.obs_t.numpy().astype(np.float32)[0]

    def _export(self, cls):
        p = _make(cls)
        with tempfile.TemporaryDirectory() as tmp:
            bp = p.to_blueprint(dest_path=tmp)
            loaded = bp.build()
        return p, loaded

    def test_exported_reset_replay(self):
        for name, cls in POLICY_CLASSES.items():
            with self.subTest(cell=name):
                _, loaded = self._export(cls)
                loaded.reset(5)
                a1, _ = loaded.sample(self.obs_np)
                loaded.reset(5)
                a2, _ = loaded.sample(self.obs_np)
                np.testing.assert_array_equal(a1, a2)

    def test_exported_stream_matches_training(self):
        """Same weights + same seed → same draws, training vs export.

        The underlying uniform draws are identical (same generator);
        the post-draw sampling math differs by at most ~1 ulp between
        the training class and the export template, so the comparison
        uses a 1e-6 envelope instead of exact equality.
        """
        for name, cls in POLICY_CLASSES.items():
            with self.subTest(cell=name):
                p, loaded = self._export(cls)
                p.reset(5)
                # Single-row obs: multinomial's RNG consumption is
                # batch-shape dependent, so batch sampling would draw
                # from a different stream offset than single-obs export.
                a_train, _ = p.sample_action(self.obs_t[:1])
                loaded.reset(5)
                a_exp, _ = loaded.sample(self.obs_np)
                np.testing.assert_allclose(
                    a_exp, a_train[0].detach().numpy(), rtol=0, atol=1e-6,
                )

    def test_exported_independent_streams(self):
        for name, cls in POLICY_CLASSES.items():
            with self.subTest(cell=name):
                p = _make(cls)
                with tempfile.TemporaryDirectory() as tmp:
                    bp = p.to_blueprint(dest_path=tmp)
                    la, lb = bp.build(), bp.build()
                la.reset(1)
                lb.reset(2)
                x1, _ = la.sample(self.obs_np)
                y1, _ = lb.sample(self.obs_np)
                la.reset(1)
                x2, _ = la.sample(self.obs_np)
                lb.reset(2)
                y2, _ = lb.sample(self.obs_np)
                np.testing.assert_array_equal(x1, x2)
                np.testing.assert_array_equal(y1, y2)
                self.assertFalse(np.array_equal(x1, y1))


if __name__ == "__main__":
    unittest.main()
