"""Tests for per-update parameter resolution (resolve_update_params).

Covers routing to CommonParams/PPOParams, unknown/blacklisted field
rejection, validation through dataclasses.replace, and the precedence
merge the loop applies (experiment hook first, CLI patches win).
"""

import dataclasses
import unittest

from baseline.framework.ppo.experiment import (
    CommonParams,
    PPOParams,
    resolve_update_params,
)


def _cp() -> CommonParams:
    return CommonParams(
        name="test_exp",
        learning_rate=3e-4,
        critic_learning_rate=1e-3,
        grad_clip_norm=1.0,
        episodes_per_update=16,
        max_updates=10,
        eval_interval=5,
        eval_episodes=8,
        video_eval_interval=0,
        rollout_workers=4,
        seed=7,
    )


def _pp() -> PPOParams:
    return PPOParams(
        clip_eps=0.2,
        target_kl=0.05,
        update_epochs=4,
        minibatch_size=64,
        early_stop_kl_window=8,
    )


class TestResolveUpdateParams(unittest.TestCase):

    def test_no_override_returns_base(self):
        base_cp, base_pp = _cp(), _pp()
        cp, pp, applied = resolve_update_params(base_cp, base_pp, None)
        self.assertIs(cp, base_cp)
        self.assertIs(pp, base_pp)
        self.assertEqual(applied, {})

        cp, pp, applied = resolve_update_params(base_cp, base_pp, {})
        self.assertIs(cp, base_cp)
        self.assertIs(pp, base_pp)
        self.assertEqual(applied, {})

    def test_pp_field_override(self):
        cp, pp, applied = resolve_update_params(
            _cp(), _pp(), {"dual_clip_c": 3.0},
        )
        self.assertEqual(pp.dual_clip_c, 3.0)
        self.assertEqual(pp.minibatch_size, 64)  # untouched
        self.assertEqual(applied, {"dual_clip_c": 3.0})

    def test_cp_field_override(self):
        cp, pp, applied = resolve_update_params(
            _cp(), _pp(), {"learning_rate": 1e-4, "episodes_per_update": 32},
        )
        self.assertEqual(cp.learning_rate, 1e-4)
        self.assertEqual(cp.episodes_per_update, 32)
        self.assertEqual(cp.max_updates, 10)
        self.assertEqual(applied, {
            "learning_rate": 1e-4, "episodes_per_update": 32,
        })

    def test_max_updates_override(self):
        cp, _, _ = resolve_update_params(_cp(), _pp(), {"max_updates": 500})
        self.assertEqual(cp.max_updates, 500)

    def test_adv_norm_override(self):
        _, pp, _ = resolve_update_params(
            _cp(), _pp(), {"adv_norm": "gauss_rank"},
        )
        self.assertEqual(pp.adv_norm, "gauss_rank")

    def test_unknown_field_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_update_params(_cp(), _pp(), {"no_such_param": 1})
        self.assertIn("no_such_param", str(ctx.exception))

    def test_blacklisted_fields_rejected(self):
        for field in ("name", "seed", "rollout_workers"):
            with self.assertRaises(ValueError, msg=field) as ctx:
                resolve_update_params(_cp(), _pp(), {field: 1})
            self.assertIn(field, str(ctx.exception))

    def test_invalid_value_rejected(self):
        # dual_clip_c in (0, 1) is invalid per PPOParams.__post_init__.
        with self.assertRaises(ValueError):
            resolve_update_params(_cp(), _pp(), {"dual_clip_c": 0.5})
        with self.assertRaises(ValueError):
            resolve_update_params(_cp(), _pp(), {"adv_norm": "bogus"})

    def test_override_is_pure_and_resumable(self):
        # Resolution is stateless: the same update index on a resumed
        # run produces the identical effective params — no event log
        # replay needed.
        base_cp, base_pp = _cp(), _pp()
        ovr = {"dual_clip_c": 3.0}
        cp1, pp1, a1 = resolve_update_params(base_cp, base_pp, ovr)
        cp2, pp2, a2 = resolve_update_params(base_cp, base_pp, ovr)
        self.assertEqual(pp1, pp2)
        self.assertEqual(cp1, cp2)
        self.assertEqual(a1, a2)
        # Base params never mutated.
        self.assertEqual(base_pp.dual_clip_c, 0.0)

    def test_cli_patch_wins_over_experiment_hook(self):
        # The loop merges {**exp_ovr, **cli_ovr} — verify the semantics.
        exp_ovr = {"dual_clip_c": 3.0, "learning_rate": 1e-4}
        cli_ovr = {"dual_clip_c": 5.0}
        merged = {**exp_ovr, **cli_ovr}
        _, pp, applied = resolve_update_params(_cp(), _pp(), merged)
        self.assertEqual(pp.dual_clip_c, 5.0)
        self.assertEqual(applied["learning_rate"], 1e-4)

    def test_patch_gating_by_update(self):
        # [(from_update, field, value)] semantics: applies when u >= from.
        patches = [(282, "dual_clip_c", 3.0)]
        for u, expect in ((281, 0.0), (282, 3.0), (300, 3.0)):
            cli_ovr = {k: v for frm, k, v in patches if frm <= u}
            _, pp, _ = resolve_update_params(_cp(), _pp(), cli_ovr)
            self.assertEqual(pp.dual_clip_c, expect, msg=f"u={u}")


if __name__ == "__main__":
    unittest.main()
