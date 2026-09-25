"""Tests for ``--set KEY=VALUE`` → experiment class-attribute patching.

``train.py`` forwards ``--set`` pairs as kwargs to the experiment
constructor; ``CombatExperimentPPOBase.__init__`` coerces each string
to the declared class attribute's type and rejects unknown keys.
"""
import unittest

from baseline.experiments_ppo import get_ppo_experiment


class TestSetKwargs(unittest.TestCase):
    def test_float_attr(self):
        e = get_ppo_experiment("standup_floor04", uncertainty_floor="0.5")
        self.assertEqual(e.uncertainty_floor, 0.5)
        self.assertIsInstance(e.uncertainty_floor, float)

    def test_int_attr(self):
        e = get_ppo_experiment("standup_floor04", seed="7")
        self.assertEqual(e.seed, 7)
        self.assertIsInstance(e.seed, int)

    def test_str_attr(self):
        e = get_ppo_experiment(
            "standup_floor04", actor_blueprint="custom.yaml")
        self.assertEqual(e.actor_blueprint, "custom.yaml")

    def test_multiple(self):
        e = get_ppo_experiment(
            "standup_floor04",
            uncertainty_floor="0.5",
            uncertainty_coef="2.0",
            explore_factor="0.3",
        )
        self.assertEqual(e.uncertainty_floor, 0.5)
        self.assertEqual(e.uncertainty_coef, 2.0)
        self.assertEqual(e.explore_factor, 0.3)

    def test_unknown_key_raises(self):
        with self.assertRaises(TypeError):
            get_ppo_experiment("standup_floor04", uncertainty_flor="0.5")

    def test_method_name_rejected(self):
        """A key matching a method is not a settable attribute."""
        with self.assertRaises(TypeError):
            get_ppo_experiment("standup_floor04", reward_channels="x")

    def test_private_name_rejected(self):
        with self.assertRaises(TypeError):
            get_ppo_experiment("standup_floor04", _private="x")

    def test_no_kwargs_unchanged(self):
        """Default construction is unaffected."""
        e = get_ppo_experiment("standup_floor04")
        self.assertEqual(e.uncertainty_floor, 0.4)
        self.assertEqual(e.name, "standup_floor04")

    def test_instance_does_not_leak_to_class(self):
        """--set values land on the instance; class default survives."""
        get_ppo_experiment("standup_floor04", uncertainty_floor="0.9")
        e2 = get_ppo_experiment("standup_floor04")
        self.assertEqual(e2.uncertainty_floor, 0.4)

    def test_bad_value_raises(self):
        with self.assertRaises(ValueError):
            get_ppo_experiment("standup_floor04", uncertainty_floor="abc")


class TestSetKwargsSAC(unittest.TestCase):
    """Same mechanism on the SAC base (incl. bool coercion)."""

    def _sac_exp_name(self):
        from baseline.experiments_sac import list_sac_experiments
        names = list_sac_experiments()
        self.assertTrue(names, "no SAC experiments registered")
        return names[0]

    def test_bool_attr(self):
        from baseline.experiments_sac import get_sac_experiment
        name = self._sac_exp_name()
        e = get_sac_experiment(name, auto_alpha="false")
        self.assertFalse(e.auto_alpha)
        e = get_sac_experiment(name, auto_alpha="true")
        self.assertTrue(e.auto_alpha)

    def test_unknown_key_raises(self):
        from baseline.experiments_sac import get_sac_experiment
        with self.assertRaises(TypeError):
            get_sac_experiment(self._sac_exp_name(), no_such_attr="1")


if __name__ == "__main__":
    unittest.main()
