"""Tests for ``PolicyEvaluator``, ``bootstrap_ci``, ``head_to_head_winrate``.

Pin the contract from ``baseline/DESIGN.md`` §3.7.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "envs" / "framework" / "tests"))

from envs.framework.env_runtime import EnvRuntime
from envs.framework.observer_plugin import BaseObserverPlugin
from conftest import MockSimulator  # type: ignore[import-not-found]

from baseline.common.eval import (
    EvalReport,
    MetricStats,
    PolicyEvaluator,
    bootstrap_ci,
    head_to_head_winrate,
)
from baseline.common.policies import TanhGaussianMLPPolicy
from baseline.common.rollout import RolloutBatch


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------
class TestBootstrapCI:
    def test_brackets_true_mean(self):
        # Sample from N(0, 1), check 95% CI brackets 0 in most runs.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        low, high = bootstrap_ci(x, n_samples=2000, alpha=0.05, rng=rng)
        assert low < 0.0 < high

    def test_empty_returns_nan(self):
        low, high = bootstrap_ci(np.array([]), n_samples=100)
        assert np.isnan(low) and np.isnan(high)

    def test_single_value_yields_degenerate_ci(self):
        low, high = bootstrap_ci(np.array([3.5]), n_samples=100)
        assert low == 3.5 and high == 3.5

    def test_seed_determinism(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = bootstrap_ci(x, n_samples=500, rng=np.random.default_rng(42))
        b = bootstrap_ci(x, n_samples=500, rng=np.random.default_rng(42))
        assert a == b


# ---------------------------------------------------------------------------
# head_to_head_winrate
# ---------------------------------------------------------------------------
class TestHeadToHead:
    def test_basic_counts(self):
        a = np.array([1.0, 2.0, 0.0, 3.0])
        b = np.array([0.0, 2.0, 1.0, 1.0])
        # A wins on idx 0, 3; tie on 1; B wins on 2.
        out = head_to_head_winrate(a, b)
        assert out.win_rate == 0.5
        assert out.draw_rate == 0.25
        assert out.loss_rate == 0.25
        assert out.n == 4
        assert out.ci_lower is None  # no bootstrap requested

    def test_bootstrap_ci_brackets_winrate(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(200)
        b = rng.standard_normal(200)
        out = head_to_head_winrate(
            a, b, bootstrap_samples=1000, alpha=0.05, seed=0,
        )
        assert out.ci_lower is not None and out.ci_upper is not None
        assert out.ci_lower <= out.win_rate <= out.ci_upper

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="align"):
            head_to_head_winrate(np.zeros(3), np.zeros(4))

    def test_empty_returns_nan_winrate(self):
        out = head_to_head_winrate(np.array([]), np.array([]))
        assert np.isnan(out.win_rate)
        assert out.n == 0


# ---------------------------------------------------------------------------
# PolicyEvaluator (integration)
# ---------------------------------------------------------------------------
OBS_DIM = 5
ACTION_DIM = 21


class _QposObserver(BaseObserverPlugin):
    def __init__(self) -> None:
        self._output = np.zeros(OBS_DIM, dtype=np.float32)

    def on_pre_episode(self, ctx) -> None:
        self._output = ctx.accessor.get_core_state()["qpos"][:OBS_DIM].astype(np.float32)

    def on_post_action_step(self, ctx) -> None:
        self._output = ctx.accessor.get_core_state()["qpos"][:OBS_DIM].astype(np.float32)

    def get_output(self) -> np.ndarray:
        return self._output.copy()


class _ConstantRewardObserver(BaseObserverPlugin):
    """Reward = constant (lets us pin per-episode return exactly)."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def on_pre_episode(self, ctx) -> None:
        pass

    def on_post_action_step(self, ctx) -> None:
        pass

    def get_output(self) -> float:
        return self._value


def _make_runtime(max_steps: int = 4) -> EnvRuntime:
    return EnvRuntime(
        simulator=MockSimulator(),
        observer_plugins={
            "robot_a_obs": _QposObserver(),
            "robot_a_reward": _ConstantRewardObserver(value=1.0),
            "robot_b_obs": _QposObserver(),
            "robot_b_reward": _ConstantRewardObserver(value=0.5),
        },
        max_steps=max_steps,
        phy_steps_per_action=1,
    )


def _make_policy() -> TanhGaussianMLPPolicy:
    torch.manual_seed(0)
    return TanhGaussianMLPPolicy(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=8, device="cpu", deterministic=True
    )


class TestEvaluator:
    def test_default_metrics_match_expected_returns(self):
        # max_steps=4 → length=4; rewards=[1,1,1,1] → return=4 for A, 2 for B.
        evaluator = PolicyEvaluator(
            runtime_factory=lambda: _make_runtime(max_steps=4),
            policy_factories={
                "robot_a": _make_policy,
                "robot_b": _make_policy,
            },
        )
        try:
            report = evaluator.evaluate(n=3, base_seed=0)
        finally:
            evaluator.close()

        assert isinstance(report, EvalReport)
        assert report.num_episodes == 3
        assert report.per_agent["robot_a"]["return"].mean == pytest.approx(4.0)
        assert report.per_agent["robot_b"]["return"].mean == pytest.approx(2.0)
        assert report.per_agent["robot_a"]["length"].mean == pytest.approx(4.0)
        # No bootstrap → ci_lower / ci_upper None
        assert report.per_agent["robot_a"]["return"].ci_lower is None

    def test_bootstrap_ci_populated_when_requested(self):
        evaluator = PolicyEvaluator(
            runtime_factory=lambda: _make_runtime(max_steps=3),
            policy_factories={
                "robot_a": _make_policy,
                "robot_b": _make_policy,
            },
        )
        try:
            report = evaluator.evaluate(
                n=8, base_seed=0,
                bootstrap_samples=500, bootstrap_alpha=0.05,
                bootstrap_seed=0,
            )
        finally:
            evaluator.close()

        stats = report.per_agent["robot_a"]["return"]
        assert stats.ci_lower is not None and stats.ci_upper is not None
        # All returns are exactly 3.0 (constant reward, length 3) so CI
        # should be degenerate around 3.0.
        assert stats.ci_lower == pytest.approx(3.0, abs=1e-9)
        assert stats.ci_upper == pytest.approx(3.0, abs=1e-9)
        assert stats.alpha == 0.05

    def test_custom_metric_overrides_default(self):
        def constant_seven(b: RolloutBatch) -> float:
            return 7.0

        evaluator = PolicyEvaluator(
            runtime_factory=lambda: _make_runtime(max_steps=2),
            policy_factories={
                "robot_a": _make_policy,
                "robot_b": _make_policy,
            },
        )
        try:
            report = evaluator.evaluate(
                n=2, base_seed=0,
                metric_fns={"return": constant_seven, "extra": constant_seven},
            )
        finally:
            evaluator.close()

        assert report.per_agent["robot_a"]["return"].mean == 7.0
        assert report.per_agent["robot_a"]["extra"].mean == 7.0

    def test_capture_agents_filters_report(self):
        evaluator = PolicyEvaluator(
            runtime_factory=lambda: _make_runtime(max_steps=2),
            policy_factories={
                "robot_a": _make_policy,
                "robot_b": _make_policy,
            },
            capture_agents=("robot_a",),
        )
        try:
            report = evaluator.evaluate(n=2, base_seed=0)
        finally:
            evaluator.close()
        assert set(report.per_agent.keys()) == {"robot_a"}

    def test_metric_stats_repr_contains_name_and_mean(self):
        s = MetricStats(name="return", mean=1.5, std=0.5, n=10)
        assert "return" in repr(s)
        assert "1.5" in repr(s)
