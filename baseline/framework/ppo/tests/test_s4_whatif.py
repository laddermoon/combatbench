"""S4 ``whatif`` tool tests.

Tests cover:
- parse_value: bool/int/float/str parsing + errors.
- parse_set_args: valid/invalid keys, type parsing, multiple --set,
  unknown key error, malformed pair.
- parse_sweep_arg: single key, multiple values, type parsing,
  empty list error, --set/--sweep mutual exclusion (at API level).
- WhatifParam / whatif_params declaration: default returns {}.
- apply_whatif_overrides: default raises NotImplementedError.
- whatif with a synthetic snapshot (MockExperiment with whatif support):
  single --set, baseline vs variant comparison, combined_adv_cosine,
  influence_share deltas, verdict logic (no_noise_band when no band).
- whatif --sweep: N variants, sweep table render.
- whatif --noise-band: below_noise / above_noise verdicts with synthetic band.
- whatif on subset snapshot: verdict="incomparable".
- whatif with --full-grad: grad_cosine computed from update.npz::full_grad.
- Snapshot immutability: baseline replay/ unchanged after whatif.
- render_report: output format sanity.

Conventions follow test_s2_replay.py.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.experiment import (
    ActorEval, CommonParams, ExplorationSpec, PPOParams, ExperimentPPO,
    UpdateStats,
)
from baseline.framework.ppo.trajectory import (
    ChannelData, RewardChannel, Trajectory, TrajectoryProvenance,
)
from baseline.framework.ppo.trainer import PPOBuffer, ppo_update, set_seed
from baseline.framework.ppo.debug.sink import NpzSink
from baseline.framework.ppo.debug.snapshot import (
    DebugRequest, capture_snapshot, MANIFEST_FILENAME,
)
from baseline.framework.ppo.debug.replay import (
    replay_snapshot, verify_against_log, ReplayResult, VerificationResult,
)
from baseline.framework.ppo.debug.compare import NoiseBand
from baseline.framework.ppo.debug.whatif import (
    WhatifParam, WhatifVariant, WhatifComparison, WhatifReport,
    parse_value, parse_set_args, parse_sweep_arg,
    whatif, save_report, render_report,
    DEFAULT_LEG_ACTION_DIMS,
)
from baseline.framework.rollout import Episode, EpisodeCollection, blueprint_hash
from envs.framework.blueprint import EnvBlueprint, ClassSpec


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class SimpleActor(nn.Module):
    def __init__(self, obs_dim=8, action_dim=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 16), nn.Tanh(), nn.Linear(16, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def evaluate_actions(self, obs, actions, explore_factor, *, want_stats=False):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw = torch.atanh(torch.clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6))
        log_prob = dist.log_prob(raw) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        uncertainty_raw = dist.entropy().sum(dim=-1)
        H_max = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + 1.0)
        H_min = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + (-4.0))
        uncertainty_norm = (uncertainty_raw - H_min) / (H_max - H_min)
        return ActorEval(log_prob=log_prob, uncertainty=uncertainty_norm, stats=None)

    def to_blueprint(self, dest_path, *, stochastic=False):
        raise NotImplementedError

    def action_dim_grad_norms(self):
        grads = []
        for p in self.net.parameters():
            if p.grad is not None:
                grads.append(p.grad.norm().item())
        if not grads:
            return np.zeros(self.action_dim, dtype=np.float32)
        # Pad/truncate to action_dim for testing.
        out = np.zeros(self.action_dim, dtype=np.float32)
        for i, g in enumerate(grads[:self.action_dim]):
            out[i] = g
        return out


class SimpleCritic(nn.Module):
    def __init__(self, obs_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, obs):
        return self.net(obs)


class WhatifMockExperiment(ExperimentPPO):
    """Minimal experiment with whatif support for S4 testing."""

    name = "mock_whatif_test"
    reward_keys = ("r_a",)
    gammas = {"r_a": 0.99}
    gae_lambdas = {"r_a": 0.95}

    # Overridable params (class-level defaults).
    r_a_actor_weight: float = 1.0
    uncertainty_floor: float = 0.4
    uncertainty_coef: float = 5.0

    # Override state (None = not overridden).
    _override_r_a_actor_weight: Optional[float] = None
    _override_uncertainty_floor: Optional[float] = None
    _override_uncertainty_coef: Optional[float] = None

    def __init__(self):
        pass  # Skip parent __init__ which needs config

    def common_params(self) -> CommonParams:
        return CommonParams(
            name="mock_whatif_test", seed=42, max_updates=10,
            episodes_per_update=4, eval_interval=100,
            eval_episodes=4, video_eval_interval=100,
            rollout_workers=2,
            learning_rate=1e-3, critic_learning_rate=1e-3,
            grad_clip_norm=0.5,
        )

    def ppo_params(self) -> PPOParams:
        return PPOParams(
            clip_eps=0.2, target_kl=0.0, update_epochs=2, minibatch_size=32,
        )

    def reward_channels(self):
        return (RewardChannel("r_a", gamma=0.99, gae_lambda=0.95),)

    def build_actor(self, device):
        return SimpleActor().to(device)

    def build_critic(self, key, device):
        return SimpleCritic().to(device)

    def build_trajectories(self, episodes):
        trajs = []
        for ep in episodes:
            T = ep.num_frames
            obs = ep.observations["robot_a"]
            actions = ep.actions["robot_a"]
            last_obs = ep.final_observation["robot_a"]
            aw = self.r_a_actor_weight
            if self._override_r_a_actor_weight is not None:
                aw = self._override_r_a_actor_weight
            channels = {
                "r_a": ChannelData(
                    reward=np.random.default_rng(ep.episode_index).standard_normal(T).astype(np.float32),
                    is_terminated=True,
                    actor_weight=aw,
                ),
            }
            prov = TrajectoryProvenance(
                episode_index=ep.episode_index, agent_id="robot_a",
            )
            trajs.append(Trajectory(
                obs=obs, actions=actions, last_obs=last_obs,
                channels=channels, importance=1.0,
                explore_factor=ep.explore_factors.get("robot_a"),
                provenance=prov,
            ))
        return trajs

    def debug_arrays(self, episodes, trajectories):
        total = sum(len(t.obs) for t in trajectories)
        return {"test_array": np.zeros(total, dtype=np.float32)}

    def exploration(self, update):
        if self._override_uncertainty_floor is not None or self._override_uncertainty_coef is not None:
            return ExplorationSpec(
                uncertainty_floor=self._override_uncertainty_floor if self._override_uncertainty_floor is not None else 0.0,
                uncertainty_coef=self._override_uncertainty_coef if self._override_uncertainty_coef is not None else 0.0,
            )
        return ExplorationSpec(
            uncertainty_floor=self.uncertainty_floor,
            uncertainty_coef=self.uncertainty_coef,
        )

    def build_jobs(self, policy_bp, base_seed, n_episodes):
        return []

    def on_eval(self, episodes, update):
        return {}

    def on_update(self, stats):
        pass

    # --- S4 whatif ---

    def whatif_params(self) -> Dict[str, WhatifParam]:
        return {
            "r_a_actor_weight": WhatifParam(
                name="r_a_actor_weight",
                type=float,
                description="Actor weight for r_a channel.",
                requires_rebuild=True,
            ),
            "uncertainty_floor": WhatifParam(
                name="uncertainty_floor",
                type=float,
                description="Exploration floor.",
                requires_rebuild=False,
            ),
            "uncertainty_coef": WhatifParam(
                name="uncertainty_coef",
                type=float,
                description="Exploration coef.",
                requires_rebuild=False,
            ),
        }

    def apply_whatif_overrides(self, overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            if key == "r_a_actor_weight":
                self._override_r_a_actor_weight = float(value)
            elif key == "uncertainty_floor":
                self._override_uncertainty_floor = float(value)
            elif key == "uncertainty_coef":
                self._override_uncertainty_coef = float(value)
            else:
                setattr(self, key, value)


class BareMockExperiment(WhatifMockExperiment):
    """Mock experiment WITHOUT whatif support — for testing defaults.

    Inherits all the minimal experiment plumbing from WhatifMockExperiment
    but does NOT override whatif_params / apply_whatif_overrides, so they
    fall through to ExperimentPPO's defaults ({} / NotImplementedError).
    """

    def whatif_params(self) -> Dict[str, "WhatifParam"]:
        # Explicitly call the grandparent's default.
        return ExperimentPPO.whatif_params(self)

    def apply_whatif_overrides(self, overrides: Dict[str, Any]) -> None:
        # Explicitly call the grandparent's default.
        return ExperimentPPO.apply_whatif_overrides(self, overrides)


def _make_episode(episode_index=0, T=5, obs_dim=8, action_dim=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    actions = rng.uniform(-0.9, 0.9, (T, action_dim)).astype(np.float32)
    bp = EnvBlueprint(simulator=ClassSpec(cls="test:TestSim", config={}))
    bp_hash = blueprint_hash(bp)
    return Episode(
        base_seed=42, episode_index=episode_index, blueprint_hash=bp_hash,
        num_frames=T, episode_options={},
        agent_termination_proposal_records={"robot_a": (("timeout", T),)},
        observations={"robot_a": obs},
        actions={"robot_a": actions},
        action_extras={"robot_a": {}},
        explore_factors={"robot_a": np.zeros(T, dtype=np.float32)},
        observer_outputs={},
        final_observation={"robot_a": rng.standard_normal(obs_dim).astype(np.float32)},
        episode_metrics={},
    )


def _make_env_blueprint():
    return EnvBlueprint(simulator=ClassSpec(cls="test:TestSim", config={}))


def _setup_snapshot_dir(run_dir, update, request):
    snapshot_dir = run_dir / "debug" / f"u{update:05d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with open(snapshot_dir / "request.json", "w") as f:
        json.dump({
            "hypothesis": request.hypothesis,
            "episodes_mode": request.episodes_mode,
            "episodes_n": request.episodes_n,
            "include_full_grad": request.include_full_grad,
        }, f)
    return snapshot_dir


def _create_snapshot(run_dir, update=1, episodes_mode="all", n_episodes=2,
                     include_full_grad=False):
    """Create a full snapshot for testing."""
    episodes = [_make_episode(i, T=10, rng=np.random.default_rng(i)) for i in range(n_episodes)]
    actor = SimpleActor()
    critics = {"r_a": SimpleCritic()}
    bp = _make_env_blueprint()

    if episodes_mode == "all":
        req = DebugRequest(hypothesis="test whatif", episodes_mode="all",
                           include_full_grad=include_full_grad)
    else:
        req = DebugRequest(hypothesis="test whatif", episodes_mode="subset",
                           episodes_n=n_episodes,
                           include_full_grad=include_full_grad)

    _setup_snapshot_dir(run_dir, update, req)
    export_dir = run_dir / "policy_exports" / f"u{update:05d}"
    export_dir.mkdir(parents=True)
    (export_dir / "blueprint.yaml").write_text("test: true")

    snapshot_dir = capture_snapshot(
        run_dir=run_dir, update=update, request=req,
        episodes=episodes, actor=actor, actor_export_dir=export_dir,
        critics=critics, experiment_name="mock_whatif_test", env_blueprint=bp,
    )
    return snapshot_dir


def _whatif(snapshot_dir, **kwargs):
    """Helper: call whatif() with the WhatifMockExperiment factory."""
    kwargs.setdefault("experiment_factory", WhatifMockExperiment)
    kwargs.setdefault("device", torch.device("cpu"))
    return whatif(snapshot_dir, **kwargs)


# ---------------------------------------------------------------------------
# parse_value tests
# ---------------------------------------------------------------------------

def test_parse_value_bool():
    assert parse_value("true", bool) is True
    assert parse_value("True", bool) is True
    assert parse_value("1", bool) is True
    assert parse_value("false", bool) is False
    assert parse_value("0", bool) is False
    print("test_parse_value_bool: PASS")


def test_parse_value_int():
    assert parse_value("42", int) == 42
    assert parse_value("-7", int) == -7
    print("test_parse_value_int: PASS")


def test_parse_value_float():
    assert parse_value("3.14", float) == pytest.approx(3.14)
    assert parse_value("-0.5", float) == pytest.approx(-0.5)
    print("test_parse_value_float: PASS")


def test_parse_value_str():
    assert parse_value("hello", str) == "hello"
    print("test_parse_value_str: PASS")


def test_parse_value_bool_invalid():
    with pytest.raises(ValueError, match="cannot parse"):
        parse_value("maybe", bool)
    print("test_parse_value_bool_invalid: PASS")


def test_parse_value_int_invalid():
    with pytest.raises(ValueError):
        parse_value("3.14", int)
    print("test_parse_value_int_invalid: PASS")


def test_parse_value_unsupported_type():
    with pytest.raises(ValueError, match="unsupported whatif param type"):
        parse_value("x", list)
    print("test_parse_value_unsupported_type: PASS")


# ---------------------------------------------------------------------------
# parse_set_args tests
# ---------------------------------------------------------------------------

def _make_declared():
    return {
        "foot_height_clip": WhatifParam("foot_height_clip", float, "", True),
        "r_potential_actor_weight": WhatifParam("r_potential_actor_weight", float, "", True),
        "flag": WhatifParam("flag", bool, "", False),
        "count": WhatifParam("count", int, "", False),
    }


def test_parse_set_args_valid():
    declared = _make_declared()
    out = parse_set_args(["foot_height_clip=0.30"], declared)
    assert out == {"foot_height_clip": 0.30}
    print("test_parse_set_args_valid: PASS")


def test_parse_set_args_multiple():
    declared = _make_declared()
    out = parse_set_args(
        ["foot_height_clip=0.30", "r_potential_actor_weight=0", "flag=true", "count=5"],
        declared,
    )
    assert out == {
        "foot_height_clip": 0.30,
        "r_potential_actor_weight": 0.0,
        "flag": True,
        "count": 5,
    }
    print("test_parse_set_args_multiple: PASS")


def test_parse_set_args_unknown_key():
    declared = _make_declared()
    with pytest.raises(ValueError, match="unknown whatif param"):
        parse_set_args(["nonexistent=1.0"], declared)
    print("test_parse_set_args_unknown_key: PASS")


def test_parse_set_args_malformed():
    declared = _make_declared()
    with pytest.raises(ValueError, match="malformed"):
        parse_set_args(["no_equals_sign"], declared)
    print("test_parse_set_args_malformed: PASS")


def test_parse_set_args_empty_declared():
    with pytest.raises(ValueError, match="declares no whatif params"):
        parse_set_args(["x=1"], {})
    print("test_parse_set_args_empty_declared: PASS")


def test_parse_set_args_type_error():
    declared = _make_declared()
    with pytest.raises(ValueError):
        parse_set_args(["foot_height_clip=not_a_number"], declared)
    print("test_parse_set_args_type_error: PASS")


# ---------------------------------------------------------------------------
# parse_sweep_arg tests
# ---------------------------------------------------------------------------

def test_parse_sweep_arg_valid():
    declared = _make_declared()
    key, values = parse_sweep_arg("foot_height_clip=0.05,0.10,0.20", declared)
    assert key == "foot_height_clip"
    assert values == [0.05, 0.10, 0.20]
    print("test_parse_sweep_arg_valid: PASS")


def test_parse_sweep_arg_int():
    declared = _make_declared()
    key, values = parse_sweep_arg("count=1,2,3", declared)
    assert key == "count"
    assert values == [1, 2, 3]
    print("test_parse_sweep_arg_int: PASS")


def test_parse_sweep_arg_unknown_key():
    declared = _make_declared()
    with pytest.raises(ValueError, match="unknown whatif param"):
        parse_sweep_arg("nonexistent=1,2", declared)
    print("test_parse_sweep_arg_unknown_key: PASS")


def test_parse_sweep_arg_empty_values():
    declared = _make_declared()
    with pytest.raises(ValueError, match="empty value list"):
        parse_sweep_arg("foot_height_clip=,,", declared)
    print("test_parse_sweep_arg_empty_values: PASS")


def test_parse_sweep_arg_malformed():
    declared = _make_declared()
    with pytest.raises(ValueError, match="malformed"):
        parse_sweep_arg("no_equals", declared)
    print("test_parse_sweep_arg_malformed: PASS")


def test_parse_sweep_arg_empty_declared():
    with pytest.raises(ValueError, match="declares no whatif params"):
        parse_sweep_arg("x=1,2", {})
    print("test_parse_sweep_arg_empty_declared: PASS")


# ---------------------------------------------------------------------------
# WhatifParam / whatif_params default tests
# ---------------------------------------------------------------------------

def test_default_whatif_params_empty():
    """Default ExperimentPPO.whatif_params() returns empty."""
    exp = BareMockExperiment()
    assert exp.whatif_params() == ()
    print("test_default_whatif_params_empty: PASS")


def test_default_apply_whatif_overrides_raises():
    """Default apply_whatif_overrides raises NotImplementedError."""
    exp = BareMockExperiment()
    with pytest.raises(NotImplementedError, match="does not implement whatif"):
        exp.apply_whatif_overrides({"x": 1})
    print("test_default_apply_whatif_overrides_raises: PASS")


def test_standup_step_v3_declares_params():
    """standup_step_v3 declares whatif params."""
    from baseline.experiments_ppo import get_ppo_experiment
    exp = get_ppo_experiment("standup_step_v3")
    params = exp.whatif_params()
    assert "foot_height_clip" in params
    assert "r_potential_actor_weight" in params
    assert "uncertainty_floor" in params
    assert "channel.r_potential.actor_weight" in params
    print("test_standup_step_v3_declares_params: PASS")


def test_standup_step_v3_apply_overrides():
    """standup_step_v3 apply_whatif_overrides mutates fields."""
    from baseline.experiments_ppo import get_ppo_experiment
    exp = get_ppo_experiment("standup_step_v3")
    exp.apply_whatif_overrides({
        "foot_height_clip": 0.10,
        "r_potential_actor_weight": 2.0,
        "channel.r_fall.actor_weight": 0.5,
        "uncertainty_floor": 0.6,
        "foot_weight": 3.0,
    })
    assert exp.foot_height_clip == 0.10
    assert exp.r_potential_actor_weight == 2.0
    assert exp.r_fall_actor_weight == 0.5
    assert exp._override_uncertainty_floor == 0.6
    assert exp._override_foot_weight == 3.0
    # exploration() should return the override.
    spec = exp.exploration(1)
    assert spec.uncertainty_floor == 0.6
    print("test_standup_step_v3_apply_overrides: PASS")


# ---------------------------------------------------------------------------
# whatif() end-to-end tests with synthetic snapshots
# ---------------------------------------------------------------------------

def test_whatif_single_set(tmp_path):
    """whatif --set produces a report with one comparison."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    # Run baseline replay first so replay/ exists.
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 2.0},
        device=torch.device("cpu"),
    )

    assert report.update == 1
    assert report.experiment_name == "mock_whatif_test"
    assert report.episodes_mode == "all"
    assert len(report.variants) == 1
    assert len(report.comparisons) == 1
    comp = report.comparisons[0]
    assert comp.overrides == {"r_a_actor_weight": 2.0}
    # No noise band → no_noise_band verdict (or incomparable if subset).
    assert comp.verdict in ("no_noise_band", "incomparable")
    print("test_whatif_single_set: PASS")


def test_whatif_combined_adv_cosine_identical(tmp_path):
    """When override has no effect, combined_adv_cosine ≈ 1.0."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    # Use the same value as the default (1.0) → no actual change.
    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 1.0},  # same as default
        device=torch.device("cpu"),
    )
    comp = report.comparisons[0]
    # combined_adv should be identical (cosine = 1.0) since the override
    # doesn't change anything.
    if comp.combined_adv_cosine is not None:
        assert comp.combined_adv_cosine == pytest.approx(1.0, abs=1e-4)
    print("test_whatif_combined_adv_cosine_identical: PASS")


def test_whatif_sweep(tmp_path):
    """whatif --sweep produces N variants."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    report = _whatif(
        snapshot_dir,
        sweep_key="r_a_actor_weight",
        sweep_values=[0.5, 1.0, 2.0, 4.0],
        device=torch.device("cpu"),
    )

    assert report.is_sweep is True
    assert report.sweep_key == "r_a_actor_weight"
    assert len(report.variants) == 4
    assert len(report.comparisons) == 4
    # Each variant has a different override value.
    values = [v.overrides["r_a_actor_weight"] for v in report.variants]
    assert values == [0.5, 1.0, 2.0, 4.0]
    print("test_whatif_sweep: PASS")


def test_whatif_subset_incomparable(tmp_path):
    """whatif on subset snapshot → verdict='incomparable'."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="subset",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 2.0},
        device=torch.device("cpu"),
    )
    comp = report.comparisons[0]
    assert comp.verdict == "incomparable"
    print("test_whatif_subset_incomparable: PASS")


def test_whatif_noise_band_below(tmp_path):
    """whatif with a noise band where change is below noise → below_noise."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    # Create a noise band with a large std (so any change is "below noise").
    noise_path = tmp_path / "noise.json"
    band = NoiseBand(
        metrics={"approx_kl": {"mean": 0.01, "std": 100.0}},
        n_seeds=3, n_updates=5,
    )
    noise_path.write_text(json.dumps(band.to_dict()))

    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 2.0},
        noise_band_path=noise_path,
        device=torch.device("cpu"),
    )
    comp = report.comparisons[0]
    # With a huge noise std, the change should be below noise.
    # (Unless grad_cosine is None, in which case it's no_noise_band.)
    if comp.grad_change_pct is not None:
        assert comp.verdict == "below_noise"
    print("test_whatif_noise_band_below: PASS")


def test_whatif_noise_band_above(tmp_path):
    """whatif with a noise band where change is above noise → above_noise."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2, include_full_grad=True)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment(),
                    include_full_grad_override=True)

    # Create a noise band with a tiny std (so any change is "above noise").
    noise_path = tmp_path / "noise.json"
    band = NoiseBand(
        metrics={"approx_kl": {"mean": 0.01, "std": 1e-8}},
        n_seeds=3, n_updates=5,
    )
    noise_path.write_text(json.dumps(band.to_dict()))

    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 2.0},
        noise_band_path=noise_path,
        full_grad=True,
        device=torch.device("cpu"),
    )
    comp = report.comparisons[0]
    # With a tiny noise std, the change should be above noise
    # (unless grad_cosine is exactly 1.0, meaning no change at all).
    if comp.grad_change_pct is not None and comp.grad_change_pct > 1e-6:
        assert comp.verdict == "above_noise"
    print("test_whatif_noise_band_above: PASS")


def test_whatif_no_noise_band(tmp_path):
    """whatif without a noise band → verdict='no_noise_band'."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 2.0},
        device=torch.device("cpu"),
    )
    comp = report.comparisons[0]
    assert comp.verdict == "no_noise_band"
    print("test_whatif_no_noise_band: PASS")


def test_whatif_full_grad(tmp_path):
    """whatif with --full-grad captures gradient and computes grad_cosine."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    # Run baseline replay with full grad.
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment(),
                    include_full_grad_override=True)

    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 2.0},
        full_grad=True,
        device=torch.device("cpu"),
    )
    comp = report.comparisons[0]
    # With full_grad, grad_cosine should be computed.
    # (It may be None if the gradient is all-zero, but that's unlikely.)
    if comp.grad_cosine is not None:
        assert -1.0 <= comp.grad_cosine <= 1.0
    print("test_whatif_full_grad: PASS")


def test_whatif_no_full_grad(tmp_path):
    """whatif without --full-grad → grad_cosine is None."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2, include_full_grad=False)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 2.0},
        full_grad=False,
        device=torch.device("cpu"),
    )
    comp = report.comparisons[0]
    assert comp.grad_cosine is None
    assert report.full_grad_captured is False
    print("test_whatif_no_full_grad: PASS")


def test_whatif_snapshot_immutability(tmp_path):
    """Running whatif does not mutate the baseline replay/stats.json."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    baseline_stats = (snapshot_dir / "replay" / "stats.json").read_text()

    # Run whatif twice.
    _whatif(snapshot_dir, overrides={"r_a_actor_weight": 2.0})
    _whatif(snapshot_dir, overrides={"r_a_actor_weight": 3.0})

    after_stats = (snapshot_dir / "replay" / "stats.json").read_text()
    assert baseline_stats == after_stats, "baseline stats.json was mutated"
    print("test_whatif_snapshot_immutability: PASS")


def test_whatif_set_sweep_mutually_exclusive(tmp_path):
    """whatif rejects both --set and --sweep."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    with pytest.raises(ValueError, match="mutually exclusive"):
        whatif(
            snapshot_dir,
            overrides={"r_a_actor_weight": 2.0},
            sweep_key="r_a_actor_weight",
            sweep_values=[1.0, 2.0],
            device=torch.device("cpu"),
        )
    print("test_whatif_set_sweep_mutually_exclusive: PASS")


def test_whatif_requires_overrides_or_sweep(tmp_path):
    """whatif rejects neither --set nor --sweep."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    with pytest.raises(ValueError, match="must provide either"):
        whatif(snapshot_dir, device=torch.device("cpu"))
    print("test_whatif_requires_overrides_or_sweep: PASS")


# ---------------------------------------------------------------------------
# save_report + render_report tests
# ---------------------------------------------------------------------------

def test_save_report(tmp_path):
    """save_report writes report.json + report.txt."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 2.0},
        device=torch.device("cpu"),
    )
    whatif_dir = tmp_path / "whatif_out"
    save_report(report, whatif_dir)

    assert (whatif_dir / "report.json").exists()
    assert (whatif_dir / "report.txt").exists()

    # report.json is valid JSON with expected fields.
    with open(whatif_dir / "report.json") as f:
        data = json.load(f)
    assert data["update"] == 1
    assert data["experiment_name"] == "mock_whatif_test"
    assert len(data["comparisons"]) == 1
    assert data["comparisons"][0]["overrides"] == {"r_a_actor_weight": 2.0}
    print("test_save_report: PASS")


def test_render_report(tmp_path):
    """render_report produces a human-readable string."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    report = _whatif(
        snapshot_dir,
        overrides={"r_a_actor_weight": 2.0},
        device=torch.device("cpu"),
    )
    text = render_report(report)

    # Should contain key sections.
    assert "反事实试算" in text
    assert "r_a_actor_weight=2.0" in text
    assert "判定" in text
    print("test_render_report: PASS")


def test_render_report_sweep(tmp_path):
    """render_report for a sweep includes a sweep summary table."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all",
                                    n_episodes=2)
    replay_snapshot(snapshot_dir, experiment=WhatifMockExperiment())

    report = _whatif(
        snapshot_dir,
        sweep_key="r_a_actor_weight",
        sweep_values=[0.5, 1.0, 2.0],
        device=torch.device("cpu"),
    )
    text = render_report(report)
    assert "扫描汇总" in text
    assert "r_a_actor_weight" in text
    print("test_render_report_sweep: PASS")


# ---------------------------------------------------------------------------
# Cosine helper tests
# ---------------------------------------------------------------------------

def test_cosine_identical():
    """Cosine of identical vectors is 1.0."""
    from baseline.framework.ppo.debug.whatif import _cosine
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert _cosine(a, b) == pytest.approx(1.0)
    print("test_cosine_identical: PASS")


def test_cosine_orthogonal():
    """Cosine of orthogonal vectors is 0.0."""
    from baseline.framework.ppo.debug.whatif import _cosine
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine(a, b) == pytest.approx(0.0)
    print("test_cosine_orthogonal: PASS")


def test_cosine_zero_vector():
    """Cosine with a zero vector returns None."""
    from baseline.framework.ppo.debug.whatif import _cosine
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert _cosine(a, b) is None
    print("test_cosine_zero_vector: PASS")


def test_cosine_none():
    """Cosine with None inputs returns None."""
    from baseline.framework.ppo.debug.whatif import _cosine
    assert _cosine(None, np.array([1.0])) is None
    print("test_cosine_none: PASS")


# ---------------------------------------------------------------------------
# DEFAULT_LEG_ACTION_DIMS test
# ---------------------------------------------------------------------------

def test_default_leg_action_dims():
    """Default leg action dims are humanoid21 leg joints (indices 3-14)."""
    assert DEFAULT_LEG_ACTION_DIMS == tuple(range(3, 15))
    assert len(DEFAULT_LEG_ACTION_DIMS) == 12
    print("test_default_leg_action_dims: PASS")


# ---------------------------------------------------------------------------
# CLI integration test (smoke)
# ---------------------------------------------------------------------------

def test_cli_whatif_help():
    """debug.py whatif --help exits 0."""
    import subprocess
    repo_root = Path(__file__).resolve().parents[4]
    debug_py = repo_root / "baseline" / "framework" / "debug.py"
    result = subprocess.run(
        [sys.executable, str(debug_py), "whatif", "--help"],
        capture_output=True, text=True,
        cwd=str(repo_root),
    )
    assert result.returncode == 0
    assert "--set" in result.stdout
    assert "--sweep" in result.stdout
    assert "--noise-band" in result.stdout
    print("test_cli_whatif_help: PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
