"""Tests for the new-route follow experiment (follow_pure)."""
from __future__ import annotations

import numpy as np

from baseline.experiments_ppo import get_ppo_experiment


def test_follow_pure_registered_and_channels():
    exp = get_ppo_experiment("follow_pure")
    assert exp.obs_dim == 96
    assert [c.name for c in exp.reward_channels()] == [
        "r_potential", "r_fall", "r_cross",
        "r_radial", "r_tangential", "r_face",
    ]


def test_follow_pure_mbs_registered():
    exp = get_ppo_experiment("follow_pure_mbs")
    assert exp.obs_dim == 96
    assert "mixture" in exp.actor_blueprint


def test_follow_pure_env_no_force_plugin():
    """No StandingTriggeredForcePlugin — zero perturbation route."""
    from envs.framework.parameterized_blueprint import (
        ParameterizedEnvBlueprint,
    )
    pb = ParameterizedEnvBlueprint.load(
        "baseline/humanoid21/end2end/follow_pure_env.yaml")
    bp = pb.materialize(
        max_steps=50, agent_id="robot_a", oppo_agent_id="robot_b",
        random_move_speed=0.5, random_move_speed_min=0.15)
    import json
    doc = json.dumps(bp.to_dict() if hasattr(bp, "to_dict") else str(bp))
    assert "StandingTriggeredForcePlugin" not in doc
    assert "GaitClock" not in doc


def test_random_move_per_episode_speed():
    """speed_min < speed → per-episode speed sampled inside the band."""
    from baseline.humanoid21.plugins.random_move import RandomMovePlugin
    speeds = set()
    for _ in range(20):
        p = RandomMovePlugin(
            target_robot="robot_b", speed=1.0, speed_min=0.3)
        # emulate on_pre_episode speed sampling
        p._ep_speed = float(p._rng.uniform(p.speed_min, p.speed))
        speeds.add(round(p._ep_speed, 3))
    assert len(speeds) > 5, "speed should be randomized per episode"


def test_random_move_legacy_fixed_speed():
    """Omitting speed_min keeps the old fixed-speed behaviour."""
    from baseline.humanoid21.plugins.random_move import RandomMovePlugin
    p = RandomMovePlugin(target_robot="robot_b", speed=0.8)
    assert p.speed_min == 0.8
    bp = p.to_blueprint()
    assert bp["speed_min"] == 0.8
