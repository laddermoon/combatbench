"""Regression test (P3-16 fix) — ConstantForcePlugin overrides not sticky.

``ConstantForcePlugin.on_pre_episode`` restores its ctor defaults before
applying ``episode_options["impulse_params"]`` — a per-episode override
no longer poisons later episodes, and ``to_blueprint`` always exports the
ctor config.

Run: PYTHONPATH=. python3 -m pytest envs/humanoid21/tests/test_audit_sticky_episode_options.py -q
"""
from __future__ import annotations

from envs.humanoid21.disturbance_plugins import ConstantForcePlugin


class _MiniCtx:
    def __init__(self, options):
        self.episode_options = options


def test_impulse_params_override_scoped_to_one_episode():
    p = ConstantForcePlugin(agent_id="robot_a", force=100.0, direction=0.0,
                            duration_action_steps=4, body_name="torso")

    # Episode 1: no override — ctor config active
    p.on_pre_episode(_MiniCtx({}))
    assert (p.force, p.direction, p.duration_action_steps, p.body_name) == \
        (100.0, 0.0, 4, "torso")

    # Episode 2: override applies
    p.on_pre_episode(_MiniCtx({
        "impulse_params": {"robot_a": {"force": 500, "direction_angle": 90,
                                       "duration_action_steps": 8,
                                       "body": "head"}},
    }))
    assert (p.force, p.direction, p.duration_action_steps, p.body_name) == \
        (500.0, 90.0, 8, "head")

    # Episode 3: no override — ctor config restored
    p.on_pre_episode(_MiniCtx({}))
    assert (p.force, p.direction, p.duration_action_steps, p.body_name) == \
        (100.0, 0.0, 4, "torso")


def test_to_blueprint_exports_ctor_config_after_override():
    p = ConstantForcePlugin(agent_id="robot_b", force=100.0)
    p.on_pre_episode(_MiniCtx({
        "impulse_params": {"robot_b": {"force": 999}},
    }))
    bp = p.to_blueprint()
    assert bp["force"] == 100.0  # ctor config, not the episode override
