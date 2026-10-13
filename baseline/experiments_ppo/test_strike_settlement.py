"""Unit tests for the strike_step settlement machinery.

Covers:
  - StrikeSettlementPlugin state machine: lock on first hand hit,
    settle-window damage accounting (locked hand only, standard
    damage formula), RETRACT transition, retract combo + hold,
    strike_done / retract_timeout terminations.
  - Non-locked-hand and foot contacts never score.
  - exp_strike_step r_strike channel: first-half payment on settle
    frames, second half on the strike_done terminal frame, withheld
    on retract_timeout.
"""
from __future__ import annotations

import numpy as np
import pytest

from baseline.humanoid21.plugins.strike_settlement import (
    HAND_LEFT,
    HAND_RIGHT,
    PHASE_PRE_HIT,
    PHASE_RETRACT,
    PHASE_SETTLE,
    StrikeSettlementPlugin,
    ELBOW_LEFT_IDX,
    ELBOW_RIGHT_IDX,
)
from envs.humanoid21.plugins import CombatScoringPlugin


# ----------------------------------------------------------------------
# Stubs — minimal ctx/accessor surface the plugin touches
# ----------------------------------------------------------------------

# body ids: 10..14 learner (robot_a), 20..23 opponent (robot_b)
_BODY_NAMES = {
    10: "hand_left_a",
    11: "hand_right_a",
    12: "foot_left_a",
    13: "torso_a",
    20: "head_b",
    21: "torso_b",
    22: "hand_left_b",
    23: "foot_left_b",
}


def _contacts(pairs):
    """Build a contacts dict from (body1, body2, aff1, aff2, force)."""
    n = len(pairs)
    return {
        "ncon": n,
        "body1": np.array([p[0] for p in pairs]),
        "body2": np.array([p[1] for p in pairs]),
        "aff1": np.array([p[2] for p in pairs]),
        "aff2": np.array([p[3] for p in pairs]),
        "force_mag": np.array([p[4] for p in pairs], dtype=np.float64),
    }


def _hit(force=400.0, attacker=10, target=20):
    """A learner(attacker body) → opponent(target body) contact."""
    return (attacker, target, 1, 2, force)


class _Accessor:
    def __init__(self, contacts=None, core=None, derived=None):
        self._contacts = contacts or {"ncon": 0}
        self._core = core or {}
        self._derived = derived or {}

    def get_derived_state(self, keys=None):
        out = {"contacts": self._contacts}
        out.update(self._derived)
        return out

    def get_static_data(self):
        return {"body_id_to_name": dict(_BODY_NAMES)}

    def get_core_state(self):
        return self._core


class _Ctx:
    def __init__(self, accessor):
        self.accessor = accessor
        self.metrics = {}
        self.events = []
        self.episode_step = 0
        self.physics_step = 0
        self.terminations = []

    def request_termination(self, reason, agent_id=None):
        self.terminations.append(reason)


def _core(root_h=1.2, elbow_l=0.8, elbow_r=0.8):
    jpn = np.zeros(21, dtype=np.float32)
    jpn[ELBOW_LEFT_IDX] = elbow_l
    jpn[ELBOW_RIGHT_IDX] = elbow_r
    return {
        "robot_a": {
            "root_pos": np.array([0.0, 0.0, root_h], dtype=np.float32),
            "joint_pos_norm": jpn,
        },
        "robot_b": {
            "root_pos": np.array([1.0, 0.0, 1.3], dtype=np.float32),
            "joint_pos_norm": jpn.copy(),
        },
    }


def _retract_derived(hand_pos=None, head=(1.0, 0.0, 1.5), torso=(1.0, 0.0, 1.0)):
    """derived state: locked hand far from opp head/torso by default."""
    if hand_pos is None:
        hand_pos = (0.0, 0.0, 1.2)   # ~1.55m from head, ~1.02m from torso
    return {
        "robot_a": {"body_xpos": {"hand_left_a": np.array(hand_pos)}},
        "robot_b": {"body_xpos": {
            "head_b": np.array(head), "torso_b": np.array(torso)}},
    }


def _plugin(**kw):
    return StrikeSettlementPlugin(agent_id="robot_a", **kw)


def _expected_damage(force, part="head"):
    w = CombatScoringPlugin.DAMAGE_RULES[part]
    eff = min(force, CombatScoringPlugin.MAX_DAMAGE_FORCE)
    return w * (eff / 100.0) ** 2 * 0.002


# ----------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------

def test_first_hit_locks_hand_and_scores():
    p = _plugin(settle_steps=3)
    ctx = _Ctx(_Accessor())
    p.on_pre_episode(ctx)
    assert p._phase == PHASE_PRE_HIT

    ctx.accessor._contacts = _contacts([_hit(400.0)])
    p.on_post_phy_step(ctx)
    assert p._phase == PHASE_SETTLE
    assert p._locked_hand == HAND_LEFT
    exp = _expected_damage(400.0, "head")
    assert ctx.metrics["robot_a_strike_step_damage"] == pytest.approx(exp)
    assert ctx.metrics["robot_a_strike_damage_total"] == pytest.approx(exp)
    assert ctx.metrics["robot_a_strike_hit_step"] == 0


def test_only_locked_hand_scores_during_settle():
    p = _plugin(settle_steps=5)
    ctx = _Ctx(_Accessor())
    p.on_pre_episode(ctx)

    # Left hand lands first.
    ctx.accessor._contacts = _contacts([_hit(400.0, attacker=10)])
    p.on_post_phy_step(ctx)
    # Right hand also touches — must NOT add damage.
    ctx.accessor._contacts = _contacts([
        _hit(800.0, attacker=11),           # right hand — ignored
        _hit(200.0, attacker=10, target=21)  # locked left hand on torso
    ])
    p.on_post_phy_step(ctx)
    exp = _expected_damage(400.0, "head") + _expected_damage(200.0, "torso")
    assert ctx.metrics["robot_a_strike_damage_total"] == pytest.approx(exp)


def test_foot_contact_never_locks():
    p = _plugin(settle_steps=5)
    ctx = _Ctx(_Accessor())
    p.on_pre_episode(ctx)
    ctx.accessor._contacts = _contacts([_hit(1000.0, attacker=12)])
    p.on_post_phy_step(ctx)
    assert p._phase == PHASE_PRE_HIT
    assert p._locked_hand == 0
    assert ctx.metrics["robot_a_strike_damage_total"] == 0.0


def test_settle_window_transitions_to_retract():
    p = _plugin(settle_steps=3)
    ctx = _Ctx(_Accessor())
    p.on_pre_episode(ctx)
    ctx.accessor._contacts = _contacts([_hit(400.0)])
    p.on_post_phy_step(ctx)

    for _ in range(2):          # steps 1,2 → still SETTLE
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
        assert p._phase == PHASE_SETTLE
    ctx.episode_step += 1       # step 3 → window done
    p.on_post_action_step(ctx)
    assert p._phase == PHASE_RETRACT


def test_no_scoring_after_retract():
    p = _plugin(settle_steps=1)
    ctx = _Ctx(_Accessor())
    p.on_pre_episode(ctx)
    ctx.accessor._contacts = _contacts([_hit(400.0)])
    p.on_post_phy_step(ctx)
    ctx.episode_step += 1
    p.on_post_action_step(ctx)          # → RETRACT
    assert p._phase == PHASE_RETRACT

    q = ctx.metrics["robot_a_strike_damage_total"]
    ctx.accessor._contacts = _contacts([_hit(1200.0)])
    p.on_post_phy_step(ctx)              # contact during RETRACT — ignored
    assert ctx.metrics["robot_a_strike_damage_total"] == q


# ----------------------------------------------------------------------
# Retract combo
# ----------------------------------------------------------------------

def _enter_retract(p, ctx):
    ctx.accessor._contacts = _contacts([_hit(400.0)])
    p.on_post_phy_step(ctx)
    for _ in range(p.settle_steps):
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
    assert p._phase == PHASE_RETRACT


def test_retract_success_terminates_strike_done():
    p = _plugin(settle_steps=1, hold_steps=3, retract_dist=0.5)
    ctx = _Ctx(_Accessor(core=_core(), derived=_retract_derived()))
    p.on_pre_episode(ctx)
    _enter_retract(p, ctx)

    for _ in range(3):
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
    assert ctx.terminations == ["strike_done"]
    assert ctx.metrics["robot_a_strike_retract_done"] == 1


def test_retract_hold_resets_when_condition_breaks():
    p = _plugin(settle_steps=1, hold_steps=3, retract_dist=0.5)
    ctx = _Ctx(_Accessor(core=_core(), derived=_retract_derived()))
    p.on_pre_episode(ctx)
    _enter_retract(p, ctx)

    ctx.episode_step += 1
    p.on_post_action_step(ctx)            # hold 1
    # Hand drifts back close to the target — condition broken.
    ctx.accessor._derived = _retract_derived(hand_pos=(0.8, 0.0, 1.0))
    ctx.episode_step += 1
    p.on_post_action_step(ctx)            # hold reset
    ctx.accessor._derived = _retract_derived()
    for _ in range(3):
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
    assert ctx.terminations == ["strike_done"]


def test_retract_requires_chambered_elbow():
    """Body may back off but an extended arm is NOT retracted."""
    p = _plugin(settle_steps=1, hold_steps=2, retract_dist=0.5)
    ctx = _Ctx(_Accessor(
        core=_core(elbow_l=-0.8),           # left arm still extended
        derived=_retract_derived()))
    p.on_pre_episode(ctx)
    _enter_retract(p, ctx)
    for _ in range(5):
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
    assert "strike_done" not in ctx.terminations


def test_retract_requires_upright():
    p = _plugin(settle_steps=1, hold_steps=2, retract_dist=0.5)
    ctx = _Ctx(_Accessor(
        core=_core(root_h=0.6),             # fell over
        derived=_retract_derived()))
    p.on_pre_episode(ctx)
    _enter_retract(p, ctx)
    for _ in range(5):
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
    assert "strike_done" not in ctx.terminations


def test_retract_timeout_terminates():
    p = _plugin(settle_steps=1, hold_steps=3, retract_timeout_steps=10)
    # Conditions never met: elbow stays extended.
    ctx = _Ctx(_Accessor(
        core=_core(elbow_l=-0.8), derived=_retract_derived()))
    p.on_pre_episode(ctx)
    _enter_retract(p, ctx)
    for _ in range(10):
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
    assert ctx.terminations == ["retract_timeout"]
    assert ctx.metrics["robot_a_strike_retract_done"] == 0


def test_observer_metrics_prefix_per_agent():
    p = StrikeSettlementPlugin(agent_id="robot_b")
    ctx = _Ctx(_Accessor())
    p.on_pre_episode(ctx)
    assert "robot_b_strike_phase" in ctx.metrics
    assert "robot_a_strike_phase" not in ctx.metrics


# ----------------------------------------------------------------------
# r_strike reward construction (through _build_agent_trajectory)
# ----------------------------------------------------------------------

def _make_episode(T, step_damage, damage_total, term_reason="strike_done"):
    """Minimal Episode with the observer fields exp_strike_step reads."""
    from baseline.framework.rollout.episode import Episode

    obs = np.zeros((T, 96), dtype=np.float32)
    obs[:, 45] = 1.2                       # standing height obs
    acts = np.zeros((T, 21), dtype=np.float32)
    foot = {
        "h_left_foot": np.zeros(T), "h_right_foot": np.zeros(T),
        "sole_clear_left": np.zeros(T), "sole_clear_right": np.zeros(T),
        "left_foot_contact": np.ones(T), "right_foot_contact": np.ones(T),
    }
    balance = {"potential": np.ones(T, dtype=np.float32)}
    strike = {
        "phase": np.zeros(T), "locked_hand": np.zeros(T),
        "step_damage": np.asarray(step_damage, dtype=np.float32),
        "damage_total": np.asarray(damage_total, dtype=np.float32),
        "retract_done": np.zeros(T),
    }
    records = {}
    if term_reason is not None:
        records = {
            "robot_a": ((term_reason, T),),
            "robot_b": ((term_reason, T),),
        }
    return Episode(
        base_seed=42, episode_index=0, blueprint_hash="t",
        num_frames=T,
        episode_options={"agent_id": "robot_a"},
        agent_termination_proposal_records=records,
        observations={"robot_a": obs, "robot_b": obs.copy()},
        actions={"robot_a": acts, "robot_b": acts.copy()},
        action_extras={"robot_a": {}, "robot_b": {}},
        explore_factors={"robot_a": np.zeros(T, np.float32),
                         "robot_b": np.zeros(T, np.float32)},
        observer_outputs={
            "foot_state_a": foot, "foot_state_b": dict(foot),
            "standing_balance_a": balance,
            "standing_balance_b": dict(balance),
            "strike_state_a": strike, "strike_state_b": dict(strike),
        },
        final_observation={"robot_a": obs[-1], "robot_b": obs[-1].copy()},
        episode_metrics={},
    )


def test_r_strike_two_half_payments():
    from baseline.experiments_ppo.exp_strike_step import StrikeStep

    T = 30
    sd = np.zeros(T); sd[10:15] = 2.0      # 5 settle frames × 2 dmg
    dt = np.cumsum(sd)
    # total = 10; terminal frame 29 gets +0.5*10 = 5
    ep = _make_episode(T, sd, dt, term_reason="strike_done")
    trajs = StrikeStep()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")
    ch = trajs[0].channels["r_strike"]

    np.testing.assert_allclose(ch.reward[10:15], 1.0, atol=1e-6)   # 0.5×2
    assert ch.reward[29] == pytest.approx(5.0)                      # 0.5×10
    assert ch.reward[:10].max() == 0.0
    assert ch.reward[15:29].max() == 0.0
    np.testing.assert_allclose(ch.actor_weight, 1.0)


def test_r_strike_timeout_withholds_second_half():
    from baseline.experiments_ppo.exp_strike_step import StrikeStep

    T = 30
    sd = np.zeros(T); sd[10:15] = 2.0
    dt = np.cumsum(sd)
    ep = _make_episode(T, sd, dt, term_reason="retract_timeout")
    trajs = StrikeStep()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")
    ch = trajs[0].channels["r_strike"]
    assert ch.reward[29] == pytest.approx(0.0)     # tail withheld
    np.testing.assert_allclose(ch.reward[10:15], 1.0, atol=1e-6)


def test_r_strike_no_hit_all_zero():
    from baseline.experiments_ppo.exp_strike_step import StrikeStep

    T = 20
    ep = _make_episode(T, np.zeros(T), np.zeros(T), term_reason="timeout")
    trajs = StrikeStep()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")
    ch = trajs[0].channels["r_strike"]
    assert ch.reward.max() == 0.0
    assert ch.is_terminated is False   # step-family: all ends bootstrap
