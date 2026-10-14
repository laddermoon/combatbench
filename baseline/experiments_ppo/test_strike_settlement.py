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


def _core(root_h=1.2):
    return {
        "robot_a": {
            "root_pos": np.array([0.0, 0.0, root_h], dtype=np.float32),
        },
        "robot_b": {
            "root_pos": np.array([1.0, 0.0, 1.3], dtype=np.float32),
        },
    }


def _retract_derived(hand_pos=None, torso_a=(0.0, 0.0, 1.0),
                     head=(1.0, 0.0, 1.5), torso=(1.0, 0.0, 1.0)):
    """derived state: locked hand home & far from opp by default."""
    if hand_pos is None:
        # 0.2m from own torso, ~1.02m from opp torso — retracted.
        hand_pos = (0.0, 0.0, 1.2)
    return {
        "robot_a": {"body_xpos": {
            "hand_left_a": np.array(hand_pos),
            "torso_a": np.array(torso_a)}},
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


def test_weak_contact_does_not_trigger():
    """A graze below min_hit_force must not lock a hand or start SETTLE
    (s42c tap-and-retract farming fix)."""
    p = _plugin(settle_steps=3, min_hit_force=250.0)
    ctx = _Ctx(_Accessor())
    p.on_pre_episode(ctx)
    ctx.accessor._contacts = _contacts([_hit(100.0)])
    p.on_post_phy_step(ctx)
    assert p._phase == PHASE_PRE_HIT
    assert p._locked_hand == 0
    # A subsequent real hit still triggers normally.
    ctx.accessor._contacts = _contacts([_hit(400.0)])
    p.on_post_phy_step(ctx)
    assert p._phase == PHASE_SETTLE
    assert p._locked_hand == HAND_LEFT
    # The graze never scored — Q contains only the real hit.
    exp = _expected_damage(400.0, "head")
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


def test_retract_requires_hand_home():
    """Body may back off but a still-extended arm is NOT retracted."""
    p = _plugin(settle_steps=1, hold_steps=2, retract_dist=0.5,
                hand_home_dist=0.55)
    ctx = _Ctx(_Accessor(
        core=_core(),
        # Opponent far away (d_opp ok) but the hand is 0.71m from the
        # own torso — arm still out, must not count as retracted.
        derived=_retract_derived(
            hand_pos=(0.7, 0.0, 1.1),
            head=(3.0, 0.0, 1.5), torso=(3.0, 0.0, 1.0))))
    p.on_pre_episode(ctx)
    _enter_retract(p, ctx)
    for _ in range(5):
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
    assert "strike_done" not in ctx.terminations


def test_retract_relaxed_hanging_arm_counts():
    """A hanging arm while disengaged IS retracted (s42a fix: the old
    elbow>=0.7 gate timed out episodes where the learner had clearly
    walked away with the arm relaxed)."""
    p = _plugin(settle_steps=1, hold_steps=2, retract_dist=0.5,
                hand_home_dist=0.55)
    ctx = _Ctx(_Accessor(
        core=_core(),
        # Hand 0.50m from own torso (arm hanging at the side) and
        # >2m from the opponent — matches the observed ep0 case.
        derived=_retract_derived(
            hand_pos=(0.45, 0.0, 0.8),
            head=(3.0, 0.0, 1.5), torso=(3.0, 0.0, 1.0))))
    p.on_pre_episode(ctx)
    _enter_retract(p, ctx)
    for _ in range(3):
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
    assert ctx.terminations == ["strike_done"]


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
    # Conditions never met: hand stays extended away from the body.
    ctx = _Ctx(_Accessor(
        core=_core(),
        derived=_retract_derived(
            hand_pos=(0.7, 0.0, 1.1),
            head=(3.0, 0.0, 1.5), torso=(3.0, 0.0, 1.0))))
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

def _make_episode(T, step_damage, damage_total, term_reason="strike_done",
                  phase=None, d_home=None, d_opp=None, approach=None):
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
        "phase": (np.zeros(T) if phase is None
                  else np.asarray(phase, dtype=np.float32)),
        "locked_hand": np.zeros(T),
        "step_damage": np.asarray(step_damage, dtype=np.float32),
        "damage_total": np.asarray(damage_total, dtype=np.float32),
        "retract_done": np.zeros(T),
        "d_home": (np.zeros(T) if d_home is None
                   else np.asarray(d_home, dtype=np.float32)),
        "d_opp": (np.zeros(T) if d_opp is None
                  else np.asarray(d_opp, dtype=np.float32)),
        "gates": np.zeros(T),
    }
    observer_outputs = {
        "foot_state_a": foot, "foot_state_b": dict(foot),
        "standing_balance_a": balance,
        "standing_balance_b": dict(balance),
        "strike_state_a": strike, "strike_state_b": dict(strike),
    }
    if approach is not None:
        observer_outputs["approach_velocity_a"] = approach
        observer_outputs["approach_velocity_b"] = dict(approach)

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
        observer_outputs=observer_outputs,
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


def test_retract_phase_masks_shaping_channels():
    """On RETRACT frames (phase==2) every shaping channel except r_fall
    and r_strike must carry reward=0 AND actor_weight=0 — otherwise the
    follow/arm rewards pay the policy for hugging the target instead of
    retracting."""
    from baseline.experiments_ppo.exp_strike_step import StrikeStep

    T = 20
    phase = np.zeros(T); phase[15:] = 2.0      # last 5 frames RETRACT
    ep = _make_episode(T, np.zeros(T), np.zeros(T),
                       term_reason="retract_timeout", phase=phase)
    trajs = StrikeStep()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")
    t = trajs[0]
    for cname in StrikeStep._RETRACT_MASKED_CHANNELS:
        ch = t.channels[cname]
        np.testing.assert_allclose(ch.reward[15:], 0.0)
        np.testing.assert_allclose(ch.actor_weight[15:], 0.0)
    # r_fall stays live (falls during retract are still punished) and
    # r_strike keeps its constant weight (the withheld half).
    assert t.channels["r_fall"].actor_weight[15:].max() != 0.0 or \
        t.channels["r_fall"].actor_weight[15:].min() != 0.0
    np.testing.assert_allclose(
        t.channels["r_strike"].actor_weight[15:], 1.0)
    # Pre-retract frames are untouched.
    np.testing.assert_allclose(
        t.channels["r_potential"].actor_weight[:15],
        t.channels["r_potential"].actor_weight[0])


def test_r_retract_potential_difference():
    """r_retract pays W·Δφ only on phase-2 frames: progress toward the
    retract combo is rewarded, parking earns nothing."""
    from baseline.experiments_ppo.exp_strike_step import StrikeStep

    T = 20
    phase = np.zeros(T); phase[10:] = 2.0        # RETRACT from t=10
    d_opp = np.full(T, 0.1)                      # close at first…
    d_opp[12:] = 0.6                             # …then separated
    d_home = np.full(T, 0.75)                    # arm out at first…
    d_home[14:] = 0.4                            # …then home
    ep = _make_episode(T, np.zeros(T), np.zeros(T),
                       term_reason="retract_timeout", phase=phase,
                       d_home=d_home, d_opp=d_opp)
    trajs = StrikeStep()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")
    ch = trajs[0].channels["r_retract"]

    # φ: d_opp term 0→1 at t=12; d_home term (0.8-d)/0.25: 0.2→1 at t=14
    W = StrikeStep.RETRACT_W
    assert ch.reward[12] == pytest.approx(W * (1.0 - 0.2))
    assert ch.reward[14] == pytest.approx(W * (1.0 - 0.2))
    # No reward on non-RETRACT frames, none while parked.
    assert ch.reward[:10].max() == 0.0
    assert ch.reward[15:].max() == 0.0
    np.testing.assert_allclose(ch.actor_weight, 1.0)


def test_sparse_zone_masking():
    """StrikeSparseStep: inside D_ENGAGE only task+fall+face channels
    stay live — feet/torso are masked; outside, the native follow
    channels carry on."""
    from baseline.experiments_ppo.exp_strike_sparse_step import (
        StrikeSparseStep)

    T = 20
    opp_x = np.full(T, 2.0)          # far for t<10 …
    opp_x[10:] = 0.4                 # … close (<=0.9) for t>=10
    approach = {
        "self_x": np.zeros(T), "self_y": np.zeros(T),
        "opp_x": opp_x, "opp_y": np.zeros(T),
    }
    ep = _make_episode(T, np.zeros(T), np.zeros(T),
                       term_reason="timeout", approach=approach)
    t = StrikeSparseStep()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a",
        "standing_balance_a")[0]

    for cname in StrikeSparseStep._CLOSE_MASKED_CHANNELS:
        ch = t.channels[cname]
        np.testing.assert_allclose(ch.reward[10:], 0.0)
        np.testing.assert_allclose(ch.actor_weight[10:], 0.0)
    # Far frames untouched: stub yields torso aw=1 and foot aw=±1.
    np.testing.assert_allclose(t.channels["r_torso"].actor_weight[:10], 1.0)
    assert t.channels["r_right_foot"].actor_weight[:10].min() != 0.0
    # Keep-alive set inside the zone: potential/fall stay full,
    # face ramps ON inside (dist_gate = 1 at 0.4m, phi=1 stub).
    np.testing.assert_allclose(
        t.channels["r_potential"].actor_weight[10:], 3.0)
    np.testing.assert_allclose(
        t.channels["r_fall"].actor_weight[10:], 2.0)
    np.testing.assert_allclose(
        t.channels["r_face"].actor_weight[10:], 1.0)
    np.testing.assert_allclose(t.channels["r_strike"].actor_weight, 1.0)


def test_sparse_explore_factor_zones():
    """Distance-gated ef: σ×0.5 fallen, σ×2.0 inside engagement,
    σ×1.0 standing outside."""
    from baseline.experiments_ppo.exp_strike_sparse_step import (
        _sparse_explore_factor)
    from baseline.experiments_ppo.exp_step import _EF_LOW, _EF_STAND

    obs = np.zeros(96, dtype=np.float32)
    obs[45] = 1.25                                   # standing
    obs[57], obs[58] = 0.4, 0.3                      # dist 0.5 ≤ 0.9
    assert _sparse_explore_factor(obs, 0) == _EF_STAND

    obs[57], obs[58] = 1.5, 0.0                      # dist 1.5 > 0.9
    assert _sparse_explore_factor(obs, 0) == 0.0

    obs[57], obs[58] = 0.3, 0.0                      # close BUT fallen
    obs[45] = 0.5
    assert _sparse_explore_factor(obs, 0) == _EF_LOW


def test_end_after_settle_terminates_strike_done():
    """Simplified scoring: settle window end → strike_done, never
    reaches PHASE_RETRACT."""
    from baseline.humanoid21.plugins.strike_settlement import (
        PHASE_SETTLE)
    p = _plugin(settle_steps=3, end_after_settle=True)
    ctx = _Ctx(_Accessor())
    p.on_pre_episode(ctx)
    ctx.accessor._contacts = _contacts([_hit(400.0)])
    p.on_post_phy_step(ctx)
    assert p._phase == PHASE_SETTLE

    for _ in range(2):
        ctx.episode_step += 1
        p.on_post_action_step(ctx)
        assert not ctx.terminations
    ctx.episode_step += 1
    p.on_post_action_step(ctx)
    assert ctx.terminations == ["strike_done"]
    # no retract: phase stays SETTLE through termination
    assert p._phase == PHASE_SETTLE


def test_simple_streams_full_damage_no_retract_channel():
    """StrikeSimpleStep: r_strike = step_damage×1.0 streamed; no
    r_retract channel; no tail payment."""
    from baseline.experiments_ppo.exp_strike_simple_step import (
        StrikeSimpleStep)

    T = 30
    sd = np.zeros(T); sd[10:15] = 2.0       # 5 settle frames × 2 dmg
    dt = np.cumsum(sd)                      # total = 10
    ep = _make_episode(T, sd, dt, term_reason="strike_done")
    trajs = StrikeSimpleStep()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")

    assert "r_retract" not in trajs[0].channels
    assert "r_left_elbow" not in trajs[0].channels
    ch = trajs[0].channels["r_strike"]
    np.testing.assert_allclose(ch.reward[10:15], 2.0, atol=1e-6)
    assert ch.reward[29] == 0.0             # NO 0.5×Q tail — streamed
    assert ch.reward[:10].max() == 0.0
    assert ch.reward[15:].max() == 0.0
    np.testing.assert_allclose(ch.actor_weight, 1.0)

    exp = StrikeSimpleStep()
    assert "r_retract" not in exp._channel_names
    assert len(exp._channel_names) == 9
