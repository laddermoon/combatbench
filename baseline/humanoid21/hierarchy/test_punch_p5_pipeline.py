"""Regression checks for the punch task's episode-start/command contract."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from baseline.experiments_ppo.exp_hier_s2_balance_mbs import HierS2BalanceMBS
from baseline.experiments_ppo.exp_punch_p5_mbs import PunchP5MBS
from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.humanoid21.hierarchy.command_simulator import (
    CommandSimulator, TGT_SCALE,
)
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from point5.contracts import Point5State, to_world
from point5.humanoid21.adapter import point5_from_view
from point5.humanoid21.punch_planner import plan_punch
from point5.trajectory import keyframe_world

from baseline.humanoid21.hierarchy.p5_world import (
    clip_world_samples, f1_from_state)

_F1_IDENTITY = (np.zeros(2), 0.0)


def _world_target_seen_by_policy(sim, rid):
    cmd = sim.get_observation()[rid][63:66]
    core = sim.get_core_state()[rid]
    quat = np.asarray(core["root_rot"])
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]])
    pelvis_name = sim.get_static_data()[rid]["keypoint_body_names"]["pelvis"]
    pelvis = sim.get_derived_state([rid])[rid]["body_xpos"][pelvis_name]
    return pelvis + rot.apply(cmd * TGT_SCALE)


def test_command_world_target_stays_fixed_as_body_moves():
    sim = CommandSimulator(cmd_version="v2")
    try:
        sim.reset(seed=3)
        rid = "robot_a"
        world = np.array([-0.4, 0.3, 1.3])
        sim.set_effector_world(rid, 1, 1, world)
        initial_cmd = sim.get_observation()[rid][63:66].copy()
        np.testing.assert_allclose(_world_target_seen_by_policy(sim, rid),
                                   world, atol=1e-5)
        core = sim.get_core_state()[rid]
        state = {k: v.copy() for k, v in core.items()}
        state["root_pos"][:2] += [0.2, -0.1]
        state["root_rot"] = Rotation.from_euler(
            "z", 0.4).as_quat()[[3, 0, 1, 2]]
        sim.set_core_state({rid: state})
        moved_cmd = sim.get_observation()[rid][63:66]
        assert np.linalg.norm(initial_cmd - moved_cmd) > 0.05
        np.testing.assert_allclose(_world_target_seen_by_policy(sim, rid),
                                   world, atol=1e-5)
    finally:
        sim.close()


def test_policy_init_and_target_precede_initial_p5_observation():
    bp = ParameterizedEnvBlueprint.load(
        "baseline/humanoid21/hierarchy/blueprints/hier_punch_p5_env.yaml")
    runtime = bp.materialize(max_steps=8).build()
    try:
        runtime.reset(seed=175217, options={
            "robot_a_strike_hand": "left",
            "robot_a_strike_dist": 0.5,
            "robot_a_strike_lat": 0.05,
            "robot_a_strike_z": 1.1,
        })
        rid = "robot_a"
        observed = runtime.get_observer_output("point5_a")
        st = point5_from_view(
            runtime.simulator.get_derived_state([rid])[rid],
            rid, with_vel=False)
        measured_local = st.pos.reshape(-1)
        measured_world = to_world(st.pos, *f1_from_state(st)).reshape(-1)
        for i in range(15):
            # Observer publishes world; metrics keep the local start P5.
            assert observed[f"p5_{i}"] == measured_world[i]
            assert runtime.ctx.metrics[
                f"{rid}_strike_p5_{i}"] == measured_local[i]
        obs, _ = runtime.get_observation()
        assert obs[61] == 1.0
        assert obs[62] == 1.0
        # The displayed target is the first waypoint of the reference
        # hand track (moving guidance), not the raw endpoint.
        sim = runtime.simulator
        track = sim._world_effector_track.get(rid)
        assert track is not None and len(track) >= 1
        np.testing.assert_allclose(
            _world_target_seen_by_policy(sim, rid), track[0], atol=1e-5)
        # After stepping, the display advances along the track and the
        # final waypoint stays world-fixed.
        rt = runtime
        zero = np.zeros(21, dtype=np.float32)
        shown = [_world_target_seen_by_policy(sim, rid)]
        for _ in range(2):
            rt.step(zero, zero)
            shown.append(_world_target_seen_by_policy(sim, rid))
        np.testing.assert_allclose(shown[1], track[min(1, len(track)-1)],
                                   atol=1e-5)
        np.testing.assert_allclose(shown[2], track[min(2, len(track)-1)],
                                   atol=1e-5)
    finally:
        runtime.close()


def test_strike_reward_scores_final_impact_not_initial_frame(monkeypatch):
    # Every post-action P5 matches its planned reference exactly.
    # In particular, the sixth action crosses T=0.28s, so it must
    # receive the impact-keyframe reward rather than be dropped.
    pos = np.array([
        [0, 0, 1.2], [0.15, -0.1, 1.05], [0.25, 0.1, 1.1],
        [-0.05, -0.15, 0.02], [0.2, 0.15, 0.01],
    ], dtype=np.float64)
    start = Point5State(pos=pos)
    target = np.array([0.7, 0.0, 1.1])
    plan = plan_punch(start, target, "left")
    assert plan.reachable
    # Identity f1: synthetic positions double as "world" coordinates.
    ref_world = clip_world_samples(plan.task, _F1_IDENTITY, 0.05)
    n_steps = len(ref_world)
    states = [ref_world[i] for i in range(n_steps)]
    states.append(states[-1] + 1.0)  # unscored tail for this unit test
    arr = np.stack(states).reshape(n_steps + 1, 15)
    metrics = {f"robot_a_strike_p5_{i}": float(v)
               for i, v in enumerate(pos.reshape(-1))}
    metrics.update({f"robot_a_strike_lt{ax}": float(v)
                    for ax, v in zip("xyz", target)})
    metrics.update({"robot_a_strike_f1_ox": 0.0,
                    "robot_a_strike_f1_oy": 0.0,
                    "robot_a_strike_f1_yaw": 0.0})
    metrics["robot_a_strike_hand_i"] = 0.0
    ep = SimpleNamespace(
        episode_metrics=metrics,
        observer_outputs={"point5_a": {
            f"p5_{i}": arr[:, i] for i in range(15)}},
    )

    def fake_parent(self, episode, agent_id, foot_key, phi_key):
        return [Trajectory(
            obs=np.zeros((n_steps + 1, 66)),
            actions=np.zeros((n_steps + 1, 21)),
            last_obs=np.zeros(66),
            channels={"r_fall": ChannelData(
                reward=np.zeros(n_steps + 1), is_terminated=False)},
        )]

    monkeypatch.setattr(
        HierS2BalanceMBS, "_build_agent_trajectory", fake_parent)
    tr = PunchP5MBS()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")[0]
    for name in ("r_strike_hand", "r_guard_hand", "r_head", "r_feet"):
        np.testing.assert_allclose(
            tr.channels[name].reward[:n_steps], 1.0)
        assert tr.channels[name].actor_weight[n_steps - 1] > 0
        assert tr.channels[name].actor_weight[n_steps] == 0
        assert tr.channels[name].is_terminated is True
    assert tr.channels["r_fall"].is_terminated is True
    assert tr.channels["r_feet"].actor_weight[2] > (
        tr.channels["r_strike_hand"].actor_weight[2])
    assert tr.channels["r_strike_hand"].actor_weight[n_steps - 2] > (
        tr.channels["r_feet"].actor_weight[n_steps - 2])

    # A missed punch still yields a graded signal, not a near-zero
    # Gaussian plateau at errors typical of a warm-start balance policy.
    ep.observer_outputs["point5_a"]["p5_6"] = (
        arr[:, 6].copy() + 0.30)
    missed = PunchP5MBS()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")[0]
    assert 0.05 < missed.channels["r_strike_hand"].reward[5] < 0.2

    # In the same world frame, moving either measured point toward its
    # forward reference must increase (not decrease) its channel reward.
    axis = plan.diag["attack_axis"]
    for point, channel in ((2, "r_strike_hand"),
                           (3 if plan.diag["moving_foot"] == "foot_right"
                            else 4, "r_feet")):
        original = arr[4, 3 * point:3 * point + 3].copy()
        def reward_at(offset):
            for i, value in enumerate(original + offset * axis):
                ep.observer_outputs["point5_a"][f"p5_{3 * point + i}"][4] = value
            return PunchP5MBS()._build_agent_trajectory(
                ep, "robot_a", "foot_state_a", "standing_balance_a")[0] \
                .channels[channel].reward[4]
        assert reward_at(-0.15) < reward_at(-0.05) < reward_at(0.0)
        reward_at(0.0)


def test_planned_foot_and_fist_never_reverse_attack_axis():
    start = Point5State(pos=np.array([
        [0, 0, 1.2], [0.15, -0.1, 1.05], [0.25, 0.1, 1.1],
        [-0.05, -0.15, 0.02], [0.2, 0.15, 0.01],
    ], dtype=np.float64))
    for hand in ("left", "right"):
        for lateral in (-0.20, 0.0, 0.20):
            target = np.array([0.8, lateral, 1.1])
            plan = plan_punch(start, target, hand)
            assert plan.reachable
            a = plan.diag["attack_axis"]
            kw = keyframe_world(plan.task, _F1_IDENTITY)
            from point5.contracts import POINT_INDEX
            for name in (f"hand_{hand}",
                         plan.task.metadata["moving_foot"]):
                projection = kw[:, POINT_INDEX[name]] @ a
                assert np.all(np.diff(projection) >= -1e-9)
