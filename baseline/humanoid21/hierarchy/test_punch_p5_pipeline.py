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
from mocap.point5.contracts import Point5State, StaticStartResolver
from mocap.point5.humanoid21.adapter import point5_from_view
from mocap.point5.punch_planner import plan_punch
from mocap.point5.trajectory import build_reference


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
    runtime = bp.materialize(max_steps=6).build()
    try:
        runtime.reset(seed=175217, options={
            "robot_a_strike_hand": "left",
            "robot_a_strike_dist": 0.5,
            "robot_a_strike_lat": 0.05,
            "robot_a_strike_z": 1.1,
        })
        rid = "robot_a"
        observed = runtime.get_observer_output("point5_a")
        measured = point5_from_view(
            runtime.simulator.get_derived_state([rid])[rid],
            rid, with_vel=False).pos.reshape(-1)
        for i, actual in enumerate(measured):
            assert observed[f"p5_{i}"] == actual
            assert runtime.ctx.metrics[f"{rid}_strike_p5_{i}"] == actual
        target = np.array([runtime.ctx.metrics[f"{rid}_strike_t{x}"]
                           for x in "xyz"])
        np.testing.assert_allclose(
            _world_target_seen_by_policy(runtime.simulator, rid), target,
            atol=1e-5)
        obs, _ = runtime.get_observation()
        assert obs[61] == 1.0
        assert obs[62] == 1.0
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
    ref = build_reference(
        plan.task, StaticStartResolver(start, np.zeros(3), np.eye(3)))
    states = [ref.sample_time(min((i + 1) * 0.05, ref.duration)).pos
              for i in range(6)]
    states.append(states[-1] + 1.0)  # unscored tail for this unit test
    arr = np.stack(states).reshape(7, 15)
    metrics = {f"robot_a_strike_p5_{i}": float(v)
               for i, v in enumerate(pos.reshape(-1))}
    metrics.update({f"robot_a_strike_t{ax}": float(v)
                    for ax, v in zip("xyz", target)})
    metrics["robot_a_strike_hand_i"] = 0.0
    ep = SimpleNamespace(
        episode_metrics=metrics,
        observer_outputs={"point5_a": {
            f"p5_{i}": arr[:, i] for i in range(15)}},
    )

    def fake_parent(self, episode, agent_id, foot_key, phi_key):
        return [Trajectory(
            obs=np.zeros((7, 66)), actions=np.zeros((7, 21)),
            last_obs=np.zeros(66),
            channels={"r_fall": ChannelData(
                reward=np.zeros(7), is_terminated=False)},
        )]

    monkeypatch.setattr(
        HierS2BalanceMBS, "_build_agent_trajectory", fake_parent)
    tr = PunchP5MBS()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")[0]
    for name in ("r_strike_hand", "r_guard_hand", "r_head", "r_feet"):
        np.testing.assert_allclose(tr.channels[name].reward[:6], 1.0)
        assert tr.channels[name].actor_weight[5] > 0
        assert tr.channels[name].actor_weight[6] == 0

    # A missed punch still yields a graded signal, not a near-zero
    # Gaussian plateau at errors typical of a warm-start balance policy.
    ep.observer_outputs["point5_a"]["p5_6"] = (
        arr[:, 6].copy() + 0.30)
    missed = PunchP5MBS()._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")[0]
    assert 0.05 < missed.channels["r_strike_hand"].reward[5] < 0.2
