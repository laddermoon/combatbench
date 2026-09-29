from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from .validation import Unsupported


ROOT = Path(__file__).resolve().parents[2]
BODY_NAMES = ("torso", "head", "pelvis", "foot_right", "foot_left", "hand_right", "hand_left")


def reward_cases():
    positions = {name: np.array([0.0, 0.0, 0.1], dtype=np.float32) for name in BODY_NAMES}
    positions["torso"][2] = 1.28
    base = {"operation": "reward", "agent": "robot_a", "positions": positions,
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "contacts": []}
    cases = []

    def add(name, data):
        cases.append({"id": name, "kind": "logic", "level": "V2", "seed": 42,
                      "episode": 0, "frame": 0, "input": copy.deepcopy(data)})

    add("airborne-stage1", base)
    prone = copy.deepcopy(base)
    prone["quaternion"] = np.array([2 ** -0.5, 0, 2 ** -0.5, 0], dtype=np.float32)
    prone["contacts"] = [{"body": "torso", "environment": "ground", "force": 20.0}]
    add("prone-stage2", prone)
    standing = copy.deepcopy(base)
    standing["contacts"] = [{"body": "foot_right", "environment": "ground", "force": 20.0}]
    add("supported-stage4", standing)
    wide = copy.deepcopy(standing)
    for hand in ("hand_right", "hand_left"):
        wide["positions"][hand][0] = 0.7
    add("wide-stage3", wide)
    for force in (0.999, 1.0, 1.001, 9.999, 10.0, 10.001):
        data = copy.deepcopy(standing)
        data["contacts"][0]["force"] = force
        add(f"force-{force}", data)
    for distance in (0.5199, 0.52, 0.5201):
        data = copy.deepcopy(standing)
        for hand in ("hand_right", "hand_left"):
            data["positions"][hand][0] = distance
        add(f"distance-{distance}", data)
    for height in (0.1499, 0.15, 0.1501, 1.2799, 1.28, 1.2801):
        data = copy.deepcopy(standing)
        data["positions"]["torso"][2] = height
        add(f"height-{height}", data)
    duplicate = copy.deepcopy(prone)
    duplicate["contacts"] *= 2
    add("duplicate-body-not-two-bodies", duplicate)
    wall = copy.deepcopy(standing)
    wall["contacts"][0]["environment"] = "wall_00"
    add("wall-not-ground", wall)
    for score in (0.7999, 0.8, 0.8001):
        data = copy.deepcopy(base)
        theta = np.arcsin(2 * score - 1)
        data["quaternion"] = np.array([np.cos(theta / 2), 0, np.sin(theta / 2), 0], dtype=np.float32)
        add(f"f-score-{score}", data)
    other = copy.deepcopy(standing)
    other["agent"] = "robot_b"
    add("robot-b-supported", other)
    return cases


class FeatureAccessor:
    def __init__(self, features):
        if set(features) != {"operation", "agent", "positions", "quaternion", "contacts"}:
            raise ValueError("unknown or missing reward feature")
        agent = features["agent"]
        if agent not in ("robot_a", "robot_b") or set(features["positions"]) != set(BODY_NAMES):
            raise ValueError("invalid agent or missing body positions")
        for name, position in features["positions"].items():
            if not isinstance(position, np.ndarray) or position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"invalid position: {name}")
        quat = features["quaternion"]
        if not isinstance(quat, np.ndarray) or quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1.0):
            raise ValueError("invalid quaternion")
        suffix = "_a" if agent == "robot_a" else "_b"
        names = {name: name + suffix for name in BODY_NAMES}
        ids = {name: i + 1 for i, name in enumerate(BODY_NAMES)}
        cv = {key: [] for key in ("aff1", "aff2", "geom1", "geom2", "body1", "body2", "force_mag")}
        for contact in features["contacts"]:
            if set(contact) != {"body", "environment", "force"}:
                raise ValueError("unknown contact feature")
            body = ids[contact["body"]]
            geom = {"ground": 0, "wall_00": 1}[contact["environment"]]
            force = contact["force"]
            if not np.isfinite(force) or force < 0:
                raise ValueError("invalid contact force")
            values = (0, 1 if agent == "robot_a" else 2, geom, body + 1, 0, body, force)
            for key, value in zip(cv, values):
                cv[key].append(value)
        cv = {key: np.asarray(value, dtype=np.float32 if key == "force_mag" else np.int32)
              for key, value in cv.items()}
        cv["ncon"] = len(features["contacts"])
        self.static = {agent: {"keypoint_body_names": names},
                       "body_id_to_name": {ids[name]: names[name] for name in names},
                       "geom_id_to_name": {0: "ground", 1: "wall_00"}}
        self.derived = {agent: {"body_xpos": {names[k]: v for k, v in features["positions"].items()},
                                "body_xquat": {names["torso"]: quat}}, "contacts": cv}

    def get_static_data(self):
        return self.static

    def get_derived_state(self, fields):
        return {key: self.derived[key] for key in fields}


def physics_case(steps=1):
    import mujoco
    from envs.humanoid21.simulator import Humanoid21Simulator

    if steps not in (1, 25):
        raise ValueError("M1 physical probes support 1 or 25 substeps")
    sim = Humanoid21Simulator()
    try:
        sim.reset(seed=42, options={"initial_distance": 2.0})
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(sim.model, spec), dtype=np.float64)
        mujoco.mj_getState(sim.model, sim.data, state, spec)
        return {"id": f"standing-{steps}-substeps", "kind": "action_sequence", "level": "V1",
                "seed": 42, "episode": 0, "frame": 0,
                "input": {"state_spec": int(spec), "integration_state": state,
                          "initial_action": sim.get_action(),
                          "actions": [{"robot_a": np.linspace(-0.2, 0.2, 21, dtype=np.float32),
                                       "robot_b": np.linspace(0.2, -0.2, 21, dtype=np.float32)}],
                          "substeps": steps, "plugins": []}}
    finally:
        sim.close()


def all_cases():
    cases = reward_cases()
    cases += [physics_case(1), physics_case(25), trajectory_case()]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return cases


def trajectory_case():
    return {"id": "standup-timeout-bootstrap", "kind": "logic", "level": "V2", "seed": 42,
            "episode": 0, "frame": 2,
            "input": {"operation": "trajectory", "observation": np.arange(192, dtype=np.float32).reshape(2, 96),
                      "action": np.zeros((2, 21), dtype=np.float32),
                      "final_observation": np.arange(96, dtype=np.float32) + 1000,
                      "potential": np.array([0.3, 0.9], dtype=np.float32)}}


def trajectory_output(inputs):
    from baseline.experiments_ppo.exp_standup_floor04 import StandupFloor04
    from baseline.framework.rollout.episode import Episode

    if set(inputs) != {"operation", "observation", "action", "final_observation", "potential"}:
        raise ValueError("unknown or missing trajectory input")
    for field, shape in (("observation", (2, 96)), ("action", (2, 21)),
                         ("final_observation", (96,)), ("potential", (2,))):
        value = inputs[field]
        if not isinstance(value, np.ndarray) or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid trajectory field: {field}")
    agents = ("robot_a", "robot_b")
    episode = Episode(base_seed=42, episode_index=0, blueprint_hash="synthetic-contract-probe",
                      num_frames=2, episode_options={},
                      agent_termination_proposal_records={a: (("timeout", 2),) for a in agents},
                      observations={a: inputs["observation"].copy() for a in agents},
                      actions={a: inputs["action"].copy() for a in agents},
                      action_extras={}, explore_factors={a: np.zeros(2, dtype=np.float32) for a in agents},
                      observer_outputs={f"standing_balance_{a[-1]}": {"potential": inputs["potential"].copy()}
                                        for a in agents},
                      final_observation={a: inputs["final_observation"].copy() for a in agents})
    trajectories = StandupFloor04().build_trajectories([episode])
    if len(trajectories) != 2:
        raise ValueError("standup must emit both agent trajectories")
    return {agent: {"obs": traj.obs, "actions": traj.actions, "last_obs": traj.last_obs,
                    "reward": traj.channels["r_potential"].reward,
                    "is_terminated": traj.channels["r_potential"].is_terminated,
                    "sampling_ctx": traj.sampling_ctx}
            for agent, traj in zip(agents, trajectories)}


# M1 oracle 的语义依赖：任务侧 envs 全部源码 + trajectory 契约实际加载的 baseline 模块 +
# 模型/蓝图/奖励实现。不纳入 ppo trainer/loop/debug 与无关 experiment——它们的改动不影响
# 这三类 fixture 的参考输出，纳入只会产生误报 stale。M5 增加 PPO 更新级 fixture 时，
# 由对应适配器另行声明 trainer/algos 依赖。
DEPENDENCY_DIRS = ("envs/framework", "envs/humanoid21")
DEPENDENCY_FILES = (
    "envs/batchframework/M0_BASELINE.md",
    "envs/batchframework/validation.py",
    "envs/batchframework/validation_cpu.py",
    "envs/humanoid21/battle_circular_v2.xml",
    "baseline/humanoid21/blueprints/standup_4stage_dense_v2_env.yaml",
    "baseline/humanoid21/rewards/standing_balance_4stage.py",
    "baseline/experiments_ppo/base.py",
    "baseline/experiments_ppo/exp_standup.py",
    "baseline/experiments_ppo/exp_standup_floor04.py",
    "baseline/framework/critic_mlp.py",
    "baseline/framework/ppo/__init__.py",
    "baseline/framework/ppo/experiment.py",
    "baseline/framework/ppo/sampling_context.py",
    "baseline/framework/ppo/stochastic_policy.py",
    "baseline/framework/ppo/trajectory.py",
    "baseline/framework/rollout/__init__.py",
    "baseline/framework/rollout/episode.py",
    "baseline/framework/rollout/job.py",
    "baseline/framework/rollout/observer_utils.py",
)


class CPUAdapter:
    backend = "mujoco-cpu-reference"
    version = "m1-v1"

    def __init__(self):
        paths = [ROOT / path for path in DEPENDENCY_FILES]
        for directory in DEPENDENCY_DIRS:
            paths.extend((ROOT / directory).glob("*.py"))
        model = ROOT / "envs/humanoid21/battle_circular_v2.xml"
        paths.extend(model.parent / node.attrib["file"] for node in ET.parse(model).iter()
                     if "file" in node.attrib)
        self.dependencies = sorted(set(paths))

    def execute(self, case):
        if case["kind"] == "logic":
            if case["input"]["operation"] == "trajectory":
                return trajectory_output(case["input"])
            if case["input"]["operation"] != "reward":
                raise Unsupported(f"unknown logic operation: {case['input']['operation']}")
            from baseline.humanoid21.rewards.standing_balance_4stage import StandingBalance4StageRewarder

            accessor = FeatureAccessor(case["input"])
            rewarder = StandingBalance4StageRewarder(case["input"]["agent"])
            ctx = SimpleNamespace(accessor=accessor)
            rewarder.on_pre_episode(ctx)
            rewarder.on_post_action_step(ctx)
            return {case["input"]["agent"]: rewarder.get_output()}
        if case["kind"] not in ("physics", "action_sequence"):
            raise Unsupported(f"CPU adapter has no executor for {case['kind']}")
        return self._physics(case["input"])

    def _physics(self, inputs):
        import mujoco
        from envs.humanoid21.simulator import Humanoid21Simulator

        if set(inputs) != {"state_spec", "integration_state", "initial_action", "actions", "substeps", "plugins"}:
            raise ValueError("unknown or missing physics input")
        if inputs["plugins"]:
            raise Unsupported(f"physics probe does not execute plugins: {inputs['plugins']}")
        if inputs["state_spec"] != int(mujoco.mjtState.mjSTATE_INTEGRATION):
            raise Unsupported("full mjSTATE_INTEGRATION required")
        if type(inputs["substeps"]) is not int or inputs["substeps"] not in (1, 25):
            raise Unsupported("only 1/25 substeps supported by M1 probes")
        if not inputs["actions"]:
            raise ValueError("action sequence must be nonempty")
        sim = Humanoid21Simulator()
        try:
            sim.reset(seed=42)
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = inputs["integration_state"]
            if state.shape != (mujoco.mj_stateSize(sim.model, spec),) or not np.isfinite(state).all():
                raise ValueError("invalid integration state")
            mujoco.mj_setState(sim.model, sim.data, state, spec)
            mujoco.mj_forward(sim.model, sim.data)
            mujoco.mj_setState(sim.model, sim.data, state, spec)
            sim.set_action(inputs["initial_action"])
            frames = []
            for action in inputs["actions"]:
                if set(action) != {"robot_a", "robot_b"}:
                    raise ValueError("both agent actions required")
                sim.set_action(action)
                for _ in range(inputs["substeps"]):
                    sim.physical_step()
                frames.append({"qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy(),
                               "observation": sim.get_observation(), "core": sim.get_core_state(),
                               "sensor": sim.get_sensor_data()})
            return {"frames": frames}
        finally:
            sim.close()


Adapter = CPUAdapter
