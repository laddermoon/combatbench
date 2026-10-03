from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from .validation import Unsupported, canonical_contacts


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
    cases += dynamics_cases()
    cases += [external_force_case(), state_io_case(), batch_isolation_case()]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return cases


def _snapshot_state(sim):
    return {"qpos": np.asarray(sim.data.qpos).copy(),
            "qvel": np.asarray(sim.data.qvel).copy()}


def _snapshot_at(dist, post_steps=0, action=None):
    """建临时 CPU sim，reset 后可选地先走几步再取状态快照。"""
    from envs.humanoid21.simulator import Humanoid21Simulator

    sim = Humanoid21Simulator()
    try:
        sim.reset(seed=42, options={"initial_distance": dist})
        if action is not None or post_steps:
            sim.set_action(action if action is not None else sim.get_action())
            for _ in range(post_steps):
                sim.physical_step()
        state = _snapshot_state(sim)
        action0 = {rid: np.asarray(sim.get_core_state()[rid]["joint_pos_norm"],
                                   dtype=np.float32)
                   for rid in ("robot_a", "robot_b")}
        return state, action0
    finally:
        sim.close()


def dynamics_cases():
    """跨后端物理对照：显式 qpos/qvel 状态 + 固定动作序列 + 子步数。

    不用 mjSTATE blob——MJX 没有 warmstart 概念，跨后端 case 只搬运
    qpos/qvel，warmstart 偏差属于已记录的固有边界。initial_action 按
    reset 同公式从状态的关节角归一化反推，保证 PD 初始无偏差。
    """
    standing, init_standing = _snapshot_at(2.0)
    tilted, init_tilted = _snapshot_at(
        2.0, post_steps=12,
        action={"robot_a": np.linspace(-0.3, 0.3, 21, dtype=np.float32),
                "robot_b": np.linspace(0.3, -0.3, 21, dtype=np.float32)})
    action_a = np.linspace(-0.2, 0.2, 21, dtype=np.float32)
    action_b = np.linspace(0.2, -0.2, 21, dtype=np.float32)
    cases = []
    for steps in (1, 25):
        cases.append({"id": f"dyn-standing-s{steps}", "kind": "action_sequence",
                      "level": "V1", "seed": 42, "episode": 0, "frame": 0,
                      "input": {"operation": "dynamics", "state": standing,
                                "initial_action": init_standing,
                                "actions": [{"robot_a": action_a,
                                             "robot_b": action_b}],
                                "substeps": steps}})
    cases.append({"id": "dyn-moving-2x5", "kind": "action_sequence", "level": "V1",
                  "seed": 42, "episode": 0, "frame": 0,
                  "input": {"operation": "dynamics", "state": tilted,
                            "initial_action": init_tilted,
                            "actions": [{"robot_a": -action_a,
                                         "robot_b": -action_b},
                                        {"robot_a": action_a,
                                         "robot_b": action_b}],
                            "substeps": 5}})
    return cases


def external_force_case():
    standing, init = _snapshot_at(2.0)
    return {"id": "extforce-torso-push", "kind": "physics", "level": "V1",
            "seed": 42, "episode": 0, "frame": 0,
            "input": {"operation": "external_force", "state": standing,
                      "initial_action": init,
                      "applies": [{"body": "torso", "robot": "robot_a",
                                   "force": np.array([80.0, 0.0, 40.0]),
                                   "torque": np.array([5.0, 0.0, 0.0])}],
                      "substeps": 3}}


def state_io_case():
    """写入 core state 后立即读回 derived（不推进物理），验证写后刷新。"""
    standing, _ = _snapshot_at(2.0)
    # 注意：写入不能产生深穿透姿态。实测把 robot_a 传送到与 robot_b 重叠的
    # 位置后，接触力 CPU≈332N / MJX≈3.7e6N——两个约束求解器在深穿透下
    # 本质分歧。该边界见 M2_RESULTS.md；本 case 只验证写后刷新读回。
    write = {"robot_a": {"root_pos": np.array([-0.7, 0.15, 1.35]),
                         "joint_pos_norm": np.linspace(-0.3, 0.3, 21)}}
    return {"id": "state-io-write-read", "kind": "physics", "level": "V1",
            "seed": 42, "episode": 0, "frame": 0,
            "input": {"operation": "state_io", "state": standing,
                      "write": write, "env_ids": [0]}}


def batch_isolation_case():
    """B=2 不同初态 + 不同动作：逐 env 输出对照，探测跨 env 泄漏。"""
    s0, i0 = _snapshot_at(1.5)
    s1, i1 = _snapshot_at(3.5)
    states = [s0, s1]
    initial_actions = [i0, i1]
    actions = [{"robot_a": np.linspace(-0.4, 0.4, 21, dtype=np.float32),
                "robot_b": np.linspace(0.4, -0.4, 21, dtype=np.float32)},
               {"robot_a": np.zeros(21, dtype=np.float32),
                "robot_b": np.zeros(21, dtype=np.float32)}]
    return {"id": "batch-isolation-2", "kind": "physics", "level": "V1",
            "seed": 42, "episode": 0, "frame": 0,
            "input": {"operation": "batch_isolation",
                      "env_states": states, "actions": actions, "substeps": 3,
                      "initial_actions": initial_actions}}


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
        op = case["input"].get("operation")
        if op == "dynamics":
            return self._dynamics(case["input"])
        if op == "external_force":
            return self._extforce(case["input"])
        if op == "state_io":
            return self._state_io(case["input"])
        if op == "batch_isolation":
            return self._batch_isolation(case["input"])
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

    # ------------------------------------------------------------------
    # 跨后端 case 执行器（M2）：显式 qpos/qvel 状态 + canonical 接触
    # ------------------------------------------------------------------
    @staticmethod
    def _restore_raw(sim, state):
        """写回显式 qpos/qvel 并 forward——跨后端状态契约，不依赖 mjSTATE blob。"""
        import mujoco

        sim.data.qpos[:] = state["qpos"]
        sim.data.qvel[:] = state["qvel"]
        # warmstart/ctrl/外力不属于跨后端状态契约：清零保证恢复求解确定性
        #（MJX 侧残留 qacc_warmstart/ctrl/xfrc 会把接触力拉偏 ~4%）。
        sim.data.qacc_warmstart[:] = 0.0
        sim.data.ctrl[:] = 0.0
        sim.data.xfrc_applied[:] = 0.0
        sim.data.qfrc_applied[:] = 0.0
        sim._data_cache.clear()
        mujoco.mj_forward(sim.model, sim.data)

    @staticmethod
    def _frame(sim):
        derived = sim.get_derived_state(["contacts", "torso_distance"])
        return {"qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy(),
                "core": sim.get_core_state(),
                "observation": sim.get_observation(),
                "torso_distance": derived["torso_distance"],
                "contacts": canonical_contacts(derived["contacts"])}

    def _dynamics(self, inputs):
        from envs.humanoid21.simulator import Humanoid21Simulator

        if set(inputs) != {"operation", "state", "initial_action", "actions", "substeps"}:
            raise ValueError("unknown or missing dynamics input")
        if type(inputs["substeps"]) is not int or inputs["substeps"] < 1:
            raise Unsupported("substeps must be a positive int")
        if not inputs["actions"]:
            raise ValueError("action sequence must be nonempty")
        sim = Humanoid21Simulator()
        try:
            sim.reset(seed=42)
            self._restore_raw(sim, inputs["state"])
            sim.set_action(inputs["initial_action"])
            frames = []
            for action in inputs["actions"]:
                if set(action) != {"robot_a", "robot_b"}:
                    raise ValueError("both agent actions required")
                sim.set_action(action)
                for _ in range(inputs["substeps"]):
                    sim.physical_step()
                frames.append(self._frame(sim))
            return {"frames": frames}
        finally:
            sim.close()

    def _extforce(self, inputs):
        from envs.humanoid21.simulator import Humanoid21Simulator

        if set(inputs) != {"operation", "state", "initial_action", "applies", "substeps"}:
            raise ValueError("unknown or missing external_force input")
        sim = Humanoid21Simulator()
        try:
            sim.reset(seed=42)
            self._restore_raw(sim, inputs["state"])
            sim.set_action(inputs["initial_action"])
            for ap in inputs["applies"]:
                sim.apply_external_force(ap["body"], np.asarray(ap["force"]),
                                         None if ap.get("torque") is None else np.asarray(ap["torque"]),
                                         ap["robot"])
            for _ in range(inputs["substeps"]):
                sim.physical_step()
            # xfrc_applied 每物理步清零——残留非零即语义错误
            residual = float(np.abs(np.asarray(sim.data.xfrc_applied)).max())
            return {"frames": [self._frame(sim)], "residual_xfrc_max": residual}
        finally:
            sim.close()

    def _state_io(self, inputs):
        from envs.humanoid21.simulator import Humanoid21Simulator

        if set(inputs) != {"operation", "state", "write", "env_ids"}:
            raise ValueError("unknown or missing state_io input")
        if inputs["env_ids"] != [0]:
            raise Unsupported("CPU oracle is single-env; env_ids must be [0]")
        sim = Humanoid21Simulator()
        try:
            sim.reset(seed=42)
            self._restore_raw(sim, inputs["state"])
            sim.set_core_state(inputs["write"])
            return {"frame": self._frame(sim)}
        finally:
            sim.close()

    def _batch_isolation(self, inputs):
        from envs.humanoid21.simulator import Humanoid21Simulator

        if set(inputs) != {"operation", "env_states", "actions", "initial_actions", "substeps"}:
            raise ValueError("unknown or missing batch_isolation input")
        n = len(inputs["env_states"])
        if n != len(inputs["actions"]) or n != len(inputs["initial_actions"]):
            raise ValueError("per-env inputs length mismatch")
        envs = []
        for i in range(n):
            sim = Humanoid21Simulator()
            try:
                sim.reset(seed=42)
                self._restore_raw(sim, inputs["env_states"][i])
                sim.set_action(inputs["initial_actions"][i])
                sim.set_action(inputs["actions"][i])
                for _ in range(inputs["substeps"]):
                    sim.physical_step()
                envs.append(self._frame(sim))
            finally:
                sim.close()
        return {"envs": envs}


# 跨后端 fixture 容差（CPU oracle 生成时写入 bundle；严格度按 M2 实测设定）。
# 依据：FP64 下动力学推进差 ~1e-13，float32 观测 ~1e-7，接触位置 ~3e-6；
# 取 atol/rtol=1e-5 既覆盖数值噪声，又远小于任何语义性偏差（如外力未清零、
# 深穿透求解分歧 ~1e6、action 未生效）。不在表内的字段（id、geom、mask 等
# 整型/结构字段）不受容差影响，仍精确比较。
CASE_TOLERANCES = {
    "dyn-standing-s1": {"field_tolerances": {"output.frames": {"atol": 1e-5, "rtol": 1e-5}}},
    "dyn-standing-s25": {"field_tolerances": {"output.frames": {"atol": 1e-5, "rtol": 1e-5}}},
    "dyn-moving-2x5": {"field_tolerances": {"output.frames": {"atol": 1e-5, "rtol": 1e-5}}},
    "extforce-torso-push": {"field_tolerances": {"output.frames": {"atol": 1e-5, "rtol": 1e-5}}},
    "state-io-write-read": {"field_tolerances": {"output.frame": {"atol": 1e-5, "rtol": 1e-5}}},
    "batch-isolation-2": {"field_tolerances": {"output.envs": {"atol": 1e-5, "rtol": 1e-5}}},
}


Adapter = CPUAdapter
