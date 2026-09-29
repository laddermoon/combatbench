"""M2 候选适配器：MjxHumanoid21Simulator。

只声明自己能执行的 case：
- ``dynamics``：显式 qpos/qvel 状态 + 固定动作序列，逐帧对照
  qpos/qvel/core/observation/canonical contacts。
- ``external_force``：挂起外力作用一个子步后自动清零的语义。
- ``state_io``：``set_core_state`` 写入后立即读回（写后 forward 刷新）。
- ``batch_isolation``：多 env 不同状态/动作逐 env 对照。

显式 unsupported（不静默降级）：
- M1 ``physics_case`` 的 ``mjSTATE_INTEGRATION`` blob——MJX 无 warmstart
  概念，属于已记录的跨后端边界。
- ``reward`` / ``trajectory`` kind——任务逻辑在宿主侧运行，由 CPU oracle
  承载；MJX 只验证物理契约。
- ``policy_eval``——留待后续里程碑。

来源指纹与 stale 检测由 ``validation.py`` 框架统一处理；本文件只负责
诚实地执行与输出。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .validation import Unsupported, canonical_contacts, CONTACT_ENTRY_KEYS

ROOT = Path(__file__).resolve().parents[2]

DEPENDENCY_FILES = (
    "envs/batchframework/mjx_simulator.py",
    "envs/batchframework/backend.py",
    "envs/batchframework/validation.py",
    "envs/batchframework/M0_BASELINE.md",
    "envs/batchframework/M2_PLAN.md",
    "envs/humanoid21/meta.py",
    "envs/humanoid21/simulator.py",
    "envs/humanoid21/battle_circular_v2.xml",
)


def _strip_batch(value, env=0):
    """把 (B, ...) 输出裁成单 env 视图。"""
    if isinstance(value, dict):
        return {k: _strip_batch(v, env) for k, v in value.items()}
    arr = np.asarray(value)
    return arr[env] if arr.ndim > 0 else arr


def _contacts_env(raw, env):
    """单 env 的 canonical 接触列表。"""
    single = {k: np.asarray(raw[k])[env] for k in CONTACT_ENTRY_KEYS}
    mask = np.asarray(raw["active_mask"])[env]
    return canonical_contacts(single, active_mask=mask)


class MJXAdapter:
    backend = "mjx-jax-fp64"
    version = "m2-v1"

    def __init__(self):
        paths = [ROOT / path for path in DEPENDENCY_FILES]
        model = ROOT / "envs/humanoid21/battle_circular_v2.xml"
        paths.extend(model.parent / node.attrib["file"] for node in ET.parse(model).iter()
                     if "file" in node.attrib)
        self.dependencies = sorted(set(paths))
        self._sims = {}

    def _sim(self, batch_size):
        from .mjx_simulator import MjxHumanoid21Simulator

        if batch_size not in self._sims:
            sim = MjxHumanoid21Simulator(batch_size=batch_size)
            sim.reset()
            self._sims[batch_size] = sim
        return self._sims[batch_size]

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
            raise Unsupported(
                "reward/trajectory logic runs on the host-side oracle; "
                "MJX adapter validates physics only")
        raise Unsupported(
            "MJX has no mjSTATE blob (warmstart-free); use 'dynamics' cases "
            "with explicit qpos/qvel")

    # ------------------------------------------------------------------
    def _frame(self, sim, env=0):
        derived = sim.get_derived_state(["contacts", "torso_distance"])
        return {
            "qpos": np.asarray(sim._mjx_data.qpos)[env],
            "qvel": np.asarray(sim._mjx_data.qvel)[env],
            "core": _strip_batch(sim.get_core_state(), env),
            "observation": _strip_batch(sim.get_observation(), env),
            "torso_distance": np.asarray(derived["torso_distance"])[env],
            "contacts": _contacts_env(derived["contacts"], env),
        }

    @staticmethod
    def _batched_action(action, batch_size):
        """{robot: (21,)} → {robot: (B, 21)} 广播为整批。"""
        return {rid: np.broadcast_to(np.asarray(v), (batch_size,) + np.asarray(v).shape).copy()
                for rid, v in action.items()}

    def _dynamics(self, inputs):
        if set(inputs) != {"operation", "state", "initial_action", "actions", "substeps"}:
            raise ValueError("unknown or missing dynamics input")
        sim = self._sim(1)
        sim.set_integration_state(inputs["state"]["qpos"][None],
                                  inputs["state"]["qvel"][None])
        sim.set_action(self._batched_action(inputs["initial_action"], 1))
        frames = []
        for action in inputs["actions"]:
            if set(action) != {"robot_a", "robot_b"}:
                raise ValueError("both agent actions required")
            sim.set_action(self._batched_action(action, 1))
            sim.physical_step(n_steps=inputs["substeps"])
            frames.append(self._frame(sim))
        return {"frames": frames}

    def _extforce(self, inputs):
        if set(inputs) != {"operation", "state", "initial_action", "applies", "substeps"}:
            raise ValueError("unknown or missing external_force input")
        sim = self._sim(1)
        sim.set_integration_state(inputs["state"]["qpos"][None],
                                  inputs["state"]["qvel"][None])
        sim.set_action(self._batched_action(inputs["initial_action"], 1))
        for ap in inputs["applies"]:
            sim.apply_external_force(
                ap["body"], np.asarray(ap["force"])[None].repeat(1, axis=0),
                None if ap.get("torque") is None else np.asarray(ap["torque"])[None],
                ap["robot"])
        sim.physical_step(n_steps=inputs["substeps"])
        residual = float(np.abs(np.asarray(sim._ext_force_jax)).max())
        return {"frames": [self._frame(sim)], "residual_xfrc_max": residual}

    def _state_io(self, inputs):
        if set(inputs) != {"operation", "state", "write", "env_ids"}:
            raise ValueError("unknown or missing state_io input")
        env_ids = list(inputs["env_ids"])
        sim = self._sim(1)
        sim.set_integration_state(inputs["state"]["qpos"][None],
                                  inputs["state"]["qvel"][None])
        write = {rid: {k: np.asarray(v)[None] for k, v in fields.items()}
                 for rid, fields in inputs["write"].items()}
        sim.set_core_state(write, env_ids=env_ids)
        return {"frame": self._frame(sim)}

    def _batch_isolation(self, inputs):
        if set(inputs) != {"operation", "env_states", "actions", "initial_actions", "substeps"}:
            raise ValueError("unknown or missing batch_isolation input")
        n = len(inputs["env_states"])
        if n != len(inputs["actions"]) or n != len(inputs["initial_actions"]):
            raise ValueError("per-env inputs length mismatch")
        sim = self._sim(n)
        for i in range(n):
            st = inputs["env_states"][i]
            sim.set_integration_state(st["qpos"][None], st["qvel"][None], env_ids=[i])
        sim.set_action({rid: np.stack([np.asarray(a[rid]) for a in inputs["initial_actions"]])
                        for rid in ("robot_a", "robot_b")})
        sim.set_action({rid: np.stack([np.asarray(a[rid]) for a in inputs["actions"]])
                        for rid in ("robot_a", "robot_b")})
        sim.physical_step(n_steps=inputs["substeps"])
        return {"envs": [self._frame(sim, env=i) for i in range(n)]}


Adapter = MJXAdapter
