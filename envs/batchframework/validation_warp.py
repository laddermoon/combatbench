"""M2 候选适配器：WarpHumanoid21Simulator（mujoco-warp 后端）。

与 ``validation_mjx.py`` 同构：只声明能执行的 case，其余显式
``Unsupported``，不静默降级。

- ``dynamics`` / ``external_force`` / ``state_io`` / ``batch_isolation``：
  同 MJX 适配器的执行路径，底层换成 NWORLDS 数据布局。
- ``reward`` / ``trajectory`` / ``physics_case``（mjSTATE blob）：沿用
  MJX 的 unsupported 语义——reward/trajectory 由 CPU oracle 承载，
  mjSTATE blob 在 warp 同样无对应物。
- **精度边界**：warp 只有 fp32。fixture 期望输出是 CPU fp64——
  回放时在调用方以文档化的 fp32 容差覆盖（见
  ``WARP_TOLERANCES`` 与 M2 结果文档），其余校验（shape、接触
  拓扑、整型语义）仍按原规则严格执行。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .validation import Unsupported, canonical_contacts, CONTACT_ENTRY_KEYS

ROOT = Path(__file__).resolve().parents[2]

DEPENDENCY_FILES = (
    "envs/batchframework/warp_simulator.py",
    "envs/batchframework/mjx_simulator.py",   # warp 子类继承的语义代码源
    "envs/batchframework/backend.py",
    "envs/batchframework/validation.py",
    "envs/batchframework/M0_BASELINE.md",
    "envs/batchframework/M2_PLAN.md",
    "envs/humanoid21/meta.py",
    "envs/humanoid21/simulator.py",
    "envs/humanoid21/battle_circular_v2.xml",
)


def _strip_batch(value, env=0):
    if isinstance(value, dict):
        return {k: _strip_batch(v, env) for k, v in value.items()}
    arr = np.asarray(value)
    return arr[env] if arr.ndim > 0 else arr


def _contacts_env(raw, env):
    single = {k: np.asarray(raw[k])[env] for k in CONTACT_ENTRY_KEYS}
    mask = np.asarray(raw["active_mask"])[env]
    return canonical_contacts(single, active_mask=mask)


class WarpAdapter:
    backend = "mujoco-warp-fp32"
    version = "m2-v1"

    def __init__(self):
        paths = [ROOT / path for path in DEPENDENCY_FILES]
        model = ROOT / "envs/humanoid21/battle_circular_v2.xml"
        paths.extend(model.parent / node.attrib["file"] for node in ET.parse(model).iter()
                     if "file" in node.attrib)
        self.dependencies = sorted(set(paths))
        self._sims = {}

    def _sim(self, batch_size):
        from .warp_simulator import WarpHumanoid21Simulator

        if batch_size not in self._sims:
            sim = WarpHumanoid21Simulator(batch_size=batch_size)
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
                "warp adapter validates physics only")
        raise Unsupported(
            "warp has no mjSTATE blob (warmstart-free); use 'dynamics' cases "
            "with explicit qpos/qvel")

    # ------------------------------------------------------------------
    def _frame(self, sim, env=0):
        derived = sim.get_derived_state(["contacts", "torso_distance"])
        # _mjx_data 属性返回 float64 快照（与 MJXAdapter._frame 同路径/dtype）
        snap = sim._mjx_data
        return {
            "qpos": np.asarray(snap.qpos)[env],
            "qvel": np.asarray(snap.qvel)[env],
            "core": _strip_batch(sim.get_core_state(), env),
            "observation": _strip_batch(sim.get_observation(), env),
            "torso_distance": np.asarray(derived["torso_distance"])[env],
            "contacts": _contacts_env(derived["contacts"], env),
        }

    @staticmethod
    def _batched_action(action, batch_size):
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


Adapter = WarpAdapter
