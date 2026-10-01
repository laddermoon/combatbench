"""Humanoid21Binding — Humanoid21 任务的宿主无关批量语义绑定（E1-W2）。

本模块持有"任务↔模型"的全部共享逻辑，供 MJX / Warp / Fake 等批量
后端引用，替代原先的 ``WarpHumanoid21Simulator(MjxHumanoid21Simulator)``
继承复用。归属原则：

- **这里放**：模型编译、meta 查找表、归一化参数、PD 表、初始姿态
  计算、core-state↔qpos/qvel 映射、观测/接触/静态数据的 host 侧提取
  公式——全部是"给定 (model, data 视图) → numpy 语义结果"的纯函数。
- **不放**：任何 jax/warp 后端对象、设备张量、episode 簿记、
  训练侧概念（Job/Episode/PPO）。

提取自 ``envs/batchframework/mjx_simulator.py``——函数体与原实现
逐行一致（``self._x`` → ``self.x``），MJX 模拟器同步改为委托调用，
保证两条后端路径的语义公式只有一份实现。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

import mujoco

from .meta import Humanoid21Meta


def quat_to_rot_np(quat: np.ndarray) -> np.ndarray:
    """Batched [w,x,y,z] → (...,3,3) rotation matrix (numpy, leading dims preserved)."""
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def sqrt_signed(v: np.ndarray, div2_inside: bool = False) -> np.ndarray:
    """CPU 观测的非线性压缩：joint_vel 用 sign*sqrt(|v|)/2，angular/kp_vel 用
    sign*sqrt(|v/2|)。div2_inside 区分两种公式（角速度 /2 在 sqrt 内）。"""
    if div2_inside:
        return np.sign(v) * np.sqrt(np.abs(v / 2.0))
    return np.sign(v) * np.sqrt(np.abs(v)) / 2.0


class Humanoid21Binding:
    """Humanoid21 模型与批量语义的宿主无关绑定。

    Args:
        initial_distance: 两机器人初始间距（米，可被 per-env options 覆盖）。
        initial_pose_a / initial_pose_b: ``INITIAL_POSES`` 中的姿态名。

    公开属性即契约字段（后端/绑定消费方直接读取，无访问器层）：
    ``model`` / ``meta`` / ``robots`` / ``ground_geom_id`` /
    ``norm_params`` / ``pd_tables`` / ``body_name_to_id``。
    """

    DT = Humanoid21Meta.DT
    ACTION_DIM = Humanoid21Meta.ACTION_DIM
    # 必须与 envs/humanoid21/simulator.py 的 ARENA_XML 指向同一个文件；
    # condim=3 摩擦接触 / impratio=10 / 圆形墙都由该模型定义，换文件即换任务。
    ARENA_XML = str(Path(__file__).resolve().parent / "battle_circular_v2.xml")
    KP = Humanoid21Meta.KP
    KD = Humanoid21Meta.KD
    CONTROLLED_JOINTS = Humanoid21Meta.CONTROLLED_JOINTS
    INITIAL_POSES = Humanoid21Meta.INITIAL_POSES

    def __init__(
        self,
        initial_distance: float = 2.0,
        initial_pose_a: str = "standing",
        initial_pose_b: str = "standing",
    ):
        self.initial_distance = initial_distance
        self.initial_pose_a = initial_pose_a
        self.initial_pose_b = initial_pose_b

        self.model = mujoco.MjSpec.from_file(self.ARENA_XML).compile()
        self.model.opt.timestep = self.DT

        self.meta = Humanoid21Meta.build_runtime_tables(self.model)
        self.robots = self.meta["robots"]
        self.ground_geom_id = self.meta["ground_geom_id"]

        self.norm_params: Dict[str, Dict[str, np.ndarray]] = {}
        for robot_id in ["robot_a", "robot_b"]:
            jnt_ranges = self.robots[robot_id]["jnt_ranges"]  # (21, 2)
            lower = jnt_ranges[:, 0]
            upper = jnt_ranges[:, 1]
            self.norm_params[robot_id] = {
                "reference": ((lower + upper) / 2.0).astype(np.float32),
                "scale": ((upper - lower) / 2.0).astype(np.float32),
            }

        self.pd_tables: Dict[str, Dict[str, np.ndarray]] = {}
        for robot_id in ["robot_a", "robot_b"]:
            act_ids = self.robots[robot_id]["actuator_ids"]
            gear = np.array(self.model.actuator_gear[act_ids, 0], dtype=np.float64)
            gear[gear == 0] = 1.0
            ctrl_lo = np.array(self.model.actuator_ctrlrange[act_ids, 0], dtype=np.float64)
            ctrl_hi = np.array(self.model.actuator_ctrlrange[act_ids, 1], dtype=np.float64)
            self.pd_tables[robot_id] = {
                "actuator_ids": act_ids,
                "gear": gear,
                "ctrl_lo": ctrl_lo,
                "ctrl_hi": ctrl_hi,
            }

        self.body_name_to_id: Dict[str, int] = {}
        for robot_id, suffix in Humanoid21Meta.ROBOT_SUFFIXES.items():
            for body_name in Humanoid21Meta.ROBOT_BODY_NAMES:
                full = f"{body_name}{suffix}"
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, full
                )
                if bid >= 0:
                    self.body_name_to_id[full] = bid

    # ------------------------------------------------------------------
    # 静态表构建（xp = np 或 jp，由调用方按后端选）
    # ------------------------------------------------------------------
    def build_statics(self, xp, dtype) -> Dict[str, Any]:
        """Build the per-robot PD table; `xp` is jp or np depending on backend."""
        statics = {}
        for robot_id in ["robot_a", "robot_b"]:
            r = self.robots[robot_id]
            norm = self.norm_params[robot_id]
            pd = self.pd_tables[robot_id]
            statics[robot_id] = {
                "qpos_indices": xp.asarray(r["qpos_indices"], dtype=xp.int32),
                "qvel_indices": xp.asarray(r["qvel_indices"], dtype=xp.int32),
                "actuator_ids": xp.asarray(r["actuator_ids"], dtype=xp.int32),
                "norm_ref": xp.asarray(norm["reference"], dtype=xp.float32),
                "norm_scale": xp.asarray(norm["scale"], dtype=xp.float32),
                "gear": xp.asarray(pd["gear"], dtype=dtype),
                "ctrl_lo": xp.asarray(pd["ctrl_lo"], dtype=dtype),
                "ctrl_hi": xp.asarray(pd["ctrl_hi"], dtype=dtype),
                "kp": xp.asarray(self.KP, dtype=dtype),
                "kd": xp.asarray(self.KD, dtype=dtype),
                "root_qpos_adr": r["root_qpos_adr"],
                "root_qvel_adr": r["root_qvel_adr"],
            }
        return statics

    def resolve_body_id(self, body_name: str, robot_id: str) -> int:
        """body 短名 + robot_id → MuJoCo body id（apply_external_force 用）。"""
        suffix = self.robots[robot_id]["suffix"]
        full = f"{body_name}{suffix}"
        bid = self.body_name_to_id.get(full)
        if bid is None:
            raise ValueError(f"Body not found: {full}")
        return bid

    # ------------------------------------------------------------------
    # Reset 姿态计算（纯 numpy，跨后端共享）
    # ------------------------------------------------------------------
    @staticmethod
    def broadcast_option(options, key, default, batch_size):
        """options 中的标量或 (B,) 序列 → 长度 B 的 per-env 列表。"""
        value = options.get(key, default) if options else default
        if isinstance(value, str) or np.isscalar(value):
            return [value] * batch_size
        seq = list(value)
        if len(seq) != batch_size:
            raise ValueError(f"options[{key}] must be scalar or length {batch_size}")
        return seq

    def compute_reset_state(self, batch_size: int, seeds=None, options=None):
        """Per-env reset 姿态计算（纯 numpy，跨后端共享）。

        返回 (qpos_all (B,nq), action_a (B,21), action_b (B,21))。
        与 CPU reset 一致：初始姿态确定性，seeds 只由调用方记录；
        ``initial_distance`` 与 ``initial_pose_a/b`` 支持标量或长度 B 的序列。
        """
        from scipy.spatial.transform import Rotation as Rot

        B = int(batch_size)
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)

        dists = [float(v) for v in self.broadcast_option(
            options, "initial_distance", self.initial_distance, B)]
        poses_a = self.broadcast_option(options, "initial_pose_a", self.initial_pose_a, B)
        poses_b = self.broadcast_option(options, "initial_pose_b", self.initial_pose_b, B)

        qpos_all = np.broadcast_to(np.asarray(data.qpos), (B, data.qpos.size)).copy()
        action_a = np.zeros((B, self.ACTION_DIM), dtype=np.float32)
        action_b = np.zeros((B, self.ACTION_DIM), dtype=np.float32)

        for i in range(B):
            for robot_id, pose_name, x_offset in [
                ("robot_a", poses_a[i], -dists[i] / 2.0),
                ("robot_b", poses_b[i], dists[i] / 2.0),
            ]:
                pose_config = self.INITIAL_POSES[pose_name]
                cache = self.robots[robot_id]
                root_qpos_adr = cache["root_qpos_adr"]
                qpos_indices = cache["qpos_indices"]

                root_pos = np.asarray(pose_config["root_pos"], dtype=np.float64).copy()
                root_pos[0] = x_offset
                qpos_all[i, root_qpos_adr : root_qpos_adr + 3] = root_pos

                # 与 CPU reset 相同的 scipy 旋转：robot_b 绕 z 轴转 180° 面向 robot_a。
                root_quat = np.asarray(pose_config["root_quat"], dtype=np.float64).copy()
                if robot_id == "robot_b":
                    q_scipy = np.array([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
                    q_new = (Rot.from_euler("z", np.pi) * Rot.from_quat(q_scipy)).as_quat()
                    root_quat = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])

                qpos_all[i, root_qpos_adr + 3 : root_qpos_adr + 7] = root_quat
                qpos_all[i, qpos_indices] = pose_config["joint_pos"]

                # 与 CPU 相同：初始动作由实际关节位置反推，而非预设近似值。
                norm = self.norm_params[robot_id]
                action = ((qpos_all[i, qpos_indices] - norm["reference"])
                          / norm["scale"]).astype(np.float32)
                if robot_id == "robot_a":
                    action_a[i] = action
                else:
                    action_b[i] = action

        return qpos_all, action_a, action_b

    # ------------------------------------------------------------------
    # core-state ↔ qpos/qvel 映射（纯 numpy，跨后端共享）
    # ------------------------------------------------------------------
    def write_core_state(self, state, env_ids, qpos_new, qvel_new):
        """core-state 字段 → 原始 qpos/qvel 数组的映射（纯 numpy，跨后端共享）。"""
        for robot_id in ["robot_a", "robot_b"]:
            if robot_id not in state:
                continue
            robot_state = state[robot_id]
            cache = self.robots[robot_id]
            norm = self.norm_params[robot_id]
            root_qa = cache["root_qpos_adr"]
            root_qva = cache["root_qvel_adr"]
            qpos_idx = cache["qpos_indices"]
            qvel_idx = cache["qvel_indices"]

            for i, eid in enumerate(env_ids):
                if "root_pos" in robot_state:
                    qpos_new[eid, root_qa : root_qa + 3] = robot_state["root_pos"][i]
                if "root_rot" in robot_state:
                    qpos_new[eid, root_qa + 3 : root_qa + 7] = robot_state["root_rot"][i]
                if "root_vel_local" in robot_state:
                    # 线速度机体系 → 世界系写回 qvel[0:3]（与 CPU set_core_state 对称）
                    quat = qpos_new[eid, root_qa + 3 : root_qa + 7]
                    R_mat = quat_to_rot_np(quat)
                    qvel_new[eid, root_qva : root_qva + 3] = R_mat @ np.asarray(
                        robot_state["root_vel_local"][i], dtype=np.float64)
                if "root_angular_vel_local" in robot_state:
                    # 角速度本就以机体系存放于 qvel[3:6]，直接写入（CPU 同样不旋转）
                    qvel_new[eid, root_qva + 3 : root_qva + 6] = np.asarray(
                        robot_state["root_angular_vel_local"][i], dtype=np.float64)
                if "joint_pos_norm" in robot_state:
                    jpn = robot_state["joint_pos_norm"][i]
                    qpos_new[eid, qpos_idx] = jpn * norm["scale"] + norm["reference"]
                if "joint_vel_norm" in robot_state:
                    jvn = robot_state["joint_vel_norm"][i]
                    qvel_new[eid, qvel_idx] = jvn * norm["scale"]

    # ------------------------------------------------------------------
    # host 侧提取（验证/回放/调试路径；非吞吐路径）
    # ------------------------------------------------------------------
    def extract_core_state(self, data, is_history: bool) -> Dict[str, Any]:
        """Extract core state from batched data（JAX/warp 快照 → numpy）。"""
        result: Dict[str, Any] = {}

        for robot_id in ["robot_a", "robot_b"]:
            r = self.robots[robot_id]
            norm = self.norm_params[robot_id]

            root_qa = r["root_qpos_adr"]
            root_qva = r["root_qvel_adr"]
            qpos_idx = r["qpos_indices"]
            qvel_idx = r["qvel_indices"]

            # Root pos and rot
            root_pos_np = np.asarray(data.qpos[..., root_qa : root_qa + 3])
            root_rot_np = np.asarray(data.qpos[..., root_qa + 3 : root_qa + 7])

            # MuJoCo free joint 的 qvel 两半坐标系不同（与 CPU get_core_state 一致）：
            # qvel[0:3] 线速度是世界系 → 乘 R^T 转机体系；
            # qvel[3:6] 角速度本就以机体系存放 → 直接取用，不做旋转。
            root_vel_np = np.asarray(data.qvel[..., root_qva : root_qva + 3])
            root_ang_vel_local = np.asarray(data.qvel[..., root_qva + 3 : root_qva + 6])

            R_inv = np.swapaxes(quat_to_rot_np(root_rot_np), -1, -2)
            root_vel_local = np.einsum("...ij,...j->...i", R_inv, root_vel_np)

            # Joint pos/vel
            joint_pos = np.asarray(data.qpos[..., qpos_idx])
            joint_vel = np.asarray(data.qvel[..., qvel_idx])

            ref = norm["reference"]
            scale = norm["scale"]
            joint_pos_norm = (joint_pos - ref) / scale
            joint_vel_norm = joint_vel / scale

            result[robot_id] = {
                "root_pos": root_pos_np.astype(np.float32),
                "root_rot": root_rot_np.astype(np.float32),
                "root_vel_local": root_vel_local.astype(np.float32),
                "root_angular_vel_local": root_ang_vel_local.astype(np.float32),
                "joint_pos_norm": joint_pos_norm.astype(np.float32),
                "joint_vel_norm": joint_vel_norm.astype(np.float32),
            }

        return result

    def extract_derived_state(
        self, data, fields: Optional[Sequence[str]], is_history: bool
    ) -> Dict[str, Any]:
        if fields is None:
            fields = ["torso_distance", "contacts", "robot_a", "robot_b"]
        else:
            fields = list(fields)
            unknown = set(fields) - {"torso_distance", "contacts", "robot_a", "robot_b"}
            if unknown:
                raise KeyError(f"extract_derived_state: unknown fields {unknown}")

        result: Dict[str, Any] = {}

        if "torso_distance" in fields:
            torso_a_id = self.robots["robot_a"]["root_body_id"]
            torso_b_id = self.robots["robot_b"]["root_body_id"]
            pos_a = np.asarray(data.xpos[..., torso_a_id, :])
            pos_b = np.asarray(data.xpos[..., torso_b_id, :])
            dist = np.linalg.norm(pos_b - pos_a, axis=-1, keepdims=True)
            result["torso_distance"] = dist.astype(np.float32)

        if "contacts" in fields:
            result["contacts"] = self.extract_contacts_batch(data)

        for rid in ("robot_a", "robot_b"):
            if rid in fields:
                opp_id = "robot_b" if rid == "robot_a" else "robot_a"
                view = self.robot_view_batch(data, rid, opp_id)
                view.update(self.collect_body_joint_arrays(data, rid))
                result[rid] = view

        return result

    def extract_contacts_batch(self, data) -> Dict[str, Any]:
        """Extract contacts from batched data.

        Uses efc_force to compute proper contact forces matching
        mujoco.mj_contactForce. For pyramidal cone with condim=3:
          normal = sum(efc[0:4])
          friction1 = efc[0] - efc[1]
          friction2 = efc[2] - efc[3]
        For condim=1 (frictionless): normal = efc[0].

        force_world = frame.T @ [normal, friction1, friction2]
        """
        contact = data._impl.contact
        geom = np.asarray(contact.geom)
        dist = np.asarray(contact.dist)
        pos = np.asarray(contact.pos)
        frame = np.asarray(contact.frame)
        efc_address = np.asarray(contact.efc_address)
        contact_dim = np.asarray(contact.dim)
        efc_force = np.asarray(data._impl.efc_force)

        # 支持任意前导维：history=False 时 (B, C)，history=True 时 (B, T, C)，
        # 无 batch 时 (C,) 统一补一维再处理。
        if geom.ndim == 1:
            geom = geom[np.newaxis, np.newaxis]
            dist = dist[np.newaxis, np.newaxis]
            pos = pos[np.newaxis, np.newaxis]
            frame = frame[np.newaxis, np.newaxis]
            efc_force = efc_force[np.newaxis, np.newaxis]
        elif geom.ndim == 2:
            geom = geom[np.newaxis]
            dist = dist[np.newaxis]
            pos = pos[np.newaxis]
            frame = frame[np.newaxis]
            efc_force = efc_force[np.newaxis]

        # efc_address and contact_dim are NOT batched (shared model metadata)
        lead_shape = dist.shape[:-1]
        max_contacts = dist.shape[-1]
        active = dist <= 0  # (B, max_contacts)

        # Body IDs from geom IDs
        geom_bodyid = np.asarray(self.model.geom_bodyid)
        body1 = geom_bodyid[geom[..., 0].astype(np.int64)]
        body2 = geom_bodyid[geom[..., 1].astype(np.int64)]

        # Affiliation
        geom_id_to_aff = self.meta["geom_id_to_aff"]
        aff_table = np.zeros(self.model.ngeom, dtype=np.int8)
        for gid, aff in geom_id_to_aff.items():
            aff_table[gid] = aff
        aff1 = aff_table[geom[..., 0].astype(np.int64)]
        aff2 = aff_table[geom[..., 1].astype(np.int64)]

        # 与 CPU 语义对齐：ncon = 每 env 的活跃接触数（CPU 是标量，batch 是 (B,)）。
        # padding 槽位由 active_mask 标记，容量由 capacity 报告。
        contact_count = np.sum(active, axis=-1).astype(np.int32)

        # Compute contact forces from efc_force.
        # For pyramidal cone with dim=3: 4 constraint rows per contact.
        # For dim=1: 1 constraint row.
        # n_rows = max(1, 2*(dim-1)) for pyramidal cone.
        n_rows = np.maximum(1, 2 * (contact_dim - 1))  # (max_contacts,)

        # Gather efc_force for each contact
        # efc_address: (max_contacts,) — NOT batched (shared model)
        # efc_force: (B, max_efc) — batched
        force_mag = np.zeros(lead_shape + (max_contacts,), dtype=np.float64)
        force_world = np.zeros(lead_shape + (max_contacts, 3), dtype=np.float64)

        for lead_idx in np.ndindex(lead_shape):
            for c in range(max_contacts):
                if not active[lead_idx + (c,)]:
                    continue
                # efc_address 语义非 batched (C,)，但 NWORLDS 后端（warp）
                # 的行址逐 world 不同，允许传入 (B, ..., C) 逐 lead 寻址。
                addr = int(efc_address[lead_idx + (c,)]
                           if efc_address.ndim > 1 else efc_address[c])
                nr = int(n_rows[c])
                ef = efc_force[lead_idx + (slice(addr, addr + nr),)]

                if nr == 1:
                    normal = ef[0]
                    f1, f2 = 0.0, 0.0
                elif nr == 4:
                    normal = ef[0] + ef[1] + ef[2] + ef[3]
                    f1 = ef[0] - ef[1]
                    f2 = ef[2] - ef[3]
                else:
                    normal = ef.sum()
                    f1 = ef[0] - ef[1] if nr >= 2 else 0.0
                    f2 = ef[2] - ef[3] if nr >= 4 else 0.0

                force_local = np.array([normal, f1, f2])
                force_mag[lead_idx + (c,)] = np.linalg.norm(force_local)
                # frame[*,c] is (3,3) with rows [normal, tangent1, tangent2]
                # force_world = frame.T @ force_local (matches original simulator)
                force_world[lead_idx + (c,)] = frame[lead_idx + (c,)].T @ force_local

        return {
            "ncon": contact_count,
            "capacity": int(max_contacts),
            "active_mask": active,
            "geom1": geom[..., 0].astype(np.int32),
            "geom2": geom[..., 1].astype(np.int32),
            "body1": body1.astype(np.int32),
            "body2": body2.astype(np.int32),
            "aff1": aff1,
            "aff2": aff2,
            "force_mag": force_mag.astype(np.float32),
            "force_world": force_world.astype(np.float32),
            "position": pos.astype(np.float32),
            "normal": frame[..., 0, :].astype(np.float32),
            "frame": frame.astype(np.float32),
        }

    def robot_view_batch(
        self, data, robot_id: str, opponent_id: str
    ) -> Dict[str, Any]:
        """Per-robot view (batched). 逐字段对齐 CPU ``_get_robot_view``。

        与 CPU 相同的语义要点：
        - 线速度 qvel[0:3] 世界系 → R^T 转机体系；角速度 qvel[3:6] 本就在机体系；
        - relative_vel 用的是**对手** torso 的 cvel 线速度（字段名带 relative
          但 CPU 语义如此，不为"修正命名"改公式）；
        - 观测中关节速度做 sign*sqrt(|v|)/2，角速度做 sign*sqrt(|v/2|)，
          对手 head 关键点速度不变换，其余 4 个做 sign*sqrt(|v|)/2；
        - 字典字段保留原始物理值，sqrt 变换只作用于扁平 observation。
        """
        cache = self.robots[robot_id]
        opp_cache = self.robots[opponent_id]
        norm = self.norm_params[robot_id]

        torso_id = cache["root_body_id"]
        opp_torso_id = opp_cache["root_body_id"]

        self_pos = np.asarray(data.xpos[..., torso_id, :])
        self_quat = np.asarray(data.xquat[..., torso_id, :])  # [w,x,y,z]
        opp_pos = np.asarray(data.xpos[..., opp_torso_id, :])
        opp_quat = np.asarray(data.xquat[..., opp_torso_id, :])

        R_self = quat_to_rot_np(self_quat)      # body→world, (...,3,3)
        R_self_inv = np.swapaxes(R_self, -1, -2)  # world→body

        height = self_pos[..., 2:3]                              # (...,1)
        projected_gravity = -R_self[..., 2, :]                   # (...,3) 与 CPU -R[2,:] 一致

        root_qva = cache["root_qvel_adr"]
        linear_vel = np.einsum(                                  # 机体系线速度
            "...ij,...j->...i", R_self_inv,
            np.asarray(data.qvel[..., root_qva : root_qva + 3]))
        angular_vel = np.asarray(data.qvel[..., root_qva + 3 : root_qva + 6])  # 本就在机体系

        feet_forces = self.feet_forces_batch(data, robot_id)
        arena_center_local = np.einsum("...ij,...j->...i", R_self_inv, -self_pos)

        # 对手基础位姿：relative_vel 取对手 torso 线速度转到自机体系（CPU 语义）。
        relative_pos_local = np.einsum("...ij,...j->...i", R_self_inv, opp_pos - self_pos)
        opp_vel_global = np.asarray(data.cvel[..., opp_torso_id, 3:6])
        relative_vel_local = np.einsum("...ij,...j->...i", R_self_inv, opp_vel_global)
        opp_forward = quat_to_rot_np(opp_quat)[..., :, 0]       # 对手局部 x 轴（世界系）
        face_vector = np.einsum("...ij,...j->...i", R_self_inv, opp_forward)

        kp = opp_cache["keypoint_body_ids"]
        kp_names = ["head", "hand_right", "hand_left", "foot_right", "foot_left"]
        kp_pos_local, kp_vel_local = {}, {}
        for name in kp_names:
            bid = kp[name]
            kp_pos_local[name] = np.einsum(
                "...ij,...j->...i", R_self_inv,
                np.asarray(data.xpos[..., bid, :]) - self_pos).astype(np.float32)
            kp_vel_local[name] = np.einsum(
                "...ij,...j->...i", R_self_inv,
                np.asarray(data.cvel[..., bid, 3:6])).astype(np.float32)

        qpos_idx = cache["qpos_indices"]
        qvel_idx = cache["qvel_indices"]
        joint_pos_norm = (np.asarray(data.qpos[..., qpos_idx]) - norm["reference"]) / norm["scale"]
        joint_vel_norm = np.asarray(data.qvel[..., qvel_idx]) / norm["scale"]

        # 96 维扁平观测：字段顺序与变换与 CPU 完全一致。
        joint_vel_obs = sqrt_signed(joint_vel_norm)
        ang_vel_obs = sqrt_signed(angular_vel, div2_inside=True)
        kp_vel_rest_obs = np.concatenate(
            [sqrt_signed(kp_vel_local[n]) for n in kp_names[1:]], axis=-1)
        observation = np.concatenate([
            joint_pos_norm, joint_vel_obs, projected_gravity, height,
            linear_vel, ang_vel_obs, feet_forces, arena_center_local,
            relative_pos_local, relative_vel_local, face_vector,
            kp_pos_local["head"], kp_pos_local["hand_right"], kp_pos_local["hand_left"],
            kp_pos_local["foot_right"], kp_pos_local["foot_left"],
            kp_vel_local["head"],                      # head 速度不变换
            kp_vel_rest_obs,
        ], axis=-1).astype(np.float32)

        return {
            "root_state": {
                "height": height.astype(np.float32),
                "projected_gravity": projected_gravity.astype(np.float32),
                "linear_vel": linear_vel.astype(np.float32),
                "angular_vel": angular_vel.astype(np.float32),
                "arena_center_local": arena_center_local.astype(np.float32),
            },
            "feet_forces": feet_forces,
            "opponent_basic_pose": {
                "relative_pos": relative_pos_local.astype(np.float32),
                "relative_vel": relative_vel_local.astype(np.float32),
                "face_vector": face_vector.astype(np.float32),
            },
            "opponent_keypoint_pos": kp_pos_local,
            "opponent_keypoint_vel": kp_vel_local,
            "observation": observation,
            "uprightness": np.asarray(R_self[..., 2, 2:3]).astype(np.float32),
            "opponent_in_local": {
                "pos": relative_pos_local.astype(np.float32),
                "vel": relative_vel_local.astype(np.float32),
                "rot": face_vector.astype(np.float32),
            },
        }

    def collect_body_joint_arrays(self, data, robot_id: str) -> Dict[str, Any]:
        """CPU ``_collect_body_joint_arrays`` 的批量版：per-body 世界系数组 + joint anchor。"""
        cache = self.robots[robot_id]
        body_ids = cache["body_ids_sorted"]
        body_names = cache["body_names"]

        def by_body(arr):
            vals = np.asarray(arr[..., body_ids, :], dtype=np.float32)
            return {name: vals[..., i, :] for i, name in enumerate(body_names)}

        cvel = np.asarray(data.cvel[..., body_ids, :], dtype=np.float32)
        joint_ids_by_name = cache["joint_ids_by_name"]
        return {
            "body_xpos": by_body(data.xpos),
            "body_xipos": by_body(data.xipos),
            "body_xquat": by_body(data.xquat),
            "body_linvel_world": {name: cvel[..., i, 3:6] for i, name in enumerate(body_names)},
            "body_angvel_world": {name: cvel[..., i, 0:3] for i, name in enumerate(body_names)},
            "joint_world_anchor": {
                name: np.asarray(data.xanchor[..., jid, :], dtype=np.float32)
                for name, jid in joint_ids_by_name.items()
            },
        }

    def feet_forces_batch(self, data, robot_id: str) -> np.ndarray:
        """Get feet contact forces, **normalized by body weight m*g** (dimensionless).

        Uses the same force_mag computed in extract_contacts_batch (from efc_force),
        matching the original simulator's _get_feet_forces which uses mj_contactForce output.

        The division by ``body_weight`` must stay in lockstep with
        ``Humanoid21Simulator._get_feet_forces`` — the two backends are
        required to produce bit-comparable observations, so a unit change
        in one without the other would silently desync them.
        """
        cache = self.robots[robot_id]
        kp = cache["keypoint_body_ids"]
        foot_right_id = kp["foot_right"]
        foot_left_id = kp["foot_left"]
        ground_gid = self.ground_geom_id
        body_weight = cache["body_weight"]

        # Extract contacts to get force_mag (same as extract_contacts_batch)
        contacts = self.extract_contacts_batch(data)

        geom1 = contacts["geom1"]  # (B, max_contacts)
        geom2 = contacts["geom2"]  # (B, max_contacts)
        body1 = contacts["body1"]  # (B, max_contacts)
        body2 = contacts["body2"]  # (B, max_contacts)
        force_mag = contacts["force_mag"]  # (B, max_contacts)
        active_mask = contacts["active_mask"]  # (B, max_contacts)

        g1_ground = geom1 == ground_gid
        g2_ground = geom2 == ground_gid
        ground_mask = (g1_ground | g2_ground) & active_mask

        other_body = np.where(g1_ground, body2, body1)

        right_force = np.sum(
            np.where(ground_mask & (other_body == foot_right_id), force_mag, 0.0),
            axis=-1,
        )
        left_force = np.sum(
            np.where(ground_mask & (other_body == foot_left_id), force_mag, 0.0),
            axis=-1,
        )

        return (
            np.stack([right_force, left_force], axis=-1) / body_weight
        ).astype(np.float32)

    def build_observation(self, data) -> Dict[str, Any]:
        """→ {"robot_a": (B,96), "robot_b": (B,96)} numpy 观测。"""
        result = {}
        for rid in ("robot_a", "robot_b"):
            opp_id = "robot_b" if rid == "robot_a" else "robot_a"
            view = self.robot_view_batch(data, rid, opp_id)
            result[rid] = view["observation"]
        return result

    def static_data(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for robot_id in ["robot_a", "robot_b"]:
            cache = self.robots[robot_id]
            body_names = list(cache["body_names"])
            body_masses = np.asarray(cache["body_masses"], dtype=np.float32)
            result[robot_id] = {
                "dof_names": list(self.CONTROLLED_JOINTS),
                "body_names": body_names,
                "body_masses_by_name": {
                    name: float(mass) for name, mass in zip(body_names, body_masses)
                },
                "joint_names": list(cache["joint_names"]),
                "controlled_joint_names": list(cache["controlled_joint_names"]),
                "root_joint_name": cache["root_joint_name"],
                "keypoint_body_names": dict(cache["keypoint_body_names"]),
                "keypoint_joint_names": dict(cache["keypoint_joint_names"]),
                "joint_limits": cache["jnt_ranges"].copy(),
            }
        result["dt"] = float(self.DT)
        result["ground_geom_name"] = "ground"
        result["ground_geom_id"] = self.ground_geom_id
        result["geom_id_to_name"] = dict(self.meta["geom_id_to_name"])
        result["body_id_to_name"] = dict(self.meta["body_id_to_name"])
        result["body_id_to_aff"] = dict(self.meta["body_id_to_aff"])
        result["geom_id_to_aff"] = dict(self.meta["geom_id_to_aff"])
        return result
