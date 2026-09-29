"""MJX (MuJoCo XLA) 批量仿真器实现。

使用 jax.lax.scan 将 n_steps 个物理步编译为单个 XLA 计算，
中间不回 Python，实现 GPU 上全向量化仿真。

所有对外接口输入输出均为 numpy array，JAX 完全封装在内部。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import mujoco
import mujoco.mjx as mjx

import jax
import jax.numpy as jp
from jax import tree_util

from envs.batchframework.backend import BaseBatchSimulator
from envs.humanoid21.meta import Humanoid21Meta


def _quat_to_rot_np(quat: np.ndarray) -> np.ndarray:
    """Batched [w,x,y,z] → (...,3,3) rotation matrix (numpy, leading dims preserved)."""
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def _sqrt_signed(v: np.ndarray, div2_inside: bool = False) -> np.ndarray:
    """CPU 观测的非线性压缩：joint_vel 用 sign*sqrt(|v|)/2，angular/kp_vel 用
    sign*sqrt(|v/2|)。div2_inside 区分两种公式（角速度 /2 在 sqrt 内）。"""
    if div2_inside:
        return np.sign(v) * np.sqrt(np.abs(v / 2.0))
    return np.sign(v) * np.sqrt(np.abs(v)) / 2.0


class MjxHumanoid21Simulator(BaseBatchSimulator):
    """基于 MJX 的 Humanoid21 批量仿真器。

    使用 jax.vmap + jax.lax.scan 实现 B 个环境的并行仿真，
    所有物理步在 GPU 上完成，仅在 get_*/set_* 时进行 host-device 传输。
    """

    DT = Humanoid21Meta.DT
    ACTION_DIM = Humanoid21Meta.ACTION_DIM
    # 必须与 envs/humanoid21/simulator.py 的 ARENA_XML 指向同一个文件；
    # condim=3 摩擦接触 / impratio=10 / 圆形墙都由该模型定义，换文件即换任务。
    ARENA_XML = str(Path(__file__).resolve().parent.parent / "humanoid21" / "battle_circular_v2.xml")
    KP = Humanoid21Meta.KP
    KD = Humanoid21Meta.KD
    CONTROLLED_JOINTS = Humanoid21Meta.CONTROLLED_JOINTS
    INITIAL_POSES = Humanoid21Meta.INITIAL_POSES

    def __init__(
        self,
        batch_size: int,
        initial_distance: float = 2.0,
        initial_pose_a: str = "standing",
        initial_pose_b: str = "standing",
        device: Optional[jax.Device] = None,
        precision: str = "fp64",
        impl: str = "jax",
    ):
        # float64 is required for close MJX/MuJoCo consistency (see validation
        # test): with float32, contact solver diverges within ~10 steps.
        # precision="fp32" 是独立的性能评估路径，不承诺逐步等价。
        # jax_enable_x64 是 JAX 全局开关，须在首次 jit/put_data 前设置；
        # 同一进程内不要混用两种精度实例。
        if precision not in ("fp64", "fp32"):
            raise ValueError(f"precision must be 'fp64' or 'fp32', got {precision}")
        jax.config.update("jax_enable_x64", precision == "fp64")
        self._precision = precision
        self._dtype = jp.float64 if precision == "fp64" else jp.float32
        # impl='jax'（默认，FP64 可选）或 'warp'（mujoco-warp，FP32 单精度）。
        if impl not in ("jax", "warp"):
            raise ValueError(f"impl must be 'jax' or 'warp', got {impl}")
        self._impl = impl
        # With float64, per-step match is ~1e-14 (machine precision).
        self._batch_size = batch_size
        self._initial_distance = initial_distance
        self._initial_pose_a = initial_pose_a
        self._initial_pose_b = initial_pose_b
        self._device = device or jax.devices()[0]

        # --- Load MuJoCo model + MJX model ---
        self._model = mujoco.MjSpec.from_file(self.ARENA_XML).compile()
        self._model.opt.timestep = self.DT
        if self._impl == "warp":
            # 版本缝隙（仅探测用）：mujoco-mjx 3.8 内置的
            # mujoco.mjx.warp.types.GraphMode 是 int 存根，put_model 期待
            # warp-lang 的真实 enum；同时 warp-lang 未在 warp.types 顶层
            # 导出 warp_type_to_np_dtype（mujoco-warp 3.8.0.3 内部引用）。
            import warp as _wp
            import warp._src.types as _wp_types
            import mujoco.mjx.warp.types as _mjxw_types
            from warp.jax_experimental.ffi import GraphMode as _WarpGraphMode
            _mjxw_types.GraphMode = _WarpGraphMode
            if not hasattr(_wp.types, "warp_type_to_np_dtype"):
                _wp.types.warp_type_to_np_dtype = _wp_types.warp_type_to_np_dtype
        self._mjx_model = mjx.put_model(self._model, impl=self._impl)

        # --- Build runtime tables from meta ---
        self._meta = Humanoid21Meta.build_runtime_tables(self._model)
        self._robots = self._meta["robots"]
        self._ground_geom_id = self._meta["ground_geom_id"]

        # --- Normalization params ---
        self._norm_params = {}
        for robot_id in ["robot_a", "robot_b"]:
            jnt_ranges = self._robots[robot_id]["jnt_ranges"]  # (21, 2)
            lower = jnt_ranges[:, 0]
            upper = jnt_ranges[:, 1]
            self._norm_params[robot_id] = {
                "reference": ((lower + upper) / 2.0).astype(np.float32),
                "scale": ((upper - lower) / 2.0).astype(np.float32),
            }

        # --- PD tables ---
        self._pd_tables = {}
        for robot_id in ["robot_a", "robot_b"]:
            act_ids = self._robots[robot_id]["actuator_ids"]
            gear = np.array(self._model.actuator_gear[act_ids, 0], dtype=np.float64)
            gear[gear == 0] = 1.0
            ctrl_lo = np.array(self._model.actuator_ctrlrange[act_ids, 0], dtype=np.float64)
            ctrl_hi = np.array(self._model.actuator_ctrlrange[act_ids, 1], dtype=np.float64)
            self._pd_tables[robot_id] = {
                "actuator_ids": act_ids,
                "gear": gear,
                "ctrl_lo": ctrl_lo,
                "ctrl_hi": ctrl_hi,
            }

        # --- Precompute static JAX arrays for PD control ---
        # Per-robot: qpos_indices, qvel_indices, actuator_ids, norm ref/scale, gear, ctrl_lo/hi, KP, KD
        self._jax_statics = self._build_jax_statics()

        # --- Body name → body id mapping for external force ---
        self._body_name_to_id = {}
        for robot_id, suffix in Humanoid21Meta.ROBOT_SUFFIXES.items():
            for body_name in Humanoid21Meta.ROBOT_BODY_NAMES:
                full = f"{body_name}{suffix}"
                bid = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_BODY, full
                )
                if bid >= 0:
                    self._body_name_to_id[full] = bid

        # --- Init JAX state ---
        self._mjx_data: Optional[mjx.Data] = None
        self._action_jax: Optional[Dict[str, jp.ndarray]] = None
        self._ext_force_jax: Optional[jp.ndarray] = None  # (B, nbody, 6) persistent
        self._history_buffer: Optional[mjx.Data] = None  # pytree of (B, n_steps, ...)
        self._history_n_steps: int = 0

        # --- JIT-compiled functions (built lazily after first reset) ---
        self._jit_step = None
        self._jit_step_scan = None

    def _build_jax_statics(self) -> Dict[str, Any]:
        """Precompute JAX arrays needed inside JIT-compiled step functions."""
        statics = {}
        for robot_id in ["robot_a", "robot_b"]:
            r = self._robots[robot_id]
            norm = self._norm_params[robot_id]
            pd = self._pd_tables[robot_id]
            statics[robot_id] = {
                "qpos_indices": jp.array(r["qpos_indices"], dtype=jp.int32),
                "qvel_indices": jp.array(r["qvel_indices"], dtype=jp.int32),
                "actuator_ids": jp.array(r["actuator_ids"], dtype=jp.int32),
                "norm_ref": jp.array(norm["reference"], dtype=jp.float32),
                "norm_scale": jp.array(norm["scale"], dtype=jp.float32),
                "gear": jp.array(pd["gear"], dtype=self._dtype),
                "ctrl_lo": jp.array(pd["ctrl_lo"], dtype=self._dtype),
                "ctrl_hi": jp.array(pd["ctrl_hi"], dtype=self._dtype),
                "kp": jp.array(self.KP, dtype=self._dtype),
                "kd": jp.array(self.KD, dtype=self._dtype),
                "root_qpos_adr": r["root_qpos_adr"],
                "root_qvel_adr": r["root_qvel_adr"],
            }
        return statics

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self._batch_size

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_physical_frequency(self) -> float:
        return 1.0 / self.DT

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    @staticmethod
    def _broadcast_option(options, key, default, batch_size):
        """options 中的标量或 (B,) 序列 → 长度 B 的 per-env 列表。"""
        value = options.get(key, default) if options else default
        if isinstance(value, str) or np.isscalar(value):
            return [value] * batch_size
        seq = list(value)
        if len(seq) != batch_size:
            raise ValueError(f"options[{key}] must be scalar or length {batch_size}")
        return seq

    def reset(
        self,
        seeds: Optional[np.ndarray] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset all envs to initial pose.

        与 CPU reset 一致：初始姿态是确定性的，``seeds`` 只被记录为来源信息，
        不改变状态（CPU 版本同样不使用 seed；逐 episode 的随机性通过
        ``options['initial_distance']`` 由调用方注入）。``initial_distance``
        与 ``initial_pose_a/b`` 支持标量或长度 B 的序列（per-env）。
        """
        from scipy.spatial.transform import Rotation as Rot

        B = self._batch_size
        data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, data)

        dists = [float(v) for v in self._broadcast_option(
            options, "initial_distance", self._initial_distance, B)]
        poses_a = self._broadcast_option(options, "initial_pose_a", self._initial_pose_a, B)
        poses_b = self._broadcast_option(options, "initial_pose_b", self._initial_pose_b, B)
        self._reset_seeds = None if seeds is None else np.asarray(seeds).copy()

        qpos_all = np.broadcast_to(np.asarray(data.qpos), (B, data.qpos.size)).copy()
        action_a = np.zeros((B, self.ACTION_DIM), dtype=np.float32)
        action_b = np.zeros((B, self.ACTION_DIM), dtype=np.float32)

        for i in range(B):
            for robot_id, pose_name, x_offset in [
                ("robot_a", poses_a[i], -dists[i] / 2.0),
                ("robot_b", poses_b[i], dists[i] / 2.0),
            ]:
                pose_config = self.INITIAL_POSES[pose_name]
                cache = self._robots[robot_id]
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
                norm = self._norm_params[robot_id]
                action = ((qpos_all[i, qpos_indices] - norm["reference"])
                          / norm["scale"]).astype(np.float32)
                if robot_id == "robot_a":
                    action_a[i] = action
                else:
                    action_b[i] = action

        # 模板 data 只提供形状/dtype；qpos/qvel/xfrc 在 broadcast 后被覆盖，
        # 由随后的 batched forward 重建全部 derived 字段（对应 CPU 的 mj_forward）。
        self._build_jit_functions()
        single = mjx.put_data(self._model, data, impl=self._impl)
        batched = jax.tree.map(
            lambda x: jp.broadcast_to(x, (B,) + x.shape), single)
        batched = batched.replace(
            qpos=jp.array(qpos_all),
            qvel=jp.zeros((B, self._model.nv), dtype=self._dtype),
            xfrc_applied=jp.zeros((B, self._model.nbody, 6), dtype=self._dtype),
            qfrc_applied=jp.zeros((B, self._model.nv), dtype=self._dtype),
        )
        self._mjx_data = self._jit_forward(batched)

        self._action_jax = {"robot_a": jp.array(action_a), "robot_b": jp.array(action_b)}
        # 挂起外力缓冲：apply_external_force 累加到这里，下一个 physical_step
        # 的第一个物理子步生效后清零（对齐 CPU 的 xfrc_applied 语义）。
        self._ext_force_jax = jp.zeros((B, self._model.nbody, 6), dtype=self._dtype)

        self._history_buffer = None
        self._history_n_steps = 0

    def _build_jit_functions(self):
        """Build JIT-compiled forward, single-step and scan-step functions."""
        mjx_model = self._mjx_model
        statics = self._jax_statics

        self._jit_forward = jax.jit(jax.vmap(lambda d: mjx.forward(mjx_model, d)))

        def _apply_pd_and_step(data, action_a, action_b, ext_force):
            """Single physics step: apply PD control, set ext force, then mjx.step."""
            # Apply persistent external forces
            data = data.replace(xfrc_applied=ext_force)

            # PD control for each robot
            ctrl = jp.zeros(data.ctrl.shape, dtype=self._dtype)
            for robot_id, action in [("robot_a", action_a), ("robot_b", action_b)]:
                s = statics[robot_id]
                target_rad = action * s["norm_scale"] + s["norm_ref"]
                current_pos = data.qpos[s["qpos_indices"]]
                current_vel = data.qvel[s["qvel_indices"]]
                torque = s["kp"] * (target_rad - current_pos) - s["kd"] * current_vel
                ctrl_val = torque / s["gear"]
                ctrl_val = jp.clip(ctrl_val, s["ctrl_lo"], s["ctrl_hi"])
                ctrl = ctrl.at[s["actuator_ids"]].set(ctrl_val)

            data = data.replace(ctrl=ctrl)
            data = mjx.step(mjx_model, data)
            return data

        def _cast_back(d, ref):
            # mjx.step may promote some int32 fields to int64; cast back to
            # match carry/input dtypes required by jax.lax.scan and jit reuse.
            return jax.tree.map(
                lambda out, inp: out.astype(inp.dtype)
                if hasattr(out, "dtype") and hasattr(inp, "dtype")
                and out.dtype != inp.dtype
                else out,
                d, ref,
            )

        def _step_once(data, action_a, action_b, ext_force):
            return _cast_back(_apply_pd_and_step(data, action_a, action_b, ext_force), data)

        self._jit_step_vmap = jax.jit(
            jax.vmap(_step_once, in_axes=(0, 0, 0, 0))
        )

        # Cache of JIT-compiled scan functions, keyed by n_steps.
        # jax.lax.scan requires length as a concrete Python int, so we
        # build a separate JIT function per n_steps value.
        # 外力按子步序列传入：CPU 的 xfrc_applied 在每个物理步后清零，
        # 因此一次 apply_external_force 只作用于下一个子步，其余子步为 0。
        self._jit_scan_cache: Dict[int, Any] = {}

        self._jit_scan_cache: Dict[Any, Any] = {}

        def _get_scan_fn(n_steps: int, keep_history: bool):
            """无历史分支不堆叠逐步 mjx.Data——否则 B=512×25 步会物化
            ~50GiB+ 中间态直接 OOM（实测）。keep_history=True 才保留栈。"""
            key = (n_steps, keep_history)
            if key not in self._jit_scan_cache:
                def _scan_single(data, action_a, action_b, ext_force_seq):
                    def body(carry, ext):
                        d = _step_once(carry, action_a, action_b, ext)
                        return d, (d if keep_history else None)
                    final, hist = jax.lax.scan(
                        body, data, ext_force_seq, length=n_steps)
                    return final, hist

                self._jit_scan_cache[key] = jax.jit(
                    jax.vmap(_scan_single, in_axes=(0, 0, 0, 0))
                )
            return self._jit_scan_cache[key]

        self._get_scan_fn = _get_scan_fn

    # ------------------------------------------------------------------
    # physical_step
    # ------------------------------------------------------------------
    def physical_step(self, n_steps: int = 1, keep_history: bool = False) -> None:
        if self._mjx_data is None:
            raise RuntimeError("Call reset() before physical_step()")

        # Clear history buffer at the start of each physical_step
        self._history_buffer = None
        self._history_n_steps = 0

        action_a = self._action_jax["robot_a"]
        action_b = self._action_jax["robot_b"]

        # pending 外力只注入第一个子步（CPU 每步后清零 xfrc_applied），之后归 0。
        pending = self._ext_force_jax
        if n_steps == 1 and not keep_history:
            self._mjx_data = self._jit_step_vmap(
                self._mjx_data, action_a, action_b, pending
            )
        else:
            ext_seq = jp.concatenate(
                [pending[:, None], jp.zeros(
                    (self._batch_size, n_steps - 1) + pending.shape[1:],
                    dtype=pending.dtype)],
                axis=1,
            )
            scan_fn = self._get_scan_fn(n_steps, keep_history)
            final, history = scan_fn(self._mjx_data, action_a, action_b, ext_seq)
            self._mjx_data = final
            if keep_history:
                self._history_buffer = history
                self._history_n_steps = n_steps
        # CPU physical_step 末尾清零 data.xfrc_applied/qfrc_applied；MJX 数据里
        # 残留的外力会污染后续 forward 求解（实测接触力偏 ~4%），必须同步清零。
        self._ext_force_jax = jp.zeros_like(pending)
        self._mjx_data = self._mjx_data.replace(
            xfrc_applied=jp.zeros_like(self._mjx_data.xfrc_applied),
            qfrc_applied=jp.zeros_like(self._mjx_data.qfrc_applied),
        )

    def _single_step(self) -> None:
        """Fallback for BaseBatchSimulator default implementation."""
        self.physical_step(n_steps=1, keep_history=False)

    # ------------------------------------------------------------------
    # get_core_state
    # ------------------------------------------------------------------
    def get_core_state(self, history: bool = False) -> Dict[str, Any]:
        if history:
            if self._history_buffer is None:
                return {}
            return self._extract_core_state(self._history_buffer, is_history=True)
        return self._extract_core_state(self._mjx_data, is_history=False)

    def _extract_core_state(self, data: mjx.Data, is_history: bool) -> Dict[str, Any]:
        """Extract core state from MJX data (JAX → numpy)."""
        result: Dict[str, Any] = {}

        for robot_id in ["robot_a", "robot_b"]:
            r = self._robots[robot_id]
            norm = self._norm_params[robot_id]

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

            R_inv = np.swapaxes(_quat_to_rot_np(root_rot_np), -1, -2)
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

    # ------------------------------------------------------------------
    # get_derived_state
    # ------------------------------------------------------------------
    def get_derived_state(
        self,
        fields: Optional[Sequence[str]] = None,
        history: bool = False,
    ) -> Dict[str, Any]:
        if history:
            if self._history_buffer is None:
                return {}
            return self._extract_derived_state(self._history_buffer, fields, is_history=True)
        return self._extract_derived_state(self._mjx_data, fields, is_history=False)

    def _extract_derived_state(
        self, data: mjx.Data, fields: Optional[Sequence[str]], is_history: bool
    ) -> Dict[str, Any]:
        if fields is None:
            fields = ["torso_distance", "contacts", "robot_a", "robot_b"]
        else:
            fields = list(fields)
            unknown = set(fields) - {"torso_distance", "contacts", "robot_a", "robot_b"}
            if unknown:
                raise KeyError(f"get_derived_state: unknown fields {unknown}")

        result: Dict[str, Any] = {}

        if "torso_distance" in fields:
            torso_a_id = self._robots["robot_a"]["root_body_id"]
            torso_b_id = self._robots["robot_b"]["root_body_id"]
            pos_a = np.asarray(data.xpos[..., torso_a_id, :])
            pos_b = np.asarray(data.xpos[..., torso_b_id, :])
            dist = np.linalg.norm(pos_b - pos_a, axis=-1, keepdims=True)
            result["torso_distance"] = dist.astype(np.float32)

        if "contacts" in fields:
            result["contacts"] = self._extract_contacts_batch(data)

        for rid in ("robot_a", "robot_b"):
            if rid in fields:
                opp_id = "robot_b" if rid == "robot_a" else "robot_a"
                view = self._get_robot_view_batch(data, rid, opp_id)
                view.update(self._collect_body_joint_arrays(data, rid))
                result[rid] = view

        return result

    def _extract_contacts_batch(self, data: mjx.Data) -> Dict[str, Any]:
        """Extract contacts from batched MJX data.

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
        geom_bodyid = np.asarray(self._model.geom_bodyid)
        body1 = geom_bodyid[geom[..., 0].astype(np.int64)]
        body2 = geom_bodyid[geom[..., 1].astype(np.int64)]

        # Affiliation
        geom_id_to_aff = self._meta["geom_id_to_aff"]
        aff_table = np.zeros(self._model.ngeom, dtype=np.int8)
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
                addr = int(efc_address[c])
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

    def _get_robot_view_batch(
        self, data: mjx.Data, robot_id: str, opponent_id: str
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
        cache = self._robots[robot_id]
        opp_cache = self._robots[opponent_id]
        norm = self._norm_params[robot_id]

        torso_id = cache["root_body_id"]
        opp_torso_id = opp_cache["root_body_id"]

        self_pos = np.asarray(data.xpos[..., torso_id, :])
        self_quat = np.asarray(data.xquat[..., torso_id, :])  # [w,x,y,z]
        opp_pos = np.asarray(data.xpos[..., opp_torso_id, :])
        opp_quat = np.asarray(data.xquat[..., opp_torso_id, :])

        R_self = _quat_to_rot_np(self_quat)      # body→world, (...,3,3)
        R_self_inv = np.swapaxes(R_self, -1, -2)  # world→body

        height = self_pos[..., 2:3]                              # (...,1)
        projected_gravity = -R_self[..., 2, :]                   # (...,3) 与 CPU -R[2,:] 一致

        root_qva = cache["root_qvel_adr"]
        linear_vel = np.einsum(                                  # 机体系线速度
            "...ij,...j->...i", R_self_inv,
            np.asarray(data.qvel[..., root_qva : root_qva + 3]))
        angular_vel = np.asarray(data.qvel[..., root_qva + 3 : root_qva + 6])  # 本就在机体系

        feet_forces = self._get_feet_forces_batch(data, robot_id)
        arena_center_local = np.einsum("...ij,...j->...i", R_self_inv, -self_pos)

        # 对手基础位姿：relative_vel 取对手 torso 线速度转到自机体系（CPU 语义）。
        relative_pos_local = np.einsum("...ij,...j->...i", R_self_inv, opp_pos - self_pos)
        opp_vel_global = np.asarray(data.cvel[..., opp_torso_id, 3:6])
        relative_vel_local = np.einsum("...ij,...j->...i", R_self_inv, opp_vel_global)
        opp_forward = _quat_to_rot_np(opp_quat)[..., :, 0]       # 对手局部 x 轴（世界系）
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
        joint_vel_obs = _sqrt_signed(joint_vel_norm)
        ang_vel_obs = _sqrt_signed(angular_vel, div2_inside=True)
        kp_vel_rest_obs = np.concatenate(
            [_sqrt_signed(kp_vel_local[n]) for n in kp_names[1:]], axis=-1)
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

    def _collect_body_joint_arrays(self, data: mjx.Data, robot_id: str) -> Dict[str, Any]:
        """CPU ``_collect_body_joint_arrays`` 的批量版：per-body 世界系数组 + joint anchor。"""
        cache = self._robots[robot_id]
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

    def _get_feet_forces_batch(self, data: mjx.Data, robot_id: str) -> np.ndarray:
        """Get feet contact forces, **normalized by body weight m*g** (dimensionless).

        Uses the same force_mag computed in _extract_contacts_batch (from efc_force),
        matching the original simulator's _get_feet_forces which uses mj_contactForce output.

        The division by ``body_weight`` must stay in lockstep with
        ``Humanoid21Simulator._get_feet_forces`` — the two backends are
        required to produce bit-comparable observations, so a unit change
        in one without the other would silently desync them. See that
        method's docstring for why the normalization exists.
        """
        cache = self._robots[robot_id]
        kp = cache["keypoint_body_ids"]
        foot_right_id = kp["foot_right"]
        foot_left_id = kp["foot_left"]
        ground_gid = self._ground_geom_id
        body_weight = cache["body_weight"]

        # Extract contacts to get force_mag (same as _extract_contacts_batch)
        contacts = self._extract_contacts_batch(data)

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

    # ------------------------------------------------------------------
    # get_observation
    # ------------------------------------------------------------------
    def get_observation(self) -> Dict[str, Any]:
        result = {}
        for rid in ("robot_a", "robot_b"):
            opp_id = "robot_b" if rid == "robot_a" else "robot_a"
            view = self._get_robot_view_batch(self._mjx_data, rid, opp_id)
            result[rid] = view["observation"]
        return result

    # ------------------------------------------------------------------
    # get_sensor_data / get_action / get_static_data
    # ------------------------------------------------------------------
    def get_sensor_data(self) -> Dict[str, Any]:
        return {}

    def get_action(self) -> Dict[str, Any]:
        if self._action_jax is None:
            return {}
        return {
            "robot_a": np.asarray(self._action_jax["robot_a"]),
            "robot_b": np.asarray(self._action_jax["robot_b"]),
        }

    def get_static_data(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for robot_id in ["robot_a", "robot_b"]:
            cache = self._robots[robot_id]
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
        result["ground_geom_id"] = self._ground_geom_id
        result["geom_id_to_name"] = dict(self._meta["geom_id_to_name"])
        result["body_id_to_name"] = dict(self._meta["body_id_to_name"])
        result["body_id_to_aff"] = dict(self._meta["body_id_to_aff"])
        result["geom_id_to_aff"] = dict(self._meta["geom_id_to_aff"])
        return result

    # ------------------------------------------------------------------
    # set_action
    # ------------------------------------------------------------------
    def set_action(self, action: Dict[str, Any]) -> None:
        for robot_id in ["robot_a", "robot_b"]:
            if robot_id in action and action[robot_id] is not None:
                act = np.asarray(action[robot_id], dtype=np.float32)
                if act.shape != (self._batch_size, self.ACTION_DIM):
                    raise ValueError(
                        f"Action for {robot_id} must have shape "
                        f"({self._batch_size}, {self.ACTION_DIM}), got {act.shape}"
                    )
                if self._action_jax is None:
                    self._action_jax = {}
                self._action_jax[robot_id] = jp.array(np.clip(act, -1.0, 1.0))

    # ------------------------------------------------------------------
    # set_core_state
    # ------------------------------------------------------------------
    def set_core_state(
        self,
        state: Dict[str, Any],
        env_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if self._mjx_data is None:
            raise RuntimeError("Call reset() before set_core_state()")

        if env_ids is None:
            env_ids = list(range(self._batch_size))

        env_ids = list(env_ids)
        n = len(env_ids)

        # Build new qpos/qvel on host for the target envs
        qpos_new = np.asarray(self._mjx_data.qpos).copy()  # (B, nq)
        qvel_new = np.asarray(self._mjx_data.qvel).copy()  # (B, nv)

        for robot_id in ["robot_a", "robot_b"]:
            if robot_id not in state:
                continue
            robot_state = state[robot_id]
            cache = self._robots[robot_id]
            norm = self._norm_params[robot_id]
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
                    R_mat = _quat_to_rot_np(quat)
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

        # Transfer to device 并重建 derived 字段——对齐 CPU 末尾的 mj_forward：
        # 写入后立刻读取观测/接触必须返回与新状态一致的数据。qacc_warmstart、
        # ctrl、xfrc/qfrc 一并清零：状态写入的跨后端契约是"全新求解"，
        # 不继承先前步进的求解偏置/外力残留（见 set_integration_state）。
        B, nv = self._batch_size, self._model.nv
        self._mjx_data = self._jit_forward(self._mjx_data.replace(
            qpos=jp.array(qpos_new), qvel=jp.array(qvel_new),
            qacc_warmstart=jp.zeros((B, nv), dtype=self._dtype),
            ctrl=jp.zeros((B, self._model.nu), dtype=self._dtype),
            xfrc_applied=jp.zeros((B, self._model.nbody, 6), dtype=self._dtype),
            qfrc_applied=jp.zeros((B, nv), dtype=self._dtype),
        ))

    # ------------------------------------------------------------------
    # set_integration_state
    # ------------------------------------------------------------------
    def set_integration_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        env_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """按原始积分状态写入 qpos/qvel（支持 per-env），随后 forward 刷新。

        CPU 侧等价物是 ``data.qpos/qvel[:] = ...; mj_forward``。与
        ``set_core_state`` 的区别：这里接受原始 MuJoCo 坐标（世界系线速度、
        原始关节弧度），不做归一化/坐标变换，用于跨后端状态搬运与回放。
        MuJoCo 的 warmstart/solver 内部态不在此契约内——``qacc_warmstart``、
        ``ctrl``、``xfrc_applied``、``qfrc_applied`` 一并清零，否则恢复的求解
        会继承上一次步进的偏置（实测残留 ctrl/xfrc 可把接触力拉偏 ~4%，
        且随调用顺序漂移）。
        """
        if self._mjx_data is None:
            raise RuntimeError("Call reset() before set_integration_state()")
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        if env_ids is None:
            if qpos.shape != (self._batch_size, self._model.nq) or \
               qvel.shape != (self._batch_size, self._model.nv):
                raise ValueError(
                    f"expected qpos (B,{self._model.nq}) / qvel (B,{self._model.nv}), "
                    f"got {qpos.shape}/{qvel.shape}")
            qpos_new, qvel_new = qpos, qvel
        else:
            env_ids = list(env_ids)
            qpos_new = np.asarray(self._mjx_data.qpos).copy()
            qvel_new = np.asarray(self._mjx_data.qvel).copy()
            if qpos.shape != (len(env_ids), self._model.nq) or \
               qvel.shape != (len(env_ids), self._model.nv):
                raise ValueError("per-env integration state shape mismatch")
            qpos_new[env_ids] = qpos
            qvel_new[env_ids] = qvel
        B, nv = self._batch_size, self._model.nv
        self._mjx_data = self._jit_forward(self._mjx_data.replace(
            qpos=jp.array(qpos_new), qvel=jp.array(qvel_new),
            qacc_warmstart=jp.zeros((B, nv), dtype=self._dtype),
            ctrl=jp.zeros((B, self._model.nu), dtype=self._dtype),
            xfrc_applied=jp.zeros((B, self._model.nbody, 6), dtype=self._dtype),
            qfrc_applied=jp.zeros((B, nv), dtype=self._dtype),
        ))

    # ------------------------------------------------------------------
    # apply_external_force
    # ------------------------------------------------------------------
    def apply_external_force(
        self,
        body_name: str,
        force: np.ndarray,
        torque: Optional[np.ndarray] = None,
        robot_id: str = "robot_a",
    ) -> None:
        suffix = self._robots[robot_id]["suffix"]
        full_body_name = f"{body_name}{suffix}"
        body_id = self._body_name_to_id.get(full_body_name)
        if body_id is None:
            raise ValueError(f"Body not found: {full_body_name}")

        # 与 CPU 一致：+= 累加到挂起缓冲；下一次 physical_step 的首个子步生效，
        # 随后自动清零。清除外力 = 不再调用（缓冲已空），而不是写一次 0 持续生效。
        force = np.asarray(force, dtype=np.float64)  # (B, 3)
        ext = np.asarray(self._ext_force_jax).copy()  # (B, nbody, 6)
        ext[:, body_id, :3] += force
        if torque is not None:
            torque = np.asarray(torque, dtype=np.float64)  # (B, 3)
            ext[:, body_id, 3:6] += torque
        self._ext_force_jax = jp.array(ext)

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._mjx_data = None
        self._history_buffer = None
