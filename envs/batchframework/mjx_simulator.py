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
from envs.humanoid21.batch_binding import Humanoid21Binding
from envs.humanoid21.meta import Humanoid21Meta


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
        _init_jax: bool = True,
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
        # _init_jax=False（warp 子类用）：跳过一切会初始化 XLA 后端的调用
        # （jax.devices()/mjx.put_model/jp.array）——XLA 默认预分配 ~75%
        # 显存，而 warp 路径根本不消费 jax 数据。共享的 numpy 表与 host
        # 提取路径不依赖 jax 后端，不受影响。
        self._device = device or (jax.devices()[0] if _init_jax else None)

        # --- 任务绑定：模型/meta/归一化/PD 表/初始姿态（跨后端单一来源） ---
        self._binding = Humanoid21Binding(
            initial_distance=initial_distance,
            initial_pose_a=initial_pose_a,
            initial_pose_b=initial_pose_b)
        # 兼容别名：本类 jax 侧方法与外部消费方仍以这些属性访问。
        self._model = self._binding.model
        self._meta = self._binding.meta
        self._robots = self._binding.robots
        self._ground_geom_id = self._binding.ground_geom_id
        self._norm_params = self._binding.norm_params
        self._pd_tables = self._binding.pd_tables
        self._body_name_to_id = self._binding.body_name_to_id
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
        self._mjx_model = mjx.put_model(self._model, impl=self._impl) \
            if _init_jax else None

        # --- Precompute static arrays for PD control ---
        # Per-robot: qpos_indices, qvel_indices, actuator_ids, norm ref/scale, gear, ctrl_lo/hi, KP, KD
        # _init_jax=False 时给 numpy 版本——warp 路径只把表当数据消费
        # （np.asarray 上载到 warp），不需要 XLA 后端。
        self._jax_statics = self._binding.build_statics(jp, self._dtype) \
            if _init_jax else self._binding.build_statics(np, np.float32)

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
        return self._binding.build_statics(jp, self._dtype)

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
        B = self._batch_size
        qpos_all, action_a, action_b = self._binding.compute_reset_state(
            B, seeds, options)
        self._reset_seeds = None if seeds is None else np.asarray(seeds).copy()

        # 模板 data 只提供形状/dtype；qpos/qvel/xfrc 在 broadcast 后被覆盖，
        # 由随后的 batched forward 重建全部 derived 字段（对应 CPU 的 mj_forward）。
        data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, data)
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

    def _compute_reset_state(self, seeds, options):
        """委托 binding——保留入口兼容既有调用方/子类。"""
        out = self._binding.compute_reset_state(
            self._batch_size, seeds, options)
        self._reset_seeds = None if seeds is None else np.asarray(seeds).copy()
        return out


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
        """委托 ``Humanoid21Binding``（语义单一来源，见 batch_binding.py）。"""
        return self._binding.extract_core_state(data, is_history)

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
        """委托 ``Humanoid21Binding``（语义单一来源，见 batch_binding.py）。"""
        return self._binding.extract_derived_state(data, fields, is_history)

    def _extract_contacts_batch(self, data: mjx.Data) -> Dict[str, Any]:
        """委托 ``Humanoid21Binding``（语义单一来源，见 batch_binding.py）。"""
        return self._binding.extract_contacts_batch(data)

    def _get_robot_view_batch(
        self, data: mjx.Data, robot_id: str, opponent_id: str
    ) -> Dict[str, Any]:
        """委托 ``Humanoid21Binding``（语义单一来源，见 batch_binding.py）。"""
        return self._binding.robot_view_batch(data, robot_id, opponent_id)

    def _collect_body_joint_arrays(self, data: mjx.Data, robot_id: str) -> Dict[str, Any]:
        """委托 ``Humanoid21Binding``（语义单一来源，见 batch_binding.py）。"""
        return self._binding.collect_body_joint_arrays(data, robot_id)

    def _get_feet_forces_batch(self, data: mjx.Data, robot_id: str) -> np.ndarray:
        """委托 ``Humanoid21Binding``（语义单一来源，见 batch_binding.py）。"""
        return self._binding.feet_forces_batch(data, robot_id)

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
        """委托 ``Humanoid21Binding``（语义单一来源，见 batch_binding.py）。"""
        return self._binding.static_data()

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

        self._write_core_state(state, env_ids, qpos_new, qvel_new)

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

    def _write_core_state(self, state, env_ids, qpos_new, qvel_new):
        """委托 ``Humanoid21Binding``（语义单一来源，见 batch_binding.py）。"""
        self._binding.write_core_state(state, env_ids, qpos_new, qvel_new)

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
