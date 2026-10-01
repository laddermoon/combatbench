"""WarpHumanoid21Simulator — humanoid21 的 warp 设备仿真器（E1-W2 解耦版）。

**不再继承 MjxHumanoid21Simulator。** 架构改为组合：

    WarpBackend (backends/warp_backend.py) — PhysicsBackend 契约，
        只拥有物理：mjw.Data、视图、wrench 缓冲、advance/capture/restore。
    Humanoid21Binding (envs/humanoid21/batch_binding.py) — 任务语义：
        模型/meta、reset 姿态、core-state 映射、观测/接触提取公式。
    本类 —— facade：把 binding 语义接到 backend 操作上，继续对外
        提供 DeviceBatchSimulator 契约的 ``dev_*`` 方法（views/reset_rows/
        set_integration_rows/force schedule/build_device_state），
        同时保留旧 host accessor（get_core_state 等，走 host_snapshot，
        仅用于验证/回放路径）。

jax/jax.numpy 完全不触碰（import os 一行禁用 XLA 预分配已挪进
WarpBackend 构造）。模型编译走 binding.compile_model()（mujoco only）。

Warmup（B=1536, s=5）实测：
  - 总显存 ~2.2GB 平稳；mass-scale 与 MMA 编译改变量无关（纯 CPU 段）
  - PD kernel 每子步 1 launch，其余纯 mjw.step
  - 吞吐 ~48.3K substep/s（2026-03-11，B=1536）
  - large-batch sweep (s=5):  B=768 ~44K | B=1536 ~48K | B=3072 ~52K
  - B=8192 (~2.3GB): ~504K sub/s

容量注意：nconmax/njmax 均为 per-world 值（put_data 内乘 nworld）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..humanoid21.batch_binding import (
    Humanoid21Binding,
    Humanoid21DeviceTables,
)
from ..humanoid21.meta import Humanoid21Meta
from .backend import BaseBatchSimulator
from .backends.warp_backend import WarpBackend
from .device_state import (
    ContactFlatNamespace,
    DeviceBatchState,
    EpisodeNamespace,
    IoNamespace,
    RngNamespace,
    SimNamespace,
)


class WarpHumanoid21Simulator(BaseBatchSimulator):
    """warp(mujoco_warp) 后端的 humanoid21 批量设备仿真器（组合式）。"""

    DT = Humanoid21Meta.DT
    ACTION_DIM = Humanoid21Meta.ACTION_DIM
    ARENA_XML = Humanoid21Binding.ARENA_XML

    def __init__(
        self,
        batch_size: int,
        nconmax_per_world: int = 48,
        njmax_per_world: int = 512,
        initial_distance: float = 2.0,
        initial_pose_a: str = "standing",
        initial_pose_b: str = "standing",
        device: str = "cuda:0",
        **_ignored,
    ):
        self._batch_size = int(batch_size)
        self._device_str = device
        self._torch_device = torch.device(device)

        self._binding = Humanoid21Binding(
            initial_distance=initial_distance,
            initial_pose_a=initial_pose_a,
            initial_pose_b=initial_pose_b,
        )
        # 兼容属性（dev-side 任务代码与验证代码读取；W4 收口为
        # 显式 TaskBinding 接口）
        self._model = self._binding.model
        self._meta = self._binding.meta
        self._norm_params = self._binding.norm_params
        self._robots = self._binding.robots
        self._ground_geom_id = self._binding.ground_geom_id

        pd_statics = self._binding.build_statics(np, np.float32)
        self._pd_statics = pd_statics
        self._backend = WarpBackend(
            self._model, self._batch_size,
            device=device,
            nconmax_per_world=nconmax_per_world,
            njmax_per_world=njmax_per_world,
            pd_statics=pd_statics)

        self._dev_state: Optional[DeviceBatchState] = None

        # host 侧动作镜像（get_action() host accessor 兼容路径）
        self._action_np = {
            "robot_a": np.zeros((self._batch_size, self.ACTION_DIM),
                                dtype=np.float64),
            "robot_b": np.zeros((self._batch_size, self.ACTION_DIM),
                                dtype=np.float64),
        }

    # ------------------------------------------------------------------
    # 属性（含 mjx 兼容 shim——validation/device_runtime 读取）
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self._batch_size

    def get_batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._torch_device

    @property
    def _wmodel(self):
        return self._backend._wmodel

    @property
    def _wdata(self):
        return self._backend._wdata

    @property
    def _ext_force_jax(self):
        """host 快照 pending wrench（验证 adapter 读取路径）。"""
        if self._backend._ext_dev is None:
            return None
        return self._backend._ext_dev.numpy().astype(np.float64)

    @property
    def _pd(self):
        return self._backend.pd_control

    @property
    def _jax_statics(self):
        return self._pd_statics

    def _wdata_snapshot(self):
        """桥到 backend.host_snapshot()（验证/回放路径）。"""
        return self._backend.host_snapshot()

    # ------------------------------------------------------------------
    # device sim 契约
    # ------------------------------------------------------------------
    def device_obs_builder(self):
        """设备端观测构造器——构造参数是 task_tables（W4 收口后
        不再读 sim 私有字段）。"""
        from .device_obs import WarpObsBuilder
        return WarpObsBuilder(self.task_tables(), self._batch_size)

    def _torch_views(self):
        """dev API 底层的视图入口（等价 backend.views()）。"""
        return self._backend.views()

    def views(self):
        """物理视图公共入口（PhysicsBackend.views 直通）。

        插件/任务代码读设备张量一律走这里（或 state.sim 命名空间），
        不再依赖 ``_torch_views`` 私有名。
        """
        return self._backend.views()

    def task_tables(self) -> "Humanoid21DeviceTables":
        """任务设备表——插件/观测器消费任务元数据的唯一接口（W4）。"""
        return self._binding.device_tables(self._torch_device)

    def reset(self, seeds: Optional[np.ndarray] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        B = self._batch_size
        if seeds is None:
            seeds = np.arange(B, dtype=np.int64)
        self._reset_seeds = np.asarray(seeds, dtype=np.int64).copy()
        qpos_all, a_np, b_np = self._binding.compute_reset_state(
            B, seeds, options)
        self._backend.initialize(
            torch.as_tensor(qpos_all, dtype=torch.float32,
                            device=self._torch_device))
        self._write_actions(a_np, b_np)
        # episode/rng 簿记清理由 runtime.reset 负责（W3 所有权边界）；
        # io.action 镜像经 _write_actions → attach_state 状态同步。

    def physical_step(self, n_steps: int = 1,
                      keep_history: bool = False) -> None:
        if keep_history:
            raise NotImplementedError("keep_history 暂不支持")
        self._backend.advance(n_steps, control=self._backend.pd_control)

    def get_physical_frequency(self) -> float:
        return 1.0 / self.DT

    # ---------------- device mutator（dev_set_* 形态） ----------------
    def _norm_consts_t(self):
        """norm ref/scale 的 torch 常量（dev_set_action 用）。"""
        if getattr(self, "_norm_t", None) is None:
            self._norm_t = {
                rid: (torch.as_tensor(p["reference"], dtype=torch.float32,
                                      device=self._torch_device),
                      torch.as_tensor(p["scale"], dtype=torch.float32,
                                      device=self._torch_device))
                for rid, p in self._norm_params.items()
            }
        return self._norm_t

    def _norm_cat_np(self):
        """拼接的 (42,) norm ref/scale（host 写 target 用）。"""
        if getattr(self, "_norm_cat", None) is None:
            self._norm_cat = (
                np.concatenate([self._norm_params["robot_a"]["reference"],
                                self._norm_params["robot_b"]["reference"]]),
                np.concatenate([self._norm_params["robot_a"]["scale"],
                                self._norm_params["robot_b"]["scale"]]),
            )
        return self._norm_cat

    def _write_actions(self, a_np: np.ndarray, b_np: np.ndarray) -> None:
        """host 动作 → 去归一化 act_target（rad）+ host 镜像。"""
        a = np.clip(np.asarray(a_np, np.float64), -1.0, 1.0)
        b = np.clip(np.asarray(b_np, np.float64), -1.0, 1.0)
        self._action_np["robot_a"] = a.copy()
        self._action_np["robot_b"] = b.copy()
        ref_cat, scale_cat = self._norm_cat_np()
        pair = np.concatenate([a, b], axis=-1).astype(np.float32)
        target = pair * scale_cat + ref_cat
        t = self._backend.ensure_act_target(2 * self.ACTION_DIM)
        t.copy_(torch.from_numpy(target.astype(np.float32))
                .to(self._torch_device))
        st = getattr(self, "_dev_state", None)
        if st is not None:
            st.io.action_a.copy_(torch.as_tensor(
                a, dtype=torch.float32, device=self._torch_device))
            st.io.action_b.copy_(torch.as_tensor(
                b, dtype=torch.float32, device=self._torch_device))

    def dev_set_action(self, action_a: torch.Tensor,
                       action_b: torch.Tensor) -> None:
        """动作 → PD target，全设备端。形状 (B,21)，clip 到 [-1,1]。

        io.action 镜像写入 runtime attach 的状态（若有）；act_target
        是物理输入寄存器，始终写。
        """
        st = self._dev_state
        norm = self._norm_consts_t()
        target = self._backend.ensure_act_target(2 * self.ACTION_DIM)
        for rid, act, cols in (("robot_a", action_a, slice(0, 21)),
                               ("robot_b", action_b, slice(21, 42))):
            a = act.clamp(-1.0, 1.0)
            if st is not None:
                getattr(st.io, f"action_{rid[-1]}").copy_(a)
            ref, scale = norm[rid]
            target[:, cols] = a * scale + ref

    def dev_add_ext_force(self, body_id: int,
                          force: torch.Tensor,
                          torque: Optional[torch.Tensor] = None) -> None:
        """设备端对 body 施加一次 force/torque——写入 pending 缓冲，
        与 schedule 无关，advance 首子步加入（对齐 CPU 语义）。"""
        v = self._backend.views()
        v["xfrc_pending"][:, body_id, :3] += force.float()
        if torque is not None:
            v["xfrc_pending"][:, body_id, 3:] += torque.float()

    def dev_upload_force_schedule(
            self, sched: Optional[torch.Tensor]) -> None:
        """上传子步外力表 (B, n_steps, nbody, 6)，下一个 advance 消费。

        与 dev_add_ext_force 共存：pending=瞬时附加，sched=子步表。
        """
        self._backend.set_wrench_schedule(sched)

    def dev_reset_rows(self, env_ids: torch.Tensor,
                       options: Optional[Dict[str, Any]] = None
                       ) -> None:
        """部分 reset：指定 env 恢复初始姿态（默认 options），其余不动。

        初始姿态在 host 计算（确定性、per-env 独立；options 按全 B
        广播后取行，与历史语义一致），仅写回 env_ids 行——qvel/ctrl/
        warmstart/qfrc/xfrc 一并清零（"全新求解"契约），io.action 与
        act_target 行恢复初始姿态对应的去归一化 PD 目标。backend
        initialize 末做 forward 重建 derived（未重置行 qpos 未变）。
        """
        ids = env_ids.to(torch.long)
        if ids.numel() == 0:
            return
        st = self._dev_state
        ids_np = ids.detach().cpu().numpy().astype(np.int64)
        qpos_all, act_a, act_b = self._binding.compute_reset_state(
            self._batch_size, None, options)
        mask = torch.zeros(self._batch_size, dtype=torch.bool,
                           device=self._torch_device)
        mask[ids] = True
        self._backend.initialize(
            torch.as_tensor(qpos_all[ids_np], dtype=torch.float32,
                            device=self._torch_device),
            mask=mask)
        st = self._dev_state
        dev = self._torch_device
        act_a_t = torch.as_tensor(act_a, dtype=torch.float32, device=dev)
        act_b_t = torch.as_tensor(act_b, dtype=torch.float32, device=dev)
        if st is not None:
            st.io.action_a[ids] = act_a_t[ids]
            st.io.action_b[ids] = act_b_t[ids]
        self._action_np["robot_a"][ids_np] = act_a[ids_np]
        self._action_np["robot_b"][ids_np] = act_b[ids_np]
        norm = self._norm_consts_t()
        ref_a, scale_a = norm["robot_a"]
        ref_b, scale_b = norm["robot_b"]
        target = self._backend.ensure_act_target(2 * self.ACTION_DIM)
        target[ids, :self.ACTION_DIM] = (
            act_a_t[ids] * scale_a + ref_a)
        target[ids, self.ACTION_DIM:] = (
            act_b_t[ids] * scale_b + ref_b)

    def dev_set_integration_rows(
            self, env_ids: torch.Tensor,
            qpos_rows: torch.Tensor,
            qvel_rows: torch.Tensor) -> None:
        """行级积分状态全写回（episode 续段/调试注入用）。"""
        ids = env_ids.to(torch.long)
        mask = torch.zeros(self._batch_size, dtype=torch.bool,
                           device=self._torch_device)
        mask[ids] = True
        self._backend.initialize(
            qpos_rows.to(torch.float32),
            qvel=qvel_rows.to(torch.float32),
            mask=mask)

    # ------------------ 数据平面（W3：runtime 拥有簿记） ------------------
    def build_sim_namespace(self) -> SimNamespace:
        """物理命名空间——backend.views() 的活视图组装（后端职责边界）。

        只含物理：qpos/qvel/ctrl/derived/wrench/接触平铺视图。
        episode/io/rng/plugin 簿记一律归 runtime（compose_state）。
        """
        v = self._backend.views()
        return SimNamespace(
            qpos=v["qpos"], qvel=v["qvel"], ctrl=v["ctrl"],
            xpos=v["xpos"], xquat=v["xquat"], xipos=v["xipos"],
            xanchor=v["xanchor"], cvel=v["cvel"],
            xfrc_applied=v["xfrc_applied"], act_target=v["act_target"],
            contacts_flat=ContactFlatNamespace(
                worldid=v["con_worldid"], geom=v["con_geom"],
                dist=v["con_dist"], pos=v["con_pos"], frame=v["con_frame"],
                dim=v["con_dim"], efc_address=v["con_efc_address"],
                efc_force=v["efc_force"], n_active=v["nacon"],
                cap_per_world=self._backend._nconmax_per_world,
            ),
        )

    def attach_state(self, state: DeviceBatchState) -> None:
        """runtime 注册其持有的 DeviceBatchState——dev_set_action /
        dev_reset_rows 的 io.action 镜像写入此对象（单源）。"""
        self._dev_state = state

    def obs_dim(self) -> int:
        """观测维度（host 路径提取一次，缓存）。"""
        if getattr(self, "_obs_dim_cached", None) is None:
            self._obs_dim_cached = int(
                self.get_observation()["robot_a"].shape[-1])
        return self._obs_dim_cached

    def build_device_state(self) -> DeviceBatchState:
        """兼容入口（standalone/测试路径）：组装完整数据平面并 attach。

        与 runtime 路径共用同一组装函数（device_state.compose_state），
        簿记分配逻辑只有一份；runtime 使用时由 BatchRuntime 组装后
        attach_state()，两者不得并存两个 state 对象。
        """
        if self._dev_state is None:
            from .device_state import compose_state
            st = compose_state(self._batch_size,
                               self.build_sim_namespace(),
                               self._torch_device,
                               self.ACTION_DIM, self.obs_dim())
            self.attach_state(st)
        return self._dev_state

    # ------------------ host accessor（验证/回放路径） ------------------
    def get_core_state(self, history: bool = False) -> Dict[str, np.ndarray]:
        assert not history, "warp backend 不支持 history"
        return self._binding.extract_core_state(
            self._backend.host_snapshot(), is_history=False)

    def get_derived_state(self, names=None,
                          history: bool = False) -> Dict[str, np.ndarray]:
        assert not history, "warp backend 不支持 history"
        return self._binding.extract_derived_state(
            self._backend.host_snapshot(), names, is_history=False)

    def get_observation(self) -> Dict[str, np.ndarray]:
        return self._binding.build_observation(self._backend.host_snapshot())

    def get_sensor_data(self) -> Dict[str, np.ndarray]:
        return self.get_derived_state()

    def get_static_data(self) -> Dict[str, np.ndarray]:
        return self._binding.static_data()

    def get_action(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._action_np.items()}

    def set_action(self, action: Dict[str, np.ndarray]) -> None:
        self._write_actions(action["robot_a"], action["robot_b"])

    def set_core_state(self, state: Dict[str, np.ndarray],
                       env_ids: Optional[np.ndarray] = None) -> None:
        """语义 core-state → qpos/qvel 映射写回（验证/回放 host 路径）。

        ``state`` 是 ``get_core_state()`` 的产物（per-robot 语义字段），
        由 binding.write_core_state 映射到原始 qpos/qvel 后走
        dev_set_integration_rows（清零求解器残留 + forward 刷新）。
        """
        if env_ids is None:
            env_ids = np.arange(self._batch_size)
        env_ids = np.asarray(env_ids, dtype=np.int64)
        v = self._backend.views()
        qpos_new = v["qpos"].detach().cpu().numpy().astype(np.float64)
        qvel_new = v["qvel"].detach().cpu().numpy().astype(np.float64)
        self._binding.write_core_state(state, env_ids, qpos_new, qvel_new)
        ids_t = torch.as_tensor(env_ids, dtype=torch.long,
                                device=self._torch_device)
        self.dev_set_integration_rows(
            ids_t,
            torch.as_tensor(qpos_new[env_ids], dtype=torch.float32,
                            device=self._torch_device),
            torch.as_tensor(qvel_new[env_ids], dtype=torch.float32,
                            device=self._torch_device))

    def set_integration_state(self, *a, **kw):
        return self.set_core_state(*a, **kw)

    def apply_external_force(self, body_name, force, torque=None,
                             robot_id: int = 0) -> None:
        """host 路径外力接口——写 pending 缓冲（下一个 step 消费）。"""
        bid = self._binding.resolve_body_id(body_name, robot_id)
        v = self._backend.views()
        f = torch.as_tensor(np.asarray(force, dtype=np.float32),
                            device=self._torch_device)
        v["xfrc_pending"][:, bid, :3] += f
        if torque is not None:
            t = torch.as_tensor(np.asarray(torque, dtype=np.float32),
                                device=self._torch_device)
            v["xfrc_pending"][:, bid, 3:] += t

    def close(self) -> None:
        self._backend.close()


# =====================================================================
# 工厂函数
# =====================================================================

def make_warp_standup_simulator(**kwargs) -> WarpHumanoid21Simulator:
    """standup 实验的默认 warp 仿真器构造（collector 便捷入口）。"""
    return WarpHumanoid21Simulator(**kwargs)
