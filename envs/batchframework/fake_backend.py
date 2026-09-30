"""FakeBatchBackend — 轻量契约后端（生命周期测试 + 后端模板）。

用普通 CPU/GPU torch 张量实现 ``DeviceBatchState`` 契约与 device
mutator 方法，不依赖 warp/mjx 编译——M3 放行条件要求生命周期测试
不必每次拉起真实物理后端。物理是最小占位（qpos += qvel·dt +
xfrc 积分漂移），只保证"状态会变、写入可见、时序正确"，不模拟
真实动力学；语义正确性由 warp 后端的 fixture/冒烟测试负责。

同时作为**新后端的实现模板**：要实现新后端，按本文件的
``build_device_state`` + ``dev_*`` + ``physical_step`` 形态对齐即可。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .device_state import (
    DeviceBatchState,
    EpisodeNamespace,
    IoNamespace,
    RngNamespace,
    SimNamespace,
)


class FakeBatchBackend:
    """最小批量后端：nq=7, nv=6, nbody=2, nu=4, obs_dim=8。"""

    DT = 0.002
    ACTION_DIM = 4
    NQ, NV, NBODY, NU, OBS_DIM = 7, 6, 2, 4, 8

    def __init__(self, batch_size: int = 4, device: str = "cpu"):
        self._B = int(batch_size)
        self._dev = torch.device(device)
        self._state: Optional[DeviceBatchState] = None
        self._pend = torch.zeros(
            self._B, self.NBODY, 6, device=self._dev)
        self._sched: Optional[torch.Tensor] = None
        self.xfrc_log = []      # 每子步的 xfrc_applied 快照（测试断言用）
        self.step_calls = 0     # physical_step 调用计数

    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self._B

    def build_device_state(self) -> DeviceBatchState:
        if self._state is not None:
            return self._state
        B, dev = self._B, self._dev
        sim = SimNamespace(
            qpos=torch.zeros(B, self.NQ, device=dev),
            qvel=torch.zeros(B, self.NV, device=dev),
            ctrl=torch.zeros(B, self.NU, device=dev),
            xpos=torch.zeros(B, self.NBODY, 3, device=dev),
            xquat=torch.zeros(B, self.NBODY, 4, device=dev),
            xipos=torch.zeros(B, self.NBODY, 3, device=dev),
            xanchor=torch.zeros(B, self.NQ, 3, device=dev),
            cvel=torch.zeros(B, self.NBODY, 6, device=dev),
            xfrc_applied=torch.zeros(B, self.NBODY, 6, device=dev),
            act_target=torch.zeros(B, self.NU, device=dev),
            contacts_flat=None,
        )
        episode = EpisodeNamespace(
            episode_steps=torch.zeros(B, dtype=torch.int64, device=dev),
            active_mask=torch.ones(B, dtype=torch.bool, device=dev),
            terminated_flag=torch.zeros(B, dtype=torch.bool, device=dev),
            term_reason=torch.full((B,), -1, dtype=torch.int8, device=dev),
            agent_terminated=torch.zeros(B, 2, dtype=torch.bool, device=dev),
            agent_term_reason=torch.full(
                (B, 2), -1, dtype=torch.int8, device=dev),
            reset_request=torch.zeros(B, dtype=torch.bool, device=dev),
            time=torch.zeros(B, dtype=torch.float32, device=dev),
        )
        io = IoNamespace(
            action_a=torch.zeros(B, self.ACTION_DIM, device=dev),
            action_b=torch.zeros(B, self.ACTION_DIM, device=dev),
            obs_a=torch.zeros(B, self.OBS_DIM, device=dev),
            obs_b=torch.zeros(B, self.OBS_DIM, device=dev),
            reward=torch.zeros(B, device=dev),
        )
        rng = RngNamespace(
            seed_offsets=torch.zeros(B, dtype=torch.int64, device=dev),
            step_counter=torch.zeros((), dtype=torch.int64, device=dev),
        )
        self._state = DeviceBatchState(B, sim, episode, io, rng)
        return self._state

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self, seeds=None, options=None) -> None:
        st = self.build_device_state()
        st.sim.qpos.zero_()
        st.sim.qvel.zero_()
        st.sim.ctrl.zero_()
        st.sim.xfrc_applied.zero_()
        st.sim.act_target.zero_()
        self._sched = None
        self.xfrc_log.clear()
        self.step_calls = 0

    def physical_step(self, n_steps: int = 1, keep_history: bool = False) -> None:
        st = self.build_device_state()
        sched = self._sched
        for i in range(n_steps):
            # PD 占位：ctrl = clamp(act_target)
            st.sim.ctrl.copy_(st.sim.act_target.clamp(-1.0, 1.0))
            # 外力组合（与 warp 后端同语义：pending 首子步 / sched 逐子步）
            st.sim.xfrc_applied.zero_()
            if i == 0:
                st.sim.xfrc_applied.add_(self._pend)
            if sched is not None:
                st.sim.xfrc_applied.add_(sched[:, i])
            self.xfrc_log.append(st.sim.xfrc_applied.clone())
            # 最小动力学占位：qpos += qvel·dt；外力让 vel 漂移
            st.sim.qvel += st.sim.xfrc_applied[:, 0, :self.NV] * self.DT
            st.sim.qpos[:, :self.NV] += st.sim.qvel * self.DT
        st.sim.xfrc_applied.zero_()
        self._pend.zero_()
        self._sched = None
        self.step_calls += 1

    # ------------------------------------------------------------------
    # device mutator
    # ------------------------------------------------------------------
    def dev_set_action(self, action_a, action_b) -> None:
        st = self.build_device_state()
        st.io.action_a.copy_(action_a.clamp(-1.0, 1.0))
        st.io.action_b.copy_(action_b.clamp(-1.0, 1.0))
        st.sim.act_target[:, :self.ACTION_DIM] = st.io.action_a
        st.sim.act_target[:, self.ACTION_DIM:] = st.io.action_b

    def dev_add_ext_force(self, body_id: int, force, torque=None) -> None:
        self._pend[:, body_id, :3] += force
        if torque is not None:
            self._pend[:, body_id, 3:6] += torque

    def dev_upload_force_schedule(self, sched) -> None:
        if sched.shape[0] != self._B or sched.shape[2:] != (self.NBODY, 6):
            raise ValueError(f"schedule must be (B,S,{self.NBODY},6)")
        self._sched = sched

    def dev_reset_rows(self, env_ids) -> None:
        st = self.build_device_state()
        env_ids = torch.as_tensor(env_ids, dtype=torch.long,
                                  device=self._dev)
        st.sim.qpos[env_ids] = 0.0
        st.sim.qvel[env_ids] = 0.0
        st.sim.ctrl[env_ids] = 0.0
        st.sim.xfrc_applied[env_ids] = 0.0
        st.sim.act_target[env_ids] = 0.0
        st.io.action_a[env_ids] = 0.0
        st.io.action_b[env_ids] = 0.0

    def dev_set_integration_rows(self, env_ids, qpos_t, qvel_t) -> None:
        st = self.build_device_state()
        env_ids = torch.as_tensor(env_ids, dtype=torch.long,
                                  device=self._dev)
        st.sim.qpos[env_ids] = qpos_t
        st.sim.qvel[env_ids] = qvel_t

    # ------------------------------------------------------------------
    def get_physical_frequency(self) -> float:
        return 1.0 / self.DT

    def close(self) -> None:
        self._state = None
