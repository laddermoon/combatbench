"""DeviceBatchState — 批量运行时的设备端数据平面契约（backend 中立）。

对应 M3_PLAN §2。所有张量第一维是 batch dim (B,)，dtype/布局在此冻结：

- 类型一律 ``torch.Tensor``（CUDA）。插件不接触 ``wp.array``、mjw.Data
  或 warp 的 flat-packed contact 布局——接触以 padded (B, cap) schema
  对插件暴露，flat 原始视图仅保留在 ``sim`` 供后端/观测内部使用。
- ``sim`` 命名空间字段是**活跃视图**：物理步进原地更新底层存储，
  视图跨步自动反映新状态。插件可在任意 hook 重读，但不允许对 sim
  字段做原地写——写一律经过 mutator 方法（保证写入时机正确）。
- 插件持久状态经 ``declare_state`` 分配进 ``plugin`` 池，按 env 行独立；
  部分 reset 由 runtime 调 ``reset_plugin_rows(env_ids)`` 清零对应行。

命名空间::

    state.sim       物理状态视图（qpos/qvel/xpos/.../contacts_flat）
    state.episode   簿记（episode_steps/active_mask/term_reason/reset_request）
    state.io        本步 IO（action/obs/reward 缓冲）
    state.rng       随机性（per-env seed_offset + 全局 step_counter）
    state.plugin    插件声明的持久张量池（plugin_name → key → Tensor）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence

import torch


class ExecutionPlane(Enum):
    """插件执行平面（声明式，runtime 据此裁决准入与计费）。

    DEVICE     — 原生：读设备张量视图，整个 action step 内零 host↔device 传输。
    HOST       — 兼容：经惰性 numpy 视图物化，每 hook 至多一次；标注 syncs。
    HOST_SLOW  — 逐子步 host 回调 / 任意 numpy 语义；仅调试与离线路径。
    """

    DEVICE = "device"
    HOST = "host"
    HOST_SLOW = "host_slow"


# ---------------------------------------------------------------------------
# Namespace containers（字段即契约；新增字段需在 docstring/表格登记）
# ---------------------------------------------------------------------------
@dataclass
class SimNamespace:
    """物理状态设备视图。全部为活跃底层存储的视图（in-place 步进可见）。

    写权限：插件不得原地修改；写操作走 mutator（set_core_state_rows 等），
    由后端保证落在正确时机（物理循环边界 / forward 刷新）。
    """

    qpos: torch.Tensor          # (B, nq) f32
    qvel: torch.Tensor          # (B, nv) f32
    ctrl: torch.Tensor          # (B, nu) f32 — PD 求解出的执行器指令
    xpos: torch.Tensor          # (B, nbody, 3) f32 — body 世界系位置
    xquat: torch.Tensor         # (B, nbody, 4) f32 — [w,x,y,z]
    xipos: torch.Tensor         # (B, nbody, 3) f32 — 质心位置
    xanchor: torch.Tensor       # (B, nj, 3) f32 — joint 世界系锚点
    cvel: torch.Tensor          # (B, nbody, 6) f32 — [ang(0:3), lin(3:6)]
    xfrc_applied: torch.Tensor  # (B, nbody, 6) f32 — 当前步生效的外力
    act_target: torch.Tensor    # (B, nu_ctrl) f32 — PD 目标角（rad）
    contacts_flat: Optional["ContactFlatNamespace"] = None


@dataclass
class ContactFlatNamespace:
    """后端原始接触视图（flat packed）。仅后端/观测内部消费。

    插件视角的 padded (B, cap) schema 由 ``refresh_padded()`` 派生：
    geom/dist/pos/frame/efc_address/dim 按 worldid 散射到 per-env 槽位，
    不活跃槽位 dist=+inf（active = dist <= 0，与 M2 契约一致）。
    槽内顺序不稳定（atomic 分配），不得依赖顺序——消费方按
    (geom, pos) 自行排序。
    """

    worldid: torch.Tensor       # (C,) i32 — 所属 env；非活跃项为负
    geom: torch.Tensor          # (C, 2) i32
    dist: torch.Tensor          # (C,) f32 — <=0 为活跃
    pos: torch.Tensor           # (C, 3) f32
    frame: torch.Tensor         # (C, 3, 3) f32 — 行 [normal, t1, t2]
    dim: torch.Tensor           # (C,) i32 — condim
    efc_address: torch.Tensor   # (C,) i32 — 逐 world 的 efc 起始行
    efc_force: torch.Tensor     # (B, nefc) f32 — 约束空间力
    n_active: torch.Tensor      # () i32 — 全局活跃接触数
    cap_per_world: int = 48
    # padded 派生缓冲（refresh_padded 填充，action-step 粒度）
    padded: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class EpisodeNamespace:
    """per-env / per-agent 簿记。runtime 拥有；插件可置位
    reset_request / term_reason / agent_terminated。

    终止是两级的（对齐旧框架 ``agent_terminated`` 语义）：
    - ``agent_terminated (B, n_agents)``：单个 agent（robot_a/b）的终止
      标记，**跨步持久**直到该 env reset；env 只在全部 agent 终止时结束。
    - ``terminated_flag (B,)``：env 级终止/复位标记（本步置位即消费）；
      也作为"全 agent 终止"的派生/直接置位出口。
    """

    episode_steps: torch.Tensor     # (B,) i64 — 本 episode 已完成的 action step
    active_mask: torch.Tensor       # (B,) bool — env 未终止
    terminated_flag: torch.Tensor   # (B,) bool — env 级终止（本步消费）
    term_reason: torch.Tensor       # (B,) i8 — TerminationReason code（-1=无）
    agent_terminated: torch.Tensor  # (B, n_agents) bool — 跨步持久
    agent_term_reason: torch.Tensor # (B, n_agents) i8
    reset_request: torch.Tensor     # (B,) bool — 插件请求 reset 该 env
    time: torch.Tensor              # (B,) f32 — episode 内累计物理秒
    n_agents: int = 2


@dataclass
class IoNamespace:
    """action-step 粒度的 IO 缓冲。plugin 可写 action（pre_action_step hook）
    与 reward（post_action_step，经 reward channel 写入）。"""

    action_a: torch.Tensor          # (B, act_dim) f32 — 当前动作 [-1,1]
    action_b: torch.Tensor
    obs_a: Optional[torch.Tensor] = None   # (B, obs_dim) f32
    obs_b: Optional[torch.Tensor] = None
    reward: Optional[torch.Tensor] = None  # (B,) f32
    reward_channels: Optional[torch.Tensor] = None  # (B, C) f32


@dataclass
class RngNamespace:
    """设备端随机性：不设全局 generator 状态。

    派生约定：per-env 随机值 = hash(seed_offset, step_counter, salt)；
    各插件的 salt 在 attach 时由 runtime 分配，保证插件间独立、
    episode 间可复现、env 间独立。部分 reset 时 seed_offset 行更新。
    """

    seed_offsets: torch.Tensor      # (B,) i64
    step_counter: torch.Tensor      # () i64 — 全局 action step 计数


class DeviceBatchState:
    """数据平面根对象。由后端 binding 构造并持有；插件经 ctx 访问。"""

    def __init__(self, batch_size: int, sim: SimNamespace,
                 episode: EpisodeNamespace, io: IoNamespace,
                 rng: RngNamespace):
        self.batch_size = int(batch_size)
        self.sim = sim
        self.episode = episode
        self.io = io
        self.rng = rng
        # plugin_name → {key: Tensor (B, ...)}
        self.plugin: Dict[str, Dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # 插件持久状态池
    # ------------------------------------------------------------------
    def declare_state(self, plugin_name: str, key: str,
                      shape: Sequence[int], dtype: torch.dtype,
                      init: float = 0.0,
                      device: Optional[torch.device] = None) -> torch.Tensor:
        """声明一个 per-env 持久张量 (B, *shape)，行独立。

        init 为填充初值。重复声明同名同型返回已有张量；形状/dtype 冲突报错。
        返回张量归插件使用，但生命周期归 runtime（partial reset 会清零行）。
        """
        pool = self.plugin.setdefault(plugin_name, {})
        if key in pool:
            t = pool[key]
            if tuple(t.shape[1:]) != tuple(shape) or t.dtype != dtype:
                raise ValueError(
                    f"declare_state({plugin_name}.{key}): existing "
                    f"{tuple(t.shape)} {t.dtype} conflicts with "
                    f"requested {(self.batch_size, *shape)} {dtype}")
            return t
        dev = device or self.sim.qpos.device
        t = torch.full((self.batch_size, *shape), init,
                       dtype=dtype, device=dev)
        pool[key] = t
        return t

    def reset_plugin_rows(self, env_ids: torch.Tensor) -> None:
        """部分 reset：把所有插件状态池中被重置 env 的行清零。"""
        for pool in self.plugin.values():
            for t in pool.values():
                t[env_ids] = 0

    # ------------------------------------------------------------------
    # episode 簿记辅助
    # ------------------------------------------------------------------
    def clear_step_flags(self) -> None:
        """每个 action step 边界：清 env 级终止/reset 请求标志。

        ``agent_terminated`` 是跨步持久的 per-episode 状态，
        不在此清理——由 partial reset 按行清零。
        """
        self.episode.terminated_flag.zero_()
        self.episode.term_reason.fill_(-1)
        self.episode.reset_request.zero_()


# ---------------------------------------------------------------------------
# runtime 侧命名空间分配（E1-W3：簿记所有权归 runtime，不归 sim/backend）
# ---------------------------------------------------------------------------
def alloc_episode_namespace(B: int, dev: torch.device,
                            n_agents: int = 2) -> EpisodeNamespace:
    return EpisodeNamespace(
        episode_steps=torch.zeros(B, dtype=torch.int64, device=dev),
        active_mask=torch.ones(B, dtype=torch.bool, device=dev),
        terminated_flag=torch.zeros(B, dtype=torch.bool, device=dev),
        term_reason=torch.full((B,), -1, dtype=torch.int8, device=dev),
        agent_terminated=torch.zeros(B, n_agents, dtype=torch.bool,
                                     device=dev),
        agent_term_reason=torch.full((B, n_agents), -1, dtype=torch.int8,
                                     device=dev),
        reset_request=torch.zeros(B, dtype=torch.bool, device=dev),
        time=torch.zeros(B, dtype=torch.float32, device=dev),
        n_agents=n_agents,
    )


def alloc_io_namespace(B: int, dev: torch.device, action_dim: int,
                       obs_dim: int) -> IoNamespace:
    return IoNamespace(
        action_a=torch.zeros(B, action_dim, device=dev),
        action_b=torch.zeros(B, action_dim, device=dev),
        obs_a=torch.zeros(B, obs_dim, device=dev),
        obs_b=torch.zeros(B, obs_dim, device=dev),
        reward=torch.zeros(B, device=dev),
    )


def alloc_rng_namespace(B: int, dev: torch.device) -> RngNamespace:
    return RngNamespace(
        seed_offsets=torch.zeros(B, dtype=torch.int64, device=dev),
        step_counter=torch.zeros((), dtype=torch.int64, device=dev),
    )


def compose_state(batch_size: int, sim_ns: SimNamespace, dev: torch.device,
                  action_dim: int, obs_dim: int,
                  n_agents: int = 2) -> DeviceBatchState:
    """runtime 组装数据平面：sim_ns 由后端提供，簿记命名空间在此分配。"""
    return DeviceBatchState(
        batch_size,
        sim_ns,
        alloc_episode_namespace(batch_size, dev, n_agents),
        alloc_io_namespace(batch_size, dev, action_dim, obs_dim),
        alloc_rng_namespace(batch_size, dev),
    )
