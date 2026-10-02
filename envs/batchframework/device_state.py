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


# 终止原因 → i8 code（内置固定码；自定义 reason 由运行期 registry
# 动态分配 ≥6 的 code——原始字符串在 ``reason_registry`` 中保留）
TERM_CODES: Dict[str, int] = {
    "timeout": 0, "ko": 1, "foul": 2, "out_of_bounds": 3, "custom": 4,
    "abandoned": 5,
}
TERM_NAMES: Dict[int, str] = {v: k for k, v in TERM_CODES.items()}


@dataclass
class EpisodeNamespace:
    """per-env / per-agent 簿记（runtime 拥有）。

    生命周期状态机（discuss.md D7.1；FREE 不占行——padding 用
    ``slot_valid=False`` 表达，FAILED 用 ``world_failed``）::

        RUNNING (world_running=1) → ENDED (world_running=0, 封存待显式 reset)
                                  → FAILED (world_failed=1, 执行错误)

    四种 mask 分离（不合并语义）：
    - ``slot_valid``      行装载了真实 job（装配时固定；padding 行=False）
    - ``world_running``   该行仍在推进物理/产生帧；ENDED/FAILED=False
    - ``agent_done``      per-agent 终止标记（跨步持久，到显式 reset）
    - ``policy_eval_mask`` 本步是否调用该 agent 策略（ended 行=False）

    终止两级模型（CPU ``agent_termination_proposals`` 对齐）：
    - ``request_termination`` **立即**写 ``agent_done``（同 phase 后续
      插件可见，CPU 同语义）、标记 ``term_pending``（本步提议观测位），
      并把 reason **立即归档**进 ``term_history``——每 (agent, reason)
      首次出现记 (code, episode_step)，异 reason 保序、同 reason 去重、
      已终止后再提出的新 reason 仍记录（CPU recorder 逐帧扫描等价）；
    - 自定义 reason 字符串经 ``reason_registry`` 分配确定性 code
      （≥6 按首见序），原始字符串不丢失；
    - phase 屏障（action-step 末；子步 hook 模式为每物理子步后）判
      env 结束 = ``agent_done.all(-1)`` 或 ``reset_request``
      （等价 CPU 的 reset-while-active → abandoned），新 ENDED 行
      封存状态快照并调度 on_post_episode；
    - ``terminated_flag`` = 本步**新进入** ENDED 的行（屏障输出，
      步内多屏障取并集；下个 step 清除）。
    """

    # --- 状态机 mask ---
    slot_valid: torch.Tensor        # (B,) bool — 非 padding
    world_running: torch.Tensor     # (B,) bool — RUNNING
    world_failed: torch.Tensor      # (B,) bool — FAILED
    fail_reason: torch.Tensor       # (B,) i8 — FAIL_CODES（-1=无）
    agent_done: torch.Tensor        # (B, n_agents) bool — 跨步持久
    policy_eval_mask: torch.Tensor  # (B, n_agents) bool

    # --- 终止 ---
    terminated_flag: torch.Tensor   # (B,) bool — 本步新 ENDED（屏障输出）
    term_reason: torch.Tensor       # (B,) i8 — env 级最近 reason code
    agent_term_reason: torch.Tensor # (B, n_agents) i8 — 最近提出 code
    term_pending: torch.Tensor      # (B, n_agents) bool — 待屏障归档
    term_pending_code: torch.Tensor # (B, n_agents) i8
    term_history: torch.Tensor      # (B, n_agents, K, 2) i32 [code,step]
    term_history_len: torch.Tensor  # (B, n_agents) i32
    term_history_overflow: torch.Tensor  # (B, n_agents) bool — 超 K 截断
    reset_request: torch.Tensor     # (B,) bool — 插件请求结束并复位该 env

    # --- 计数器（分离；不假定互相换算） ---
    episode_steps: torch.Tensor     # (B,) i64 — 完成的 action step
    action_call_index: torch.Tensor # (B,) i64 — episode 内 step() 调用序号
    physics_steps: torch.Tensor     # (B,) i64 — episode 内已执行物理子步
    substep_index: torch.Tensor     # (B,) i32 — 当前子步（仅子步循环内有效）
    time: torch.Tensor              # (B,) f32 — episode 内累计物理秒
    n_agents: int = 2
    term_history_k: int = 8
    # reason str → i32 code 注册表（装配级共享；自定义 reason 按首见顺序
    # 确定性分配 ≥6 的 code，term_history 只存 code，导出时反查字符串）
    reason_registry: Dict[str, int] = field(
        default_factory=lambda: dict(TERM_CODES))

    # --- 兼容别名（E1 及更早调用点；新代码用规范名） ---
    @property
    def active_mask(self) -> torch.Tensor:
        """``world_running`` 的旧名（返回张量本体，原位写仍生效）。"""
        return self.world_running

    @property
    def agent_terminated(self) -> torch.Tensor:
        """``agent_done`` 的旧名。"""
        return self.agent_done


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

    派生约定：per-env 随机值 = hash(seed_offset, draw_counter, salt)；
    各插件的 salt 由 runtime 在 attach 时分配（声明 ``rng_salt`` 或
    按 unit 名哈希），保证插件间独立、episode 间可复现、env 间独立。
    种子绑定 **job identity**（base_seed），不绑 GPU slot/行号——
    分片重排不改变 job 的随机序列（E4 多卡前置）。
    """

    seed_offsets: torch.Tensor      # (B,) i64 — job 绑定的 per-env 种子
    step_counter: torch.Tensor      # () i64 — 全局 action step 计数


def _lshr64(x: torch.Tensor, s: int) -> torch.Tensor:
    """int64 逻辑右移（torch 的 >> 对负数是算术右移）。"""
    return (x >> s) & ((1 << (64 - s)) - 1)


def splitmix64(x: torch.Tensor) -> torch.Tensor:
    """splitmix64（int64 环绕语义）——per-env 独立可复现随机源。"""
    i64 = torch.int64
    z = x + torch.tensor(-7046029254386353131, dtype=i64,
                       device=x.device)  # 0x9E3779B97F4A7C15
    z = (z ^ _lshr64(z, 30)) * torch.tensor(-4658895280553007687, dtype=i64,
                                         device=x.device)  # 0xBF58476D1CE4E5B9
    z = (z ^ _lshr64(z, 27)) * torch.tensor(-7723592293110705685, dtype=i64,
                                         device=x.device)  # 0x94D049BB133111EB
    return z ^ _lshr64(z, 31)


# unit_seed 派生乘子——与 device_standup._draw_actions 的历史约定一致
_SEED_COUNTER_MULT = 6364136223846793005  # Knuth MMIX LCG 乘子


class RngView:
    """per-unit 确定性随机派生视图（挂到 ctx.rng）。

    ``unit_seed(env_ids, counter)`` 返回该 unit 随机域内、per-env 的
    i64 基础种子；插件自行决定 counter（reset 次数 / step_counter /
    子步索引……）与后续 splitmix 网格展开。运行时只负责 salt 分配与
    seed_offsets 按行发布——这样 E4 分片到不同 GPU slot 的同一 job
    抽到相同序列。
    """

    __slots__ = ("_state", "salt")

    def __init__(self, state: "DeviceBatchState", salt: int):
        self._state = state
        self.salt = int(salt)

    def unit_seed(self, env_ids: Optional[torch.Tensor] = None,
                  counter: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(M,) i64 基础种子 = seed_offsets[ids] + counter·MULT + salt。"""
        seeds = self._state.rng.seed_offsets
        if env_ids is not None:
            seeds = seeds[env_ids]
        if counter is None:
            counter = torch.zeros_like(seeds)
        elif counter.dim() == 0:
            counter = counter.expand_as(seeds)
        return seeds + counter * _SEED_COUNTER_MULT + self.salt


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
        """每个 action step 边界：清本步消费的标志。

        ``agent_done`` / ``term_history`` 是跨步持久的 per-episode 状态，
        不在此清理——由 reset_rows 按行清零。``terminated_flag`` 是
        "本步新 ENDED" 的一步期输出，每步开头清除。
        """
        ep = self.episode
        ep.terminated_flag.zero_()
        ep.term_reason.fill_(-1)
        ep.reset_request.zero_()
        ep.term_pending.zero_()
        ep.term_pending_code.fill_(-1)

    def reset_episode_rows(self, env_ids: torch.Tensor) -> None:
        """按行清 episode 簿记（显式 reset_rows 用）。

        只清 episode scope 状态；``slot_valid``/策略身份/跨 episode
        的插件外部计数不动。
        """
        ep = self.episode
        ep.world_running[env_ids] = ep.slot_valid[env_ids]
        ep.world_failed[env_ids] = False
        ep.fail_reason[env_ids] = -1
        ep.agent_done[env_ids] = False
        ep.policy_eval_mask[env_ids] = True
        ep.terminated_flag[env_ids] = False
        ep.term_reason[env_ids] = -1
        ep.agent_term_reason[env_ids] = -1
        ep.term_pending[env_ids] = False
        ep.term_pending_code[env_ids] = -1
        ep.term_history[env_ids] = -1
        ep.term_history_len[env_ids] = 0
        ep.term_history_overflow[env_ids] = False
        ep.reset_request[env_ids] = False
        ep.episode_steps[env_ids] = 0
        ep.action_call_index[env_ids] = 0
        ep.physics_steps[env_ids] = 0
        ep.substep_index[env_ids] = 0
        ep.time[env_ids] = 0.0


# ---------------------------------------------------------------------------
# runtime 侧命名空间分配（E1-W3：簿记所有权归 runtime，不归 sim/backend）
# ---------------------------------------------------------------------------
def alloc_episode_namespace(B: int, dev: torch.device,
                            n_agents: int = 2,
                            term_history_k: int = 8) -> EpisodeNamespace:
    i8, i32, i64, bl = torch.int8, torch.int32, torch.int64, torch.bool
    return EpisodeNamespace(
        slot_valid=torch.ones(B, dtype=bl, device=dev),
        world_running=torch.ones(B, dtype=bl, device=dev),
        world_failed=torch.zeros(B, dtype=bl, device=dev),
        fail_reason=torch.full((B,), -1, dtype=i8, device=dev),
        agent_done=torch.zeros(B, n_agents, dtype=bl, device=dev),
        policy_eval_mask=torch.ones(B, n_agents, dtype=bl, device=dev),
        terminated_flag=torch.zeros(B, dtype=bl, device=dev),
        term_reason=torch.full((B,), -1, dtype=i8, device=dev),
        agent_term_reason=torch.full((B, n_agents), -1, dtype=i8,
                                     device=dev),
        term_pending=torch.zeros(B, n_agents, dtype=bl, device=dev),
        term_pending_code=torch.full((B, n_agents), -1, dtype=i8,
                                     device=dev),
        term_history=torch.full((B, n_agents, term_history_k, 2), -1,
                                dtype=i32, device=dev),
        term_history_len=torch.zeros(B, n_agents, dtype=i32, device=dev),
        term_history_overflow=torch.zeros(B, n_agents, dtype=bl, device=dev),
        reset_request=torch.zeros(B, dtype=bl, device=dev),
        episode_steps=torch.zeros(B, dtype=i64, device=dev),
        action_call_index=torch.zeros(B, dtype=i64, device=dev),
        physics_steps=torch.zeros(B, dtype=i64, device=dev),
        substep_index=torch.zeros(B, dtype=i32, device=dev),
        time=torch.zeros(B, dtype=torch.float32, device=dev),
        n_agents=n_agents,
        term_history_k=term_history_k,
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
