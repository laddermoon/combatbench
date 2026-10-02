"""BaseDevicePlugin + DeviceCtx — 设备端原生插件契约（M3 W2）。

与 ``batch_plugin.py``（numpy 契约原型）的关系：hook 序列与权限语义
相同，但 ctx 暴露的是 **设备张量视图**（torch.Tensor, CUDA）。DEVICE
插件在整个 action step 内不得触发任何 host↔device 传输（``Tensor.cpu()``
/ ``.item()`` / ``print(tensor)`` 均违规）——由约定 + 性能审计强制，
不在解释层拦截。

终止编码（episode.term_reason i8）::

    -1 = 无, 0 = TIMEOUT, 1 = KO, 2 = FOUL, 3 = OUT_OF_BOUNDS, 4 = CUSTOM,
    5 = ABANDONED（reset-while-active / 显式 abandon）

终止模型（对齐 CPU ``agent_termination_proposals``，见
LIFECYCLE_TRACE.md §2）：

- ``request_termination`` **立即**置 ``agent_done``——同 phase 后续
  插件可见（CPU 提出即生效语义）；同时写 ``term_pending`` 待归档。
- 每个 phase 结束后的屏障把 pending 归档进 ``term_history``
  （per-agent，每 reason 首次出现记 (code, step)，去重保序）。
- env 结束判定在屏障：``agent_done.all(-1)`` 或 env 级/reset_request。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .device_state import (
    DeviceBatchState,
    ExecutionPlane,
    TERM_CODES,
    TERM_NAMES,
)

# 执行错误码（episode.fail_reason i8；FAILED 是执行错误不是任务失败）
FAIL_CODES: Dict[str, int] = {"capacity": 0, "non_finite": 1}
FAIL_NAMES: Dict[int, str] = {v: k for k, v in FAIL_CODES.items()}


# ---------------------------------------------------------------------------
# Mutator — 可写 hook 中授予的写接口
# ---------------------------------------------------------------------------
class DeviceMutator:
    """对后端 device 写方法的窄封装。只暴露契约内操作。

    按 hook/插件声明收窄（E2-W4）：runtime 授予时带 ``allowed`` 动词
    集合；None = 全部允许（兼容旧插件），否则越权调用抛
    ``PermissionError``。hook 结束随 ctx._revoke_mutator 一并失效。
    """

    __slots__ = ("_sim", "_allowed")

    _VERBS = ("set_action", "add_ext_force", "upload_force_schedule",
              "reset_rows", "set_integration_rows")

    def __init__(self, sim):
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_allowed", None)

    def __setattr__(self, name, value):
        raise AttributeError("DeviceMutator is immutable")

    def _set_scope(self, allowed) -> None:
        object.__setattr__(self, "_allowed", allowed)

    def _check(self, verb: str) -> None:
        al = self._allowed
        if al is not None and verb not in al:
            raise PermissionError(
                f"mutator verb {verb!r} not granted for this hook/plugin "
                f"(allowed: {sorted(al)})")

    def set_action(self, action_a: torch.Tensor, action_b: torch.Tensor) -> None:
        """(B,21) 动作张量 → PD target。on_pre_action_step 内可调用。"""
        self._check("set_action")
        self._sim.dev_set_action(action_a, action_b)

    def add_ext_force(self, body_id: int, force: torch.Tensor,
                      torque: Optional[torch.Tensor] = None) -> None:
        """累加挂起外力 (B,3)[,(B,3)]；作用于下一物理块的首个子步。"""
        self._check("add_ext_force")
        self._sim.dev_add_ext_force(body_id, force, torque)

    def upload_force_schedule(self, sched: torch.Tensor) -> None:
        """(B, n_steps, nbody, 6) 逐子步外力表；下一 physical_step 消费。"""
        self._check("upload_force_schedule")
        self._sim.dev_upload_force_schedule(sched)

    def reset_rows(self, env_ids: torch.Tensor) -> None:
        """部分 reset：恢复初始姿态并重建 derived（forward）。"""
        self._check("reset_rows")
        self._sim.dev_reset_rows(env_ids)

    def set_integration_rows(self, env_ids: torch.Tensor,
                             qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        """写原始 qpos/qvel 行 + forward 刷新（跨后端搬运/回放用）。"""
        self._check("set_integration_rows")
        self._sim.dev_set_integration_rows(env_ids, qpos, qvel)


# ---------------------------------------------------------------------------
# DeviceCtx — hook 收到的上下文
# ---------------------------------------------------------------------------
class DeviceCtx:
    """设备端插件上下文。

    - ``state``    DeviceBatchState（sim 视图只读；episode/io 见权限）
    - ``mutator``  可写 hook 中由 runtime 授予，否则为 None
    - ``pstate``   本插件在 plugin_pool 中的张量字典
    - ``reset_env_ids`` / ``terminated_env_ids``：本事件的 env 行索引
      (device long tensor; 空长 tensor = 无/全量的语义按 hook 文档)
    """

    __slots__ = ("state", "mutator", "plugin_name", "reset_env_ids",
                 "terminated_env_ids", "phy_substeps", "rng",
                 "_mutator_impl")

    def __init__(self, state: DeviceBatchState, plugin_name: str = ""):
        self.state = state
        self.plugin_name = plugin_name
        self.mutator: Optional[DeviceMutator] = None
        self.reset_env_ids: Optional[torch.Tensor] = None
        self.terminated_env_ids: Optional[torch.Tensor] = None
        self.phy_substeps: int = 0   # 本物理块子步数（pre/post_batch_step 有效）
        self.rng = None              # RngView——声明 rng_salt 的 unit 才有
        self._mutator_impl: Optional[DeviceMutator] = None

    @property
    def sim(self):
        return self.state.sim

    @property
    def episode(self):
        return self.state.episode

    @property
    def io(self):
        return self.state.io

    @property
    def pstate(self) -> Dict[str, torch.Tensor]:
        return self.state.plugin.setdefault(self.plugin_name, {})

    @property
    def batch_size(self) -> int:
        return self.state.batch_size

    @property
    def substep_index(self) -> torch.Tensor:
        """(B,) i32 当前物理子步位置——仅 on_pre/post_phy_step 内有意义。"""
        return self.state.episode.substep_index

    def request_termination(
        self,
        env_ids: torch.Tensor,
        reason: str,
        agents: Optional[Sequence[int]] = None,
    ) -> None:
        """提出终止请求（提出即生效、即归档——CPU 同语义）。

        - ``agent_done`` / ``agent_term_reason`` / ``term_pending`` 立即
          写入：同 phase 的后续插件立即可见（对齐 CPU
          ``ctx.agent_terminated`` 直写）。
        - ``agents=None``：env 级终止（全员终止，等价旧框架
          ``agent_id=None``）；``agents=[0]`` 仅终止指定 agent——env 在
          全部 agent 终止后才由屏障判为 ENDED
          （``all_agents_terminated`` 语义）。
        - reason 立即记入 ``term_history``（每 (agent, reason) 首次出现
          记 (code, 当前 episode_step)；同 reason 去重、异 reason 保序、
          已终止后再提出的新 reason 仍记录——CPU recorder 逐帧扫描语义）。
        - 自定义 reason 字符串经 ``reason_registry`` 分配确定性 code，
          原始字符串不丢失（导出端反查）。
        - env 是否 ENDED 由 phase 屏障判定（见
          ``BatchRuntime._consume_terminations``）。
        """
        ep = self.state.episode
        ids = torch.as_tensor(env_ids, dtype=torch.long,
                              device=ep.agent_done.device).reshape(-1)
        if ids.numel() == 0:
            return
        reg = ep.reason_registry
        code = reg.get(reason)
        if code is None:
            code = len(reg)
            reg[reason] = code
        if agents is None:
            agents = range(ep.n_agents)
            ep.term_reason[ids] = code
        for a in agents:
            ep.agent_done[ids, a] = True
            ep.agent_term_reason[ids, a] = code
            ep.term_pending[ids, a] = True
            ep.term_pending_code[ids, a] = code
            _archive_reason(ep, ids, a, code)


    # runtime 内部使用
    def _grant_mutator(self, allowed=None) -> None:
        if self._mutator_impl is not None:
            self._mutator_impl._set_scope(allowed)
        self.mutator = self._mutator_impl

    def _revoke_mutator(self) -> None:
        if self._mutator_impl is not None:
            self._mutator_impl._set_scope(None)
        self.mutator = None


def _archive_reason(ep, env_ids: torch.Tensor, agent: int,
                    code: int) -> None:
    """(env_ids, agent, code) 首次出现 → term_history 追加 (code, step)。

    同 code 去重（已存在不重复记）；超 ``term_history_k`` 置 overflow
    标志（显式截断，不静默丢）。提出即归档——记录时刻 = 当前
    episode_step（与 CPU recorder 帧扫描的提议时刻一致）。
    """
    K = ep.term_history_k
    hist = ep.term_history[env_ids, agent]              # (M,K,2)
    exists = (hist[..., 0] == code).any(-1)             # (M,)
    hlen = ep.term_history_len[env_ids, agent]          # (M,)
    fresh = ~exists
    ep.term_history_overflow[env_ids, agent] |= (fresh & (hlen >= K))
    add = env_ids[fresh & (hlen < K)]
    if add.numel():
        slots = ep.term_history_len[add, agent]
        ep.term_history[add, agent, slots, 0] = code
        ep.term_history[add, agent, slots, 1] = \
            ep.episode_steps[add].to(torch.int32)
        ep.term_history_len[add, agent] += 1


# ---------------------------------------------------------------------------
# BaseDevicePlugin
# ---------------------------------------------------------------------------
class BaseDevicePlugin:
    """设备端原生插件基类。

    声明式属性：

    - ``plane``           ExecutionPlane（DEVICE 插件由 runtime 计 sync=0）
    - ``priority``        越大越先执行（observer dispatcher = 1_000_000）
    - ``require_mutator`` 最小权限：False 时即使可写 hook mutator 也为 None

    生命周期：

    - ``declare_state(state)``  attach 时调用一次，分配持久张量
      （``state.declare_state(name, key, shape, dtype)``）
    - hook 签名均为 ``(ctx: DeviceCtx) -> None``
    - ``on_envs_reset(ctx)``    本插件状态池行已清零后的回调
      （env_ids 在 ctx.reset_env_ids；默认 no-op——约定优于隐式发现）

    hook 表（与 batch_plugin.py 同序；E2-W3 起含子步粒度）::

        on_pre_episode      env 重置后、第一步前           read-write
        on_pre_action_step  action 入队后、物理块前      read-write(io.action)
        on_pre_batch_step   physical_step(n) 前          read-write(外力)
        on_pre_phy_step     每个物理子步前               read-write(子步力)
        on_post_phy_step    每个物理子步后               read-write(投影)
        on_post_batch_step  physical_step(n) 后          read-write(受限投影)
        on_post_action_step 每 action step               read-only(+term/reward)
        on_post_episode     env 终止后                   read-only

    子步 hook 仅 DEVICE plane 可声明；有任一插件覆写时 runtime 退化为
    逐子步驱动 physical_step(1)（warp 后端本就是 python 级子步循环，
    无额外 host 同步约束）。插件内不得 .cpu()/.item()（热路径禁令）。

    声明式能力面（E2-W4，装配期校验；默认空 = 不校验向后兼容）：

    - ``declared_reads``    sim namespace 字段名集（装配期对
      backend.describe().fields 白名单校验）
    - ``declared_writes``   mutator 动词集（set_action/add_ext_force/
      upload_force_schedule/reset_rows/set_integration_rows）
    - ``per_hook_mutator``  ``{hook_name: frozenset(verbs)}``——按 hook
      收窄 mutator 动词；None = 不收窄（兼容）
    - ``rng_salt``          int——声明后 runtime 给 ctx.rng 分配绑定
      RngView；None = 不使用随机服务
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def plane(self) -> ExecutionPlane:
        return ExecutionPlane.DEVICE

    @property
    def priority(self) -> int:
        return 0

    @property
    def require_mutator(self) -> bool:
        return False

    # --- 声明式能力面（装配期校验） ---
    @property
    def declared_reads(self) -> Sequence[str]:
        return ()

    @property
    def declared_writes(self) -> Sequence[str]:
        return ()

    @property
    def per_hook_mutator(self) -> Optional[Dict[str, frozenset]]:
        return None

    @property
    def rng_salt(self) -> Optional[int]:
        return None

    # --- attach 期 ---
    def declare_state(self, state: DeviceBatchState) -> None:
        """分配插件持久张量。默认 no-op。"""
        return None

    def on_attach(self) -> None:
        return None

    def on_detach(self) -> None:
        return None

    def set_episode_seeds(self, seeds: torch.Tensor) -> None:
        """(B,) i64 per-env 种子；reset 前调用。默认 no-op。"""
        return None

    # --- episode 指标导出（E1-W4：显式 schema，替代采集端嗅探 pool 键） ---
    def export_episode_metrics(self, state: DeviceBatchState
                               ) -> Dict[str, torch.Tensor]:
        """→ {metric_name: (B, ...) torch.Tensor}——按行取值的 episode
        指标表。由采集端在波末统一物化到 host 并写入 Episode.metrics。
        默认空集。键名需唯一（跨插件冲突时后 attach 覆盖先 attach）。
        """
        return {}

    # --- 生命周期 hooks（默认全 no-op） ---
    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        return None

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        """部分 reset 后：本插件 pool 行已清零，可在此重新初始化。"""
        return None

    def on_pre_action_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_pre_batch_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_pre_phy_step(self, ctx: DeviceCtx) -> None:
        """每个物理子步前（DEVICE 插件；ctx.substep_index 为子步位置）。"""
        return None

    def on_post_phy_step(self, ctx: DeviceCtx) -> None:
        """每个物理子步后。"""
        return None

    def on_post_batch_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        return None


# ---------------------------------------------------------------------------
# Observer：只读单元（reward/指标），由 dispatcher 统一刷新
# ---------------------------------------------------------------------------
class BaseDeviceObserver:
    """设备端 observer——只读，输出入 get_output()。

    约定：reward unit 返回 (B,) tensor；debug/metric unit 返回任意值。
    """

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        return None

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        return None

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        return None

    @property
    def output_schema(self):
        """``{leaf: (dtype, shape_suffix)}``——声明式输出契约（E3-W3）。

        ``None`` = 未声明（兼容路径：RecordStore 只收 (B,) 张量叶）。
        声明后 store 校验每帧每叶存在且形状匹配，缺失/不符即报错。
        """
        return None

    def get_output(self) -> Any:
        raise NotImplementedError


OBSERVER_DISPATCHER_PRIORITY = 1_000_000


class DeviceObserverDispatcher(BaseDevicePlugin):
    """observer 调度器——抢占用户插件之前刷新（priority 最高）。

    保证同一步内下游 reward/终止插件读到的 observer 输出对应当前状态。
    """

    def __init__(self):
        self.observers: Dict[str, BaseDeviceObserver] = {}

    @property
    def name(self) -> str:
        return "device_observer_dispatcher"

    @property
    def priority(self) -> int:
        return OBSERVER_DISPATCHER_PRIORITY

    def set_observer(self, name: str, unit: Optional[BaseDeviceObserver]) -> None:
        if unit is None:
            self.observers.pop(name, None)
        else:
            self.observers[name] = unit

    def get_output(self, name: str) -> Any:
        u = self.observers.get(name)
        return u.get_output() if u is not None else None

    def on_pre_episode(self, ctx):
        for u in self.observers.values():
            u.on_pre_episode(ctx)

    def on_envs_reset(self, ctx):
        for u in self.observers.values():
            u.on_envs_reset(ctx)

    def on_post_action_step(self, ctx):
        for u in self.observers.values():
            u.on_post_action_step(ctx)

    def on_post_episode(self, ctx):
        for u in self.observers.values():
            u.on_post_episode(ctx)
