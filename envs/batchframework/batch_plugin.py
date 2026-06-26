"""Batch plugin system — vectorized lifecycle hooks for batch simulation.

与旧 framework 的 plugin.py 对应，但面向批量（MJX）仿真。

核心区别：
    旧: 6 个生命周期 hook + 2 个管理 hook
    新: 6 个生命周期 hook + 2 个管理 hook（但 hook 语义不同）

    旧: on_pre_phy_step / on_post_phy_step 在每个物理步前后触发
    新: on_pre_batch_step / on_post_batch_step 在整个 n_steps 物理推进前后触发一次
        （因为 physical_step(n_steps) 在 GPU 上连续执行，中间不回 Python）

    旧: set_episode_seed(seed: int)
    新: set_episode_seeds(seeds: np.ndarray)  # (B,) int array

    旧: ctx.request_termination(reason) → 整个 episode 停止
    新: ctx.request_termination(env_id, reason) → 单个 env 终止

    旧: ctx.mutator 操作标量数据
    新: ctx.mutator 操作 (B, ...) 批量数据

设计原则（来自 discuss.md 的两层架构）：
1. Plugin 运行在 **Python 编排层**，不在 JIT 编译的物理步内部。
2. 物理推进（n_steps）在 GPU 上连续执行，plugin 只在 action step 边界触发。
3. 终止是 per-env 的：plugin 标记某些 env 终止，runtime 负责重置。
4. 权限模型与旧 framework 一致：require_mutator + grant/revoke。
5. Priority 排序与旧 framework 一致：大的先执行。

Hook 时序与权限：
| Hook                   | 时机                              | 权限       | 典型用途                     |
|------------------------|-----------------------------------|------------|------------------------------|
| on_pre_episode         | env 重置后、第一步前              | read-write | 初始化状态、清零累加器       |
| on_pre_action_step     | action 设置后、物理步前           | read-write | 动作映射/裁剪                |
| on_pre_batch_step      | physical_step(n_steps) 前         | read-write | 注入扰动（持续外力）         |
| on_post_batch_step     | physical_step(n_steps) 后         | read-write | 硬约束投影                   |
| on_post_action_step    | 所有物理步完成后                  | read-only  | 指标聚合、终止判定、reward   |
| on_post_episode        | env 终止后                        | read-only  | episode 级汇总               |
| on_attach              | 插件附加到 runtime 时             | 无 ctx     | 分配一次性资源               |
| on_detach              | 插件从 runtime 分离时             | 无 ctx     | 释放资源                     |
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .batch_context import (
    BatchSimContext,
    ReadOnlyBatchSimContext,
    TerminationReason,
)


# ---------------------------------------------------------------------------
# Observer dispatcher priority (shared constant)
# ---------------------------------------------------------------------------
# 与旧 framework 的 OBSERVER_DISPATCHER_PRIORITY 保持一致。
# Observer dispatcher 默认抢占所有用户 plugin 之前刷新，
# 这样下游的奖励 / 终止 / 指标 plugin 在同一钩子里读到的 observer 输出
# 总是对应当前步的状态。
OBSERVER_DISPATCHER_PRIORITY: int = 1_000_000


# ---------------------------------------------------------------------------
# Base batch plugin
# ---------------------------------------------------------------------------
class BaseBatchPlugin:
    """批量仿真扩展插件基类。

    与旧 ``BasePlugin`` 的对应关系：
        name / priority / require_mutator  → 完全一致
        set_episode_seed(seed: int)        → set_episode_seeds(seeds: (B,) int)
        on_pre_phy_step / on_post_phy_step → on_pre_batch_step / on_post_batch_step
        其余 hook                           → 签名一致，ctx 类型变为 BatchSimContext

    职责：
    1. 在批量仿真的特定生命周期点注入自定义逻辑。
    2. 通过 ctx.accessor 读取批量数据（所有返回值第一维是 B）。
    3. 通过 ctx.mutator 修改批量数据（仅在可写 hook 中可用）。
    4. 通过 ctx.request_termination(env_id, reason) 标记单个 env 终止。
    """

    @property
    def name(self) -> str:
        return "unnamed_batch_plugin"

    @property
    def priority(self) -> int:
        """Plugin dispatch priority — larger values run first.

        约定与旧 framework 一致：
        * 默认 0。大多数 plugin（reward / termination / data collection）保持默认。
        * 需要在 observer 之前执行的 plugin，设置 priority 严格大于
          :data:`OBSERVER_DISPATCHER_PRIORITY`。
        """
        return 0

    @property
    def require_mutator(self) -> bool:
        """是否请求数据修改权限。

        False 时，即使在可写 hook 中 ctx.mutator 也为 None。
        遵循最小权限原则。
        """
        return False

    # ==========================================
    # Randomness Hook
    # ==========================================

    def set_episode_seeds(self, seeds: np.ndarray) -> None:
        """[Timing]: BatchRuntime.reset 前、on_pre_episode 之前。
        [Responsibility]: 拥有 RNG 的 plugin 应在此立即重建 RNG。

        Args:
            seeds: (B,) int array。每个 env 的随机种子。

        Default no-op: 不消费随机性的 plugin 无需覆写。
        """
        pass

    # ==========================================
    # Lifecycle Hooks
    # ==========================================

    def on_pre_episode(self, ctx: BatchSimContext) -> None:
        """[Timing]: env 重置后、第一步前。
        [Responsibility]: 初始化状态（resetters）、清零历史统计。
        [Permission]: Read-write（ctx.mutator 可用）。

        注意：在批量模式下，reset 可以是部分的（仅重置某些 env）。
        通过 ctx.reset_env_ids 查看本次重置了哪些 env。
        如果 ctx.reset_env_ids 为空，表示全量重置（所有 env）。
        """
        pass

    def on_pre_action_step(self, ctx: BatchSimContext) -> None:
        """[Timing]: 收到外部 action 后、发送到物理引擎前。
        [Responsibility]: 控制模式映射、动作空间映射、动作裁剪。
        [Permission]: Read-write（ctx.mutator 可用；action 可被修改）。

        与旧 framework 的区别：action 是 (B, action_dim) 批量数组。
        """
        pass

    def on_pre_batch_step(self, ctx: BatchSimContext) -> None:
        """[Timing]: physical_step(n_steps) 之前。
        [Responsibility]: 注入持续外力扰动。
        [Permission]: Read-write（ctx.mutator 可用；外力可被修改）。

        与旧 on_pre_phy_step 的区别：
        - 旧：每个物理步触发一次（N 次）
        - 新：整个 n_steps 块触发一次（1 次）
        - 因此外力是"持续"的——设置后在 n_steps 内持续作用，无需每步重设。
        """
        pass

    def on_post_batch_step(self, ctx: BatchSimContext) -> None:
        """[Timing]: physical_step(n_steps) 之后、状态已更新。
        [Responsibility]: 硬状态约束投影、高频数据收集。
        [Permission]: Read-write（ctx.mutator 可用；状态可被覆盖投影）。

        与旧 on_post_phy_step 的区别：
        - 旧：每个物理步后触发（N 次），可以看到中间状态
        - 新：n_steps 完成后触发一次，只能看到最终状态
        - 需要中间状态的 plugin 应使用 keep_history=True 读取历史
        """
        pass

    def on_post_action_step(self, ctx: BatchSimContext) -> None:
        """[Timing]: 一个 action step 的所有物理步完成后。
        [Responsibility]: 指标聚合、终止判定、reward 计算。
        [Permission]: Read-only（ctx.mutator 为 None）。

        终止判定通过 ctx.request_termination(env_id, reason) 标记单个 env。
        """
        pass

    def on_post_episode(self, ctx: BatchSimContext) -> None:
        """[Timing]: env 终止后。
        [Responsibility]: Episode 级日志聚合和数据上报。
        [Permission]: Read-only（ctx.mutator 为 None）。

        与旧 framework 的区别：
        - 旧：整个 episode 终止时触发一次
        - 新：每个终止的 env 触发一次，通过 ctx.terminated_env_ids 查看哪些 env 终止
        - 批量模式下可能只有部分 env 终止，其余继续运行
        """
        pass

    # ==========================================
    # Management Hooks
    # ==========================================

    def on_attach(self) -> None:
        """[Timing]: 插件附加到 runtime 时（一次性）。
        [Responsibility]: 分配一次性资源——打开文件、建立连接、预编译 kernel。
        [Permission]: 无 ctx，不要访问仿真状态。
        [Default]: no-op.
        """
        pass

    def on_detach(self) -> None:
        """[Timing]: 插件从 runtime 分离时（一次性）。
        [Responsibility]: 释放在 on_attach 中获取的资源。
        [Permission]: 无 ctx，不要访问仿真状态。
        [Default]: no-op.
        """
        pass


# ---------------------------------------------------------------------------
# Batch plugin manager
# ---------------------------------------------------------------------------
class _BatchPluginManager:
    """批量插件管理器——按 priority 降序调度 plugin hooks。

    与旧 ``_PluginManager`` 的设计一致：
    - attach / detach 管理 plugin 列表
    - invoke 按 priority 降序调用 hook
    - require_mutator 控制 mutator 授予
    - strict 模式控制异常传播

    区别：ctx 类型为 BatchSimContext，hook 签名适配批量语义。
    """

    def __init__(self, strict: bool = True):
        self._plugins: List[BaseBatchPlugin] = []
        self._strict = bool(strict)

    def attach(self, plugin: BaseBatchPlugin) -> None:
        if plugin in self._plugins:
            return
        self._plugins.append(plugin)
        self._plugins.sort(
            key=lambda p: p.priority, reverse=True
        )
        plugin.on_attach()

    def detach(self, plugin: BaseBatchPlugin) -> None:
        if plugin in self._plugins:
            self._plugins.remove(plugin)
            plugin.on_detach()

    def clear(self) -> None:
        for plugin in list(self._plugins):
            self.detach(plugin)

    def iter_plugins(self) -> Tuple[BaseBatchPlugin, ...]:
        return tuple(self._plugins)

    def invoke(
        self,
        hook_name: str,
        ctx: BatchSimContext,
        allow_mutator: bool = False,
    ) -> None:
        """按 priority 降序调用所有 plugin 的指定 hook。"""
        for plugin in self._plugins:
            hook = getattr(plugin, hook_name, None)
            if hook is None:
                continue
            if allow_mutator and plugin.require_mutator:
                ctx._grant_mutator()
            else:
                ctx._revoke_mutator()
            _safe_call(
                plugin, hook_name, self._strict,
                f"BatchPlugin '{plugin.name}'",
                ctx,
            )
        ctx._revoke_mutator()


# ---------------------------------------------------------------------------
# Safe call helper
# ---------------------------------------------------------------------------
def _safe_call(
    obj: Any,
    method_name: str,
    strict: bool,
    label: str,
    *args: Any,
) -> None:
    """调用 obj.method_name(*args)，根据 strict 模式处理异常。

    与旧 framework 的 _safe_call 行为一致：
    - strict=True: 异常直接传播
    - strict=False: 打印 traceback 并吞掉异常
    """
    if strict:
        getattr(obj, method_name)(*args)
    else:
        try:
            getattr(obj, method_name)(*args)
        except Exception:
            print(f"[{label}] Error in {method_name}:")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Batch observer dispatcher plugin
# ---------------------------------------------------------------------------
class _BatchObserverDispatcherPlugin(BaseBatchPlugin):
    """Observer dispatcher for batch mode.

    与旧 ``_ObserverDispatcherPlugin`` 对应：
    - priority = OBSERVER_DISPATCHER_PRIORITY（抢占用户 plugin 之前）
    - require_mutator = False（observer 永远只读）
    - 在 on_pre_episode / on_post_action_step / on_post_episode 刷新 observer
    """

    def __init__(self):
        self.observer_plugins: Dict[str, Optional["BaseBatchRuntimeUnit"]] = {}
        self._last_process_token: Optional[Tuple] = None

    @property
    def name(self) -> str:
        return "batch_observer_dispatcher"

    @property
    def priority(self) -> int:
        return OBSERVER_DISPATCHER_PRIORITY

    @property
    def require_mutator(self) -> bool:
        return False

    def set_observer_plugin(
        self, name: str, observer_plugin: Optional["BaseBatchRuntimeUnit"]
    ) -> None:
        self.observer_plugins[name] = observer_plugin
        self._last_process_token = None

    def remove_observer_plugin(self, name: str) -> None:
        self.observer_plugins.pop(name, None)
        self._last_process_token = None

    def get_output(self, name: str) -> Any:
        op = self.observer_plugins.get(name)
        return op.get_output() if op is not None else None

    def on_pre_episode(self, ctx: BatchSimContext) -> None:
        self._process_ctx(ctx, "on_pre_episode")

    def on_pre_action_step(self, ctx: BatchSimContext) -> None:
        pass

    def on_pre_batch_step(self, ctx: BatchSimContext) -> None:
        pass

    def on_post_batch_step(self, ctx: BatchSimContext) -> None:
        pass

    def on_post_action_step(self, ctx: BatchSimContext) -> None:
        self._process_ctx(ctx, "on_post_action_step")

    def on_post_episode(self, ctx: BatchSimContext) -> None:
        self._process_ctx(ctx, "on_post_episode")

    def on_attach(self) -> None:
        self._last_process_token = None

    def on_detach(self) -> None:
        self._last_process_token = None

    def refresh(self, ctx: BatchSimContext, force: bool = False) -> None:
        self._process_ctx(ctx, "on_manual_refresh", force=force)

    def _process_ctx(
        self, ctx: BatchSimContext, trigger_name: str, force: bool = False
    ) -> None:
        readonly_ctx = ReadOnlyBatchSimContext(ctx)
        process_token = (
            trigger_name,
            readonly_ctx.action_step,
            tuple(readonly_ctx.terminated_env_ids),
            readonly_ctx.is_any_terminated,
        )
        if not force and process_token == self._last_process_token:
            return
        self._last_process_token = process_token
        for unit in self._iter_runtime_units():
            getattr(unit, trigger_name)(readonly_ctx)

    def _iter_runtime_units(self):
        seen: set = set()
        for unit in list(self.observer_plugins.values()):
            if unit is None:
                continue
            unit_id = id(unit)
            if unit_id in seen:
                continue
            seen.add(unit_id)
            yield unit


# ---------------------------------------------------------------------------
# Batch runtime unit (observer base)
# ---------------------------------------------------------------------------
class BaseBatchRuntimeUnit:
    """Observer-side unit invoked by ``_BatchObserverDispatcherPlugin``.

    与旧 ``BaseRuntimeUnit`` 对应，但 ctx 类型为 ReadOnlyBatchSimContext。

    生命周期：
    * on_pre_episode — episode 开始时初始化
    * on_post_action_step — 每步刷新内部状态
    * on_post_episode — episode 结束时收尾

    get_output() 返回值约定：
    * Reward units — 返回 (B,) 数组或 dict 含 "reward" key 的 (B,) 数组
    * Debug / metric units — 返回任意 JSON-safe 值
    """

    def on_pre_episode(self, ctx: ReadOnlyBatchSimContext) -> None:
        """Episode-start preparation. Default: no-op."""
        return None

    def on_post_action_step(self, ctx: ReadOnlyBatchSimContext) -> None:
        """Per-step refresh. Default: no-op — subclasses override."""
        return None

    def on_post_episode(self, ctx: ReadOnlyBatchSimContext) -> None:
        """Episode-end finalisation. Default: no-op."""
        return None

    def on_manual_refresh(self, ctx: ReadOnlyBatchSimContext) -> None:
        """Out-of-band refresh. Default: delegate to on_post_action_step."""
        self.on_post_action_step(ctx)

    def get_output(self) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in: BatchTimeoutPlugin
# ---------------------------------------------------------------------------
class BatchTimeoutPlugin(BaseBatchPlugin):
    """Per-env timeout termination plugin.

    与旧 ``TimeoutPlugin`` 对应，但终止是 per-env 的：
    每个 env 独立计数，达到 max_steps 时仅终止该 env。
    """

    def __init__(self, max_steps: int):
        self._max_steps = int(max_steps)

    @property
    def name(self) -> str:
        return "batch_timeout"

    @property
    def require_mutator(self) -> bool:
        return False

    def on_pre_episode(self, ctx: BatchSimContext) -> None:
        pass  # env_episode_steps 已在 ctx.clear_episode_state 中清零

    def on_post_action_step(self, ctx: BatchSimContext) -> None:
        for env_id in range(ctx.batch_size):
            if not ctx.active_mask[env_id]:
                continue
            if ctx.env_episode_steps[env_id] >= self._max_steps:
                ctx.request_termination(env_id, TerminationReason.TIMEOUT)
