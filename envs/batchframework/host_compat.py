"""HOST / HOST_SLOW 兼容层——旧插件经数据转换复用（M3 W3）。

两种适配器：

- ``HostBatchCompatAdapter`` (plane=HOST)：包装 ``BaseBatchPlugin``
  （``batch_plugin.py`` 的 numpy 批量契约）。通过 ``_CachingSimProxy``
  把 warp sim 的 host 快照 API 包一层 per-hook 缓存 + SyncStats 计量；
  旧插件见到的 ``BatchSimContext`` 与原型路径完全一致。
- ``LegacyPluginAdapter`` (plane=HOST_SLOW)：包装单 env ``BasePlugin``
  （``envs/framework/plugin.py``）。每 hook 逐 env 构造 SimContext
  视图循环调用——B× 物化，语义精确但慢；仅调试/对照/低频路径，
  runtime 默认拒绝接入（``allow_host_slow=True`` 才放行）。
  **覆写了 on_pre_phy_step / on_post_phy_step 的旧插件在 attach 时
  直接拒绝**——逐子步 host 回调在融合步进中无法忠实执行，
  须转换为 schedule 上传或设备原生实现，不做静默降级。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .device_plugin import BaseDeviceObserver, BaseDevicePlugin, DeviceCtx
from .device_state import ExecutionPlane


# ---------------------------------------------------------------------------
# SyncStats — host↔device 传输计量
# ---------------------------------------------------------------------------
class SyncStats:
    """按插件统计 host 物化次数与字节数（runtime 审计/报告用）。"""

    def __init__(self):
        self.transfers: Dict[str, int] = {}
        self.bytes_moved: Dict[str, int] = {}

    def note(self, plugin: str, nbytes: int) -> None:
        self.transfers[plugin] = self.transfers.get(plugin, 0) + 1
        self.bytes_moved[plugin] = self.bytes_moved.get(plugin, 0) + nbytes

    def summary(self) -> Dict[str, Dict[str, int]]:
        return {k: {"transfers": self.transfers[k],
                    "bytes": self.bytes_moved.get(k, 0)}
                for k in self.transfers}


def _tree_nbytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_tree_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tree_nbytes(v) for v in obj)
    return 0


# ---------------------------------------------------------------------------
# _CachingSimProxy — per-dispatch 缓存的批量 host 视图
# ---------------------------------------------------------------------------
class _CachingSimProxy:
    """转发 ``BaseBatchSimulator`` 的 get_*，在 flush() 之间 memoize。

    每个结果计一次 SyncStats 传输；mutator 写操作（set_*/apply_*）使
    缓存失效（写入后读必须看到新状态）。
    """

    _GETTERS = ("get_static_data", "get_core_state", "get_derived_state",
                "get_sensor_data", "get_action", "get_observation",
                "get_broadcastview_image")

    def __init__(self, sim, stats: SyncStats, owner: str):
        self._sim = sim
        self._stats = stats
        self._owner = owner
        self._cache: Dict[str, Any] = {}

    def flush(self) -> None:
        self._cache.clear()

    def __getattr__(self, name: str):
        if name in self._GETTERS:
            def cached(*args, **kwargs):
                key = (name, str(args), str(sorted(kwargs.items())))
                if key not in self._cache:
                    out = getattr(self._sim, name)(*args, **kwargs)
                    self._stats.note(self._owner, _tree_nbytes(out))
                    self._cache[key] = out
                return self._cache[key]
            return cached
        attr = getattr(self._sim, name)
        if name.startswith(("set_", "apply_")) or name in (
                "reset", "physical_step"):
            def write_then_flush(*a, **kw):
                out = attr(*a, **kw)
                self._stats.note(self._owner, _tree_nbytes(a) + _tree_nbytes(kw))
                self.flush()
                return out
            return write_then_flush
        return attr


# ---------------------------------------------------------------------------
# HostBatchCompatAdapter — BaseBatchPlugin（numpy 批量契约）→ device
# ---------------------------------------------------------------------------
class HostBatchCompatAdapter(BaseDevicePlugin):
    """把 ``BaseBatchPlugin`` 挂进设备 runtime。

    被包装插件收到的仍是 ``BatchSimContext``（绑定到 caching proxy），
    接口契约与 ``batch_plugin.py`` 原型逐字一致；每 hook 结束后把
    ctx 上的终止/事件标记回流到设备 episode 簿记。
    """

    def __init__(self, plugin, sim, stats: Optional[SyncStats] = None):
        from .batch_context import BatchSimContext
        self._p = plugin
        self._stats = stats or SyncStats()
        self._proxy = _CachingSimProxy(sim, self._stats, owner=plugin.name)
        self._bctx = BatchSimContext(self._proxy)
        self.last_events: List[Any] = []

    @property
    def name(self) -> str:
        return self._p.name

    @property
    def plane(self) -> ExecutionPlane:
        return ExecutionPlane.HOST

    @property
    def priority(self) -> int:
        return self._p.priority

    @property
    def require_mutator(self) -> bool:
        return self._p.require_mutator

    # --- ctx 同步 ---
    def _pull_ctx(self, ctx: DeviceCtx) -> None:
        """device episode 簿记 → batch ctx（每次 dispatch 前）。

        episode 簿记本身也是 host↔device 传输，一并计入 SyncStats。
        """
        ep = ctx.state.episode
        self._bctx.env_episode_steps = ep.episode_steps.cpu().numpy()
        self._bctx.active_mask = ep.active_mask.cpu().numpy()
        self._bctx.action_step = int(ctx.state.rng.step_counter.item())
        self._bctx.reset_env_ids = ([] if ctx.reset_env_ids is None else
                                    ctx.reset_env_ids.cpu().tolist())
        self._stats.note(self.name, int(ep.episode_steps.nbytes
                                      + ep.active_mask.nbytes))

    def _push_ctx(self, ctx: DeviceCtx) -> None:
        """batch ctx 上累积的终止请求 → device 标记。"""
        if self._bctx.terminated_env_ids:
            ids = torch.as_tensor(self._bctx.terminated_env_ids,
                                  dtype=torch.long,
                                  device=ctx.state.episode.terminated_flag.device)
            reasons = self._bctx.termination_reasons
            # 逐 env 原因可能不同——按 reason 分组一次写入
            for env_id in self._bctx.terminated_env_ids:
                ctx.request_termination(
                    torch.as_tensor([env_id], device=ids.device),
                    str(reasons.get(env_id, "custom")))
            self._bctx.terminated_env_ids.clear()
            self._bctx.termination_reasons.clear()
        if self._bctx.events:
            self.last_events.extend(self._bctx.events)
            self._bctx.events.clear()

    def _dispatch(self, ctx: DeviceCtx, hook: str, writable: bool) -> None:
        self._proxy.flush()
        self._pull_ctx(ctx)
        if writable:
            self._bctx._grant_mutator()
        try:
            getattr(self._p, hook)(self._bctx)
        finally:
            self._bctx._revoke_mutator()
        self._push_ctx(ctx)
        self._bctx.clear_step_state()

    def set_episode_seeds(self, seeds: torch.Tensor) -> None:
        self._p.set_episode_seeds(seeds.cpu().numpy())

    # --- hooks ---
    def on_attach(self) -> None:
        if hasattr(self._p, "on_attach"):
            self._p.on_attach()

    def on_detach(self) -> None:
        if hasattr(self._p, "on_detach"):
            self._p.on_detach()

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_pre_episode", writable=True)

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        # 旧契约无部分 reset 区分：按 on_pre_episode 语义转发，
        # reset_env_ids 已通过 _pull_ctx 注入 batch ctx。
        self._dispatch(ctx, "on_pre_episode", writable=True)

    def on_pre_action_step(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_pre_action_step", writable=True)

    def on_pre_batch_step(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_pre_batch_step", writable=True)

    def on_post_batch_step(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_post_batch_step", writable=True)

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_post_action_step", writable=False)

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_post_episode", writable=False)


# ---------------------------------------------------------------------------
# LegacyPluginAdapter — 单 env BasePlugin → device（HOST_SLOW）
# ---------------------------------------------------------------------------
class _SingleEnvSimView:
    """把批量 sim 投影成单 env 视图（env i 的行切片）。

    accessor 方法返回去掉 batch 维的 numpy；mutator 方法只写 env i 行。
    每次调用都穿透到底层 sim 的 host 快照——不经额外缓存
    （HOST_SLOW 路径，语义正确优先于快）。
    """

    def __init__(self, sim, env_id: int):
        self._sim = sim
        self._env_id = int(env_id)

    # --- helpers ---
    def _strip(self, obj: Any) -> Any:
        i = self._env_id
        if isinstance(obj, np.ndarray) and obj.ndim >= 1 \
                and obj.shape[0] == self._sim.batch_size:
            return obj[i]
        if isinstance(obj, dict):
            return {k: self._strip(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._strip(v) for v in obj)
        return obj

    def _rebatch(self, obj: Any, template: Any = None) -> Any:
        """单 env dict → batch=1 dict（各 (...,) 加 leading 1 维）。"""
        if isinstance(obj, np.ndarray):
            return obj[np.newaxis]
        if isinstance(obj, dict):
            return {k: self._rebatch(v) for k, v in obj.items()}
        return obj

    # --- IDataAccessor ---
    @property
    def batch_size(self) -> int:
        return 1

    def get_static_data(self):
        return self._sim.get_static_data()

    def get_core_state(self, history: bool = False):
        return self._strip(self._sim.get_core_state(history=history))

    def get_derived_state(self, fields=None, history: bool = False):
        return self._strip(
            self._sim.get_derived_state(fields=fields, history=history))

    def get_sensor_data(self):
        return self._sim.get_sensor_data()

    def get_action(self):
        return self._strip(self._sim.get_action())

    def get_observation(self):
        return self._strip(self._sim.get_observation())

    def get_broadcastview_image(self, env_ids=None):
        return self._sim.get_broadcastview_image(env_ids=[self._env_id])

    def get_physical_frequency(self):
        return self._sim.get_physical_frequency()

    # --- IDataMutator（仅 env i 行） ---
    def set_core_state(self, state, env_ids=None):
        self._sim.set_core_state(self._rebatch(state), env_ids=[self._env_id])

    def set_action(self, action):
        cur = self._sim.get_action()
        for rid, a in action.items():
            full = np.asarray(cur[rid]).copy()
            full[self._env_id] = np.asarray(a, dtype=np.float32)
            cur[rid] = full
        self._sim.set_action(cur)

    def apply_external_force(self, body_name, force, torque=None,
                             robot_id="robot_a"):
        B = self._sim.batch_size
        f = np.zeros((B, 3)); f[self._env_id] = np.asarray(force)
        t = None
        if torque is not None:
            t = np.zeros((B, 3)); t[self._env_id] = np.asarray(torque)
        self._sim.apply_external_force(body_name, f, torque=t,
                                       robot_id=robot_id)


class LegacyPluginAdapter(BaseDevicePlugin):
    """单 env ``BasePlugin`` → 设备 runtime（HOST_SLOW）。

    attach 时拒绝覆写 per-substep hook 的插件（无法在融合步进中
    忠实执行）；其余 hook 逐 env 循环，每 env 一个独立 SimContext
    （含独立 metrics/events/agent_termination_proposals）。
    """

    _SUBSTEP_HOOKS = ("on_pre_phy_step", "on_post_phy_step")

    def __init__(self, plugin, sim, phy_substeps: int = 25):
        from envs.framework.context import SimContext
        from envs.framework.plugin import BasePlugin
        for h in self._SUBSTEP_HOOKS:
            if h in type(plugin).__dict__:
                raise ValueError(
                    f"LegacyPluginAdapter: '{plugin.name}' overrides {h} — "
                    f"per-substep host hooks cannot run inside the fused "
                    f"physics loop. Convert to force-schedule upload or a "
                    f"device-native plugin; silent block-boundary fallback "
                    f"is not allowed.")
        self._p = plugin
        self._sim = sim
        self._substeps = int(phy_substeps)
        self._env_ctxs = [SimContext(_SingleEnvSimView(sim, i))
                          for i in range(sim.batch_size)]
        self.last_events: List[Any] = []

    @property
    def name(self) -> str:
        return self._p.name

    @property
    def plane(self) -> ExecutionPlane:
        return ExecutionPlane.HOST_SLOW

    @property
    def priority(self) -> int:
        return self._p.priority

    @property
    def require_mutator(self) -> bool:
        return self._p.require_mutator

    def set_episode_seeds(self, seeds: torch.Tensor) -> None:
        # 已知限制：旧插件实例是单 env 语义——共享实例持有单个 RNG，
        # 逐 env 调 set_episode_seed 只保留最后一次的值。
        # 依赖 per-env 独立随机性的旧插件不能经此路径忠实复用，
        # 必须原生转换（rng.seed_offsets + salt 派生）。
        seeds_np = seeds.cpu().numpy()
        for i, c in enumerate(self._env_ctxs):
            c.base_seed = int(seeds_np[i])
            if hasattr(self._p, "set_episode_seed"):
                self._p.set_episode_seed(int(seeds_np[i]))

    def _dispatch(self, ctx: DeviceCtx, hook: str, writable: bool) -> None:
        ep = ctx.state.episode
        steps = ep.episode_steps.cpu().numpy()
        act = ep.active_mask.cpu().numpy()
        reset_ids = (None if ctx.reset_env_ids is None
                     else set(ctx.reset_env_ids.cpu().tolist()))
        term_ids = (set() if ctx.terminated_env_ids is None
                    else set(ctx.terminated_env_ids.cpu().tolist()))
        # 只向受影响的 env 分发：post_episode 只见终止 env；
        # pre_episode 带 reset_ids（部分 reset）时只见 reset env；
        # 常规 hook 见全部存活 env。
        if hook == "on_post_episode":
            targets = term_ids
        elif hook == "on_pre_episode" and reset_ids is not None:
            targets = reset_ids
        else:
            targets = set(range(ctx.batch_size))
        for i in sorted(targets):
            if not act[i] and hook not in ("on_post_episode",):
                continue
            c = self._env_ctxs[i]
            c.episode_step = int(steps[i])
            c.physics_step = int(steps[i]) * self._substeps
            if writable:
                c._grant_mutator()
            try:
                getattr(self._p, hook)(c)
            finally:
                c._revoke_mutator()
            self._drain_env_ctx(ctx, i, c)

    def _drain_env_ctx(self, ctx: DeviceCtx, env_id: int, c) -> None:
        """单 env ctx 的 per-agent 终止 → 设备行标记。"""
        dev = ctx.state.episode.terminated_flag.device
        for aid_i, aid in enumerate(("robot_a", "robot_b")):
            if c.agent_terminated.get(aid):
                reasons = c.agent_termination_proposals.get(aid) or ["custom"]
                ctx.request_termination(
                    torch.as_tensor([env_id], device=dev),
                    str(reasons[-1]), agents=[aid_i])
                c.agent_terminated[aid] = False
                c.agent_termination_proposals[aid].clear()
        if c.events:
            self.last_events.extend((env_id, e) for e in c.events)
            c.events.clear()

    # --- hooks ---
    def on_attach(self) -> None:
        if hasattr(self._p, "on_attach"):
            self._p.on_attach()

    def on_detach(self) -> None:
        if hasattr(self._p, "on_detach"):
            self._p.on_detach()

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        for c in self._env_ctxs:
            c.clear_episode_state()
        self._dispatch(ctx, "on_pre_episode", writable=True)

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        ids = ctx.reset_env_ids
        if ids is None:
            return
        for i in ids.cpu().tolist():
            self._env_ctxs[i].clear_episode_state()
        self._dispatch(ctx, "on_pre_episode", writable=True)

    def on_pre_action_step(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_pre_action_step", writable=True)

    def on_pre_batch_step(self, ctx: DeviceCtx) -> None:
        # 块级边界近似：旧插件的"每子步前"已无法逐子步表达；
        # 仅在块前调用一次，语义差异已在类 docstring 声明。
        self._dispatch(ctx, "on_pre_phy_step", writable=True)

    def on_post_batch_step(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_post_phy_step", writable=True)

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_post_action_step", writable=False)

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_post_episode", writable=False)


class LegacyObserverAdapter(BaseDeviceObserver):
    """单 env ``BaseObserverPlugin`` → 设备 observer（HOST_SLOW）。

    ``get_output()`` 返回 list[B]——per-env 输出列表；消费方自行聚合。
    """

    def __init__(self, observer, sim, phy_substeps: int = 25):
        from envs.framework.context import ReadOnlySimContext, SimContext
        self._o = observer
        self._substeps = int(phy_substeps)
        self._env_ctxs = [SimContext(_SingleEnvSimView(sim, i))
                          for i in range(sim.batch_size)]

    def _dispatch(self, ctx: DeviceCtx, hook: str) -> None:
        from envs.framework.context import ReadOnlySimContext
        ep = ctx.state.episode
        steps = ep.episode_steps.cpu().numpy()
        for i, c in enumerate(self._env_ctxs):
            c.episode_step = int(steps[i])
            c.physics_step = int(steps[i]) * self._substeps
            ro = ReadOnlySimContext.from_sim_context(c)
            getattr(self._o, hook)(ro)
            # observer 只读，无终止回流

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        for c in self._env_ctxs:
            c.clear_episode_state()
        self._dispatch(ctx, "on_pre_episode")

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        if ctx.reset_env_ids is None:
            return
        for i in ctx.reset_env_ids.cpu().tolist():
            self._env_ctxs[i].clear_episode_state()
        self._dispatch(ctx, "on_pre_episode")

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_post_action_step")

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        self._dispatch(ctx, "on_post_episode")

    def get_output(self) -> Any:
        return self._o.get_output()
