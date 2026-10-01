"""BatchRuntime — 设备端批量运行时（M3 W2）。

驱动契约（每个 ``step()`` = 一个 action step = n_substeps 物理步）::

    clear_step_flags
    [dev_set_action(actions)]            ← runtime 写入
    on_pre_action_step   (rw: io.action)
    on_pre_batch_step    (rw: pending force / force schedule)
    sim.physical_step(n_substeps)        ← 设备内连续推进
    on_post_batch_step   (rw: 受限投影)
    episode_steps += 1, time += dt·n, obs 构建 → io.obs
    on_post_action_step  (ro；可置 term/reset_request/写 reward)
    ── 终止消费 ──
    on_post_episode      (ro; terminated_env_ids)
    dev_reset_rows + plugin_pool 行清零 + episode 行复位
    on_envs_reset        (rw; reset_env_ids) —— 插件重初始化本 env 行
    on_pre_episode       (rw; reset_env_ids 非 None = 部分 reset)

per-env 终止是行级语义：KO/timeout 只影响对应 env；其余 env 的
物理状态、插件状态、episode 计数完全不受扰动。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .device_plugin import (
    BaseDeviceObserver,
    BaseDevicePlugin,
    DeviceCtx,
    DeviceMutator,
    DeviceObserverDispatcher,
    TERM_NAMES,
)
from .device_state import DeviceBatchState, ExecutionPlane


class BatchRuntime:
    """设备端批量运行时。

    Args:
        sim: 后端模拟器。必须实现 ``build_device_state()`` +
            dev_set_action/dev_add_ext_force/dev_upload_force_schedule/
            dev_reset_rows/dev_set_integration_rows + ``physical_step``。
        obs_builder: 每 action step 的观测构建器（``build(state)``），
            在 on_post_action_step 之前运行。
        phy_substeps: 每 action step 的物理子步数。
        strict: hook 异常直接传播（False 时打印并继续——仅调试用）。
        allow_host_slow: 是否允许 plane=HOST_SLOW 插件接入
            （训练 rollout 应为 False）。
    """

    def __init__(self, sim, obs_builder=None, phy_substeps: int = 25,
                 strict: bool = True, allow_host_slow: bool = False):
        self.sim = sim
        self.obs_builder = obs_builder
        self.phy_substeps = int(phy_substeps)
        self._strict = bool(strict)
        self._allow_host_slow = bool(allow_host_slow)
        self._plugins: List[BaseDevicePlugin] = []
        self._ctxs: Dict[int, DeviceCtx] = {}   # id(plugin) → ctx
        self._mutator = DeviceMutator(sim)
        self.dispatcher = DeviceObserverDispatcher()
        self.attach(self.dispatcher)

    # ------------------------------------------------------------------
    # 插件管理
    # ------------------------------------------------------------------
    @property
    def state(self) -> DeviceBatchState:
        """runtime 拥有的数据平面（E1-W3）。

        物理命名空间来自 sim.build_sim_namespace()；episode/io/rng
        簿记由 compose_state（device_state 模块）分配——sim/backend
        不再创建或持有簿记张量。组装后经 sim.attach_state 注册，
        dev_* mutator 的 io.action 镜像写入此对象（单一来源）。
        """
        if getattr(self, "_state", None) is None:
            from .device_state import compose_state
            obs_dim = (self.obs_builder.obs_dim()
                       if hasattr(self.obs_builder, "obs_dim")
                       else self.sim.obs_dim())
            dev = getattr(self.sim, "device", None)
            st = compose_state(
                self.sim.batch_size, self.sim.build_sim_namespace(),
                dev, self.sim.ACTION_DIM, obs_dim)
            self.sim.attach_state(st)
            self._state = st
        return self._state

    def attach(self, plugin: BaseDevicePlugin) -> None:
        if plugin.plane is ExecutionPlane.HOST_SLOW and not self._allow_host_slow:
            raise ValueError(
                f"plugin '{plugin.name}' declares plane=HOST_SLOW but runtime "
                f"disallows it (allow_host_slow=False)")
        if plugin in self._plugins:
            return
        plugin.declare_state(self.state)
        ctx = DeviceCtx(self.state, plugin_name=plugin.name)
        ctx._mutator_impl = self._mutator
        self._ctxs[id(plugin)] = ctx
        self._plugins.append(plugin)
        self._plugins.sort(key=lambda p: p.priority, reverse=True)
        plugin.on_attach()

    def detach(self, plugin: BaseDevicePlugin) -> None:
        if plugin in self._plugins:
            self._plugins.remove(plugin)
            self._ctxs.pop(id(plugin), None)
            plugin.on_detach()

    def set_observer(self, name: str, unit: Optional[BaseDeviceObserver]) -> None:
        self.dispatcher.set_observer(name, unit)

    def get_observer_output(self, name: str) -> Any:
        return self.dispatcher.get_output(name)

    def export_episode_metrics(self) -> Dict[str, torch.Tensor]:
        """聚合全部已 attach 插件声明的 episode 指标（显式 schema）。"""
        out: Dict[str, torch.Tensor] = {}
        for p in self._plugins:
            out.update(p.export_episode_metrics(self.state))
        return out

    # ------------------------------------------------------------------
    # Hook 调度
    # ------------------------------------------------------------------
    def _invoke(self, hook: str, writable: bool,
                reset_env_ids=None, terminated_env_ids=None) -> None:
        for p in self._plugins:
            ctx = self._ctxs[id(p)]
            ctx.reset_env_ids = reset_env_ids
            ctx.terminated_env_ids = terminated_env_ids
            if writable and p.require_mutator:
                ctx._grant_mutator()
            else:
                ctx._revoke_mutator()
            try:
                getattr(p, hook)(ctx)
            except Exception:
                if self._strict:
                    raise
                import traceback
                print(f"[DevicePlugin '{p.name}'] error in {hook}:")
                traceback.print_exc()
            finally:
                ctx._revoke_mutator()
                ctx.reset_env_ids = None
                ctx.terminated_env_ids = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self, seeds: Optional[torch.Tensor] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        """全量 reset 所有 env。"""
        st = self.state
        seeds_np = None if seeds is None else np_seeds(seeds)
        self.sim.reset(seeds=seeds_np, options=options)
        ep = st.episode
        ep.episode_steps.zero_()
        ep.active_mask.fill_(True)
        ep.terminated_flag.zero_()
        ep.term_reason.fill_(-1)
        ep.agent_terminated.zero_()
        ep.agent_term_reason.fill_(-1)
        ep.reset_request.zero_()
        ep.time.zero_()
        if seeds is not None:
            st.rng.seed_offsets.copy_(
                torch.as_tensor(seeds, dtype=torch.int64,
                                device=st.rng.seed_offsets.device).reshape(-1))
        st.rng.step_counter.zero_()
        seed_args = st.rng.seed_offsets
        for p in self._plugins:
            p.set_episode_seeds(seed_args)
        self._invoke("on_pre_episode", writable=True, reset_env_ids=None)

    def step(self, actions=None,
             n_substeps: Optional[int] = None) -> None:
        """一个 action step。

        Args:
            actions: 可选 (action_a (B,21), action_b (B,21)) torch 张量；
                None 表示沿用上一动作。
            n_substeps: 物理子步数（默认构造时的 phy_substeps）。
        """
        st = self.state
        ep = st.episode
        n = int(n_substeps or self.phy_substeps)

        st.clear_step_flags()
        if actions is not None:
            self.sim.dev_set_action(actions[0], actions[1])
        self._invoke("on_pre_action_step", writable=True)
        for c in self._ctxs.values():
            c.phy_substeps = n
        self._invoke("on_pre_batch_step", writable=True)
        self.sim.physical_step(n)
        self._invoke("on_post_batch_step", writable=True)

        ep.episode_steps[ep.active_mask] += 1
        ep.time += float(self.sim.DT) * n
        st.rng.step_counter += 1
        if self.obs_builder is not None:
            self.obs_builder.build(st)

        self._invoke("on_post_action_step", writable=False)

        # --- 终止 / reset 消费（行级） ---
        # env 终止 = 显式 env 级 flag 或全部 agent 均已终止（旧框架
        # all_agents_terminated 语义）；~active_mask 防御覆盖未消费行。
        need_reset = (ep.terminated_flag | ep.reset_request
                      | ep.agent_terminated.all(dim=-1) | ~ep.active_mask)
        term_mask = ep.terminated_flag | ep.agent_terminated.all(dim=-1)
        if not bool(need_reset.any()):
            return
        term_ids = torch.nonzero(term_mask, as_tuple=False).squeeze(-1)
        if term_ids.numel():
            self._invoke("on_post_episode", writable=False,
                         terminated_env_ids=term_ids)
        reset_ids = torch.nonzero(need_reset, as_tuple=False).squeeze(-1)

        self.sim.dev_reset_rows(reset_ids)
        st.reset_plugin_rows(reset_ids)
        ep.episode_steps[reset_ids] = 0
        ep.time[reset_ids] = 0.0
        ep.active_mask[reset_ids] = True
        ep.terminated_flag[reset_ids] = False
        ep.term_reason[reset_ids] = -1
        ep.agent_terminated[reset_ids] = False
        ep.agent_term_reason[reset_ids] = -1
        ep.reset_request[reset_ids] = False

        self._invoke("on_envs_reset", writable=True, reset_env_ids=reset_ids)
        self._invoke("on_pre_episode", writable=True, reset_env_ids=reset_ids)

    # ------------------------------------------------------------------
    def terminated_info(self) -> Dict[str, Any]:
        """当前终止快照（host 读取——低频诊断用，非热路径）。"""
        ep = self.state.episode
        ids = torch.nonzero(ep.terminated_flag, as_tuple=False).squeeze(-1)
        return {int(i): TERM_NAMES.get(int(ep.term_reason[i]), "custom")
                for i in ids.cpu()}


def np_seeds(seeds):
    """torch/list → (B,) int64 numpy。"""
    import numpy as np
    if torch.is_tensor(seeds):
        return seeds.detach().cpu().numpy().astype(np.int64)
    return np.asarray(seeds, dtype=np.int64)


# ---------------------------------------------------------------------------
# 内置：per-env timeout
# ---------------------------------------------------------------------------
class DeviceTimeoutPlugin(BaseDevicePlugin):
    """per-env timeout——等效旧框架 TimeoutPlugin 的批量设备版。"""

    def __init__(self, max_steps: int):
        self._max = int(max_steps)

    @property
    def name(self) -> str:
        return "device_timeout"

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        ep = ctx.episode
        ids = torch.nonzero(
            ep.active_mask & (ep.episode_steps >= self._max),
            as_tuple=False).squeeze(-1)
        if ids.numel():
            ctx.request_termination(ids, "timeout")
