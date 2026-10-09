"""BatchRuntime — 设备端批量运行时（E2 生命周期）。

驱动契约（每个 ``step()`` = 一个 action step = n_substeps 物理步）::

    clear_step_flags（本步消费标志）
    policy_eval_mask 刷新（world_running × post_termination_action）
    [dev_set_action(actions)]            ← runtime 写入
    on_pre_action_step   (rw: io.action)            【屏障：提出即生效】
    on_pre_batch_step    (rw: pending force / force schedule)
    for s in substep:（仅当有插件覆写子步 hook 才逐子步驱动）
        on_pre_phy_step → physical_step(1) → on_post_phy_step
    否则 sim.physical_step(n_substeps) 整块推进
    on_post_batch_step   (rw: 受限投影)
    episode_steps/physics_steps/... += 1（仅 RUNNING 行）
    obs 构建 → io.obs
    on_post_action_step  (ro；可提终止/写 reward)
    ── 终止屏障 ──
    pending → term_history 归档；ENDED 行封存（capture 快照 +
    每步 restore 冻结——warp 无 masked-step，见 physics.py 契约头注）
    on_post_episode      (ro; terminated_env_ids=本步新 ENDED 行)

    step() **不做 reset**。ENDED 行保持封存直到 collector 显式
    ``reset_rows``；对 RUNNING 行直接 reset 是契约错误，须先
    ``abandon``。语义依据见 LIFECYCLE_TRACE.md。

per-env 终止是行级语义：KO/timeout 只影响对应 env；其余 env 的
物理状态、插件状态、episode 计数完全不受扰动。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .device_plugin import (
    _archive_reason,
    BaseDeviceObserver,
    BaseDevicePlugin,
    DeviceCtx,
    DeviceMutator,
    DeviceObserverDispatcher,
    FAIL_CODES,
    TERM_CODES,
    TERM_NAMES,
)
from .device_state import DeviceBatchState, ExecutionPlane, RngView
from .physics import ContractError


class BatchRuntime:
    """设备端批量运行时。

    Args:
        sim: 后端模拟器。必须实现 ``build_sim_namespace()``/``attach_state`` +
            dev_set_action/dev_add_ext_force/dev_upload_force_schedule/
            dev_reset_rows/dev_set_integration_rows + ``physical_step``
            + ``capture``/``restore``（sealed-ENDED 冻结原语）。
        obs_builder: 每 action step 的观测构建器（``build(state)``），
            在 on_post_action_step 之前运行。
        phy_substeps: 每 action step 的物理子步数。
        strict: hook 异常直接传播（False 时打印并继续——仅调试用）。
        allow_host_slow: 是否允许 plane=HOST_SLOW 插件接入
            （训练 rollout 应为 False）。
        post_termination_action: ``"policy"``（默认，已终止 agent 继续
            采样——CPU EpisodeRunner 同语义）或 ``"hold"``（已终止 agent
            不采样，回放最后动作由 collector 负责）。
    """

    def __init__(self, sim, obs_builder=None, phy_substeps: int = 25,
                 strict: bool = True, allow_host_slow: bool = False,
                 post_termination_action: str = "policy"):
        if post_termination_action not in ("policy", "hold"):
            raise ValueError(
                f"post_termination_action must be 'policy' or 'hold', "
                f"got {post_termination_action!r}")
        self.sim = sim
        self.obs_builder = obs_builder
        self.phy_substeps = int(phy_substeps)
        self._strict = bool(strict)
        self._allow_host_slow = bool(allow_host_slow)
        self.post_termination_action = post_termination_action
        self._plugins: List[BaseDevicePlugin] = []
        self._ctxs: Dict[int, DeviceCtx] = {}   # id(plugin) → ctx
        self._substep_units: List[BaseDevicePlugin] = []  # 覆写子步 hook 的
        self._rng_salts: Dict[int, str] = {}    # salt → unit name（防撞）
        self._seal_snap: Optional[Dict[str, torch.Tensor]] = None
        self._mutator = DeviceMutator(sim)
        self.dispatcher = DeviceObserverDispatcher()
        # 同步点分类记账（E3-W5）：每个 host sync（bool(.any())/D2H 读回）
        # 记一类——先显式计量，优化归 E7。
        # E7-W0：barrier 内部逐 sync 记数（*_syncs），hook 按名累计
        # host 提交时间（hook_timing），consume 内部耗时
        # （barrier_time）——physics_wall 内含的子步 hook/屏障可用
        # 减法还原 kernel 估计。
        self.sync_stats = {"term_barrier": 0, "term_barrier_syncs": 0,
                           "freeze_check": 0, "freeze_syncs": 0,
                           "any_running": 0, "reset": 0}
        self.hook_timing: Dict[str, float] = {}
        # E7-W0：hook 内逐插件耗时——``{hook}::{plugin.name}`` → 秒。
        # 注意是 host 提交墙钟：插件内的隐式 host sync 会把前面异步
        # 工作的等待时间也计入该插件名下，归因以"谁发起同步"为准。
        self.hook_plugin_timing: Dict[str, float] = {}
        self.barrier_time: float = 0.0
        self.seg_timing: Dict[str, float] = {
            "physics_wall": 0.0, "obs_build": 0.0}
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
        if any(p.name == plugin.name for p in self._plugins):
            raise ValueError(
                f"duplicate plugin name {plugin.name!r} — plugin state pool "
                f"is keyed by name; give the second instance a distinct name")
        self._validate_unit_spec(plugin)
        plugin.declare_state(self.state)
        ctx = DeviceCtx(self.state, plugin_name=plugin.name)
        ctx._mutator_impl = self._mutator
        if plugin.rng_salt is not None:
            salt = int(plugin.rng_salt)
            clash = self._rng_salts.get(salt)
            if clash is not None and clash != plugin.name:
                raise ValueError(
                    f"rng salt {salt:#x} clash between {clash!r} and "
                    f"{plugin.name!r}")
            self._rng_salts[salt] = plugin.name
            ctx.rng = RngView(self.state, salt)
        self._ctxs[id(plugin)] = ctx
        self._plugins.append(plugin)
        self._plugins.sort(key=lambda p: p.priority, reverse=True)
        # 子步 hook 覆写检测（默认实现是 no-op——覆写才进子步驱动）
        if (type(plugin).on_pre_phy_step is not BaseDevicePlugin.on_pre_phy_step
                or type(plugin).on_post_phy_step
                is not BaseDevicePlugin.on_post_phy_step):
            if plugin.plane is not ExecutionPlane.DEVICE:
                raise ValueError(
                    f"plugin '{plugin.name}': substep hooks require "
                    f"plane=DEVICE (declared {plugin.plane})")
            self._substep_units.append(plugin)
        plugin.on_attach()

    def _validate_unit_spec(self, plugin: BaseDevicePlugin) -> None:
        """装配期校验声明式能力面（E2-W4）。

        声明为空集 → 跳过校验（向后兼容）；声明非空则必须全部合法。
        """
        writes = set(plugin.declared_writes or ())
        unknown = writes - set(DeviceMutator._VERBS)
        if unknown:
            raise ValueError(
                f"plugin '{plugin.name}' declares unknown mutator verbs "
                f"{sorted(unknown)}; legal: {DeviceMutator._VERBS}")
        if writes and not plugin.require_mutator:
            raise ValueError(
                f"plugin '{plugin.name}' declares writes {sorted(writes)} "
                f"but require_mutator=False")
        phm = plugin.per_hook_mutator
        if phm:
            for hook, verbs in phm.items():
                bad = set(verbs) - set(DeviceMutator._VERBS)
                if bad:
                    raise ValueError(
                        f"plugin '{plugin.name}'.per_hook_mutator[{hook!r}] "
                        f"has unknown verbs {sorted(bad)}")
        reads = set(plugin.declared_reads or ())
        if reads:
            try:
                fields = set(self.sim.describe().fields.keys())
            except Exception:
                fields = set()
            ns_fields = set(vars(self.sim.build_sim_namespace()).keys())
            unknown_r = reads - fields - ns_fields
            if unknown_r:
                raise ValueError(
                    f"plugin '{plugin.name}' declares reads on unknown "
                    f"fields {sorted(unknown_r)}")

    def detach(self, plugin: BaseDevicePlugin) -> None:
        if plugin in self._plugins:
            self._plugins.remove(plugin)
            self._ctxs.pop(id(plugin), None)
            if plugin in self._substep_units:
                self._substep_units.remove(plugin)
            plugin.on_detach()

    def set_observer(self, name: str, unit: Optional[BaseDeviceObserver]) -> None:
        self.dispatcher.set_observer(name, unit)

    def get_observer_output(self, name: str) -> Any:
        return self.dispatcher.get_output(name)

    def export_episode_metrics(self) -> Dict[str, torch.Tensor]:
        """聚合全部已 attach 插件声明的 episode 指标（显式 schema）。

        跨插件键名冲突 = 装配错误——静默覆盖会让采集端丢指标，显式拒绝。
        """
        out: Dict[str, torch.Tensor] = {}
        owners: Dict[str, str] = {}
        for p in self._plugins:
            for k, v in p.export_episode_metrics(self.state).items():
                if k in owners:
                    raise ValueError(
                        f"episode metric {k!r} exported by both "
                        f"{owners[k]!r} and {p.name!r}")
                owners[k] = p.name
                out[k] = v
        return out

    # ------------------------------------------------------------------
    # Hook 调度
    # ------------------------------------------------------------------
    def _invoke(self, hook: str, writable: bool,
                reset_env_ids=None, terminated_env_ids=None,
                units: Optional[List[BaseDevicePlugin]] = None) -> None:
        import time as _time
        _t0 = _time.perf_counter()
        for p in (self._plugins if units is None else units):
            ctx = self._ctxs[id(p)]
            ctx.reset_env_ids = reset_env_ids
            ctx.terminated_env_ids = terminated_env_ids
            phm = p.per_hook_mutator
            allowed = phm.get(hook) if phm else None
            if writable and p.require_mutator:
                ctx._grant_mutator(allowed)
            else:
                ctx._revoke_mutator()
            _tp = _time.perf_counter()
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
            key = f"{hook}::{p.name}"
            self.hook_plugin_timing[key] = (
                self.hook_plugin_timing.get(key, 0.0)
                + _time.perf_counter() - _tp)
        self.hook_timing[hook] = (
            self.hook_timing.get(hook, 0.0)
            + _time.perf_counter() - _t0)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def _publish_episode_options(self, st: DeviceBatchState,
                                 env_ids: torch.Tensor,
                                 options: Optional[Dict[str, Any]]
                                 ) -> None:
        """把 reset options 物化为 per-env 行张量快照（E9 G4）。

        options 形如 ``{白名单键: (B,) 数组}``（collector 已广播补
        padding）；快照供 ``ctx.episode_options`` 在 hook 内只读。
        """
        if not options:
            return
        dev = st.sim.qpos.device
        B = st.batch_size
        for k, v in options.items():
            t = torch.as_tensor(v, device=dev).reshape(-1)
            slot = st.episode_options.get(k)
            if (slot is None or slot.dtype != t.dtype
                    or slot.shape[0] != B):
                slot = torch.zeros(B, dtype=t.dtype, device=dev)
                st.episode_options[k] = slot
            if t.shape[0] == B:
                slot[env_ids] = t[env_ids]
            else:
                slot[env_ids] = t[: env_ids.shape[0]]

    def reset(self, seeds: Optional[torch.Tensor] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        """全量 reset 所有 env。"""
        st = self.state
        self.sync_stats["reset"] += 1
        seeds_np = None if seeds is None else np_seeds(seeds)
        self.sim.reset(seeds=seeds_np, options=options)
        all_ids = torch.arange(st.batch_size, device=st.sim.qpos.device)
        st.reset_episode_rows(all_ids)
        st.reset_shared_rows(all_ids)
        st.reset_events_rows(all_ids)
        self._publish_episode_options(st, all_ids, options)
        self._seal_snap = None
        if seeds is not None:
            st.rng.seed_offsets.copy_(
                torch.as_tensor(seeds, dtype=torch.int64,
                                device=st.rng.seed_offsets.device).reshape(-1))
        st.rng.step_counter.zero_()
        seed_args = st.rng.seed_offsets
        for p in self._plugins:
            p.set_episode_seeds(seed_args)
        self._invoke("on_pre_episode", writable=True, reset_env_ids=None)

    def reset_rows(self, env_ids: torch.Tensor,
                   seeds: Optional[torch.Tensor] = None,
                   options: Optional[Dict[str, Any]] = None) -> None:
        """显式部分 reset：只允许 ENDED/FAILED 行。

        对 RUNNING 行调用是契约错误（丢弃未完成 episode）——必须先
        ``abandon``。顺序对齐 CPU reset 链：backend 写初态 → 插件行
        清零/episode 簿记复位 → on_envs_reset → on_pre_episode。
        """
        st = self.state
        ep = st.episode
        ids = torch.as_tensor(env_ids, dtype=torch.long,
                              device=ep.episode_steps.device).reshape(-1)
        if ids.numel() == 0:
            return
        running = ep.world_running[ids] & ep.slot_valid[ids]
        if bool(running.any()):
            raise ContractError(
                f"reset_rows on RUNNING rows "
                f"{ids[running].tolist()} — call abandon() first")
        self.sim.dev_reset_rows(ids)
        st.reset_plugin_rows(ids)
        st.reset_shared_rows(ids)
        st.reset_events_rows(ids)
        st.reset_episode_rows(ids)
        self._publish_episode_options(st, ids, options)
        if seeds is not None:
            st.rng.seed_offsets[ids] = torch.as_tensor(
                seeds, dtype=torch.int64,
                device=st.rng.seed_offsets.device).reshape(-1)
        self._invoke("on_envs_reset", writable=True, reset_env_ids=ids)
        self._invoke("on_pre_episode", writable=True, reset_env_ids=ids)

    def abandon(self, env_ids: torch.Tensor,
                reason: str = "abandoned") -> None:
        """显式终止（不 reset）：记 "abandoned" 终止 + post_episode + 封存。

        用于诊断/取消/预算耗尽；不得冒充 timeout。ENDED 行经后续
        ``reset_rows`` 才能复用。
        """
        ep = self.state.episode
        ids = torch.as_tensor(env_ids, dtype=torch.long,
                              device=ep.episode_steps.device).reshape(-1)
        reg = ep.reason_registry
        code = reg.get(reason)
        if code is None:
            code = len(reg)
            reg[reason] = code
        live = ids[ep.world_running[ids] & ep.slot_valid[ids]]
        if live.numel():
            ep.term_reason[live] = code
            for a in range(ep.n_agents):
                ep.agent_done[live, a] = True
                ep.agent_term_reason[live, a] = code
                ep.term_pending[live, a] = True
                ep.term_pending_code[live, a] = code
                _archive_reason(ep, live, a, code)
        self._consume_terminations()

    # ------------------------------------------------------------------
    # 终止屏障 + sealed-ENDED
    # ------------------------------------------------------------------
    def _consume_terminations(self, *, freeze: bool = False) -> None:
        """phase 屏障：判 env 结束；新 ENDED 行封存 + on_post_episode。

        reason 归档发生在 ``request_termination`` 提出时刻（每 (agent,
        reason) 首次出现记 (code, action_call_index)，同 code 去重、异 code
        保序、超 K 置 ``term_history_overflow``）；本屏障不搬运 reason，
        只做：reset_request→abandoned 提议、env 级 ENDED 判定、
        状态快照封存、post_episode 调度、冻结写回。

        ``freeze=True`` 的屏障点（物理子步后 / post-action）在进入时把
        "已 ENDED/FAILED 行的漂移撤销"与 pending 检查融进同一次 host
        sync——sealed 行物理上仍被整批 advance，必须每步写回封存态
        （此前仅在 new-ENDED 分支顺带写回，无新终止的步不冻结，是
        漏写）。freeze=False 的屏障点（物理前）无漂移可撤，跳过。
        """
        import time as _time
        _t0 = _time.perf_counter()
        st = self.state
        ep = st.episode

        self.sync_stats["term_barrier"] += 1
        # W1 融合快路径：单 bool(.any()) 覆盖全部提议来源——
        # request_termination 写入的 agent_done、直写兼容口
        # terminated_flag / reset_request。有提议才进慢路径（此时
        # 内部检查次数无所谓——真实终止每 episode 至多几次）。
        running_valid = ep.world_running & ep.slot_valid
        pending = ((ep.reset_request | ep.terminated_flag
                    | ep.agent_done.all(dim=-1)) & running_valid)
        sealed = None
        if freeze and self._seal_snap is not None:
            sealed = ep.slot_valid & ~ep.world_running
            # 单次 sync 同时取 pending/sealed 两个判定（.tolist() 一次
            # D2H），避免 freeze 检查成为第二个排队排空点。
            self.sync_stats["term_barrier_syncs"] += 1
            need_term, need_freeze = torch.stack(
                [pending.any(), sealed.any()]).tolist()
            if need_freeze:
                self._freeze_ended_rows(sealed)
        else:
            self.sync_stats["term_barrier_syncs"] += 1
            need_term = bool(pending.any())
        if not need_term:
            self.barrier_time += _time.perf_counter() - _t0
            return

        # reset_request → 等价 CPU reset-while-active：对全员提 "abandoned"
        rr = ep.reset_request & running_valid
        self.sync_stats["term_barrier_syncs"] += 1
        if bool(rr.any()):
            rr_ids = torch.nonzero(rr, as_tuple=False).squeeze(-1)
            aband = TERM_CODES["abandoned"]
            ep.term_reason[rr_ids] = aband
            for a in range(ep.n_agents):
                ep.agent_done[rr_ids, a] = True
                ep.agent_term_reason[rr_ids, a] = aband
                ep.term_pending[rr_ids, a] = True
                ep.term_pending_code[rr_ids, a] = aband
                _archive_reason(ep, rr_ids, a, aband)

        # env 结束 = 全 agent done | env 级 flag 直写（插件兼容口）
        # ——仅限 RUNNING 行；ENDED/FAILED 不重复触发。
        newly = ((ep.agent_done.all(dim=-1) | ep.terminated_flag)
                 & running_valid)
        self.sync_stats["term_barrier_syncs"] += 1
        if not bool(newly.any()):
            self.barrier_time += _time.perf_counter() - _t0
            return
        ep.world_running[newly] = False
        ep.policy_eval_mask[newly] = False
        # terminated_flag = "本步新 ENDED"（步内多次屏障取并集；
        # 插件直写的旧兼容口也被并进来，屏障仍是唯一出口）
        ep.terminated_flag |= newly

        # 封存：捕获终止时刻的完整积分态（ENDED 行本步参与了 advance，
        # 其终止状态 = 刚跑完的那步——正确语义）
        snap = self.sim.capture(newly)
        if self._seal_snap is None:
            self._seal_snap = {
                k: torch.zeros(
                    (st.batch_size,) + v.shape[1:], dtype=v.dtype,
                    device=v.device)
                for k, v in snap.items()}
        for k, v in snap.items():
            self._seal_snap[k][newly] = v

        ids = torch.nonzero(newly, as_tuple=False).squeeze(-1)
        self._invoke("on_post_episode", writable=False,
                     terminated_env_ids=ids)
        # post_episode 中插件仍可提终止（request_termination 即归档；
        # 对仍 RUNNING 的其他行于下个屏障生效——CPU 无跨 env 对应物）
        # （本步新 ENDED 行的漂移撤销由下一个 freeze=True 屏障承担；
        # 刚封存的行此处 restore 是 no-op，不再顺带调用）
        self.barrier_time += _time.perf_counter() - _t0

    def _freeze_ended_rows(self, sealed: torch.Tensor) -> None:
        """把封存态写回 ENDED/FAILED 行，撤销本步 advance 造成的漂移。

        warp 无 masked-step（W0 实测），冻结 = 每 action step 边界一次
        capture/restore write-back。ENDED 行物理仍被推进但每步末被
        写回封存态 → 状态不漂移、接触有界，单行失稳不再撑爆 nconmax。
        调用方（freeze=True 屏障）已在融合检查中确认 sealed 非空且
        ``_seal_snap`` 存在。"""
        self.sync_stats["freeze_check"] += 1
        dense = {k: b[sealed] for k, b in self._seal_snap.items()}
        self.sim.restore(sealed, dense)

    def mark_failed(self, env_ids: torch.Tensor,
                    reason: str = "capacity") -> None:
        """把行标记为 FAILED（执行错误）。显式传播——不静默吞掉。

        FAILED 行同样被冻结封存；collector 应在波界检查
        ``failed_mask`` 并使 collect 失败。
        """
        ep = self.state.episode
        ids = torch.as_tensor(env_ids, dtype=torch.long,
                              device=ep.episode_steps.device).reshape(-1)
        ep.world_failed[ids] = True
        ep.world_running[ids] = False
        ep.policy_eval_mask[ids] = False
        ep.fail_reason[ids] = FAIL_CODES.get(reason, 0)
        # FAILED 行也封存：捕获当前态快照供冻结写回（state 完整性）
        st = self.state
        mask = torch.zeros(st.batch_size, dtype=torch.bool,
                           device=ep.episode_steps.device)
        mask[ids] = True
        snap = self.sim.capture(mask)
        if self._seal_snap is None:
            self._seal_snap = {
                k: torch.zeros(
                    (st.batch_size,) + v.shape[1:], dtype=v.dtype,
                    device=v.device)
                for k, v in snap.items()}
        for k, v in snap.items():
            self._seal_snap[k][mask] = v

    @property
    def failed_mask(self) -> torch.Tensor:
        return self.state.episode.world_failed

    def any_running(self) -> bool:
        """是否仍有 RUNNING 行（collector 波提前退出判断；低频）。"""
        self.sync_stats["any_running"] += 1
        ep = self.state.episode
        return bool((ep.world_running & ep.slot_valid).any())

    def step(self, actions=None,
             n_substeps: Optional[int] = None) -> None:
        """一个 action step。

        Args:
            actions: 可选 (action_a (B,21), action_b (B,21)) torch 张量；
                None 表示沿用上一动作。
            n_substeps: 物理子步数（默认构造时的 phy_substeps）。

        注意：``step()`` **不 auto-reset**（E2-W2）。ENDED 行封存；
        终止/reset 语义见模块 docstring 与 LIFECYCLE_TRACE.md。
        """
        st = self.state
        ep = st.episode
        n = int(n_substeps or self.phy_substeps)

        st.clear_step_flags()
        # 进入本步的行快照——episode_steps / action_call_index 按"进入
        # step() 的行"无条件计数（CPU 对齐：episode_step 在 step() 尾部
        # 无条件 +1，即使该步的动作从未驱动物理）。action_call_index
        # 步首即增——它是本步的 1-based 帧序号，步内任意提议点都等于
        # CPU 记录帧扫描时读到的 episode_step（_archive_reason /
        # seal_rows 的边界值来源）。
        entered = ep.world_running & ep.slot_valid
        entered_i64 = entered.to(torch.int64)
        ep.action_call_index += entered_i64
        # policy_eval_mask：ENDED 行不采样；"hold" 下已终 agent 也不采样
        run2 = entered.unsqueeze(-1)
        if self.post_termination_action == "hold":
            ep.policy_eval_mask.copy_(run2 & ~ep.agent_done)
        else:
            ep.policy_eval_mask.copy_(
                run2.expand(-1, ep.n_agents))
        if actions is not None:
            self.sim.dev_set_action(actions[0], actions[1])
        self._invoke("on_pre_action_step", writable=True)
        for c in self._ctxs.values():
            c.phy_substeps = n
        self._invoke("on_pre_batch_step", writable=True)
        # pre-action 终止屏障（CPU 对齐：on_pre_action_step / 首个
        # on_pre_phy_step 的全员终止提议在物理前结束本步——ended 行的
        # 终止帧物理增量为 0，是记录但不进轨迹的"退化帧"）
        self._consume_terminations()

        run_i64 = (ep.world_running & ep.slot_valid).to(torch.int64)
        dt = float(self.sim.DT)
        import time as _time
        _t0 = _time.perf_counter()
        if self._substep_units:
            # 有插件覆写子步 hook → 回调进后端子步循环（不能拆成
            # physical_step(1)×n——pending wrench 只在块首子步消费一次）
            subs = self._substep_units
            ep_ = ep

            def _pre(i: int) -> None:
                ep_.substep_index.fill_(i)
                self._invoke("on_pre_phy_step", writable=True, units=subs)
                # pre_phy 终止屏障（CPU 对齐：on_pre_phy_step 的提议在
                # 本子步物理前生效——ended 行由冻结写回保持不漂移，且
                # 不再累计本子步 physics_steps）
                self._consume_terminations()

            def _post(i: int) -> None:
                # 物理记账在 post_phy 检查前（CPU 对齐：physics_step
                # ++ 在终止检查之前——本子步提议终止的行仍计入本子步）
                m = ep_.world_running & ep_.slot_valid
                ep_.physics_steps += m.to(torch.int64)
                ep_.time += dt * m.to(torch.float32)
                self._invoke("on_post_phy_step", writable=True, units=subs)
                # 子步级终止屏障（CPU 对齐：每物理子步后检查 env 结束；
                # 新 ENDED 行立即封存，余下子步由冻结写回保持不漂移）
                self._consume_terminations(freeze=True)

            self.sim.physical_step(n, pre_step=_pre, post_step=_post)
        else:
            self.sim.physical_step(n)
            ep.physics_steps += run_i64 * n
            ep.time += dt * n * run_i64.to(torch.float32)
        self.seg_timing["physics_wall"] += _time.perf_counter() - _t0
        self._invoke("on_post_batch_step", writable=True)

        # episode_steps = 进入 step() 的无条件调用计数（含本步内已
        # ENDED 的行——CPU：终止帧照常 +1，terminated-flag 信息由
        # term_records/frame boundary 承担，不由计数器隐含）
        ep.episode_steps += entered_i64
        st.rng.step_counter += 1
        _t0 = _time.perf_counter()
        if self.obs_builder is not None:
            self.obs_builder.build(st)
        self.seg_timing["obs_build"] += _time.perf_counter() - _t0

        self._invoke("on_post_action_step", writable=False)
        self._consume_terminations(freeze=True)

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
        # mask 提议——热路径无 nonzero/host 读取（E9-W3），终止判定
        # 仍由随后的 term barrier 结算。
        ctx.request_termination_mask(
            ep.active_mask & (ep.episode_steps >= self._max),
            "timeout")
