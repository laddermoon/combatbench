"""DeviceRollouter — 设备端波次同步 episode 采集器（E3 重构）。

与 ``ParallelRollouter`` 同契约：``collect(jobs) -> List[Episode]``，
jobs 同序返回。下游（build_trajectories / PPOBuffer / GAE / dump）
对数据来源无感知。

执行模型（ROADMAP §6"同步定长模式"）：

- jobs 按**同构键**（env_bp + policy 对 + stochastic）分组，组内按
  ``batch_size`` 切波，波内 B 个 env lockstep 推进；跨组顺序执行，
  返回顺序与输入 jobs 一致；
- 波内 env 提前终止：该行进入 **sealed-ENDED**——物理封存冻结
  （E2-W2；帧截取到首个 env 终止步）；
- 热路径无逐步 host 数据依赖；波末一次批量 D2H 组装 Episode。

结构与依赖（E3）：

- 后端经 ``binding_registry.resolve(env_bp.simulator.cls)`` 解析——
  collector 不 import 具体 simulator；
- 策略经 ``PolicyExecutorCache``（有界 LRU，版本 = 权重内容 hash）；
- 波内缓冲归 ``RecordStore``（io_schema 声明形状，不写死维度）；
- Episode 组装经 ``episode_exporter``（向量化切片，
  ``from_buffer_frames`` 仅作等价测试参考）。

支持边界（不满足即 raise，不静默降级）：sampling spec 需求超出
executor 声明的 ctx 字段集 → 拒绝；blueprint 插件/observer 必须在
capability_registry 中为 NATIVE；per-agent 早停按
``post_termination_action="policy"`` 语义（CPU EpisodeRunner 默认）。
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from baseline.framework.ppo.sampling_context import SamplingContext
from baseline.framework.rollout.episode import Episode, blueprint_hash
from baseline.framework.rollout.job import Job
from envs.framework.blueprint import EnvBlueprint

from .binding_registry import resolve_binding
from .coordinator import group_jobs
from .capability_registry import resolve_observer, resolve_plugin
from .device_plugin import BaseDevicePlugin
from .device_runtime import BatchRuntime, DeviceTimeoutPlugin
from .episode_exporter import export_episode
from .policy_executor import PolicyExecutor, PolicyExecutorCache
from .record_store import ObserverLeaf, RecordStore
from .wave_runner import run_wave


# ---------------------------------------------------------------------------
# 记录插件 → RecordStore 薄适配器
# ---------------------------------------------------------------------------
class _RecorderAdapter(BaseDevicePlugin):
    """把 hook 时刻的 observer 输出/终止事件搬运进 RecordStore。

    缓冲所有权归 store（不被 partial reset 清理）。priority 最低——
    读到的是本步 dispatcher 已刷新 + timeout 已标记后的完整输出。
    """

    def __init__(self, observer_names: Sequence[str],
                 store: RecordStore):
        self._names = list(observer_names)
        self._store = store
        self._rt: Optional[BatchRuntime] = None
        self._t = 0

    @property
    def name(self):
        return "wave_recorder"

    @property
    def priority(self):
        return -10_000  # 最后执行——读到本步完整输出

    def set_runtime(self, rt: BatchRuntime) -> None:
        self._rt = rt

    def begin_wave(self) -> None:
        self._t = 0
        self._store.begin_wave()

    def on_post_action_step(self, ctx) -> None:
        outs = {n: self._rt.get_observer_output(n) for n in self._names}
        self._store.write_observer_step(self._t, outs)
        self._t += 1

    def on_post_episode(self, ctx) -> None:
        """env 结束行（本步新 ENDED）：CPU 序——插件 post_episode/observer
        刷新之后才记末帧（observer 输出取 post_episode 刷新值）。"""
        ids = ctx.terminated_env_ids
        if ids is None or len(ids) == 0:
            return
        st = self._store
        fresh = ids[st.env_term_step[ids] < 0]
        if fresh.numel() == 0:
            return
        st.seal_rows(fresh, ctx.episode, ctx.io)
        # 末帧 observer 输出用 post_episode 刷新值覆写
        outs = {n: self._rt.get_observer_output(n) for n in self._names}
        st.write_observer_step(self._t - 1, outs, rows=fresh)


# ---------------------------------------------------------------------------
# Job 校验
# ---------------------------------------------------------------------------
def _spec_ef(spec_dict: Dict[str, Any], tag: str):
    """提取 (ef_scalar, df_scalar)；类型合法性校验（能力拒绝在
    ``PolicyExecutor.check_spec``）。"""
    ef = spec_dict.get("explore_factor", 0.0)
    if callable(ef):
        raise ValueError(
            f"job[{tag}]: callable explore_factor not supported on device "
            f"collector (per-frame host evaluation required)")
    return (float(ef), float(spec_dict.get("delta_factor", 0.0)))


# ---------------------------------------------------------------------------
# DeviceRollouter
# ---------------------------------------------------------------------------
class DeviceRollouter:
    """波次同步设备采集器。``collect(jobs)`` 与 ParallelRollouter 同契约。"""

    def __init__(self, batch_size: int = 64, device: str = "cuda",
                 policy_cache_size: int = 8):
        self.batch_size = int(batch_size)
        # device 归一化为显式索引（"cuda" → "cuda:<current>"）——
        # warp/torch 的 kernel 绑定与视图都需要确定的设备 id（W5）。
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            idx = torch.cuda.current_device() if torch.cuda.is_available() \
                else 0
            dev = torch.device("cuda", idx)
        self.device = str(dev)
        self._sim = None
        self._rt: Optional[BatchRuntime] = None
        self._binding = None
        self._io_schema = None
        self._env_key: Optional[str] = None
        self._recorder: Optional[_RecorderAdapter] = None
        self._store: Optional[RecordStore] = None
        self._policies = PolicyExecutorCache(capacity=policy_cache_size)
        # 计时分解——按 collect 调用累计
        self.timing = {"reset": 0.0, "policy": 0.0, "step": 0.0,
                       "assemble": 0.0, "n_waves": 0}
        # 同步点分类记账（E3-W5）：值随每次 collect 重置
        self.sync_stats = {"early_exit_check": 0, "health_check": 0,
                           "d2h_export": 0}
        # 最近一次 collect 的资源报告（D10.3：不只报 simulator 显存）
        self.last_collect_report: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _build_runtime(self, env_bp: EnvBlueprint, T: int) -> None:
        """按 env_bp 装配设备 runtime（blueprint 变化才重建）。"""
        binding = resolve_binding(env_bp.simulator.cls)
        sim = binding.make_sim(self.batch_size, self.device)
        sim.reset()
        rt = BatchRuntime(
            sim, obs_builder=sim.device_obs_builder(),
            phy_substeps=env_bp.phy_steps_per_action)
        for spec in env_bp.plugins:
            rt.attach(resolve_plugin(spec.cls, spec.config, sim))
        if env_bp.max_steps:
            rt.attach(DeviceTimeoutPlugin(int(env_bp.max_steps)))
        for name, spec in env_bp.observer_plugins.items():
            rt.set_observer(name, resolve_observer(spec.cls, spec.config, sim))

        io_schema = binding.io_schema(sim)
        # observer schema：声明过的走严格校验；未声明的走 (B,) 兼容路径
        obs_schemas: Dict[str, Dict[str, ObserverLeaf]] = {}
        for name, spec in env_bp.observer_plugins.items():
            obs = rt.dispatcher.observers.get(name)
            declared = getattr(obs, "output_schema", None)
            if declared:
                obs_schemas[name] = {
                    leaf: ObserverLeaf(dtype=dt, shape_suffix=tuple(sfx))
                    for leaf, (dt, sfx) in declared.items()}
        store = RecordStore(io_schema, T, self.batch_size,
                            torch.device(self.device), obs_schemas)
        rec = _RecorderAdapter(list(env_bp.observer_plugins), store)
        rec.set_runtime(rt)
        rt.attach(rec)
        self._sim, self._rt, self._binding = sim, rt, binding
        self._io_schema = io_schema
        self._recorder, self._store = rec, store
        self._env_key = json.dumps(env_bp.to_dict(), sort_keys=True)

    def _executor(self, bp_dict: Dict[str, Any]) -> PolicyExecutor:
        return self._policies.get(bp_dict, self.device)

    def _policy(self, bp_dict: Dict[str, Any]):
        """→ 底层策略模块（测试/dump 兼容入口——执行请走 _executor）。"""
        return self._executor(bp_dict).policy

    # ------------------------------------------------------------------
    def collect(self, jobs: Sequence[Job]) -> List[Episode]:
        if not jobs:
            raise ValueError("jobs must not be empty")

        # 同构键分组：env_bp + policy 对 + stochastic（每波内统一；
        # 跨组顺序执行，返回顺序=输入顺序）——键函数与多卡
        # coordinator 共用（coordinator.homogeneous_key）。
        groups = group_jobs(jobs)
        for k in self.sync_stats:
            self.sync_stats[k] = 0

        episodes: List[Optional[Episode]] = [None] * len(jobs)
        for key, idxs in groups.items():
            g_jobs = [jobs[i] for i in idxs]
            env_bp = g_jobs[0].env_bp
            env_key = key[0]
            T = int(env_bp.max_steps or 0)
            if T <= 0:
                raise ValueError(
                    "device collector requires env_bp.max_steps "
                    "(wave-synchronous fixed horizon)")
            if self._env_key != env_key:
                self._teardown()
                self._build_runtime(env_bp, T)
            rt = self._rt

            exec_a = self._executor(g_jobs[0].policy_a_bp.to_dict())
            exec_b = (exec_a if g_jobs[0].policy_a_bp.to_dict()
                      == g_jobs[0].policy_b_bp.to_dict()
                      else self._executor(
                          g_jobs[0].policy_b_bp.to_dict()))
            stochastic = bool(g_jobs[0].stochastic)
            # 能力检查：spec 需要的 ctx 字段 ⊆ executor 声明
            for gi, j in enumerate(g_jobs):
                exec_a.check_spec(j.sampling_a.to_dict(), f"{gi}/a")
                exec_b.check_spec(j.sampling_b.to_dict(), f"{gi}/b")
            efs = [(_spec_ef(j.sampling_a.to_dict(), f"{gi}/a"),
                    _spec_ef(j.sampling_b.to_dict(), f"{gi}/b"))
                   for gi, j in enumerate(g_jobs)]

            B = self.batch_size
            for w0 in range(0, len(g_jobs), B):
                wave_idx = idxs[w0:w0 + B]
                self._run_wave(g_jobs[w0:w0 + B], wave_idx,
                              efs[w0:w0 + B], exec_a, exec_b,
                              stochastic, env_bp, rt, T, episodes)

        self._write_collect_report()
        return [e for e in episodes]

    # ------------------------------------------------------------------
    def _run_wave(self, wave, wave_idx, efs, exec_a, exec_b, stochastic,
                 env_bp, rt, T, episodes):
        sim, rec, store = self._sim, self._recorder, self._store
        B, n = self.batch_size, len(wave)
        st = rt.state
        dev = torch.device(self.device)

        # per-env ef → (B,) ctx 张量（pad 行填末值）
        ef_pad = list(efs) + [efs[-1]] * max(0, B - n)
        ef_a_t = torch.tensor([e[0][0] for e in ef_pad],
                              dtype=torch.float32, device=dev)
        ef_b_t = torch.tensor([e[1][0] for e in ef_pad],
                              dtype=torch.float32, device=dev)
        ctx_a = SamplingContext(explore_factor=ef_a_t)
        ctx_b = SamplingContext(explore_factor=ef_b_t)
        # self-play 合并前向时 obs 是 (2B,·)——ctx 也要合并
        ctx_ab = SamplingContext(
            explore_factor=torch.cat([ef_a_t, ef_b_t]))

        t0 = time.perf_counter()
        seeds = torch.tensor(
            [j.seed for j in wave] + [wave[-1].seed] * (B - n),
            dtype=torch.int64, device=dev)
        # per-env options：白名单键做 (B,) 广播，未知键拒绝；
        # 同一键必须全波都在（混合缺省语义不明确，拒绝而非猜测）
        opt_keys = {k for j in wave for k in (j.episode_options or {})}
        unknown = opt_keys - set(self._binding.episode_options_keys)
        if unknown:
            raise ValueError(
                f"episode_options keys {sorted(unknown)} not supported "
                f"by device binding {self._binding.name!r}")
        options = {}
        for k in opt_keys:
            vals = [(j.episode_options or {}).get(k) for j in wave]
            if any(v is None for v in vals):
                raise ValueError(
                    f"episode_options[{k!r}] must be set for all jobs "
                    f"or none")
            options[k] = np.asarray(vals + [vals[0]] * (B - n))
        rt.reset(seeds=seeds, options=options or None)
        rt.obs_builder.build(st)          # obs_0
        self.timing["reset"] += time.perf_counter() - t0

        rec.begin_wave()
        run_wave(rt, exec_a, exec_b, store, stochastic=stochastic,
                 ctx_a=ctx_a, ctx_b=ctx_b, ctx_ab=ctx_ab,
                 T=T, timing=self.timing, sync=self.sync_stats)
        self.timing["n_waves"] += 1

        # 波界健康检查：FAILED 行（容量溢出/非有限态）显式失败
        self.sync_stats["health_check"] += 1
        if bool(rt.failed_mask.any()):
            frows = torch.nonzero(rt.failed_mask).squeeze(-1).tolist()
            freasons = rt.state.episode.fail_reason[
                rt.failed_mask].tolist()
            raise RuntimeError(
                f"device collect failed: FAILED rows {frows} "
                f"reasons={freasons}")

        # --- 波末组装：一次批量 D2H + 向量化导出 ---
        t0 = time.perf_counter()
        store.finalize(st.episode, st.io)
        self.sync_stats["d2h_export"] += 1
        agent_ids = self._io_schema.agent_ids
        np_bufs = dict(
            obs={rid: store.obs[rid].cpu().numpy() for rid in agent_ids},
            act={rid: store.act[rid].cpu().numpy() for rid in agent_ids},
            log_prob={rid: store.log_prob[rid].cpu().numpy()
                      for rid in agent_ids},
            final_obs={rid: store.final_obs[rid].cpu().numpy()
                       for rid in agent_ids},
            obs_out={name: {k: v.cpu().numpy() for k, v in bufs.items()}
                     for name, bufs in store.obs_out.items()},
            env_term=store.env_term_step.cpu().numpy(),
        )
        # 插件 episode 指标——显式 schema 导出（W4），不嗅探 pool 键名
        metrics_np = {k: v.detach().cpu().numpy()
                      for k, v in rt.export_episode_metrics().items()}

        env_hash = blueprint_hash(env_bp)
        for i, (job, gi) in enumerate(zip(wave, wave_idx)):
            episodes[gi] = export_episode(
                np_bufs, job=job, ep_index=gi, row=i,
                agent_ids=agent_ids, ef_pair=efs[i],
                stochastic=stochastic, env_hash=env_hash,
                term_records=store.term_records[i],
                metrics_np=metrics_np, T=T)
        self.timing["assemble"] += time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _write_collect_report(self) -> None:
        if self._store is None:
            return
        rt_sync = (dict(self._rt.sync_stats) if self._rt is not None
                   else {})
        self.last_collect_report = {
            "record_store_bytes": self._store.n_bytes(),
            "policy_versions": sorted({ex.version for _, (_, ex)
                                       in self._policies._items.items()}),
            "binding": self._binding.name if self._binding else None,
            "sync_stats": {**self.sync_stats, **{
                f"rt.{k}": v for k, v in rt_sync.items()}},
        }

    # ------------------------------------------------------------------
    def _teardown(self):
        if self._sim is not None:
            close = getattr(self._sim, "close", None)
            if callable(close):
                close()
        self._sim = self._rt = self._recorder = self._store = None
        self._binding = self._io_schema = None
        self._env_key = None

    def close(self):
        self._teardown()
        self._policies.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False
