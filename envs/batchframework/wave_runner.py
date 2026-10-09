"""wave_runner — 后端无关的波次采样循环（E3-W3）。

把 `collect` 的热循环从 `DeviceRollouter` 中提出：入参只有
``BatchRuntime`` + 两个 ``PolicyExecutor`` + ``RecordStore`` +
每 agent 的 SamplingContext——**不依赖 warp**，FakeBackend 可注入
（混长波/早退/FAILED 契约测试因此无 GPU 可测）。

语义不变量（与 CPU EpisodeRunner 对齐）：

- 每步顺序：写帧输入 obs → executor.act → rt.step（含屏障/observer
  刷新/记录 hook）→ 视情况早退；
- 全 ENDED 早退只是省算力——ENDED 行已封存，多走几步不改变
  导出数据（``env_term_step``/``final_obs`` 在屏障时刻已定）。
  ``check_every`` 控制检查频率（每 k 步而非每步一次 host sync）；
- ``stochastic=False`` 波不传 ctx、不记录 log_prob（CPU 语义）。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from .device_runtime import BatchRuntime
from .device_state import _SEED_COUNTER_MULT, rng_uniform
from .policy_executor import PolicyExecutor
from .record_store import RecordStore
from baseline.framework.ppo.sampling_context import SamplingContext

# 策略噪声注入盐（E6-W1）：per-agent 独立流，与插件 rng_salt 域分离。
_U_SALT_A = 0xAC710A
_U_SALT_B = 0xAC710B


def run_wave(rt: BatchRuntime,
             exec_a: PolicyExecutor,
             exec_b: PolicyExecutor,
             store: RecordStore,
             *,
             stochastic: bool,
             ctx_a=None, ctx_b=None, ctx_ab=None,
             ef_sched_a=None, ef_sched_b=None,
             T: int,
             timing: Optional[Dict[str, float]] = None,
             sync: Optional[Dict[str, int]] = None,
             check_every: int = 8,
             step_hook=None) -> None:
    """一个 wave：lockstep T 步或全 ENDED 早退。

    ``exec_a is exec_b``（self-play 共享）时传 ``ctx_ab``（合并前向）。
    ``ef_sched_*``：声明式 ef 程序的逐帧调度器（``_EfSchedule``，见
    device_rollouter）——提供后覆盖静态 ctx，每步对当前 obs 求值；
    双 schedule 必须同时给或同时不给。
    ``timing`` 收集 "policy"/"step" 段耗时；``check_every`` 控制
    ``any_running()`` host 检查频率（语义无关——ENDED 行已封存）。
    """
    st = rt.state
    B = st.batch_size
    dev = st.sim.qpos.device
    shared = exec_a is exec_b
    ep = st.episode

    for t in range(T):
        store.write_inputs(t, st, ep.world_running & ep.slot_valid)
        tp = time.perf_counter()
        u_a = u_b = u_ab = None
        if stochastic:
            # E6-W1：动作噪声 job-keyed——f(job_seed, episode_step, salt)。
            # 同一 job 单独重跑/任意分片得到相同外生流；已 ENDED 行的
            # u 照常计算（动作不驱动物理，无害）。
            base = (st.rng.seed_offsets
                    + ep.episode_steps.to(torch.int64)
                    * _SEED_COUNTER_MULT)
            wa = exec_a.noise_cols(st.io.action_a.shape[-1])
            wb = exec_b.noise_cols(st.io.action_b.shape[-1])
            u_a = rng_uniform(base + _U_SALT_A, wa)
            u_b = rng_uniform(base + _U_SALT_B, wb)
            if shared:
                u_ab = torch.cat([u_a, u_b], dim=0)
        # 声明式 ef 程序：逐帧对当前 obs 求值（设备侧向量化）；
        # 静态路径不受影响——sched 为 None 时沿用传入的常量 ctx。
        c_a, c_b, c_ab = ctx_a, ctx_b, ctx_ab
        if stochastic and ef_sched_a is not None:
            efa = ef_sched_a.eval(st.io.obs_a)
            efb = ef_sched_b.eval(st.io.obs_b)
            c_a = SamplingContext(explore_factor=efa)
            c_b = SamplingContext(explore_factor=efb)
            if shared:
                c_ab = SamplingContext(
                    explore_factor=torch.cat([efa, efb]))
        a_a, a_b, lp_a, lp_b = exec_a.act(
            st.io.obs_a, st.io.obs_b, stochastic=stochastic,
            ctx_a=c_a, ctx_b=c_b, ctx_ab=c_ab,
            shared=shared, u_a=u_a, u_b=u_b, u_ab=u_ab)
        if timing is not None:
            timing["policy"] += time.perf_counter() - tp
        ef_a = ef_b = None
        if stochastic and c_a is not None:
            ef_a, ef_b = c_a.explore_factor, c_b.explore_factor
        store.write_actions(
            t, a_a, a_b, lp_a, lp_b, ef_a=ef_a, ef_b=ef_b)
        tp = time.perf_counter()
        rt.step((a_a, a_b))
        if timing is not None:
            timing["step"] += time.perf_counter() - tp
        if step_hook is not None:
            step_hook(t, rt)          # E6-W2 按需捕获：帧 t 已写完整
        if (t + 1) % check_every == 0 or t == T - 1:
            if sync is not None:
                sync["early_exit_check"] += 1
                sync["health_scan"] = sync.get("health_scan", 0) + 1
            _health_scan(rt)
            if not rt.any_running():
                break


def _health_scan(rt: BatchRuntime) -> None:
    """E6-W5：容量溢出与非有限态检测（挂在既有检查 cadence 上）。

    - contacts：per-world 活跃接触数 ≥ cap → 该 world 静默截断可能
      （warp 全局 capped，per-world 饱和是唯一可检信号）→ FAILED(capacity)；
    - 非有限：qpos/qvel 出现 NaN/Inf → FAILED(non_finite)。
    行级隔离，不影响其他行；FAILED 行在波末健康检查显式报错。
    """
    st = rt.state
    ep = st.episode
    running = ep.world_running & ep.slot_valid
    if not bool(running.any()):
        return

    cf = getattr(st.sim, "contacts_flat", None)
    if cf is not None and cf.worldid is not None:
        n = int(cf.n_active.reshape(()).item())  # 已在 sync 点，无额外成本
        if n > 0:
            counts = torch.bincount(
                cf.worldid[:n].long(), minlength=st.batch_size)
            over = (counts >= cf.cap_per_world) & running
            if bool(over.any()):
                rt.mark_failed(
                    torch.nonzero(over).squeeze(-1), "capacity")

    bad = (~torch.isfinite(st.sim.qpos).all(dim=-1)
           | ~torch.isfinite(st.sim.qvel).all(dim=-1)) & running
    if bool(bad.any()):
        rt.mark_failed(torch.nonzero(bad).squeeze(-1), "non_finite")
