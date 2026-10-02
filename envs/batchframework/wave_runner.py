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
from .policy_executor import PolicyExecutor
from .record_store import RecordStore


def run_wave(rt: BatchRuntime,
             exec_a: PolicyExecutor,
             exec_b: PolicyExecutor,
             store: RecordStore,
             *,
             stochastic: bool,
             ctx_a=None, ctx_b=None, ctx_ab=None,
             T: int,
             timing: Optional[Dict[str, float]] = None,
             sync: Optional[Dict[str, int]] = None,
             check_every: int = 8) -> None:
    """一个 wave：lockstep T 步或全 ENDED 早退。

    ``exec_a is exec_b``（self-play 共享）时传 ``ctx_ab``（合并前向）。
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
        a_a, a_b, lp_a, lp_b = exec_a.act(
            st.io.obs_a, st.io.obs_b, stochastic=stochastic,
            ctx_a=ctx_a, ctx_b=ctx_b, ctx_ab=ctx_ab,
            shared=shared)
        if timing is not None:
            timing["policy"] += time.perf_counter() - tp
        store.write_actions(t, a_a, a_b, lp_a, lp_b)
        tp = time.perf_counter()
        rt.step((a_a, a_b))
        if timing is not None:
            timing["step"] += time.perf_counter() - tp
        if (t + 1) % check_every == 0 or t == T - 1:
            if sync is not None:
                sync["early_exit_check"] += 1
            if not rt.any_running():
                break
