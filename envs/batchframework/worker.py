"""worker — 多卡采样的 worker 进程入口（E4-W2）。

每个 worker 是一个 ``spawn`` 子进程（不 fork——CUDA context 不可
继承）。启动后先 ``torch.cuda.set_device`` 绑定本卡，再初始化
torch/warp，实例化 ``DeviceRollouter``（单卡采样实现零分叉）。

协议（mp.Queue，全 pickle 载荷——E4-W0 实测带宽富余）：

- 上行 ``{"kind": "hello", ...}``：握手自检（device/pid/版本）；
- 下行 ``("collect", collect_id, [ShardPlan, ...])``；
- 上行 ``{"kind": "collect_done", "collect_id", "results":
  [ShardResult,...]}``——每 shard 一个 ShardResult；
- 下行 ``("close",)`` → ``dr.close()`` + 退出；
- 任何异常 → ``{"kind": "error", "tb": ...}``（coordinator 判
  ``WorkerLost``）。

worker 内策略缓存沿用 PolicyExecutorCache（有界 LRU）；上行 report
携带 executor 版本集合，coordinator 侧校验一致性。
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, List

from .coordinator import ShardPlan, ShardResult


def worker_main(device_index: int, batch_size: int, cmd_q, out_q,
                policy_cache_size: int = 8) -> None:
    """spawn 入口：device_index 是物理卡序号（无遮罩直选）。"""
    try:
        import torch
        torch.cuda.set_device(device_index)
        from .device_rollouter import DeviceRollouter
        dr = DeviceRollouter(batch_size=batch_size,
                             device=f"cuda:{device_index}",
                             policy_cache_size=policy_cache_size)
    except Exception:
        out_q.put({"kind": "error", "phase": "init",
                   "device": device_index,
                   "tb": traceback.format_exc()})
        return

    out_q.put({"kind": "hello", "device": device_index,
               "phys": torch.cuda.get_device_name(device_index),
               "torch": torch.__version__,
               "pid": __import__("os").getpid()})

    while True:
        try:
            msg = cmd_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        if not isinstance(msg, tuple) or not msg:
            continue
        cmd = msg[0]
        if cmd == "close":
            break
        if cmd != "collect":
            out_q.put({"kind": "error", "phase": "dispatch",
                       "tb": f"unknown cmd {cmd!r}"})
            continue
        _, collect_id, shards = msg[:3]
        capture_req = msg[3] if len(msg) > 3 else None
        try:
            results: List[ShardResult] = []
            for shard in shards:
                assert isinstance(shard, ShardPlan)
                cap = None
                if capture_req is not None:
                    # 全局 job_ref → shard 内局部下标；out_dir 按 worker
                    # 分子目录（多卡写同一目录会互相覆盖 manifest）。
                    from .debug_capture import CaptureRequest
                    local = tuple(i for i, ref in enumerate(shard.job_refs)
                                  if ref in capture_req.job_refs)
                    if local:
                        cap = CaptureRequest(
                            job_refs=local, frames=capture_req.frames,
                            out_dir=str(
                                Path(capture_req.out_dir)
                                / f"worker{device_index}"),
                            level=capture_req.level)
                eps = dr.collect(list(shard.jobs), capture=cap)
                # episode_index = 全局 JobRef（dr.collect 给的是 shard
                # 内局部序号）
                import dataclasses
                eps = [dataclasses.replace(ep, episode_index=ref)
                       for ep, ref in zip(eps, shard.job_refs)]
                results.append(ShardResult(
                    collect_id=collect_id,
                    group_key=shard.group_key,
                    job_refs=shard.job_refs,
                    episodes=eps,
                    report=dict(dr.last_collect_report)))
            out_q.put({"kind": "collect_done", "collect_id": collect_id,
                       "device": device_index, "results": results})
        except BaseException:
            out_q.put({"kind": "error", "phase": "collect",
                       "collect_id": collect_id, "device": device_index,
                       "tb": traceback.format_exc()})
    try:
        dr.close()
    except Exception:
        out_q.put({"kind": "error", "phase": "close",
                   "device": device_index, "tb": traceback.format_exc()})
