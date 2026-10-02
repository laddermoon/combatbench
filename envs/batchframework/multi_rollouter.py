"""MultiDeviceRollouter — 多卡 collect facade（E4-W3）。

对外契约与 ``DeviceRollouter``/``ParallelRollouter`` 相同：
``collect(jobs) -> List[Episode]``，输入序返回；``close()`` 幂等。

事务模型（D12.2 / D13.1）：

- coordinator（本类）固定输入序 → ``plan_shards`` 确定性分片 →
  每 worker 一列 ``ShardPlan`` → 等齐 → ``merge_results`` 校验；
- 任一 worker error/死亡/违例 → 整个 collect 抛 ``WorkerLost``/
  ``ShardError``——不返回部分结果、不自动重试；
- 背压：cmd_q ``maxsize=1``——上一次 collect 未完成不派新事务
  （同步 on-policy 语义）；
- close 顺序：下发 close → worker 内 ``dr.close()`` → join（超时
  → ``terminate`` 并报告资源清理失败）。

单设备请直接用进程内 ``DeviceRollouter``——本类至少 1 worker，
``force_workers=True`` 时单设备也走进程（故障隔离/等价测试用）。
"""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Dict, List, Optional, Sequence

from baseline.framework.rollout.episode import Episode
from baseline.framework.rollout.job import Job

from .debug_capture import CaptureRequest
from .coordinator import (
    ShardError, ShardPlan, ShardResult, WorkerLost,
    merge_results, plan_shards)
from .worker import worker_main


class MultiDeviceRollouter:
    """多 GPU worker 采集器。"""

    def __init__(self, devices: Sequence[int],
                 batch_size_per_worker: int = 64,
                 policy_cache_size: int = 8,
                 hello_timeout: float = 300.0):
        if not devices:
            raise ValueError("devices must be non-empty")
        self.devices = [int(d) for d in devices]
        self.batch_size = int(batch_size_per_worker)
        self._ctx = mp.get_context("spawn")
        self._procs: List[mp.Process] = []
        self._cmd_qs: List[Any] = []
        self._out_qs: List[Any] = []
        self._collect_id = 0
        self._closed = False
        self.last_collect_report: Dict[str, Any] = {}
        self.timing = {"plan": 0.0, "collect_wait": 0.0, "merge": 0.0,
                       "n_collects": 0}
        for dev in self.devices:
            cq = self._ctx.Queue(maxsize=1)
            oq = self._ctx.Queue()
            p = self._ctx.Process(
                target=worker_main,
                args=(dev, self.batch_size, cq, oq, policy_cache_size),
                daemon=True)
            p.start()
            self._procs.append(p)
            self._cmd_qs.append(cq)
            self._out_qs.append(oq)
        # 启动握手自检：每个 worker 报 hello（含物理卡名）；
        # init 失败的 worker 上行 error → 立即 WorkerLost。
        self.hello: List[Dict[str, Any]] = []
        for i, (oq, p) in enumerate(zip(self._out_qs, self._procs)):
            try:
                msg = oq.get(timeout=hello_timeout)
            except Exception as e:
                self._teardown()
                raise WorkerLost(
                    f"worker cuda:{self.devices[i]} handshake timeout/"
                    f"dead ({e}); alive={p.is_alive()}") from e
            if msg.get("kind") != "hello":
                self._teardown()
                raise WorkerLost(
                    f"worker cuda:{self.devices[i]} init failed: "
                    f"{msg.get('tb', msg)}")
            self.hello.append(msg)

    # ------------------------------------------------------------------
    def collect(self, jobs: Sequence[Job],
                capture: Optional["CaptureRequest"] = None
                ) -> List[Episode]:
        if self._closed:
            raise RuntimeError("MultiDeviceRollouter is closed")
        if not jobs:
            raise ValueError("jobs must not be empty")
        self._collect_id += 1
        cid = self._collect_id
        t0 = time.perf_counter()
        plans = plan_shards(jobs, len(self._procs), collect_id=cid)
        self.timing["plan"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for w, shards in enumerate(plans):
            self._cmd_qs[w].put(("collect", cid, shards, capture))
        pending = set(range(len(self._procs)))
        results: List[ShardResult] = []
        # 等齐：轮询 queue + 存活检查（死进程 → WorkerLost）
        while pending:
            progressed = False
            for w in list(pending):
                try:
                    msg = self._out_qs[w].get_nowait()
                except Exception:
                    msg = None
                if msg is None:
                    if not self._procs[w].is_alive():
                        raise WorkerLost(
                            f"worker cuda:{self.devices[w]} died "
                            f"during collect {cid} "
                            f"(exitcode={self._procs[w].exitcode})")
                    continue
                progressed = True
                if msg.get("kind") == "collect_done":
                    if msg["collect_id"] != cid:
                        raise ShardError(
                            f"worker cuda:{self.devices[w]} returned "
                            f"collect_id={msg['collect_id']} "
                            f"(expected {cid})")
                    results.extend(msg["results"])
                    pending.discard(w)
                elif msg.get("kind") == "error":
                    raise WorkerLost(
                        f"worker cuda:{self.devices[w]} collect {cid} "
                        f"failed:\n{msg.get('tb', msg)}")
                else:
                    raise ShardError(
                        f"worker cuda:{self.devices[w]} unexpected "
                        f"message {msg.get('kind')!r}")
            if pending and not progressed:
                time.sleep(0.01)
        self.timing["collect_wait"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        episodes = merge_results(results, len(jobs), cid)
        self.timing["merge"] += time.perf_counter() - t0
        self.timing["n_collects"] += 1
        # per-worker 报告汇总（版本集合一致性由 merge 保证覆盖性；
        # 版本一致性由同 bp 路径 + executor hash 语义保证）
        self.last_collect_report = {
            "collect_id": cid,
            "workers": len(self._procs),
            "devices": list(self.devices),
            "per_worker_reports": [
                r.report for r in results],
            "n_shards": len(results),
        }
        return episodes

    # ------------------------------------------------------------------
    def _teardown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for cq in self._cmd_qs:
            try:
                cq.put_nowait(("close",))
            except Exception:
                pass
        deadline = time.perf_counter() + 30.0
        for p in self._procs:
            p.join(timeout=max(0.1, deadline - time.perf_counter()))
        stuck = [p for p in self._procs if p.is_alive()]
        for p in stuck:
            p.terminate()
        for p in stuck:
            p.join(timeout=5.0)
        still = [p.pid for p in self._procs if p.is_alive()]
        if still:
            print(f"[MultiDeviceRollouter] WARNING: workers still "
                  f"alive after terminate: {still}", flush=True)

    def close(self) -> None:
        self._teardown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False
