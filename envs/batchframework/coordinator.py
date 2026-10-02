"""coordinator — 多卡 collect 的分片协议（E4-W1）。

纯函数/纯数据层：**不碰 CUDA、不起进程**——可独立单测。

职责（D12.2）：

- ``homogeneous_key(job)``：同构键（env_bp + policy 对 + stochastic）
  ——E3 collect 的分组逻辑提到这里，``DeviceRollouter`` 与
  coordinator 共用单一实现；
- ``ShardPlan``：一个 worker 一次 collect 的输入——同构组内按
  输入序切连续块，JobRef = 全局输入下标；
- ``plan_shards``：jobs → 每 worker 的 shard 列表（确定性）；
- ``merge_results``：shard 结果 → 按输入序重排的 Episode 列表 +
  完整性校验（每 JobRef 恰好一次、collect_id 一致、版本一致、
  Episode schema 合法）。任一违例 → 整个 collect 失败。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from baseline.framework.rollout.episode import Episode
from baseline.framework.rollout.job import Job


class ShardError(RuntimeError):
    """分片/合并契约违例——collect 整体失败的显式错误。"""


class WorkerLost(ShardError):
    """worker 进程死亡/失联/上报错误——collect 整体失败（D13.1）。"""


# ---------------------------------------------------------------------------
# 同构键（E3 collect 内联逻辑的单一实现）
# ---------------------------------------------------------------------------
def homogeneous_key(job: Job) -> Tuple:
    """同构键：同键 job 可共波（同 env_bp + 同 policy 对 + 同
    stochastic）。序列化形式稳定可比较。"""
    return (json.dumps(job.env_bp.to_dict(), sort_keys=True),
            json.dumps(job.policy_a_bp.to_dict(), sort_keys=True),
            json.dumps(job.policy_b_bp.to_dict(), sort_keys=True),
            bool(job.stochastic))


def group_jobs(jobs: Sequence[Job]) -> "Dict[Tuple, List[int]]":
    """→ {同构键: [输入下标,...]}（保持输入序）。"""
    groups: Dict[Tuple, List[int]] = {}
    for i, j in enumerate(jobs):
        groups.setdefault(homogeneous_key(j), []).append(i)
    return groups


def _group_key_hash(key: Tuple) -> str:
    return hashlib.sha256(
        "|".join(str(x) for x in key).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Shard 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShardPlan:
    """发给一个 worker 的工作单元（同构组内的一段连续 job 块）。"""

    collect_id: int
    group_key: str                  # 同构键 hash（诊断/manifest）
    job_refs: Tuple[int, ...]       # 全局输入下标（JobRef）
    jobs: Tuple[Job, ...]           # 载荷（可 pickle）


@dataclass
class ShardResult:
    """worker 上行：一个 shard 的结果 + 资源/版本报告。"""

    collect_id: int
    group_key: str
    job_refs: Tuple[int, ...]
    episodes: List[Episode]
    report: Mapping[str, Any]       # policy_versions/store_bytes/…


# ---------------------------------------------------------------------------
# 分片规划（确定性）
# ---------------------------------------------------------------------------
def plan_shards(jobs: Sequence[Job], n_workers: int,
                collect_id: int = 0) -> List[List[ShardPlan]]:
    """jobs → ``[worker][shard]`` 计划。

    - 按同构键分组（保持输入序）；
    - 每组按输入序切成至多 ``n_workers`` 个**连续块**，第 i 块给
      worker i（组小于 worker 数时末尾 worker 本组无 shard——
      首版静态分片，不做负载均衡）；
    - 返回长度恒为 ``n_workers``（无活 worker 得空列表）。
    """
    if n_workers < 1:
        raise ShardError("n_workers must be >= 1")
    groups = group_jobs(jobs)
    per_worker: List[List[ShardPlan]] = [[] for _ in range(n_workers)]
    for key, idxs in groups.items():
        n = len(idxs)
        n_chunks = min(n_workers, n)
        # 连续块尽量均匀：前 r 块多一个
        base, rem = divmod(n, n_chunks)
        pos = 0
        for w in range(n_chunks):
            cnt = base + (1 if w < rem else 0)
            chunk = tuple(idxs[pos:pos + cnt])
            pos += cnt
            per_worker[w].append(ShardPlan(
                collect_id=collect_id,
                group_key=_group_key_hash(key),
                job_refs=chunk,
                jobs=tuple(jobs[i] for i in chunk)))
    return per_worker


# ---------------------------------------------------------------------------
# 结果合并 + 完整性校验
# ---------------------------------------------------------------------------
def merge_results(results: Sequence[ShardResult], n_jobs: int,
                  collect_id: int) -> List[Episode]:
    """shard 结果 → 输入序 Episode 列表。任一违例 → ShardError。

    校验（D12.2.6）：
    - 每个返回的 job_ref 合法（0<=ref<n_jobs）、无重复；
    - 全覆盖：n_jobs 个 ref 恰好各一次；
    - collect_id 一致；
    - Episode schema 基础合法（num_frames>0、final_observation
      含全部 agent）。
    """
    seen: Dict[int, Episode] = {}
    versions: set = set()
    for res in results:
        if res.collect_id != collect_id:
            raise ShardError(
                f"stale/mismatched collect_id {res.collect_id} "
                f"(expected {collect_id})")
        if len(res.episodes) != len(res.job_refs):
            raise ShardError(
                f"shard {res.group_key}: {len(res.episodes)} episodes "
                f"for {len(res.job_refs)} job_refs")
        for ref, ep in zip(res.job_refs, res.episodes):
            if not (0 <= ref < n_jobs):
                raise ShardError(f"job_ref {ref} out of range "
                                 f"(n_jobs={n_jobs})")
            if ref in seen:
                raise ShardError(f"job_ref {ref} delivered twice")
            if ep.num_frames <= 0:
                raise ShardError(
                    f"job_ref {ref}: num_frames={ep.num_frames}<=0")
            if not ep.final_observation:
                raise ShardError(
                    f"job_ref {ref}: missing final_observation")
            seen[ref] = ep
        vs = (res.report or {}).get("policy_versions")
        if vs:
            versions.update(tuple(vs))
    if len(seen) != n_jobs:
        missing = sorted(set(range(n_jobs)) - set(seen))
        raise ShardError(f"missing job_refs: {missing[:8]}"
                         f"{'…' if len(missing) > 8 else ''} "
                         f"({len(seen)}/{n_jobs} delivered)")
    return [seen[i] for i in range(n_jobs)]
