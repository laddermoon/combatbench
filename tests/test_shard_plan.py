"""E4-W5 分片协议契约测试（无 GPU）。

``plan_shards``/``merge_results`` 是纯函数——确定性、覆盖性、
校验拒绝路径都在这里守住。
"""
from types import SimpleNamespace

import pytest

from envs.batchframework.coordinator import (
    ShardError, ShardResult, group_jobs, merge_results, plan_shards)


class _BP:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


def _job(i, env="e1", pa="p1", pb="p1", st=True):
    return SimpleNamespace(env_bp=_BP({"e": env}),
                           policy_a_bp=_BP({"p": pa}),
                           policy_b_bp=_BP({"p": pb}),
                           stochastic=st, seed=i,
                           episode_options={})


class _Ep:
    def __init__(self, n=5):
        self.num_frames = n
        self.final_observation = {"a": 1}


def _mk_results(plans, cid=1):
    return [ShardResult(collect_id=cid, group_key=s.group_key,
                        job_refs=s.job_refs,
                        episodes=[_Ep() for _ in s.job_refs],
                        report={})
            for ps in plans for s in ps]


def test_grouping_order_preserved():
    jobs = [_job(i, env="a") for i in range(3)] + \
           [_job(10 + i, env="b") for i in range(2)] + \
           [_job(20, env="a")]
    groups = group_jobs(jobs)
    # 同构键分桶保输入序（注意 job20 也归 a 组，不与前面连号）
    assert len(groups) == 2
    keys = list(groups)
    assert groups[keys[0]] == [0, 1, 2, 5]
    assert groups[keys[1]] == [3, 4]


def test_shards_cover_all_refs_once():
    jobs = [_job(i) for i in range(7)]
    for nw in (1, 2, 3, 4, 8):
        plans = plan_shards(jobs, nw, collect_id=1)
        refs = sorted(r for ps in plans for s in ps for r in s.job_refs)
        assert refs == list(range(7))
        assert len(plans) == nw
        # 连续块、保序
        for ps in plans:
            for s in ps:
                assert list(s.job_refs) == sorted(s.job_refs)


def test_shards_deterministic():
    jobs = [_job(i, env="e" + str(i % 2)) for i in range(10)]
    a = plan_shards(jobs, 3, collect_id=1)
    b = plan_shards(jobs, 3, collect_id=1)
    assert [[s.job_refs for s in ps] for ps in a] == \
           [[s.job_refs for s in ps] for ps in b]


def test_shard_count_min_workers_jobs():
    # 3 jobs / 4 workers → 只有 3 个 worker 拿到 shard
    plans = plan_shards([_job(i) for i in range(3)], 4, collect_id=1)
    nonempty = [ps for ps in plans if ps]
    assert len(nonempty) == 3


def test_merge_reorders_to_input():
    jobs = [_job(i) for i in range(6)]
    plans = plan_shards(jobs, 2, collect_id=9)
    res = _mk_results(plans, cid=9)
    res.reverse()   # 乱序到达也应按 job_ref 重排
    eps = merge_results(res, 6, collect_id=9)
    assert len(eps) == 6


def test_merge_rejects_duplicate_missing_stale():
    jobs = [_job(i) for i in range(4)]
    plans = plan_shards(jobs, 2, collect_id=1)
    res = _mk_results(plans)
    # 重复
    with pytest.raises(ShardError, match="twice"):
        merge_results(res + [res[0]], 4, collect_id=1)
    # 缺失
    with pytest.raises(ShardError, match="missing"):
        merge_results(res[:-1], 4, collect_id=1)
    # 旧 collect_id
    bad = [ShardResult(collect_id=99, group_key="g", job_refs=(0,),
                       episodes=[_Ep()], report={})]
    with pytest.raises(ShardError, match="collect_id"):
        merge_results(bad, 4, collect_id=1)
    # 空帧 episode
    bad2 = _mk_results(plans)
    bad2[0].episodes[0].num_frames = 0
    with pytest.raises(ShardError, match="num_frames"):
        merge_results(bad2, 4, collect_id=1)


def test_zero_jobs():
    assert all(not ps for ps in plan_shards([], 2, collect_id=1))
