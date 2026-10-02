"""MultiDeviceRollouter（E4 W3/W5）测试（GPU-gated，需 ≥2 卡）。

验证多卡 collect 契约：

- spawn worker × N 各自绑物理卡；collect 返回与输入同序；
- 每 JobRef 恰好一次（merge_results 保证），seed/options 归属正确；
- padding 不产出额外 Episode；
- worker kill → collect 显式 WorkerLost（不挂起/不部分返回）；
- close 幂等、worker 全 join。

注：warp 物理非逐位确定（E1-W0 实测 ~1e-7/10 步漂移），
多卡 vs 单卡**不做逐字段等价**——验契约级不变量（帧数/终止
记录/形状/有限值/job 身份），不做数组相等断言。
"""
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).resolve().parent.parent


def _gpu_count() -> int:
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


pytestmark = pytest.mark.skipif(_gpu_count() < 2,
                                reason="needs >=2 CUDA devices")

_MAX_STEPS = 32


@pytest.fixture(scope="module")
def env_bp():
    from envs.framework.parameterized_blueprint import (
        ParameterizedEnvBlueprint)
    pb = ParameterizedEnvBlueprint.load(
        project_root / "baseline/humanoid21/blueprints"
        / "standup_4stage_dense_v2_env.yaml")
    return pb.materialize(max_steps=_MAX_STEPS)


@pytest.fixture(scope="module")
def policy_bp(tmp_path_factory):
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)
    pol = TruncatedNormalPolicy(obs_dim=96, action_dim=21, hidden_dim=256,
                                device="cpu")
    dest = tmp_path_factory.mktemp("pol") / "export"
    return pol.to_blueprint(str(dest))


def _jobs(env_bp, policy_bp, n, seed0=0):
    from baseline.framework.rollout.job import Job, SamplingSpec
    return [
        Job(policy_a_bp=policy_bp, policy_b_bp=policy_bp, env_bp=env_bp,
            seed=seed0 + i,
            episode_options={"initial_distance": 2.0 + 0.05 * i},
            sampling_a=SamplingSpec(0.0), sampling_b=SamplingSpec(0.0),
            stochastic=True)
        for i in range(n)]


def _check_contract(eps, n, seed0, T=_MAX_STEPS):
    assert len(eps) == n
    for i, ep in enumerate(eps):
        assert ep.episode_index == i
        assert ep.base_seed == seed0 + i
        assert ep.episode_options["initial_distance"] == 2.0 + 0.05 * i
        assert ep.num_frames == T
        for rid in ("robot_a", "robot_b"):
            assert ep.observations[rid].shape == (T, 96)
            assert ep.actions[rid].shape == (T, 21)
            assert np.isfinite(ep.observations[rid]).all()
            assert ep.agent_termination_proposal_records[rid] == (
                ("timeout", T),)


def test_two_gpu_collect(env_bp, policy_bp):
    """2 worker × 独立物理卡，8 jobs 同序返回。"""
    from envs.batchframework.multi_rollouter import MultiDeviceRollouter
    with MultiDeviceRollouter(devices=[0, 1],
                              batch_size_per_worker=4) as mdr:
        assert len(mdr.hello) == 2
        assert {h["device"] for h in mdr.hello} == {0, 1}
        eps = mdr.collect(_jobs(env_bp, policy_bp, 8, seed0=7))
    _check_contract(eps, 8, seed0=7)


def test_two_gpu_padding_and_identity(env_bp, policy_bp):
    """末 shard 不满 batch_size → padding 不产出；job 身份不错位。"""
    from envs.batchframework.multi_rollouter import MultiDeviceRollouter
    with MultiDeviceRollouter(devices=[0, 1],
                              batch_size_per_worker=4) as mdr:
        eps = mdr.collect(_jobs(env_bp, policy_bp, 5, seed0=50))
    _check_contract(eps, 5, seed0=50)


def test_worker_death_fails_collect(env_bp, policy_bp):
    """kill 一个 worker → collect 显式 WorkerLost，不挂起。"""
    from envs.batchframework.coordinator import WorkerLost
    from envs.batchframework.multi_rollouter import MultiDeviceRollouter
    mdr = MultiDeviceRollouter(devices=[0, 1], batch_size_per_worker=4)
    try:
        # 杀掉 worker 0 的进程
        mdr._procs[0].kill()
        mdr._procs[0].join(timeout=10)
        with pytest.raises(WorkerLost):
            mdr.collect(_jobs(env_bp, policy_bp, 8))
    finally:
        mdr.close()
    # close 后所有进程已退出
    assert not any(p.is_alive() for p in mdr._procs)


def test_close_idempotent(env_bp, policy_bp):
    from envs.batchframework.multi_rollouter import MultiDeviceRollouter
    mdr = MultiDeviceRollouter(devices=[0], batch_size_per_worker=4)
    mdr.close()
    mdr.close()   # 幂等
    assert not any(p.is_alive() for p in mdr._procs)
    with pytest.raises(RuntimeError, match="closed"):
        mdr.collect(_jobs(env_bp, policy_bp, 2))
