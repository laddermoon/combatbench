"""DeviceRollouter（M5 W1/W2）测试（GPU-gated）。

验证设备端采集器与 ``ParallelRollouter`` 的 Episode 契约等价：

- collect(jobs) -> List[Episode]，job 同序返回；
- Episode 结构与 CPU `_run_job` 产出同构（帧数/终止记录/extras
  含 sctx__ 字段/observer 输出/final_observation/options）；
- rollout 记录的 log_prob 与训练侧 evaluate_actions 重算一致
  （采样分布 = 训练分布的可证伪检查）；
- 实验的 build_trajectories + PPOBuffer 管线无差别消费；
- padding 波丢弃填充行且保持 job 顺序；不支持的 spec 显式拒绝。
"""
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).resolve().parent.parent


def _gpu_ok() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_ok(), reason="CUDA unavailable")

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
    """当前代码格式的 file: 策略导出（随机初始化权重即可——
    本测试验契约不验行为）。"""
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
        for i in range(n)
    ]


def _assert_episode_contract(ep, T=_MAX_STEPS):
    from baseline.framework.rollout.episode import Episode
    assert isinstance(ep, Episode)
    assert ep.num_frames == T
    for rid in ("robot_a", "robot_b"):
        assert ep.observations[rid].shape == (T, 96)
        assert ep.actions[rid].shape == (T, 21)
        assert ep.final_observation[rid].shape == (96,)
        assert ep.explore_factors[rid].shape == (T,)
        ex = ep.action_extras[rid]
        for k in ("log_prob", "explore_factor",
                  "sctx__delta_factor"):
            assert k in ex and ex[k].shape == (T,)
        assert np.isfinite(ex["log_prob"]).all()
        assert ep.agent_termination_proposal_records[rid] == (
            ("timeout", T),)
    for name in ("standing_balance_a", "standing_balance_b"):
        out = ep.observer_outputs[name]
        for k in ("potential", "stage", "h_torso", "f_score",
                  "contact_score", "d_score", "w_foot"):
            # CPU 侧标量叶子堆叠为 list（每帧 Python float）——保持同构
            v = np.asarray(out[k], dtype=np.float64)
            assert v.shape == (T,)
            assert np.isfinite(v).all()
        pot = np.asarray(out["potential"], dtype=np.float64)
        assert np.all((pot >= 0.0) & (pot <= 1.0))


def test_collect_episode_contract(env_bp, policy_bp):
    from envs.batchframework.device_rollouter import DeviceRollouter
    with DeviceRollouter(batch_size=4) as dr:
        eps = dr.collect(_jobs(env_bp, policy_bp, 4))
    assert len(eps) == 4
    for i, ep in enumerate(eps):
        assert ep.episode_index == i
        assert ep.base_seed == i
        assert ep.episode_options["initial_distance"] == 2.0 + 0.05 * i
        _assert_episode_contract(ep)


def test_padding_wave_order(env_bp, policy_bp):
    """末波不足 B 时填充行不产出 episode，job 顺序不变。"""
    from envs.batchframework.device_rollouter import DeviceRollouter
    with DeviceRollouter(batch_size=4) as dr:
        eps = dr.collect(_jobs(env_bp, policy_bp, 5, seed0=100))
    assert len(eps) == 5
    for i, ep in enumerate(eps):
        assert ep.episode_index == i and ep.base_seed == 100 + i
        assert ep.episode_options["initial_distance"] == 2.0 + 0.05 * i
        _assert_episode_contract(ep)


def test_logprob_replay_parity(env_bp, policy_bp):
    """rollout 记录 log_prob 与 evaluate_actions(ctx 重放) 一致。"""
    from envs.batchframework.device_rollouter import DeviceRollouter
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)
    from baseline.framework.ppo.sampling_context import SamplingContext

    jobs = _jobs(env_bp, policy_bp, 2)
    with DeviceRollouter(batch_size=2) as dr:
        eps = dr.collect(jobs)
        # 与采集器同一权重对象——保证 replay 在 theta_rollout 下评估
        actor = dr._policy(jobs[0].policy_a_bp.to_dict())
    assert isinstance(actor, TruncatedNormalPolicy)

    for ep in eps:
        for rid in ("robot_a", "robot_b"):
            obs = torch.as_tensor(ep.observations[rid],
                                  dtype=torch.float32, device="cuda")
            act = torch.as_tensor(ep.actions[rid],
                                  dtype=torch.float32, device="cuda")
            ctx = SamplingContext.from_fields({
                k: torch.as_tensor(v, dtype=torch.float32, device="cuda")
                for k, v in ep.sampling_contexts[rid].items()})
            with torch.no_grad():
                ev = actor.evaluate_actions(obs, act, ctx=ctx)
            rec = torch.as_tensor(ep.action_extras[rid]["log_prob"],
                                  dtype=torch.float32, device="cuda")
            diff = (ev.log_prob - rec).abs().max().item()
            assert diff < 1e-4, f"{rid} log_prob replay diff {diff}"


def test_ppo_pipeline_compat(env_bp, policy_bp):
    """device episode 无差别流经 build_trajectories + PPOBuffer。"""
    from envs.batchframework.device_rollouter import DeviceRollouter
    from baseline.experiments_ppo.exp_standup_floor04 import (
        StandupFloor04)
    from baseline.framework.ppo.trainer import PPOBuffer
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)

    jobs = _jobs(env_bp, policy_bp, 4)
    with DeviceRollouter(batch_size=4) as dr:
        eps = dr.collect(jobs)
        actor = dr._policy(jobs[0].policy_a_bp.to_dict())
    assert isinstance(actor, TruncatedNormalPolicy)

    trajs = StandupFloor04().build_trajectories(eps)
    assert len(trajs) == 8  # 双 agent 各一条
    for t in trajs:
        assert t.obs.shape == (_MAX_STEPS, 96)
        assert t.actions.shape == (_MAX_STEPS, 21)
        assert t.last_obs.shape == (96,)
        assert t.channels["r_potential"].reward.shape == (_MAX_STEPS,)
        assert t.channels["r_potential"].is_terminated is False
        assert sorted(t.sampling_ctx) == [
            "delta_factor", "explore_factor"]
    buf = PPOBuffer(trajectories=trajs, actor=actor, device="cuda",
                    reward_keys=["r_potential"])
    rec = np.concatenate([
        np.asarray(e.action_extras[r]["log_prob"], dtype=np.float32)
        for e in eps for r in ("robot_a", "robot_b")])
    assert np.abs(buf.log_probs - rec).max() < 1e-4
    assert np.isfinite(buf.log_probs).all()


def test_unsupported_specs_rejected(env_bp, policy_bp):
    from envs.batchframework.device_rollouter import DeviceRollouter
    from baseline.framework.rollout.job import Job, SamplingSpec

    dr = DeviceRollouter(batch_size=4)
    with pytest.raises(ValueError, match="callable explore_factor"):
        dr.collect([Job(
            policy_a_bp=policy_bp, policy_b_bp=policy_bp, env_bp=env_bp,
            seed=0, episode_options={},
            sampling_a=SamplingSpec(lambda o, t: 0.1),
            sampling_b=SamplingSpec(0.0), stochastic=True)])
    with pytest.raises(ValueError, match="not supported"):
        dr.collect([Job(
            policy_a_bp=policy_bp, policy_b_bp=policy_bp, env_bp=env_bp,
            seed=0, episode_options={"bogus_key": 1},
            sampling_a=SamplingSpec(0.0), sampling_b=SamplingSpec(0.0),
            stochastic=True)])
    dr.close()
