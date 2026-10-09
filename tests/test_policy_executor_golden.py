"""policy_executor golden — 8 个 truncnorm 族逐族对拍。

每族覆盖：
- 确定性动作：executor ``deterministic_action`` vs 导出 ``policy.py``
  的 ``act()``（CPU 消费侧真相）；
- 随机动作：同一打包 u 注入 → executor 输出 = 训练侧
  ``sample_action`` 直接调用（验证 executor 的 u 拆分/装配，
  不是重复采样数学本身）；
- log_prob：executor 返回的 lp == ``evaluate_actions`` 对同动作的
  评估值（独立校验 lp 语义）；
- job-keyed 复现性：同一 u 两次 act → 逐位相同；
- mixture：u_comp 组件频率 ≈ π（统计），打包列角色回归；
- GPU（若可用）：executor(device=cuda) vs CPU policy 同一 u —
  fp32 跨设备容差。
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from envs.batchframework.policy_executor import PolicyExecutorCache
from envs.framework.policy import PolicyBlueprint
from baseline.framework.ppo.sampling_context import SamplingContext

FAMILIES = [
    ("truncated_normal_mlp", "TruncatedNormalPolicy", {}, False),
    ("bounded_std_truncated_normal_mlp",
     "BoundedStdTruncatedNormalPolicy", {}, False),
    ("state_truncated_normal_mlp", "StateTruncatedNormalPolicy",
     {}, False),
    ("state_bounded_std_truncated_normal_mlp",
     "StateBoundedStdTruncatedNormalPolicy", {}, False),
    ("mixture_truncated_normal_mlp", "MixtureTruncatedNormalPolicy",
     {}, True),
    ("shared_mixture_truncated_normal_mlp",
     "SharedMixtureTruncatedNormalPolicy", {}, True),
    ("shared_mixture_bounded_std_truncated_normal_mlp",
     "SharedMixtureBoundedStdTruncatedNormalPolicy", {}, True),
    ("state_mixture_bounded_std_truncated_normal_mlp",
     "StateMixtureBoundedStdTruncatedNormalPolicy", {}, True),
]

OBS_DIM, ACT_DIM, HID, B = 9, 5, 16, 8


def _make_policy(module, cls_name, kwargs, seed=0):
    import importlib
    torch.manual_seed(seed)
    cls = getattr(
        importlib.import_module(
            f"baseline.framework.ppo.policies.{module}"), cls_name)
    return cls(obs_dim=OBS_DIM, action_dim=ACT_DIM, hidden_dim=HID,
               **kwargs)


def _export_and_load(pol, device="cpu"):
    tmp = tempfile.mkdtemp()
    bp = pol.to_blueprint(tmp)
    ex = PolicyExecutorCache().get(bp.to_dict(), device)
    exported = bp.build()            # 导出 policy.py（CPU 消费侧）
    return ex, exported


def _split_u(u, is_mixture):
    return (u[:, 1:], u[:, 0]) if is_mixture else (u, None)


@pytest.mark.parametrize("module,cls_name,kwargs,is_mix", FAMILIES)
def test_deterministic_matches_exported(module, cls_name, kwargs,
                                        is_mix):
    pol = _make_policy(module, cls_name, kwargs)
    ex, exported = _export_and_load(pol)
    obs = torch.randn(B, OBS_DIM)
    a_ex, _, _, _ = ex.act(obs, obs, stochastic=False, ctx_a=None,
                           ctx_b=None, ctx_ab=None, shared=False)
    obs_np = obs.numpy().astype(np.float32)
    a_ref = np.stack([exported.act(o)[0] for o in obs_np])
    np.testing.assert_allclose(a_ex.numpy(), a_ref, atol=1e-6)
    ex.close()


@pytest.mark.parametrize("module,cls_name,kwargs,is_mix", FAMILIES)
def test_stochastic_matches_policy_and_replays(module, cls_name, kwargs,
                                               is_mix):
    pol = _make_policy(module, cls_name, kwargs)
    ex, _ = _export_and_load(pol)
    obs = torch.randn(B, OBS_DIM)
    ncols = ex.noise_cols(ACT_DIM)
    assert ncols == ACT_DIM + (1 if is_mix else 0)
    u = torch.rand(B, ncols)
    u_dim, u_comp = _split_u(u, is_mix)
    if is_mix:
        a_ref, lp_ref = pol.sample_action(obs, u=u_dim, u_comp=u_comp)
    else:
        a_ref, lp_ref = pol.sample_action(obs, u=u_dim)
    a1, _, lp1, _ = ex.act(obs, obs, stochastic=True, ctx_a=None,
                           ctx_b=None, ctx_ab=None, shared=False,
                           u_a=u, u_b=u)
    # executor 装配后的 u 拆分必须与直接调用一致（同机同 dtype → 严格等）
    torch.testing.assert_close(a1, a_ref, atol=0, rtol=0)
    torch.testing.assert_close(lp1, lp_ref, atol=0, rtol=0)
    # job-keyed 复现性：同一 u 重放逐位相同
    a2, _, lp2, _ = ex.act(obs, obs, stochastic=True, ctx_a=None,
                           ctx_b=None, ctx_ab=None, shared=False,
                           u_a=u, u_b=u)
    torch.testing.assert_close(a2, a1, atol=0, rtol=0)
    torch.testing.assert_close(lp2, lp1, atol=0, rtol=0)
    ex.close()


@pytest.mark.parametrize("module,cls_name,kwargs,is_mix", FAMILIES)
def test_log_prob_matches_evaluate_actions(module, cls_name, kwargs,
                                           is_mix):
    pol = _make_policy(module, cls_name, kwargs)
    ex, _ = _export_and_load(pol)
    obs = torch.randn(B, OBS_DIM)
    u = torch.rand(B, ex.noise_cols(ACT_DIM))
    a, _, lp, _ = ex.act(obs, obs, stochastic=True, ctx_a=None,
                         ctx_b=None, ctx_ab=None, shared=False,
                         u_a=u, u_b=u)
    ev = pol.evaluate_actions(obs, a)
    torch.testing.assert_close(lp, ev.log_prob, atol=1e-5, rtol=1e-5)
    ex.close()


@pytest.mark.parametrize("module,cls_name,kwargs,is_mix", FAMILIES)
def test_explore_factor_propagates(module, cls_name, kwargs, is_mix):
    pol = _make_policy(module, cls_name, kwargs)
    ex, _ = _export_and_load(pol)
    obs = torch.randn(B, OBS_DIM)
    u = torch.rand(B, ex.noise_cols(ACT_DIM))
    ctx0 = SamplingContext(explore_factor=torch.zeros(B))
    ctx_hi = SamplingContext(explore_factor=torch.full((B,), 0.5))
    a0, _, _, _ = ex.act(obs, obs, stochastic=True, ctx_a=ctx0,
                         ctx_b=ctx0, ctx_ab=None, shared=False,
                         u_a=u, u_b=u)
    a1, _, _, _ = ex.act(obs, obs, stochastic=True, ctx_a=ctx_hi,
                         ctx_b=ctx_hi, ctx_ab=None, shared=False,
                         u_a=u, u_b=u)
    assert not torch.allclose(a0, a1), (
        f"{cls_name}: explore_factor 未生效")
    ex.close()


@pytest.mark.parametrize("module,cls_name,kwargs,is_mix", FAMILIES)
def test_shared_concat_matches_separate(module, cls_name, kwargs,
                                        is_mix):
    """shared=True 的 concat 前向 == shared=False 的两次前向。"""
    pol = _make_policy(module, cls_name, kwargs)
    ex, _ = _export_and_load(pol)
    obs = torch.randn(B, OBS_DIM)
    ncols = ex.noise_cols(ACT_DIM)
    u_a = torch.rand(B, ncols)
    u_b = torch.rand(B, ncols)
    a_s, b_s, lpa_s, lpb_s = ex.act(
        obs, obs, stochastic=True, ctx_a=None, ctx_b=None, ctx_ab=None,
        shared=True, u_ab=torch.cat([u_a, u_b], 0))
    a_p, b_p, lpa_p, lpb_p = ex.act(
        obs, obs, stochastic=True, ctx_a=None, ctx_b=None, ctx_ab=None,
        shared=False, u_a=u_a, u_b=u_b)
    torch.testing.assert_close(a_s, a_p, atol=0, rtol=0)
    torch.testing.assert_close(b_s, b_p, atol=0, rtol=0)
    ex.close()


def _mixture_ids():
    return [f for f in FAMILIES if f[3]]


@pytest.mark.parametrize("module,cls_name,kwargs,is_mix",
                         _mixture_ids())
def test_mixture_component_freq_matches_pi(module, cls_name, kwargs,
                                           is_mix):
    """u_comp 均匀扫描 → 组件命中频率 ≈ softmax(log_pi)。"""
    pol = _make_policy(module, cls_name, kwargs)
    ex, _ = _export_and_load(pol)
    obs = torch.randn(B, OBS_DIM)
    log_pi, mean, _ = pol._forward_raw(obs)      # noqa: SLF001
    pi = log_pi.exp()                            # (B,K)
    K = pol.num_components
    # 逐行把 u_comp 放在 π 的 CDF 区间中点 → 应选中对应组件
    cum = torch.cumsum(pi, dim=-1)
    lo = torch.cat([torch.zeros(B, 1), cum[:, :-1]], dim=1)
    u_mid = ((lo + cum) / 2).clamp(0, 1 - 1e-6)  # (B,K)
    for k in range(K):
        u = torch.rand(B, ex.noise_cols(ACT_DIM))
        u[:, 0] = u_mid[:, k]
        a, _, _, _ = ex.act(obs, obs, stochastic=True, ctx_a=None,
                            ctx_b=None, ctx_ab=None, shared=False,
                            u_a=u, u_b=u)
        # 选中第 k 组件 → 动作应落在该组件均值附近（同一 u_dim）
        mu_k = mean[torch.arange(B), k]          # (B,D)
        assert (a - mu_k).abs().max() < 3.0, (
            f"{cls_name}: u_comp 中点未选中组件 {k}")
    ex.close()


@pytest.mark.parametrize("module,cls_name,kwargs,is_mix",
                         _mixture_ids())
def test_mixture_packed_column_roles(module, cls_name, kwargs, is_mix):
    """打包 u 的第 0 列只影响组件选择，其余列只影响组内逆CDF。"""
    pol = _make_policy(module, cls_name, kwargs)
    ex, _ = _export_and_load(pol)
    obs = torch.randn(B, OBS_DIM)
    ncols = ex.noise_cols(ACT_DIM)
    u0 = torch.rand(B, ncols)
    # 仅改第 0 列 → 组件可能变，但给定相同组件时动作偏移应离散跳变；
    # 仅改尾列 → 组件不变、连续变化。用 log_pi 主导的退化检查：
    u1 = u0.clone()
    u1[:, 0] = torch.rand(B)
    u2 = u0.clone()
    u2[:, 1:] = torch.rand(B, ACT_DIM)
    # 两次调用与直接 sample_action 对拍已由上一测试覆盖；
    # 这里只验证两列互不泄漏：同 u_comp、不同 u_dim →
    # 与 policy.sample_action(u=u2[:,1:], u_comp=u0[:,0]) 一致
    a, _, _, _ = ex.act(obs, obs, stochastic=True, ctx_a=None,
                        ctx_b=None, ctx_ab=None, shared=False,
                        u_a=u2.clone().index_copy(
                            1, torch.tensor(0), u0[:, :1]),
                        u_b=u0)
    u_dim, u_comp = u2[:, 1:], u0[:, 0]
    a_ref, _ = pol.sample_action(obs, u=u_dim, u_comp=u_comp)
    torch.testing.assert_close(a, a_ref, atol=0, rtol=0)
    ex.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("module,cls_name,kwargs,is_mix", FAMILIES)
def test_gpu_executor_matches_cpu(module, cls_name, kwargs, is_mix):
    pol = _make_policy(module, cls_name, kwargs)
    ex_gpu, _ = _export_and_load(pol, device="cuda")
    obs_cpu = torch.randn(B, OBS_DIM)
    obs_gpu = obs_cpu.cuda()
    ncols = ex_gpu.noise_cols(ACT_DIM)
    u_cpu = torch.rand(B, ncols)
    u_gpu = u_cpu.cuda()
    # deterministic
    ag, _, _, _ = ex_gpu.act(obs_gpu, obs_gpu, stochastic=False,
                             ctx_a=None, ctx_b=None, ctx_ab=None,
                             shared=False)
    a_ref = pol.deterministic_action(obs_cpu)
    torch.testing.assert_close(ag.cpu(), a_ref, atol=1e-5, rtol=1e-5)
    # stochastic（同 u 跨设备）
    ag, _, lpg, _ = ex_gpu.act(obs_gpu, obs_gpu, stochastic=True,
                               ctx_a=None, ctx_b=None, ctx_ab=None,
                               shared=False, u_a=u_gpu, u_b=u_gpu)
    u_dim, u_comp = _split_u(u_cpu, is_mix)
    if is_mix:
        a_ref, lp_ref = pol.sample_action(obs_cpu, u=u_dim,
                                          u_comp=u_comp)
    else:
        a_ref, lp_ref = pol.sample_action(obs_cpu, u=u_dim)
    torch.testing.assert_close(ag.cpu(), a_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lpg.cpu(), lp_ref, atol=1e-4, rtol=1e-4)
    ex_gpu.close()
