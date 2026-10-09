"""policy_executor dispatch — payload policy_class 路由与显式拒绝。

覆盖 W1 契约：
- 注册表按导出 payload 的 ``policy_class`` 分发；
- 未注册类（含 pre_tanh 族）显式 ValueError，信息含类名与已注册清单，
  绝不静默错载；
- payload 缺 ``policy_class`` → 拒绝为非合法导出；
- 已注册类（TruncatedNormalPolicy）经 cache 正常加载并可 act。
"""
import tempfile
from pathlib import Path

import pytest
import torch

from envs.batchframework.policy_executor import (
    PolicyExecutorCache, TruncatedNormalExecutor, _default_factory)


def _bp_dict(tmpdir: str) -> dict:
    return {"cls": f"file:{tmpdir}/policy.py:ExportedX"}


def _write_payload(tmpdir: str, payload: dict) -> dict:
    torch.save(payload, Path(tmpdir) / "model.pt")
    return _bp_dict(tmpdir)


def test_unknown_policy_class_rejected_explicitly():
    with tempfile.TemporaryDirectory() as d:
        bp = _write_payload(d, {
            "format_version": 1,
            "policy_class": "NoSuchPolicy",
            "arch": {"obs_dim": 4, "action_dim": 3, "hidden_dim": 8},
            "state_dict": {},
        })
        with pytest.raises(ValueError, match="NoSuchPolicy"):
            _default_factory(bp, "cpu")


def test_pre_tanh_export_rejected_not_misloaded():
    """pre_tanh 族不做支持——必须显式拒绝而非错载成 TN。"""
    with tempfile.TemporaryDirectory() as d:
        bp = _write_payload(d, {
            "format_version": 1,
            "policy_class": "PreTanhNormalPolicy",
            "arch": {"obs_dim": 4, "action_dim": 3, "hidden_dim": 8},
            "state_dict": {},
        })
        with pytest.raises(ValueError) as ei:
            _default_factory(bp, "cpu")
        msg = str(ei.value)
        assert "PreTanhNormalPolicy" in msg
        assert "TruncatedNormalPolicy" in msg  # 错误信息列出已注册类


def test_payload_missing_policy_class_rejected():
    with tempfile.TemporaryDirectory() as d:
        bp = _write_payload(d, {"format_version": 1, "state_dict": {}})
        with pytest.raises(ValueError, match="not a policy export"):
            _default_factory(bp, "cpu")


def test_non_file_blueprint_rejected():
    with pytest.raises(ValueError, match="file:"):
        _default_factory({"cls": "random"}, "cpu")


def test_registered_truncated_normal_loads_via_cache():
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)
    torch.manual_seed(0)
    pol = TruncatedNormalPolicy(obs_dim=9, action_dim=5, hidden_dim=16)
    with tempfile.TemporaryDirectory() as d:
        bp = pol.to_blueprint(d).to_dict()
        cache = PolicyExecutorCache()
        ex = cache.get(bp, "cpu")
        assert isinstance(ex, TruncatedNormalExecutor)
        obs = torch.randn(4, 9)
        a_a, a_b, lp_a, lp_b = ex.act(
            obs, obs, stochastic=False, ctx_a=None, ctx_b=None,
            ctx_ab=None, shared=False)
        assert a_a.shape == (4, 5) and lp_a is None
        a_a, a_b, lp_a, lp_b = ex.act(
            obs, obs, stochastic=True, ctx_a=None, ctx_b=None,
            ctx_ab=None, shared=False)
        assert lp_a.shape == (4,)
        cache.clear()
