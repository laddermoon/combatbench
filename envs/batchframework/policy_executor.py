"""policy_executor — 设备端策略适配层（E3-W2）。

collector 不再直接实例化/缓存策略类。职责切分：

- ``PolicyExecutor``：批量 `act` 的执行接口 + ``capabilities()`` 声明
  （stochastic/deterministic/支持的 SamplingContext 字段/是否有隐状态）。
  采样 spec 的支持性检查由"spec 需要的 ctx 字段 ⊆ executor 声明字段"
  表达——拒绝来源是能力声明，不是 collector 内的硬编码白名单。
- ``PolicyExecutorCache``：有界 LRU（默认 8）。缓存键 = model.pt 路径
  （进程内文件身份）；命中时 stat 校验 mtime/size，变了即重载——
  路径不是版本保证，真正的版本是 ``executor.version``（state_dict
  内容 sha256），记入 collect manifest 供 provenance 审计。

新策略接入 = 实现 executor + ``register_executor(kind, factory)``。
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch

from baseline.framework.ppo.sampling_context import SamplingContext


class PolicyCapabilities(NamedTuple):
    """executor 能力面——采样 spec 兼容性检查的数据来源。"""

    stochastic: bool
    deterministic: bool
    ctx_fields: frozenset        # 支持的 SamplingContext 字段名
    stateful: bool               # 含需 reset_rows 的隐状态


def required_ctx_fields(spec_dict: Dict[str, Any]) -> frozenset:
    """job sampling spec → 需要的 ctx 字段集。

    - ``explore_factor``（标量）总是需要；
    - callable ef / reference / delta_factor≠0 → 需要对应字段，
      本阶段无 executor 声明这些能力 → 显式拒绝。
    """
    req = {"explore_factor"}
    ef = spec_dict.get("explore_factor", 0.0)
    if callable(ef):
        # callable 需要逐帧 host 评估——无 executor 声明该能力
        req.add("callable_explore_factor")
    if spec_dict.get("reference") is not None:
        req.add("reference_action")
    if float(spec_dict.get("delta_factor", 0.0)) != 0.0:
        req.add("delta")
    return frozenset(req)


# ---------------------------------------------------------------------------
# Executor 基类
# ---------------------------------------------------------------------------
class PolicyExecutor:
    """设备端批量策略执行器。

    ``act()`` 返回 ``(action_a, action_b, log_prob_a, log_prob_b)``；
    ``stochastic=False`` 时 log_prob 为 None（CPU 语义：deterministic
    波不伪造 log_prob=0）。
    """

    #: 由子类设置
    capabilities = PolicyCapabilities(
        stochastic=False, deterministic=False,
        ctx_fields=frozenset(), stateful=False)
    version: str = "unknown"

    def check_spec(self, spec_dict: Dict[str, Any], tag: str) -> None:
        """spec 需要的 ctx 字段 ⊆ 声明能力，否则显式拒绝。"""
        missing = required_ctx_fields(spec_dict) - self.capabilities.ctx_fields
        if missing:
            readable = [m.replace("callable_explore_factor",
                                  "callable explore_factor")
                        for m in sorted(missing)]
            raise ValueError(
                f"job[{tag}]: sampling spec requires ctx fields "
                f"{readable} not supported by executor "
                f"{type(self).__name__} (supports "
                f"{sorted(self.capabilities.ctx_fields)})")

    def act(self, obs_a: torch.Tensor, obs_b: torch.Tensor, *,
            stochastic: bool,
            ctx_a: Optional[SamplingContext],
            ctx_b: Optional[SamplingContext],
            ctx_ab: Optional[SamplingContext],
            shared: bool) -> Tuple[torch.Tensor, torch.Tensor,
                                   Optional[torch.Tensor],
                                   Optional[torch.Tensor]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class TruncatedNormalExecutor(PolicyExecutor):
    """TruncatedNormalPolicy 的设备端适配（当前唯一 executor 实现）。

    file: 导出蓝图 → 重建 TruncatedNormalPolicy；版本 = state_dict
    内容 sha256（前 16  hex），不以路径/mtime 充当版本保证。
    """

    def __init__(self, bp_dict: Dict[str, Any], device: str):
        cls = bp_dict.get("cls", "")
        if not cls.startswith("file:"):
            raise ValueError(
                f"device executor requires file: policy export, "
                f"got {cls!r}")
        path = cls[5:].rsplit(":", 1)[0]
        self._model_path = Path(path).with_name("model.pt")
        payload = torch.load(self._model_path, map_location="cpu",
                             weights_only=False)
        arch = payload["arch"]
        from baseline.framework.ppo.policies.truncated_normal_mlp import (
            TruncatedNormalPolicy)
        self.policy = TruncatedNormalPolicy(
            obs_dim=int(arch["obs_dim"]),
            action_dim=int(arch["action_dim"]),
            hidden_dim=int(arch["hidden_dim"]),
            device=device)
        self.policy.load_state_dict(payload["state_dict"])
        self.policy.eval()
        self._device = device
        self.version = self._hash_state_dict(payload["state_dict"])

    @staticmethod
    def _hash_state_dict(sd: Dict[str, torch.Tensor]) -> str:
        h = hashlib.sha256()
        for k in sorted(sd):
            v = sd[k]
            h.update(k.encode())
            t = v.detach().cpu().contiguous()
            h.update(t.numpy().tobytes())
        return h.hexdigest()[:16]

    @property
    def capabilities(self) -> PolicyCapabilities:
        return PolicyCapabilities(
            stochastic=True, deterministic=True,
            ctx_fields=frozenset({"explore_factor", "delta_factor"}),
            stateful=False)

    def act(self, obs_a, obs_b, *, stochastic, ctx_a, ctx_b, ctx_ab,
            shared) -> Tuple[torch.Tensor, torch.Tensor,
                             Optional[torch.Tensor],
                             Optional[torch.Tensor]]:
        with torch.no_grad():
            if stochastic:
                if shared:
                    B = obs_a.shape[0]
                    both = torch.cat([obs_a, obs_b], dim=0)
                    a_all, lp_all = self.policy.sample_action(
                        both, ctx=ctx_ab)
                    return a_all[:B], a_all[B:], lp_all[:B], lp_all[B:]
                a_a, lp_a = self.policy.sample_action(obs_a, ctx=ctx_a)
                a_b, lp_b = self.policy.sample_action(obs_b, ctx=ctx_b)
                return a_a, a_b, lp_a, lp_b
            return (self.policy.deterministic_action(obs_a),
                    self.policy.deterministic_action(obs_b),
                    None, None)

    def close(self) -> None:
        del self.policy


# ---------------------------------------------------------------------------
# 有界版本化缓存
# ---------------------------------------------------------------------------
_EXECUTOR_FACTORIES: Dict[str, Any] = {}


def register_executor(kind: str, factory) -> None:
    """kind 供 manifest/诊断引用（当前只有 truncated_normal_file）。"""
    _EXECUTOR_FACTORIES[kind] = factory


def _default_factory(bp_dict: Dict[str, Any], device: str) -> PolicyExecutor:
    return TruncatedNormalExecutor(bp_dict, device)


class PolicyExecutorCache:
    """有界 LRU：键 = model.pt 路径；命中校验 stat，变了重载。"""

    def __init__(self, capacity: int = 8):
        self._cap = int(capacity)
        self._items: "OrderedDict[str, Tuple[Any, PolicyExecutor]]" = \
            OrderedDict()

    def get(self, bp_dict: Dict[str, Any],
            device: str) -> PolicyExecutor:
        cls = bp_dict.get("cls", "")
        if not cls.startswith("file:"):
            raise ValueError(
                f"device executor requires file: policy export, got {cls!r}")
        path = Path(cls[5:].rsplit(":", 1)[0]).with_name("model.pt")
        stat = (path.stat().st_mtime_ns, path.stat().st_size)
        hit = self._items.get(str(path))
        if hit is not None and hit[0] == stat:
            self._items.move_to_end(str(path))
            return hit[1]
        if hit is not None:
            hit[1].close()
            del self._items[str(path)]
        ex = _default_factory(bp_dict, device)
        self._items[str(path)] = (stat, ex)
        while len(self._items) > self._cap:
            _, (_, evicted) = self._items.popitem(last=False)
            evicted.close()
        return ex

    def clear(self) -> None:
        for _, (_, ex) in self._items.items():
            ex.close()
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
