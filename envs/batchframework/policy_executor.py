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

新策略接入 = 实现 executor + ``register_executor(policy_class, factory)``；
分发键是导出 payload 的 ``policy_class``——未注册类显式拒绝，
绝不静默错载到别的 executor。
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
            shared: bool,
            u_a: Optional[torch.Tensor] = None,
            u_b: Optional[torch.Tensor] = None,
            u_ab: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor,
                       Optional[torch.Tensor],
                       Optional[torch.Tensor]]:
        raise NotImplementedError

    def noise_cols(self, action_dim: int) -> int:
        """随机波每行需要的 uniform 列数（默认 = action_dim）。

        mixture 等额外噪声通道的 executor 覆写此方法；wave_runner
        按此宽度生成 job-keyed u。
        """
        return action_dim

    def close(self) -> None:
        return None


def _load_payload(
        bp_dict: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    """file: 蓝图 → (model.pt 路径, payload)；非导出蓝图显式拒绝。"""
    cls = bp_dict.get("cls", "")
    if not cls.startswith("file:"):
        raise ValueError(
            f"device executor requires file: policy export, got {cls!r}")
    path = Path(cls[5:].rsplit(":", 1)[0]).with_name("model.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "policy_class" not in payload:
        raise ValueError(
            f"{path}: not a policy export payload (missing "
            f"'policy_class')")
    return path, payload


class _FamilySpec(NamedTuple):
    """一族策略的 executor 装配参数。"""
    module: str            # baseline.framework.ppo.policies 下的模块名
    cls_name: str          # 训练侧策略类名
    bounded: bool = False  # 需要 payload 顶层 sigma_min/max/init_std/explore_alpha
    is_mixture: bool = False  # sample_action 需要 u_comp 组件选择噪声


# payload policy_class → 装配参数。新增策略族 = 这里加一行。
_FAMILY_SPECS: Dict[str, _FamilySpec] = {
    "TruncatedNormalPolicy": _FamilySpec(
        "truncated_normal_mlp", "TruncatedNormalPolicy"),
    "BoundedStdTruncatedNormalPolicy": _FamilySpec(
        "bounded_std_truncated_normal_mlp",
        "BoundedStdTruncatedNormalPolicy", bounded=True),
    "StateTruncatedNormalPolicy": _FamilySpec(
        "state_truncated_normal_mlp", "StateTruncatedNormalPolicy"),
    "StateBoundedStdTruncatedNormalPolicy": _FamilySpec(
        "state_bounded_std_truncated_normal_mlp",
        "StateBoundedStdTruncatedNormalPolicy", bounded=True),
    "MixtureTruncatedNormalPolicy": _FamilySpec(
        "mixture_truncated_normal_mlp", "MixtureTruncatedNormalPolicy",
        is_mixture=True),
    "SharedMixtureTruncatedNormalPolicy": _FamilySpec(
        "shared_mixture_truncated_normal_mlp",
        "SharedMixtureTruncatedNormalPolicy", is_mixture=True),
    "SharedMixtureBoundedStdTruncatedNormalPolicy": _FamilySpec(
        "shared_mixture_bounded_std_truncated_normal_mlp",
        "SharedMixtureBoundedStdTruncatedNormalPolicy",
        bounded=True, is_mixture=True),
    "StateMixtureBoundedStdTruncatedNormalPolicy": _FamilySpec(
        "state_mixture_bounded_std_truncated_normal_mlp",
        "StateMixtureBoundedStdTruncatedNormalPolicy",
        bounded=True, is_mixture=True),
}

_BOUNDED_KEYS = ("sigma_min", "sigma_max", "init_std", "explore_alpha")


class TorchPolicyExecutor(PolicyExecutor):
    """训练侧策略类的通用设备端适配。

    file: 导出蓝图 → 按 ``_FamilySpec`` 重建对应 policy 类：
    ctor kwargs = ``arch`` + （bounded 族）payload 顶层 σ 界配置；
    版本 = state_dict 内容 sha256（前 16 hex）。

    mixture 族：注入噪声打包为 ``(B, action_dim+1)``——第 0 列是
    组件选择 uniform（u_comp），其余是逐维逆CDF uniform（u）。
    """

    def __init__(self, bp_dict: Dict[str, Any], device: str,
                 payload: Optional[Dict[str, Any]] = None,
                 spec: Optional[_FamilySpec] = None):
        if payload is None:
            _, payload = _load_payload(bp_dict)
        if spec is None:
            spec = _FAMILY_SPECS[payload["policy_class"]]
        self._spec = spec
        kwargs = dict(payload["arch"])
        if spec.bounded:
            missing = [k for k in _BOUNDED_KEYS if k not in payload]
            if missing:
                raise ValueError(
                    f"bounded-σ export missing {missing} in payload "
                    f"top-level (policy_class="
                    f"{payload['policy_class']!r}) — re-export with "
                    f"current code")
            kwargs.update({k: payload[k] for k in _BOUNDED_KEYS})
        import importlib
        mod = importlib.import_module(
            f"baseline.framework.ppo.policies.{spec.module}")
        cls = getattr(mod, spec.cls_name)
        self.policy = cls(**kwargs, device=device)
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

    def noise_cols(self, action_dim: int) -> int:
        """每行需要的 uniform 列数；mixture 多 1 列组件选择。"""
        return action_dim + (1 if self._spec.is_mixture else 0)

    @staticmethod
    def _split_u(u: torch.Tensor, is_mixture: bool):
        """打包 u → (u_dim, u_comp)；非 mixture 原样返回。"""
        if u is None or not is_mixture:
            return u, None
        return u[:, 1:], u[:, 0]

    def _sample(self, obs, ctx, u):
        u_dim, u_comp = self._split_u(u, self._spec.is_mixture)
        if u_comp is None:
            return self.policy.sample_action(obs, ctx=ctx, u=u_dim)
        return self.policy.sample_action(
            obs, ctx=ctx, u=u_dim, u_comp=u_comp)

    def act(self, obs_a, obs_b, *, stochastic, ctx_a, ctx_b, ctx_ab,
            shared, u_a=None, u_b=None, u_ab=None
            ) -> Tuple[torch.Tensor, torch.Tensor,
                       Optional[torch.Tensor],
                       Optional[torch.Tensor]]:
        with torch.no_grad():
            if stochastic:
                if shared:
                    B = obs_a.shape[0]
                    both = torch.cat([obs_a, obs_b], dim=0)
                    a_all, lp_all = self._sample(both, ctx_ab, u_ab)
                    return a_all[:B], a_all[B:], lp_all[:B], lp_all[B:]
                a_a, lp_a = self._sample(obs_a, ctx_a, u_a)
                a_b, lp_b = self._sample(obs_b, ctx_b, u_b)
                return a_a, a_b, lp_a, lp_b
            return (self.policy.deterministic_action(obs_a),
                    self.policy.deterministic_action(obs_b),
                    None, None)

    def close(self) -> None:
        del self.policy


class TruncatedNormalExecutor(TorchPolicyExecutor):
    """TruncatedNormalPolicy 的设备端适配（保留类名兼容既有引用）。"""

    def __init__(self, bp_dict: Dict[str, Any], device: str,
                 payload: Optional[Dict[str, Any]] = None):
        super().__init__(bp_dict, device, payload,
                         spec=_FAMILY_SPECS["TruncatedNormalPolicy"])


# ---------------------------------------------------------------------------
# 有界版本化缓存
# ---------------------------------------------------------------------------
_EXECUTOR_FACTORIES: Dict[str, Any] = {}


def register_executor(policy_class: str, factory) -> None:
    """注册 ``policy_class``（导出 payload 的 policy_class 字段）
    → executor 工厂。工厂签名 ``(bp_dict, device, payload)``。"""
    _EXECUTOR_FACTORIES[policy_class] = factory


def _default_factory(bp_dict: Dict[str, Any], device: str) -> PolicyExecutor:
    path, payload = _load_payload(bp_dict)
    policy_class = payload["policy_class"]
    factory = _EXECUTOR_FACTORIES.get(policy_class)
    if factory is None:
        raise ValueError(
            f"{path}: no device executor registered for policy_class "
            f"{policy_class!r}; registered: "
            f"{sorted(_EXECUTOR_FACTORIES)}. Implement an executor and "
            f"register_executor() it, or re-export as a supported "
            f"policy class.")
    return factory(bp_dict, device, payload)


def _torch_factory(spec: _FamilySpec):
    return lambda bp_dict, device, payload: TorchPolicyExecutor(
        bp_dict, device, payload, spec)


for _pcls, _spec in _FAMILY_SPECS.items():
    register_executor(_pcls, _torch_factory(_spec))
register_executor(
    "TruncatedNormalPolicy",
    lambda bp_dict, device, payload: TruncatedNormalExecutor(
        bp_dict, device, payload))


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
