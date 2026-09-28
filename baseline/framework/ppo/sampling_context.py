"""SamplingContext — 策略采样 / 评估的外生逐帧输入容器。

``SamplingContext`` 是 ``StochasticPolicy.sample()`` 与
``TrainablePolicy.evaluate_actions()`` 唯一的探索控制入参，取代原来
裸传的 ``explore_factor`` 关键字参数。

设计契约
--------

1. **框架构造并记录，策略解释。** 框架在 rollout 时逐帧构造 ctx、写入
   ``action_extras``、经 ``Episode → Trajectory → PPOBuffer`` 原样回放；
   训练时 minibatch 切片重建。框架不解释字段含义——怎么用是策略自己的事
   （与 ``explore_factor`` 的既有分工一致）。

2. **字段即契约。** rollout 与 PPO 重算必须使用同一套字段值，这是
   importance ratio 正确性的前提。``None`` 字段不进入记录。

3. **rollout 侧为 numpy，训练侧为 torch。** 本类型对字段值类型保持
   ``Any``——标量 ef 在两侧重放好是 ``np.float32`` 标量 / ``(B,)``
   tensor；``reference_action`` 是 ``(D,)`` / ``(B, D)``。

字段语义
--------

- ``explore_factor``：逐帧探索强度 ∈ [-1, 1]（0 = 中性），映射由策略定义。
- ``reference_action``：历史策略在当前 obs 上加权确定性动作 ``(D,)``；
  ``None`` = 无参考机制。策略自行决定如何消费（本阶段未消费）。
- ``delta_factor``：Δ→σ 标定系数 c（spec 静态值，逐帧记录供断言）。
- ``delta_mix``：原尺度/Δ尺度混合权重 λ（spec 静态值，逐帧记录）。
"""
from __future__ import annotations

from dataclasses import dataclass, fields as _dc_fields
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["CTX_SCHEMA_VERSION", "SamplingContext"]

#: Dump npz 中 ctx 序列化格式的版本号。
CTX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SamplingContext:
    """Per-frame exogenous inputs to stochastic sampling / evaluation."""

    explore_factor: Any = 0.0
    reference_action: Optional[Any] = None
    delta_factor: Any = 0.0
    delta_mix: Any = 0.0

    # ------------------------------------------------------------------
    # Rollout-side: extras recording
    # ------------------------------------------------------------------
    def record_fields(self) -> Dict[str, np.ndarray]:
        """Serialize non-None fields to per-frame numpy values.

        Scalars stay 0-d float32 arrays; ``reference_action`` stays
        ``(D,)`` float32.  ``None`` fields produce no key — missing keys
        on replay mean "mechanism off" rather than "value zero".
        """
        out: Dict[str, np.ndarray] = {}
        for f in _dc_fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            out[f.name] = np.asarray(v, dtype=np.float32)
        return out

    # ------------------------------------------------------------------
    # Train-side: rebuild from batched recorded fields
    # ------------------------------------------------------------------
    @classmethod
    def from_batch(cls, batched_fields: Dict[str, Any], sl) -> "SamplingContext":
        """Rebuild a ctx for a batch slice.

        ``batched_fields`` maps field name → batched tensor/array
        ``(N, ...)``; ``sl`` is the slice/index selecting this call's
        frames.  Keys that are not dataclass fields are ignored, so a
        record dict may carry future fields without breaking callers.
        """
        names = {f.name for f in _dc_fields(cls)}
        kwargs = {
            k: v[sl] for k, v in batched_fields.items() if k in names
        }
        return cls(**kwargs)

    @classmethod
    def from_fields(cls, fields: Dict[str, Any]) -> "SamplingContext":
        """Rebuild a ctx from already-sliced field values (same filtering)."""
        names = {f.name for f in _dc_fields(cls)}
        return cls(**{k: v for k, v in fields.items() if k in names})

    # ------------------------------------------------------------------
    # Policy-side helpers
    # ------------------------------------------------------------------
    def has_delta(self) -> bool:
        """True iff the delta-scale mechanism is active this frame/batch.

        Policies must short-circuit when False (return σ_ef untouched) —
        that is what makes the λ=0 / no-reference path bit-identical to
        the pre-delta behavior.  ``delta_mix`` may be a scalar or a batched
        tensor; batched ⇒ checked elementwise-conservatively (any ≠ 0
        activates, the mix itself stays per-element).
        """
        if self.reference_action is None:
            return False
        dm = self.delta_mix
        if hasattr(dm, "any"):  # ndarray / torch.Tensor
            return bool((dm != 0).any())
        return dm != 0
