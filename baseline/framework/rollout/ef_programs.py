"""ef_programs — 声明式 explore_factor 程序（device-rollout 可迁）。

``SamplingSpec.explore_factor`` 除标量/callable 外，接受第三种形态：
**声明式程序**——带 ``"kind"`` 键的 mapping，经本模块注册表求值。

为什么需要它：device collector 无法每帧回调 host 上的 Python
callable（会破坏 job-keyed 确定性 + 每步 D2H 同步）。把 ef 表达为
可序列化程序后，同一 spec 在 CPU 路径（逐帧求值）与 device 路径
（波内向量化逐帧求值）语义一致。

求值约定：evaluator 只用 numpy/torch 通吃的算术表达式（布尔掩码
×差值+基底），因此同一函数可处理 (D,) numpy 行与 (B,D) torch
批量张量，返回标量或 (B,) 张量。

新 kind 接入 = ``register_ef_program(kind, fn)`` 显式登记；
未知 kind 一律显式拒绝，不静默降级。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

#: evaluator: fn(params: Mapping, obs, step) -> scalar 或 (B,)
_EF_PROGRAMS: Dict[str, Callable[..., Any]] = {}

_OPS = {
    "ge": ">=",
    "gt": ">",
    "le": "<=",
    "lt": "<",
    "eq": "==",
    "ne": "!=",
}


def register_ef_program(kind: str, fn: Callable[..., Any]) -> None:
    _EF_PROGRAMS[str(kind)] = fn


def eval_ef_program(spec: Mapping[str, Any], obs: Any,
                    step: Optional[int] = None) -> Any:
    """对 obs 求值 ef 程序 → 标量（(D,) obs）或 (B,)（(B,D) obs）。"""
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"ef program must be a Mapping, got {type(spec).__name__}")
    kind = spec.get("kind")
    fn = _EF_PROGRAMS.get(kind)
    if fn is None:
        raise ValueError(
            f"unknown ef program kind {kind!r}; registered: "
            f"{sorted(_EF_PROGRAMS)}")
    return fn(spec, obs, step)


def _obs_threshold(spec: Mapping[str, Any], obs: Any,
                   step: Optional[int]) -> Any:
    """``{index, threshold, op, then, else}``：观测阈值二段切换。

    ``op`` 默认 ``"ge"``（obs[index] >= threshold → then 否则 else）。
    纯算术实现：mask*(then-else)+else——numpy/torch 通用。
    """
    try:
        idx = int(spec["index"])
        thr = float(spec["threshold"])
        hi = float(spec["then"])
        lo = float(spec["else"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"obs_threshold ef program malformed: {spec!r} ({e})")
    op = spec.get("op", "ge")
    if op not in _OPS:
        raise ValueError(
            f"obs_threshold: unknown op {op!r}; allowed {sorted(_OPS)}")
    col = obs[..., idx]
    mask = {"ge": col >= thr, "gt": col > thr, "le": col <= thr,
            "lt": col < thr, "eq": col == thr, "ne": col != thr}[op]
    return mask * (hi - lo) + lo


def ef_program_to_source(spec: Mapping[str, Any]) -> str:
    """程序 → 等价 Python 函数源码（dump/replay 自含需要）。

    仅支持可完整源码化的 kind；不支持的 kind 显式拒绝。
    """
    kind = spec.get("kind")
    if kind == "obs_threshold":
        idx = int(spec["index"])
        thr = float(spec["threshold"])
        hi = float(spec["then"])
        lo = float(spec["else"])
        op = _OPS.get(spec.get("op", "ge"))
        if op is None:
            raise ValueError(
                f"obs_threshold: unknown op {spec.get('op')!r}")
        return (f"def _explore_factor(obs, step):\n"
                f"    return {hi!r} if float(obs[{idx}]) {op} "
                f"{thr!r} else {lo!r}\n")
    raise ValueError(
        f"ef program kind {kind!r} has no source emitter — extend "
        f"ef_program_to_source or keep the dict and call "
        f"eval_ef_program at replay")


register_ef_program("obs_threshold", _obs_threshold)
