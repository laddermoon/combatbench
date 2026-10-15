"""Slice contract verifiers — experiment-owned reward/gate recompute.

Registered by ``versions.reward_semantics``; consumed by
``analysis.slice_verification`` via late import (framework never imports
this module at load time). Each verifier receives the dump batch's numpy
rows plus the captured ``spec.slice_contract`` (or ``None`` for legacy
dumps) and returns a list of check dicts.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

# semantics -> verifier(batch_np, channels, contract) -> List[check dict]
_SLICE_VERIFIERS: Dict[str, Callable] = {}


def register_slice_verifier(semantics: str) -> Callable:
    def _wrap(fn: Callable) -> Callable:
        _SLICE_VERIFIERS[semantics] = fn
        return fn
    return _wrap


def get_slice_verifier(semantics: str) -> Optional[Callable]:
    return _SLICE_VERIFIERS.get(str(semantics))


def known_semantics() -> List[str]:
    return sorted(_SLICE_VERIFIERS)


def _top_violating_rows(err: np.ndarray, ids: np.ndarray, k: int = 20) -> List[int]:
    idx = np.argsort(-np.abs(err))[:k]
    return [int(ids[i]) for i in idx]


def _scaled_clip_verify(
    batch: Mapping[str, np.ndarray],
    *,
    channel_idx: int,
    spec: Mapping[str, Any],
    ids: np.ndarray,
    atol: float,
) -> Optional[Dict[str, Any]]:
    source = spec.get("source", "")
    facts = batch["task_facts"] if source.startswith("task_facts.") else batch["reward_features"]
    key = source.split(".", 1)[1]
    src = facts.get(key)
    if src is None:
        return {
            "name": f"reward_source_missing:{source}",
            "status": "fail",
            "detail": f"contract source {source!r} absent from batch",
        }
    clip_lo, clip_hi = spec.get("clip", [0.0, 1.0])
    expected = float(spec.get("scale", 1.0)) * np.clip(
        np.asarray(src, dtype=np.float64), clip_lo, clip_hi
    )
    actual = np.asarray(batch["rewards"][:, channel_idx], dtype=np.float64)
    err = np.abs(expected - actual)
    return {
        "name": f"reward_recompute:{source}",
        "status": "pass" if float(err.max(initial=0.0)) <= atol else "fail",
        "metric": {"max_abs_err": float(err.max(initial=0.0)), "atol": atol},
        "violating_samples": _top_violating_rows(err, ids)
        if float(err.max(initial=0.0)) > atol else [],
        "detail": f"rewards[:, {channel_idx}] ?= {spec.get('scale')}*clip({source})",
    }


def _identity_verify(
    batch: Mapping[str, np.ndarray],
    *,
    channel_idx: int,
    spec: Mapping[str, Any],
    ids: np.ndarray,
    atol: float,
) -> Optional[Dict[str, Any]]:
    source = spec.get("source", "")
    facts = batch["task_facts"] if source.startswith("task_facts.") else batch["reward_features"]
    key = source.split(".", 1)[1]
    src = facts.get(key)
    if src is None:
        return {
            "name": f"reward_source_missing:{source}",
            "status": "fail",
            "detail": f"contract source {source!r} absent from batch",
        }
    expected = np.asarray(src, dtype=np.float64)
    actual = np.asarray(batch["rewards"][:, channel_idx], dtype=np.float64)
    err = np.abs(expected - actual)
    return {
        "name": f"reward_recompute:{source}",
        "status": "pass" if float(err.max(initial=0.0)) <= atol else "fail",
        "metric": {"max_abs_err": float(err.max(initial=0.0)), "atol": atol},
        "violating_samples": _top_violating_rows(err, ids)
        if float(err.max(initial=0.0)) > atol else [],
        "detail": f"rewards[:, {channel_idx}] ?= {source}",
    }


def _gate_phi_squared(
    batch: Mapping[str, np.ndarray],
    *,
    channels: Sequence[str],
    gate_spec: Mapping[str, Any],
    ids: np.ndarray,
    atol: float,
    contract_source: str,
) -> List[Dict[str, Any]]:
    """w = base_weights * (1, phi^2); actor_weight = gate / sum(gate)."""
    weights = np.asarray(gate_spec.get("weights", [3.0, 1.0]), dtype=np.float64)
    clip_lo, clip_hi = gate_spec.get("clip", [0.0, 1.0])
    pre_src = str(gate_spec.get("pre_source", "task_facts.phi_pre"))
    post_src = str(gate_spec.get("post_source", "task_facts.phi_post_reference"))
    facts = batch["task_facts"]
    phi_pre = facts.get(pre_src.split(".", 1)[1])
    phi_post = facts.get(post_src.split(".", 1)[1])
    checks: List[Dict[str, Any]] = []
    for label, phi, field in (
        ("current", phi_pre, "actor_weight"),
        ("next", phi_post, "actor_weight_next"),
    ):
        if phi is None:
            checks.append({
                "name": f"actor_weight_gate_recompute:{label}",
                "status": "fail",
                "detail": f"gate source missing for {label} ({pre_src if label=='current' else post_src})",
            })
            continue
        phi_c = np.clip(np.asarray(phi, dtype=np.float64), clip_lo, clip_hi)
        gates = np.stack(
            [np.full_like(phi_c, weights[0]), weights[1] * phi_c ** 2],
            axis=1,
        )
        expected = gates / gates.sum(axis=1, keepdims=True)
        actual = np.asarray(batch[field], dtype=np.float64)
        # channel order in the contract: [r_fall, r_cross] → weights[0],
        # weights[1]*phi^2. Verify against recorded per-channel weights.
        err = np.abs(expected - actual)
        row_err = err.max(axis=1)
        checks.append({
            "name": f"actor_weight_gate_recompute:{label}",
            "status": "pass" if float(row_err.max(initial=0.0)) <= atol else "fail",
            "metric": {
                "max_abs_err": float(row_err.max(initial=0.0)),
                "atol": atol,
                "contract_source": contract_source,
            },
            "violating_samples": _top_violating_rows(row_err, ids)
            if float(row_err.max(initial=0.0)) > atol else [],
            "detail": (
                f"{field} ?= normalize([{weights[0]}, {weights[1]}*phi^2]) "
                f"via {pre_src if label == 'current' else post_src}"
            ),
        })
    return checks


def _gate_constant_ones(
    batch: Mapping[str, np.ndarray],
    *,
    channels: Sequence[str],
    ids: np.ndarray,
    atol: float,
    contract_source: str,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for label, field in (("current", "actor_weight"), ("next", "actor_weight_next")):
        actual = np.asarray(batch[field], dtype=np.float64)
        expected = np.full_like(actual, 1.0 / max(len(channels), 1))
        err = np.abs(expected - actual).max(axis=1)
        checks.append({
            "name": f"actor_weight_gate_recompute:{label}",
            "status": "pass" if float(err.max(initial=0.0)) <= atol else "fail",
            "metric": {
                "max_abs_err": float(err.max(initial=0.0)),
                "atol": atol,
                "contract_source": contract_source,
            },
            "violating_samples": _top_violating_rows(err, ids)
            if float(err.max(initial=0.0)) > atol else [],
            "detail": f"{field} ?= uniform 1/{len(channels)} (constant_ones gate)",
        })
    return checks


@register_slice_verifier("standup_4stage_dense_v2")
def verify_standup_4stage_dense_v2(
    batch: Mapping[str, np.ndarray],
    channels: Sequence[str],
    contract: Optional[Mapping[str, Any]],
    *,
    atol: float,
    contract_source: str,
) -> List[Dict[str, Any]]:
    ids = np.asarray(batch["sample_ids"])
    checks: List[Dict[str, Any]] = []
    rewards = (contract or {}).get("rewards") or {
        "r_potential": {
            "kind": "scaled_clip",
            "source": "task_facts.potential",
            "scale": 1.0 - 0.99,
            "clip": [0.0, 1.0],
        },
    }
    for c, ch in enumerate(channels):
        spec = rewards.get(ch)
        if spec is None:
            checks.append({
                "name": f"reward_contract_missing:{ch}",
                "status": "fail",
                "detail": f"no reward contract for channel {ch!r}",
            })
            continue
        res = _scaled_clip_verify(
            batch, channel_idx=c, spec=spec, ids=ids, atol=atol,
        )
        if res is not None:
            res["metric"]["contract_source"] = contract_source
            checks.append(res)
    checks.extend(_gate_constant_ones(
        batch, channels=channels, ids=ids, atol=atol,
        contract_source=contract_source,
    ))
    return checks


@register_slice_verifier("basic_balance_phi_cross_v1")
def verify_basic_balance_phi_cross_v1(
    batch: Mapping[str, np.ndarray],
    channels: Sequence[str],
    contract: Optional[Mapping[str, Any]],
    *,
    atol: float,
    contract_source: str,
) -> List[Dict[str, Any]]:
    ids = np.asarray(batch["sample_ids"])
    checks: List[Dict[str, Any]] = []
    rewards = (contract or {}).get("rewards") or {
        "r_fall": {
            "kind": "scaled_clip",
            "source": "task_facts.phi_post_reference",
            "scale": 0.01,
            "clip": [0.0, 1.0],
        },
        "r_cross": {
            "kind": "identity",
            "source": "reward_features.cross_support",
        },
    }
    for c, ch in enumerate(channels):
        spec = rewards.get(ch)
        if spec is None:
            checks.append({
                "name": f"reward_contract_missing:{ch}",
                "status": "fail",
                "detail": f"no reward contract for channel {ch!r}",
            })
            continue
        if spec.get("kind") == "scaled_clip":
            res = _scaled_clip_verify(
                batch, channel_idx=c, spec=spec, ids=ids, atol=atol,
            )
        else:
            res = _identity_verify(
                batch, channel_idx=c, spec=spec, ids=ids, atol=atol,
            )
        if res is not None:
            res["metric"]["contract_source"] = contract_source
            checks.append(res)
    gate_spec = (contract or {}).get("gate") or {
        "kind": "phi_squared",
        "weights": [3.0, 1.0],
        "pre_source": "task_facts.phi_pre",
        "post_source": "task_facts.phi_post_reference",
        "clip": [0.0, 1.0],
    }
    checks.extend(_gate_phi_squared(
        batch, channels=channels, gate_spec=gate_spec, ids=ids,
        atol=atol, contract_source=contract_source,
    ))
    return checks
