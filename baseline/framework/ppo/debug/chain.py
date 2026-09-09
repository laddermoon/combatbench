"""``chain`` — signal chain profile (九环剖面).

S3: The most important debug tool.  It turns the nine-ring causal
chain mental model into a readable table, automatically identifying
the main breakpoint and explicitly ruling out change categories that
won't help.

Three data sources:
1. **Training log** (④⑦⑧ rings): S0 aggregates from ``__RAW_STATS__``
2. **Snapshot replay** (②③⑤⑥ rings): per-frame quantiles from .npz
3. **Probe** (①⑨ rings): behavior occurrence rate + pass rate

See ``DEBUG_GUIDE.md`` §3.2 ``chain``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .attribute import parse_log_stats


# ---------------------------------------------------------------------------
# Ring definitions and thresholds
# ---------------------------------------------------------------------------

# Judgment symbols
_OK = "✓"
_WARN = "⚠"
_FAIL = "✗"
_NA = "—"

# Threshold rules (user chose: fixed thresholds per ring)
THRESHOLDS = {
    # ①物理: behavior occurrence rate from probe
    "physics_rate_fail": 0.05,   # <5% → ✗
    "physics_rate_warn": 0.20,   # <20% → ⚠
    # ②观测: non-zero frame ratio
    "obs_nonzero_warn": 0.50,    # <50% → ⚠
    # ③奖励: P50 near zero + low non-zero ratio
    "reward_nonzero_warn": 0.30, # <30% non-zero → ⚠
    # ④预测: explained variance
    "ev_fail": 0.30,             # <0.3 → ✗
    "ev_warn": 0.60,             # <0.6 → ⚠
    # ⑤优势: SNR = |mean| / std
    "adv_snr_fail": 0.01,        # <0.01 → ✗
    "adv_snr_warn": 0.05,        # <0.05 → ⚠
    # ⑥门控: aw non-zero ratio + mean sign
    "gate_nonzero_warn": 0.50,   # <50% → ⚠
    # ⑦合成: influence share
    "combine_fail": 0.05,        # <5% → ✗
    "combine_warn": 0.10,        # <10% → ⚠
    # ⑧梯度: per-dim gradient ratio to max
    "grad_fail": 0.01,           # <1% of max → ✗
    # ⑨行为: probe pass rate
    "behavior_fail": 0.01,       # ~0% → ✗
    "behavior_warn": 0.50,       # <50% → ⚠
}


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------

@dataclass
class RingResult:
    """One ring's assessment."""
    ring: int          # 1-9
    name: str          # "物理", "观测", etc.
    quantity: str      # human-readable quantity description
    judgment: str      # "✓", "⚠", "✗", "—"
    detail: str = ""   # extra detail (values, thresholds)


@dataclass
class ChainReport:
    """Full nine-ring chain profile."""
    channel: str
    update: int
    rings: List[RingResult] = field(default_factory=list)
    main_breakpoint: Optional[int] = None  # ring number (1-9) or None
    diagnosis: str = ""

    @property
    def secondary_breakpoints(self) -> List[int]:
        """Rings with ✗ that are not the main breakpoint."""
        return [
            r.ring for r in self.rings
            if r.judgment == _FAIL and r.ring != self.main_breakpoint
        ]


# ---------------------------------------------------------------------------
# Per-ring assessment functions
# ---------------------------------------------------------------------------

def _assess_physics(probe_rate: Optional[float]) -> RingResult:
    """①物理: behavior occurrence rate from probe."""
    if probe_rate is None:
        return RingResult(1, "物理", "行为出现率 未测", _NA)
    pct = probe_rate * 100
    if probe_rate < THRESHOLDS["physics_rate_fail"]:
        j = _FAIL
    elif probe_rate < THRESHOLDS["physics_rate_warn"]:
        j = _WARN
    else:
        j = _OK
    return RingResult(1, "物理", f"行为出现率 {pct:.1f}%", j,
                      f"阈值: <{THRESHOLDS['physics_rate_fail']*100:.0f}%✗ "
                      f"<{THRESHOLDS['physics_rate_warn']*100:.0f}%⚠")


def _assess_observation(arrays: Dict[str, np.ndarray], channel: str) -> RingResult:
    """②观测: non-zero frame ratio of debug arrays (h_left, etc.)."""
    # Use h_left or h_right as proxy for observation activity.
    key = None
    for k in ("debug.h_left", "debug.h_right"):
        if k in arrays:
            key = k
            break
    if key is None:
        return RingResult(2, "观测", "observer 一致性 未提供 debug_arrays", _NA)
    arr = np.asarray(arrays[key], dtype=np.float32)
    nonzero_ratio = float(np.mean(np.abs(arr) > 1e-6))
    lo, hi = float(arr.min()), float(arr.max())
    if nonzero_ratio < THRESHOLDS["obs_nonzero_warn"]:
        j = _WARN
    else:
        j = _OK
    return RingResult(2, "观测",
                      f"非零帧 {nonzero_ratio*100:.0f}% 量级 [{lo:.3f}, {hi:.3f}]",
                      j, f"阈值: 非零帧<{THRESHOLDS['obs_nonzero_warn']*100:.0f}%⚠")


def _assess_reward(arrays: Dict[str, np.ndarray], channel: str) -> RingResult:
    """③奖励: non-zero frame ratio + P50 of reward."""
    # Reward is in the buffer or gae stage.  We look for rewards in
    # the gae stage (advantages/returns are derived from rewards).
    # Actually rewards are not directly stored in the sink — we use
    # the returns as a proxy for reward signal presence.
    key = f"gae.returns.{channel}"
    if key not in arrays:
        return RingResult(3, "奖励", "reward 分布 需快照", _NA)
    arr = np.asarray(arrays[key], dtype=np.float32)
    nonzero_ratio = float(np.mean(np.abs(arr) > 1e-6))
    p50 = float(np.median(np.abs(arr)))
    if nonzero_ratio < THRESHOLDS["reward_nonzero_warn"] and p50 < 1e-4:
        j = _WARN
    else:
        j = _OK
    return RingResult(3, "奖励",
                      f"非零帧 {nonzero_ratio*100:.0f}% P50={p50:.4f}",
                      j, f"阈值: 非零帧<{THRESHOLDS['reward_nonzero_warn']*100:.0f}%⚠")


def _assess_prediction(ev: Optional[float]) -> RingResult:
    """④预测: explained variance from log."""
    if ev is None:
        return RingResult(4, "预测", "EV 未在日志中", _NA)
    if ev < THRESHOLDS["ev_fail"]:
        j = _FAIL
    elif ev < THRESHOLDS["ev_warn"]:
        j = _WARN
    else:
        j = _OK
    return RingResult(4, "预测", f"EV {ev:.3f}", j,
                      f"阈值: <{THRESHOLDS['ev_fail']}✗ <{THRESHOLDS['ev_warn']}⚠")


def _assess_advantage(arrays: Dict[str, np.ndarray], channel: str) -> RingResult:
    """⑤优势: adv_std, |mean|, SNR."""
    key = f"gae.advantages.{channel}"
    if key not in arrays:
        return RingResult(5, "优势", "adv 统计 需快照", _NA)
    arr = np.asarray(arrays[key], dtype=np.float32)
    std = float(arr.std())
    mean_abs = float(np.abs(arr).mean())
    snr = mean_abs / max(std, 1e-12)
    if snr < THRESHOLDS["adv_snr_fail"]:
        j = _FAIL
    elif snr < THRESHOLDS["adv_snr_warn"]:
        j = _WARN
    else:
        j = _OK
    return RingResult(5, "优势",
                      f"adv_std {std:.4f} |mean| {mean_abs:.4f} SNR {snr:.4f}",
                      j, f"阈值: SNR<{THRESHOLDS['adv_snr_fail']}✗ "
                      f"<{THRESHOLDS['adv_snr_warn']}⚠")


def _assess_gating(arrays: Dict[str, np.ndarray], channel: str) -> RingResult:
    """⑥门控: aw non-zero ratio + mean sign."""
    key = f"combine.aw_frame.{channel}"
    if key not in arrays:
        return RingResult(6, "门控", "aw 统计 需快照", _NA)
    arr = np.asarray(arrays[key], dtype=np.float32)
    nonzero_ratio = float(np.mean(np.abs(arr) > 1e-12))
    mean = float(arr.mean())
    parts = []
    if nonzero_ratio < THRESHOLDS["gate_nonzero_warn"]:
        parts.append(f"非零帧 {nonzero_ratio*100:.0f}%⚠")
    if mean < 0:
        parts.append(f"mean {mean:.4f}⚠ (净抑制)")
    j = _WARN if parts else _OK
    detail = f"非零帧 {nonzero_ratio*100:.0f}% mean {mean:.4f}"
    return RingResult(6, "门控", detail, j, " ".join(parts) if parts else "")


def _assess_combination(influence: Optional[float]) -> RingResult:
    """⑦合成: influence share from log."""
    if influence is None:
        return RingResult(7, "合成", "影响份额 未在日志中", _NA)
    pct = influence * 100
    if influence < THRESHOLDS["combine_fail"]:
        j = _FAIL
    elif influence < THRESHOLDS["combine_warn"]:
        j = _WARN
    else:
        j = _OK
    return RingResult(7, "合成", f"影响份额 {pct:.1f}%", j,
                      f"阈值: <{THRESHOLDS['combine_fail']*100:.0f}%✗ "
                      f"<{THRESHOLDS['combine_warn']*100:.0f}%⚠")


def _assess_gradient(grad_norms: Optional[np.ndarray]) -> RingResult:
    """⑧梯度: per-dim gradient ratio to max."""
    if grad_norms is None:
        return RingResult(8, "梯度", "按维度梯度 不可用", _NA)
    grads = np.asarray(grad_norms, dtype=np.float32)
    max_grad = float(grads.max())
    if max_grad <= 0:
        return RingResult(8, "梯度", "梯度全零", _FAIL)
    min_ratio = float(grads.min()) / max_grad
    if min_ratio < THRESHOLDS["grad_fail"]:
        j = _FAIL
    else:
        j = _OK
    return RingResult(8, "梯度",
                      f"min/max 梯度比 {min_ratio:.4f} "
                      f"(max={max_grad:.4f})",
                      j, f"阈值: <{THRESHOLDS['grad_fail']*100:.0f}%✗")


def _assess_behavior(probe_rate: Optional[float]) -> RingResult:
    """⑨行为: probe pass rate."""
    if probe_rate is None:
        return RingResult(9, "行为", "probe 通过率 未测", _NA)
    pct = probe_rate * 100
    if probe_rate < THRESHOLDS["behavior_fail"]:
        j = _FAIL
    elif probe_rate < THRESHOLDS["behavior_warn"]:
        j = _WARN
    else:
        j = _OK
    return RingResult(9, "行为", f"probe 通过率 {pct:.1f}%", j)


# ---------------------------------------------------------------------------
# Diagnosis text generation
# ---------------------------------------------------------------------------

_DIAGNOSIS = {
    1: "这是探索问题，不是奖励问题。修 reward 权重不会有效果。\n"
       "      → 建议：debug.py whatif 试算探索类改动；或先用 probe 确认行为可达性。",
    2: "观测层可能有问题。用 frame --where 抽检 observer 输出。\n"
       "      → 检查 observer 是否正确记录了目标物理量。",
    3: "检查 reward 分布而非均值（§7.1）。\n"
       "      → 用 frame --where 抽检 reward 是否出现在正确的帧。",
    4: "critic 没学到。EV 过低意味着 V(s) 预测不准。\n"
       "      → 检查 critic 学习率、训练 epochs、reward scale。",
    5: "优势信噪比过低。advantage 几乎是噪声。\n"
       "      → 检查 reward scale、GAE lambda、normalize 设置。",
    6: "门控层有问题。actor_weight 配置可能未生效或方向错误。\n"
       "      → 用 attribute 看实际影响份额；用 whatif 试算权重改动。",
    7: "信号在此死亡。影响份额过低意味着该通道几乎不参与策略更新。\n"
       "      → 用 attribute 看实际影响份额；用 whatif --sweep 试算权重。",
    8: "梯度层有问题。目标关节的梯度接近零。\n"
       "      → 用 attribute --by action-dim 确认目标关节是否有梯度。",
    9: "指标可能失真。\n"
       "      → 用 metric --verify 验证你的指标测的是你以为的东西。",
}


def _generate_diagnosis(rings: List[RingResult]) -> Tuple[Optional[int], str]:
    """Find main breakpoint and generate diagnosis text."""
    main_bp = None
    for r in rings:
        if r.judgment == _FAIL:
            main_bp = r.ring
            break

    if main_bp is None:
        # No hard fail — check for warnings.
        warnings = [r for r in rings if r.judgment == _WARN]
        if warnings:
            return None, ("无硬性断点。以下环有警告：\n      "
                          + "; ".join(f"{r.ring}{r.name}" for r in warnings))
        return None, "所有环正常。"

    diag = _DIAGNOSIS.get(main_bp, "未知断点。")
    secondary = [r for r in rings if r.judgment == _FAIL and r.ring != main_bp]
    lines = [f"主断点在 {main_bp}（{rings[main_bp-1].name}）。"]
    if secondary:
        lines.append(f"次断点在 {', '.join(str(r.ring) for r in secondary)}。")
    lines.append(f"      {diag}")
    return main_bp, "\n      ".join(lines)


# ---------------------------------------------------------------------------
# Probe integration
# ---------------------------------------------------------------------------

def _run_probes_for_chain(
    run_dir: Path,
    update: int,
    n_workers: int = 2,
) -> Tuple[Optional[float], Optional[float]]:
    """Run probe suite and return (physics_rate, behavior_rate).

    physics_rate = best probe pass rate (①环 proxy: did any target
    behavior ever happen?).
    behavior_rate = pass rate of the most relevant probe (⑨环).
    """
    from .probes import (
        list_available_updates,
        load_experiment_from_run,
        load_policy_blueprint,
        run_probe_suite,
    )

    try:
        experiment = load_experiment_from_run(run_dir)
        suites = experiment.probe_suites()
        if not suites:
            return None, None

        # Find the update's policy export.
        available = list_available_updates(run_dir)
        if update not in available:
            # Use the nearest available update.
            if not available:
                return None, None
            update = max(u for u in available if u <= update) if any(u <= update for u in available) else min(available)

        policy_bp = load_policy_blueprint(run_dir, update)

        # Run the first suite (locomotion).
        suite = suites[0]
        result = run_probe_suite(experiment, policy_bp, suite, n_workers=n_workers)

        # physics_rate = max pass rate across all probes (did any behavior happen?)
        # behavior_rate = pass rate of the last probe (most advanced behavior)
        if result.probe_results:
            physics_rate = max(pr.pass_rate for pr in result.probe_results)
            behavior_rate = result.probe_results[-1].pass_rate
            return physics_rate, behavior_rate
    except Exception as e:
        # Probe failure is non-fatal — chain degrades gracefully.
        print(f"[chain] probe 跳过: {e}", flush=True)

    return None, None


# ---------------------------------------------------------------------------
# Main chain function
# ---------------------------------------------------------------------------

def chain(
    run_dir: Path,
    *,
    channel: str,
    update: Optional[int] = None,
    behavior: Optional[str] = None,
    snapshot_dir: Optional[Path] = None,
    run_probes: bool = True,
    n_workers: int = 2,
) -> ChainReport:
    """Build a nine-ring chain profile for one channel.

    Args:
        run_dir: Training run directory.
        channel: Reward channel name (e.g. "r_left_foot").
        update: Update number.  If None, uses the latest from the log.
        behavior: Optional behavior name (for display only).
        snapshot_dir: Snapshot directory.  If None, auto-finds
            ``<run_dir>/debug/u{update:05d}/``.
        run_probes: If True, auto-run probe suite for ①⑨ rings.
        n_workers: Number of rollout workers for probes.

    Returns:
        :class:`ChainReport`.
    """
    run_dir = Path(run_dir)

    # --- Determine update ---
    if update is None:
        stats = parse_log_stats(run_dir)
        if stats is None:
            raise FileNotFoundError(
                f"No __RAW_STATS__ in {run_dir}/train.log — cannot determine update"
            )
        # Need the raw update number — parse_log_stats returns just stats.
        # Re-parse to get update.
        import json, re
        from .attribute import _RAW_STATS_RE
        log_path = run_dir / "train.log"
        latest_update = 0
        with open(log_path) as f:
            for line in f:
                m = _RAW_STATS_RE.search(line)
                if not m:
                    continue
                try:
                    data = json.loads(m.group(1))
                    latest_update = data.get("update", latest_update)
                except json.JSONDecodeError:
                    continue
        update = latest_update

    # --- Source 1: Training log (④⑦⑧) ---
    log_stats = parse_log_stats(run_dir, update)
    ev = None
    influence = None
    grad_norms = None
    if log_stats:
        ev = log_stats.get(f"explained_variance_{channel}")
        if ev is not None:
            ev = float(ev)
        influence = log_stats.get(f"influence_share_{channel}")
        if influence is not None:
            influence = float(influence)
        # Per-dim gradients.
        grad_dims = []
        i = 0
        while f"grad_dim_{i:02d}" in log_stats:
            grad_dims.append(float(log_stats[f"grad_dim_{i:02d}"]))
            i += 1
        if grad_dims:
            grad_norms = np.array(grad_dims, dtype=np.float32)

    # --- Source 2: Snapshot replay (②③⑤⑥) ---
    arrays: Dict[str, np.ndarray] = {}
    if snapshot_dir is None:
        snapshot_dir = run_dir / "debug" / f"u{update:05d}"
    snapshot_dir = Path(snapshot_dir)
    replay_dir = snapshot_dir / "replay"
    if replay_dir.exists():
        from .where import build_frame_arrays
        arrays = build_frame_arrays(replay_dir)

    # --- Source 3: Probe (①⑨) ---
    physics_rate = None
    behavior_rate = None
    if run_probes:
        physics_rate, behavior_rate = _run_probes_for_chain(
            run_dir, update, n_workers=n_workers,
        )

    # --- Assess each ring ---
    rings: List[RingResult] = [
        _assess_physics(physics_rate),
        _assess_observation(arrays, channel),
        _assess_reward(arrays, channel),
        _assess_prediction(ev),
        _assess_advantage(arrays, channel),
        _assess_gating(arrays, channel),
        _assess_combination(influence),
        _assess_gradient(grad_norms),
        _assess_behavior(behavior_rate),
    ]

    # --- Diagnosis ---
    main_bp, diagnosis = _generate_diagnosis(rings)

    return ChainReport(
        channel=channel,
        update=update,
        rings=rings,
        main_breakpoint=main_bp,
        diagnosis=diagnosis,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_report(report: ChainReport) -> str:
    """Render a ChainReport as a human-readable string."""
    lines: List[str] = []
    lines.append(f"{report.channel} 信号链剖面 @ update {report.update}")
    lines.append("")
    lines.append(f"环             量                              判定")

    ring_names = {
        1: "①物理", 2: "②观测", 3: "③奖励", 4: "④预测",
        5: "⑤优势", 6: "⑥门控", 7: "⑦合成", 8: "⑧梯度", 9: "⑨行为",
    }

    for r in report.rings:
        name = ring_names.get(r.ring, f"{r.ring}")
        # Pad quantity to ~30 chars for alignment.
        qty = r.quantity
        lines.append(f"{name:<10s} {qty:<32s} {r.judgment}")
        if r.detail:
            lines.append(f"{'':>10s} {r.detail}")

    lines.append("")
    if report.main_breakpoint is not None:
        lines.append(f"诊断：{report.diagnosis}")
    else:
        lines.append(f"诊断：{report.diagnosis}")

    return "\n".join(lines)


__all__ = [
    "ChainReport",
    "RingResult",
    "chain",
    "render_report",
    "THRESHOLDS",
]
