"""``attribute`` — update attribution tool.

S3: Reads S0 aggregates from the training log's ``__RAW_STATS__``
lines and renders an attribution report:

- Channel influence share (bar chart)
- Dead frame ratio
- Per-action-dimension gradient norms (top/bottom 3)

This is a **log-only** tool — no snapshot needed.  It answers "who is
driving the policy this update?" using the S0 aggregates already
logged by ``ppo_update``.

See ``DEBUG_GUIDE.md`` §3.3 ``attribute``.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Log parsing (shared pattern — not a second parser, just the stats extractor)
# ---------------------------------------------------------------------------

_RAW_STATS_RE = re.compile(r"__RAW_STATS__\s*(\{.*\})")


def parse_log_stats(run_dir: Path, target_update: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Find the latest (or specific) ``__RAW_STATS__`` entry in train.log.

    Args:
        run_dir: Training run directory (contains train.log).
        target_update: If given, find the entry for this update number.
            If None, return the latest entry.

    Returns:
        The parsed ``stats`` dict (the inner ``"stats"`` key), or None
        if no matching entry is found.
    """
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return None

    latest: Optional[Dict[str, Any]] = None
    with open(log_path) as f:
        for line in f:
            m = _RAW_STATS_RE.search(line)
            if not m:
                continue
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if target_update is not None:
                if data.get("update") == target_update:
                    return data.get("stats", data)
            else:
                latest = data

    if latest is None:
        return None
    return latest.get("stats", latest)


def parse_log_stats_window(run_dir: Path, window: int) -> List[Dict[str, Any]]:
    """Return the last ``window`` ``__RAW_STATS__`` stats dicts."""
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    with open(log_path) as f:
        for line in f:
            m = _RAW_STATS_RE.search(line)
            if not m:
                continue
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            entries.append(data.get("stats", data))

    return entries[-window:]


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------

@dataclass
class AttributeReport:
    update: int
    influence_shares: Dict[str, float] = field(default_factory=dict)
    aw_normed: Dict[str, float] = field(default_factory=dict)
    dead_frame_ratio: float = 0.0
    action_dim_grad_norms: Optional[np.ndarray] = None
    n_channels: int = 0


def build_report(
    run_dir: Path,
    *,
    update: Optional[int] = None,
    window: int = 1,
) -> AttributeReport:
    """Build an attribution report from the training log.

    Args:
        run_dir: Training run directory.
        update: Specific update number.  If None, uses the latest.
        window: Number of recent updates to average (when update is None).
    """
    if update is not None:
        stats = parse_log_stats(run_dir, update)
        if stats is None:
            raise FileNotFoundError(
                f"No __RAW_STATS__ entry for update {update} in {run_dir}/train.log"
            )
        return _stats_to_report(stats, update)

    # Window mode: average over last N updates.
    entries = parse_log_stats_window(run_dir, window)
    if not entries:
        raise FileNotFoundError(
            f"No __RAW_STATS__ entries in {run_dir}/train.log"
        )

    if len(entries) == 1:
        return _stats_to_report(entries[0], 0)

    # Average across entries.
    return _average_stats_to_report(entries)


def _stats_to_report(stats: Dict[str, Any], update: int) -> AttributeReport:
    """Convert one stats dict to an AttributeReport."""
    influence: Dict[str, float] = {}
    aw_normed: Dict[str, float] = {}
    for key, val in stats.items():
        if key.startswith("influence_share_"):
            ch = key[len("influence_share_"):]
            influence[ch] = float(val)
        elif key.startswith("aw_normed_"):
            ch = key[len("aw_normed_"):]
            aw_normed[ch] = float(val)

    dead_frame = float(stats.get("dead_frame_ratio", 0.0))

    # Per-dim gradients: grad_dim_00, grad_dim_01, ...
    grad_dims: List[float] = []
    i = 0
    while f"grad_dim_{i:02d}" in stats:
        grad_dims.append(float(stats[f"grad_dim_{i:02d}"]))
        i += 1
    action_dim_grads = np.array(grad_dims, dtype=np.float32) if grad_dims else None

    return AttributeReport(
        update=update,
        influence_shares=influence,
        aw_normed=aw_normed,
        dead_frame_ratio=dead_frame,
        action_dim_grad_norms=action_dim_grads,
        n_channels=len(influence),
    )


def _average_stats_to_report(entries: List[Dict[str, Any]]) -> AttributeReport:
    """Average multiple stats dicts into one report."""
    # Collect all channel keys.
    influence: Dict[str, List[float]] = {}
    aw_normed: Dict[str, List[float]] = {}
    dead_frames: List[float] = []
    grad_dim_keys: Optional[List[str]] = None

    for stats in entries:
        for key, val in stats.items():
            if key.startswith("influence_share_"):
                ch = key[len("influence_share_"):]
                influence.setdefault(ch, []).append(float(val))
            elif key.startswith("aw_normed_"):
                ch = key[len("aw_normed_"):]
                aw_normed.setdefault(ch, []).append(float(val))
        dead_frames.append(float(stats.get("dead_frame_ratio", 0.0)))
        # Collect grad_dim keys from first entry that has them.
        if grad_dim_keys is None:
            gk = sorted(k for k in stats if k.startswith("grad_dim_"))
            if gk:
                grad_dim_keys = gk

    avg_influence = {k: sum(v) / len(v) for k, v in influence.items()}
    avg_aw = {k: sum(v) / len(v) for k, v in aw_normed.items()}
    avg_dead = sum(dead_frames) / len(dead_frames) if dead_frames else 0.0

    action_dim_grads = None
    if grad_dim_keys:
        grad_sums = {k: 0.0 for k in grad_dim_keys}
        count = 0
        for stats in entries:
            if all(k in stats for k in grad_dim_keys):
                for k in grad_dim_keys:
                    grad_sums[k] += float(stats[k])
                count += 1
        if count > 0:
            action_dim_grads = np.array(
                [grad_sums[k] / count for k in grad_dim_keys],
                dtype=np.float32,
            )

    return AttributeReport(
        update=0,  # averaged
        influence_shares=avg_influence,
        aw_normed=avg_aw,
        dead_frame_ratio=avg_dead,
        action_dim_grad_norms=action_dim_grads,
        n_channels=len(avg_influence),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_report(report: AttributeReport, *, by_action_dim: bool = False) -> str:
    """Render an AttributeReport as a human-readable string."""
    lines: List[str] = []

    if report.update > 0:
        lines.append(f"更新归因 @ update {report.update}")
    else:
        lines.append("更新归因（窗口平均）")
    lines.append("")

    # --- Channel influence share (bar chart) ---
    lines.append("通道影响份额（Σ|aw_normed × conf × normed_adv|，归一化）")
    if report.influence_shares:
        max_share = max(report.influence_shares.values()) if report.influence_shares else 1.0
        for ch, share in sorted(report.influence_shares.items(), key=lambda x: -x[1]):
            bar_len = int(share / max(max_share, 1e-12) * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            pct = share * 100
            lines.append(f"  {ch:<16s} {bar} {pct:5.1f}%")
    else:
        lines.append("  (无 influence_share 数据)")
    lines.append("")

    # --- Dead frame ratio ---
    lines.append(f"无梯度帧占比  {report.dead_frame_ratio * 100:.1f}%"
                 f"   （Σ|aw| = 0，这些帧对 actor 完全无贡献）")
    lines.append("")

    # --- Per-action-dim gradients ---
    if by_action_dim and report.action_dim_grad_norms is not None:
        grads = report.action_dim_grad_norms
        n = len(grads)
        if n > 0:
            max_grad = float(grads.max())
            if max_grad > 0:
                ratios = grads / max_grad
            else:
                ratios = grads

            # Top 3 and bottom 3.
            indices = np.argsort(grads)
            top3 = indices[-3:][::-1]
            bot3 = indices[:3]

            lines.append("按动作维度的梯度分布（top / bottom 3）")
            top_strs = [f"dim_{i} {grads[i]:.4f}" for i in top3]
            lines.append(f"  最大  {' | '.join(top_strs)}")
            bot_strs = [f"dim_{i} {grads[i]:.4f}" for i in bot3]
            lines.append(f"  最小  {' | '.join(bot_strs)}")

            # Warning for near-zero gradients.
            min_grad = float(grads[bot3[0]])
            if min_grad < 0.01 * max_grad:
                lines.append(f"        ⚠ 部分维度梯度接近零——对应行为在物理上无法通过当前梯度改变")
    elif by_action_dim:
        lines.append("按动作维度的梯度分布：不可用（实验未提供 action_dim_grad_norms）")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attribute(
    run_dir: Path,
    *,
    update: Optional[int] = None,
    window: int = 1,
    by_action_dim: bool = False,
) -> str:
    """Run the attribution tool and return the rendered report.

    Args:
        run_dir: Training run directory.
        update: Specific update number.  If None, uses latest.
        window: Number of recent updates to average (when update is None).
        by_action_dim: Include per-action-dimension gradient breakdown.

    Returns:
        Human-readable report string.
    """
    report = build_report(run_dir, update=update, window=window)
    return render_report(report, by_action_dim=by_action_dim)


__all__ = [
    "AttributeReport",
    "build_report",
    "render_report",
    "attribute",
    "parse_log_stats",
    "parse_log_stats_window",
]
