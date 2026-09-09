"""``timeline`` — event timeline tool (S6).

S6: Parses the training log for significant events (resume, new best,
floor transitions, early stop, seed overrides) and overlays them with
metric sparklines on the same time axis.  This establishes causal
judgment: "曲线在这里转折，是因为我做了什么？" (DEBUG_GUIDE.md §3.10).

Reuses ``_sparkline`` from ``analyze_training.py`` (P2 — no logic
duplication).

See ``DEBUG_GUIDE.md`` §3.10 ``timeline``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """A single event on the training timeline."""
    update: int
    """Update number at which the event occurred (0 for pre-training)."""
    description: str
    """Human-readable description of the event."""
    kind: str
    """Event kind: 'resume', 'reset_update', 'best', 'floor', 'early_stop',
    'seed'."""


def parse_events(run_dir: Path) -> List[TimelineEvent]:
    """Parse significant events from the training log.

    Recognized event patterns:
    - ``[checkpoint] resuming from update X`` → resume event
    - ``[checkpoint] update counter reset to 0`` → reset_update event
    - ``[eval N] ... [new_best]`` → best checkpoint event at update N
    - ``[floor] ... disabling floor loss at update X`` → floor transition
    - ``[early_stop] ... stopping at update X`` → early stop event
    - ``[seed] overridden to X`` → seed override event

    Args:
        run_dir: Training run directory.

    Returns:
        List of TimelineEvent sorted by update number.
    """
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return []

    events: List[TimelineEvent] = []
    current_update = 0

    with open(log_path) as f:
        for line in f:
            # Track current update from [update N] markers
            um = re.search(r"\[update\s+(\d+)\]", line)
            if um:
                current_update = int(um.group(1))

            # Resume
            if "[checkpoint] resuming from update" in line:
                m = re.search(r"resuming from update (\d+)", line)
                u = int(m.group(1)) if m else current_update
                events.append(TimelineEvent(
                    update=u, description="resume from checkpoint",
                    kind="resume",
                ))

            # Reset update
            if "[checkpoint] update counter reset to 0" in line:
                m = re.search(r"was (\d+)", line)
                was = m.group(1) if m else "?"
                events.append(TimelineEvent(
                    update=0,
                    description=f"reset update counter (was {was})",
                    kind="reset_update",
                ))

            # New best
            if "[new_best]" in line:
                m = re.search(r"\[eval\s+(\d+)\]", line)
                u = int(m.group(1)) if m else current_update
                events.append(TimelineEvent(
                    update=u, description="best checkpoint",
                    kind="best",
                ))

            # Floor transition
            if "[floor]" in line and "disabling floor loss" in line:
                m = re.search(r"update (\d+)", line)
                u = int(m.group(1)) if m else current_update
                events.append(TimelineEvent(
                    update=u, description="floor 阶段切换 (floor loss 关闭)",
                    kind="floor",
                ))

            # Early stop (training-level, not per-minibatch PPO early stop)
            if "[early_stop]" in line and "stopping at update" in line:
                m = re.search(r"update (\d+)", line)
                u = int(m.group(1)) if m else current_update
                events.append(TimelineEvent(
                    update=u, description="early stop (无改善)",
                    kind="early_stop",
                ))

            # Seed override
            if "[seed] overridden to" in line:
                m = re.search(r"overridden to (\d+)", line)
                seed = m.group(1) if m else "?"
                events.append(TimelineEvent(
                    update=0, description=f"seed overridden to {seed}",
                    kind="seed",
                ))

    # Sort by update, then by insertion order (stable)
    events.sort(key=lambda e: e.update)
    return events


# ---------------------------------------------------------------------------
# Metric series extraction
# ---------------------------------------------------------------------------

_RAW_STATS_RE = re.compile(r"__RAW_STATS__\s*(\{.*\})")


def parse_metric_series(
    run_dir: Path,
    metric_names: Optional[List[str]] = None,
) -> Tuple[List[int], Dict[str, List[float]]]:
    """Extract metric time series from the training log.

    Args:
        run_dir: Training run directory.
        metric_names: If given, only extract these metrics.  Otherwise,
            extract a default set of common metrics.

    Returns:
        (updates, metrics) where:
        - updates: list of update numbers
        - metrics: dict mapping metric name to list of values
    """
    if metric_names is None:
        metric_names = ["uncertainty", "std_mean", "approx_kl",
                        "max_pot", "success", "step"]

    log_path = run_dir / "train.log"
    if not log_path.exists():
        return [], {m: [] for m in metric_names}

    updates: List[int] = []
    series: Dict[str, List[float]] = {m: [] for m in metric_names}

    with open(log_path) as f:
        for line in f:
            m = _RAW_STATS_RE.search(line)
            if not m:
                continue
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            u = data.get("update", 0)
            updates.append(u)
            stats = data.get("stats", {})
            eval_info = data.get("eval_info", {})
            for metric in metric_names:
                val = None
                if metric in stats and isinstance(stats[metric], (int, float)):
                    val = float(stats[metric])
                elif metric in eval_info and isinstance(eval_info[metric], (int, float)):
                    val = float(eval_info[metric])
                series[metric].append(val if val is not None else 0.0)

    return updates, series


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_timeline(
    events: List[TimelineEvent],
    updates: List[int],
    series: Dict[str, List[float]],
    overlay: Optional[List[str]] = None,
) -> str:
    """Render the event timeline with optional metric sparklines.

    Format per DEBUG_GUIDE.md §3.10::

        u0000  ├─ resume from checkpoint
        u0110  ├─ floor 阶段切换
        u0250  └─ 当前

        uncertainty  ▁▂▃▅▆▇▇▇▆▅▄▃▃▂▂▂▂
        max_pot      ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
    """
    # Import sparkline from analyze_training.py (P2 — no duplication)
    from baseline.framework.analyze_training import _sparkline

    lines: List[str] = []
    run_name = ""  # Will be filled by caller if needed

    # --- Event tree ---
    if events:
        lines.append("事件时间线")
        lines.append("")
        for i, ev in enumerate(events):
            is_last = (i == len(events) - 1)
            prefix = "└─" if is_last else "├─"
            lines.append(f"u{ev.update:04d}  {prefix} {ev.description}")

    # Current position
    if updates:
        current = updates[-1]
        if not events or events[-1].update != current:
            lines.append(f"u{current:04d}  └─ 当前")

    # --- Metric sparklines ---
    if overlay:
        metrics_to_show = [m for m in overlay if m in series]
    else:
        metrics_to_show = [m for m in series if any(v != 0 for v in series[m])]

    if metrics_to_show:
        lines.append("")
        for metric in metrics_to_show:
            vals = series.get(metric, [])
            if vals:
                spark = _sparkline(vals, width=40)
                lines.append(f"{metric:<20s} {spark}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def timeline(
    run_dir: Path,
    overlay: Optional[List[str]] = None,
) -> str:
    """Generate an event timeline with optional metric sparklines.

    Args:
        run_dir: Training run directory.
        overlay: Optional list of metric names to show as sparklines.
            If None, shows all non-zero metrics.

    Returns:
        Human-readable timeline string.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    events = parse_events(run_dir)

    # Determine which metrics to extract
    metric_names = overlay if overlay else None
    updates, series = parse_metric_series(run_dir, metric_names=metric_names)

    return render_timeline(events, updates, series, overlay=overlay)
