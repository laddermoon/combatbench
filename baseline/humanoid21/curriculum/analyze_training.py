#!/usr/bin/env python3
"""
Universal PPO Training Log Analyzer.

Parses ``__RAW_STATS__`` JSON lines emitted by the curriculum training
loop and provides:

1. **PPO health diagnostics** — exploration collapse, early-stop, critic
   blindness, episode-length death spiral.  These checks use fixed
   thresholds that apply to *any* PPO experiment.

2. **Auto-discovered metric trends** — every key in ``bsum``, ``esum``,
   ``sinfo``, and ``rsum`` is automatically tracked and displayed with
   short-term (window) and long-term (full history) trends.  No
   experiment-specific configuration is needed.

This replaces the per-experiment ``analyze_*_logs.py`` scripts.  New
experiments that emit ``__RAW_STATS__`` are automatically supported.

Usage::

    # One-shot analysis of a log file
    python3 analyze_training.py balance.log

    # Real-time watch mode (like tail -f)
    python3 analyze_training.py balance.log --watch

    # Custom window size (default 10 updates)
    python3 analyze_training.py fight.log --window 20

    # Show only diagnostics (no trend table)
    python3 analyze_training.py fight.log --diagnostics-only

    # Full-history sparkline charts for all metrics
    python3 analyze_training.py fight.log --history

    # Filter history to specific metrics (substring match)
    python3 analyze_training.py fight.log --history survived
    python3 analyze_training.py fight.log --history ev

    # List all discovered metric names
    python3 analyze_training.py fight.log --list-metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers — nested dict path access
# ---------------------------------------------------------------------------

def _resolve(d: Any, path: str) -> Any:
    """Resolve a dotted path (e.g. ``stats.std_min``) inside a nested dict."""
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _series(history: deque, path: str) -> List[float]:
    """Extract a float time-series for a dotted path."""
    out: List[float] = []
    for d in history:
        v = _resolve(d, path)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _avg(history: deque, path: str, default: float = 0.0) -> float:
    vals = _series(history, path)
    return sum(vals) / len(vals) if vals else default


def _trend_slope(series: List[float]) -> float:
    """Simple linear regression slope."""
    n = len(series)
    if n < 2:
        return 0.0
    x = list(range(n))
    mx = sum(x) / n
    my = sum(series) / n
    num = sum((x[i] - mx) * (series[i] - my) for i in range(n))
    den = sum((xi - mx) ** 2 for xi in x)
    return num / den if den != 0 else 0.0


def _fmt_trend_arrow(slope: float, scale: float = 1.0) -> str:
    """Return a coloured arrow indicating trend direction."""
    s = slope * scale
    if abs(s) < 1e-6:
        return f"{DIM}→{RESET}"
    if s > 0:
        return f"{GREEN}↑{RESET}"
    return f"{RED}↓{RESET}"


def _fmt_float(v: float, precision: int = 3) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{precision}f}"


# ---------------------------------------------------------------------------
# ASCII sparkline (unicode block elements)
# ---------------------------------------------------------------------------

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float], width: int = 60) -> str:
    """Render a compact unicode sparkline for a value series.

    Downsamples to ``width`` points and maps each to a block character.
    """
    if not values:
        return ""
    n = len(values)
    if n <= width:
        # Pad with leading spaces to align right
        sampled = values
        pad = width - n
    else:
        # Downsample by averaging buckets
        bucket = n / width
        sampled = []
        for i in range(width):
            lo = int(i * bucket)
            hi = max(lo + 1, int((i + 1) * bucket))
            chunk = values[lo:hi]
            sampled.append(sum(chunk) / len(chunk) if chunk else values[lo])
        pad = 0

    vmin = min(sampled)
    vmax = max(sampled)
    vrange = vmax - vmin
    if vrange < 1e-12:
        # All values identical — render a flat line
        return "─" * len(sampled)

    chars = []
    for v in sampled:
        idx = int((v - vmin) / vrange * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(len(_SPARK_CHARS) - 1, idx))
        chars.append(_SPARK_CHARS[idx])

    return " " * pad + "".join(chars)


def _downsample_updates(updates: List[int], width: int) -> List[int]:
    """Downsample update numbers to match sparkline width."""
    n = len(updates)
    if n <= width:
        return updates
    bucket = n / width
    result = []
    for i in range(width):
        lo = int(i * bucket)
        result.append(updates[lo])
    return result


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------

class TrainingLogAnalyzer:
    """Sliding-window diagnostic engine for any PPO curriculum experiment."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: deque = deque(maxlen=200)

    # -- public API --------------------------------------------------------

    def feed_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Feed a raw log line; return parsed dict if it was a ``__RAW_STATS__`` line."""
        if "__RAW_STATS__" not in line:
            return None
        try:
            json_str = line.split("__RAW_STATS__", 1)[1].strip()
            data = json.loads(json_str)
            self.history.append(data)
            return data
        except Exception:
            return None

    def recent(self) -> deque:
        """Return the most recent ``window_size`` entries as a deque."""
        return deque(list(self.history)[-self.window_size:], maxlen=self.window_size)

    # -- diagnostics -------------------------------------------------------

    def run_diagnostics(self) -> List[Dict[str, Any]]:
        """Run PPO health checks over the current window."""
        if len(self.history) < 3:
            return []

        win = self.recent()
        u_start = win[0]["update"]
        u_end   = win[-1]["update"]
        conclusions: List[Dict[str, Any]] = []

        # ---- Check 1: Exploration / Std Collapse ----
        std_mins = _series(win, "stats.std_min")
        avg_std_min = sum(std_mins) / len(std_mins) if std_mins else 1.0
        if avg_std_min <= 0.145:
            conclusions.append({
                "severity": "CRITICAL",
                "title": "Exploration Collapse — std_min at floor",
                "conclusion": (
                    "At least one joint's action std has hit the minimum floor. "
                    "That DOF is deterministic and can no longer explore."
                ),
                "evidence": (
                    f"  Window u{u_start}–u{u_end} ({len(win)} updates)\n"
                    f"  avg std_min = {avg_std_min:.4f}\n"
                    f"  series: {[round(x, 4) for x in std_mins]}"
                ),
                "remedy": (
                    "1. Raise log_std_min in the experiment config (e.g. from -2.7 to -2.0).\n"
                    "2. Increase entropy_coef to encourage exploration."
                ),
            })

        # ---- Check 2: PPO Early Stop (KL trust-region rupture) ----
        epochs_dones = _series(win, "stats.epochs_done")
        avg_epochs = sum(epochs_dones) / len(epochs_dones) if epochs_dones else 0
        # Read update_epochs from the last entry if available (not in stats,
        # but we can infer from max epochs_done seen across history).
        max_epochs_seen = max(_series(self.history, "stats.epochs_done") + [4])
        if avg_epochs <= 1.2:
            kls = _series(win, "stats.approx_kl")
            conclusions.append({
                "severity": "WARNING",
                "title": "PPO Early Stop — KL exceeds target every update",
                "conclusion": (
                    "The policy diverges so fast that PPO early-stops after ~1 epoch "
                    "every update. Data efficiency is very low."
                ),
                "evidence": (
                    f"  avg epochs_done = {avg_epochs:.1f} (typical target: {max_epochs_seen})\n"
                    f"  series: {epochs_dones}\n"
                    f"  KL values: {[round(x, 4) for x in kls]}"
                ),
                "remedy": (
                    "1. Lower the actor learning rate (e.g. halve it).\n"
                    "2. Verify advantage normalization is functioning.\n"
                    "3. If fine-tuning from a very different checkpoint, the initial "
                    "KL is naturally large — this may resolve after a few updates."
                ),
            })

        # ---- Check 3: Critic Blindness (EV <= 0) ----
        last = win[-1]
        stats = last.get("stats", {})
        ev_keys = sorted(k for k in stats if k.startswith("ev_"))
        for ev_key in ev_keys:
            comp = ev_key[3:]  # strip "ev_"
            evs = _series(win, f"stats.{ev_key}")
            avg_ev = sum(evs) / len(evs) if evs else 0.0
            if avg_ev <= 0.0:
                vlosses = _series(win, f"stats.vloss_{comp}")
                avg_vloss = sum(vlosses) / len(vlosses) if vlosses else 0.0
                conclusions.append({
                    "severity": "WARNING",
                    "title": f"Critic Blind — EV({comp}) <= 0",
                    "conclusion": (
                        f"The critic for '{comp}' has non-positive explained variance. "
                        "Its advantage estimates are noise, polluting the policy gradient."
                    ),
                    "evidence": (
                        f"  avg EV = {avg_ev:+.4f}\n"
                        f"  series: {[round(x, 3) for x in evs]}\n"
                        f"  avg value_loss = {avg_vloss:.4f}"
                    ),
                    "remedy": (
                        "1. Increase the critic learning rate (2–4x actor LR).\n"
                        f"2. Lower gamma for '{comp}' to reduce prediction horizon.\n"
                        "3. Early in training EVs can temporarily go negative; "
                        "check long-term trends before panicking."
                    ),
                })

        # ---- Check 4: Episode Length Death Spiral ----
        ep_means = _series(win, "stats.ep_len_mean")
        avg_ep_mean = sum(ep_means) / len(ep_means) if ep_means else 1000.0
        if avg_ep_mean <= 30.0:
            conclusions.append({
                "severity": "CRITICAL",
                "title": "Episode Death Spiral — avg length critically short",
                "conclusion": (
                    "Episodes are ending almost immediately. Rollout data is dominated "
                    "by failure trajectories with no successful signal for the policy "
                    "to learn from."
                ),
                "evidence": (
                    f"  avg ep_len_mean = {avg_ep_mean:.1f} steps\n"
                    f"  series: {[round(x, 1) for x in ep_means]}"
                ),
                "remedy": (
                    "1. Reduce curriculum difficulty (lower perturbation, simpler opponent).\n"
                    "2. Mix in easier episodes (mixed-batch curriculum replay).\n"
                    "3. Add shaping rewards to provide incremental signal."
                ),
            })

        # ---- Check 5: Reward signal vanishing (adv_std near 0) ----
        for ev_key in ev_keys:
            comp = ev_key[3:]
            adv_stds = _series(win, f"stats.adv_std_{comp}")
            if adv_stds:
                avg_adv_std = sum(adv_stds) / len(adv_stds)
                if avg_adv_std < 1e-4:
                    conclusions.append({
                        "severity": "WARNING",
                        "title": f"Vanishing Advantage — adv_std({comp}) ≈ 0",
                        "conclusion": (
                            f"The advantage signal for '{comp}' has effectively zero variance. "
                            "The critic is producing flat values, so this reward component "
                            "contributes no gradient direction."
                        ),
                        "evidence": (
                            f"  avg adv_std = {avg_adv_std:.6f}\n"
                            f"  series: {[round(x, 6) for x in adv_stds]}"
                        ),
                        "remedy": (
                            f"1. Check if the reward '{comp}' is actually varying across steps.\n"
                            "2. The critic may have converged to a constant — try resetting "
                            "it or increasing its learning rate."
                        ),
                    })

        return conclusions

    # -- trend reporting ---------------------------------------------------

    def discover_metric_groups(self) -> Dict[str, List[str]]:
        """Auto-discover all metric keys from the most recent entry.

        Returns a dict of group_name -> list of metric keys.
        Groups: ``sinfo``, ``bsum``, ``esum``, ``rsum``, ``ppo``.
        """
        if not self.history:
            return {}

        last = self.history[-1]
        groups: Dict[str, List[str]] = {}

        # sinfo — scheduler info (experiment-specific)
        sinfo = last.get("sinfo") or {}
        if sinfo:
            groups["sinfo"] = sorted(sinfo.keys())

        # bsum — batch/rollout summary (episode metrics)
        bsum = last.get("bsum") or {}
        if bsum:
            groups["bsum"] = sorted(bsum.keys())

        # esum — eval summary (same structure as bsum, may be absent on some
        # updates due to eval_interval; search recent history for an entry that has it)
        esum = last.get("esum")
        if not esum:
            for d in reversed(list(self.history)):
                esum = d.get("esum")
                if esum:
                    break
        if esum:
            groups["esum"] = sorted(esum.keys())

        # rsum — reward summary (r_{key}_mean / r_{key}_std pairs)
        rsum = last.get("rsum") or {}
        if rsum:
            # Group by reward key, only show _mean variants in the table
            reward_keys = sorted(
                k[:-5] for k in rsum if k.endswith("_mean")
            )
            groups["rsum"] = reward_keys

        # ppo — core PPO stats (fixed set)
        stats = last.get("stats") or {}
        ppo_keys = [
            k for k in (
                "policy_loss", "value_loss", "approx_kl", "max_kl",
                "epochs_done", "entropy", "std_mean", "std_min",
                "ep_len_mean", "n_batches", "total_steps",
            ) if k in stats
        ]
        if ppo_keys:
            groups["ppo"] = ppo_keys

        # per-critic EV — auto-discovered
        ev_keys = sorted(k for k in stats if k.startswith("ev_"))
        if ev_keys:
            groups["ev"] = ev_keys

        return groups

    def render_trend_table(self) -> str:
        """Render a compact trend table for all auto-discovered metrics."""
        if len(self.history) < 2:
            return f"{DIM}  (need at least 2 updates for trend){RESET}"

        groups = self.discover_metric_groups()
        if not groups:
            return f"{DIM}  (no metrics discovered){RESET}"

        win = self.recent()
        full = self.history
        lines: List[str] = []

        group_labels = {
            "sinfo": "Scheduler",
            "bsum":  "Rollout",
            "esum":  "Eval",
            "rsum":  "Reward/step",
            "ppo":   "PPO",
            "ev":    "Critic EV",
        }
        group_order = ["sinfo", "bsum", "esum", "rsum", "ppo", "ev"]

        for group in group_order:
            if group not in groups:
                continue
            label = group_labels.get(group, group)
            lines.append(f"  {BOLD}{CYAN}── {label} ──{RESET}")

            for key in groups[group]:
                if group == "rsum":
                    path_mean = f"rsum.{key}_mean"
                    path_std  = f"rsum.{key}_std"
                    win_vals = _series(win, path_mean)
                    full_vals = _series(full, path_mean)
                    if not win_vals:
                        continue
                    cur = win_vals[-1]
                    win_avg = sum(win_vals) / len(win_vals)
                    slope = _trend_slope(full_vals) if len(full_vals) >= 5 else 0.0
                    std_val = _avg(win, path_std, 0.0)
                    arrow = _fmt_trend_arrow(slope)
                    lines.append(
                        f"    {key:<16} cur={_fmt_float(cur):>10}  "
                        f"win={_fmt_float(win_avg):>10}  "
                        f"±{_fmt_float(std_val):>8}  {arrow}"
                    )
                elif group == "ev":
                    comp = key[3:]  # strip "ev_"
                    win_vals = _series(win, f"stats.{key}")
                    full_vals = _series(full, f"stats.{key}")
                    if not win_vals:
                        continue
                    cur = win_vals[-1]
                    win_avg = sum(win_vals) / len(win_vals)
                    slope = _trend_slope(full_vals) if len(full_vals) >= 5 else 0.0
                    arrow = _fmt_trend_arrow(slope)
                    # Color EV values: green >0, red <=0
                    color = GREEN if cur > 0 else RED if cur <= 0 else YELLOW
                    lines.append(
                        f"    {comp:<16} {color}cur={cur:+.3f}{RESET}  "
                        f"win={win_avg:+.3f}  {arrow}"
                    )
                else:
                    prefix = f"{group}." if group in ("sinfo", "bsum", "esum") else "stats."
                    path = f"{prefix}{key}"
                    win_vals = _series(win, path)
                    full_vals = _series(full, path)
                    if not win_vals:
                        continue
                    cur = win_vals[-1]
                    win_avg = sum(win_vals) / len(win_vals)
                    slope = _trend_slope(full_vals) if len(full_vals) >= 5 else 0.0
                    arrow = _fmt_trend_arrow(slope)
                    lines.append(
                        f"    {key:<16} cur={_fmt_float(cur):>10}  "
                        f"win={_fmt_float(win_avg):>10}  {arrow}"
                    )
            lines.append("")

        return "\n".join(lines)

    def render_history(self, metric_filter: Optional[str] = None, width: int = 60) -> str:
        """Render full-history sparkline charts for all auto-discovered metrics.

        Args:
            metric_filter: If given, only show metrics whose name contains this
                           substring (case-insensitive). None = show all.
            width: Sparkline width in characters.
        """
        if len(self.history) < 3:
            return f"  {DIM}(need at least 3 updates for history){RESET}"

        groups = self.discover_metric_groups()
        if not groups:
            return f"  {DIM}(no metrics discovered){RESET}"

        updates = [d["update"] for d in self.history]
        u_first = updates[0]
        u_last = updates[-1]
        n = len(updates)

        lines: List[str] = []
        lines.append(
            f"  {DIM}{n} updates (u{u_first} → u{u_last}), "
            f"sparkline width={width}{RESET}\n"
        )

        group_labels = {
            "sinfo": "Scheduler",
            "bsum":  "Rollout",
            "esum":  "Eval",
            "rsum":  "Reward/step",
            "ppo":   "PPO",
            "ev":    "Critic EV",
        }
        group_order = ["sinfo", "bsum", "esum", "rsum", "ppo", "ev"]

        filt = metric_filter.lower() if metric_filter else None

        for group in group_order:
            if group not in groups:
                continue
            label = group_labels.get(group, group)
            group_lines: List[str] = []

            for key in groups[group]:
                if filt and filt not in key.lower():
                    continue

                if group == "rsum":
                    path = f"rsum.{key}_mean"
                elif group == "ev":
                    path = f"stats.{key}"
                elif group in ("sinfo", "bsum", "esum"):
                    path = f"{group}.{key}"
                else:
                    path = f"stats.{key}"

                vals = _series(self.history, path)
                if len(vals) < 3:
                    continue

                spark = _sparkline(vals, width=width)
                vmin = min(vals)
                vmax = max(vals)
                cur = vals[-1]
                first = vals[0]

                # For EV, color the current value
                if group == "ev":
                    color = GREEN if cur > 0 else RED if cur <= 0 else YELLOW
                    cur_str = f"{color}{cur:+.3f}{RESET}"
                else:
                    cur_str = _fmt_float(cur)

                # Show range on the right
                lines.append(
                    f"    {key:<16} {spark}  "
                    f"{DIM}[{_fmt_float(vmin)} ~ {_fmt_float(vmax)}]{RESET}  "
                    f"cur={cur_str}  "
                    f"{DIM}Δ={_fmt_float(cur - first, 3)}{RESET}"
                )
                group_lines.append(key)

            if group_lines:
                lines.insert(
                    len(lines) - len(group_lines),
                    f"  {BOLD}{CYAN}── {label} ──{RESET}"
                )
                lines.append("")

        if not any(c for c in lines if c.strip()):
            return f"  {DIM}No metrics matching '{metric_filter}'.{RESET}"

        return "\n".join(lines)

    def render_weights(self) -> str:
        """Render the current reward weights."""
        if not self.history:
            return ""
        last = self.history[-1]
        weights = last.get("weights") or []
        if not weights:
            return ""
        return f"  {BOLD}Weights{RESET}: [{', '.join(f'{w:.2f}' for w in weights)}]"

    def render_summary_header(self) -> str:
        """Render a one-line summary of the latest update."""
        if not self.history:
            return ""
        last = self.history[-1]
        u = last.get("update", 0)
        stats = last.get("stats", {})
        ep_len = stats.get("ep_len_mean", 0.0)
        kl = stats.get("approx_kl", 0.0)
        epochs = stats.get("epochs_done", 0)
        entropy = stats.get("entropy", 0.0)
        return (
            f"  {BOLD}Update {u}{RESET}  "
            f"ep_len={_fmt_float(ep_len, 1)}  "
            f"kl={kl:.4f}  epochs={epochs}  "
            f"entropy={entropy:.2f}"
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "CRITICAL": RED,
    "WARNING":  YELLOW,
    "INFO":     BLUE,
}


def render_diagnostics(conclusions: List[Dict[str, Any]]) -> str:
    if not conclusions:
        return f"  {GREEN}✓ All PPO health checks passed.{RESET}\n"

    lines: List[str] = []
    for c in conclusions:
        sev = c.get("severity", "INFO")
        color = SEVERITY_COLORS.get(sev, RESET)
        lines.append(f"  {color}{BOLD}[{sev}]{RESET} {color}{c['title']}{RESET}")
        lines.append(f"    {c['conclusion']}")
        lines.append(f"    {DIM}Evidence:{RESET}")
        for ev_line in c["evidence"].strip().split("\n"):
            lines.append(f"    {DIM}{ev_line}{RESET}")
        lines.append(f"    {DIM}Remedy:{RESET}")
        for rem_line in c["remedy"].strip().split("\n"):
            lines.append(f"    {DIM}{rem_line}{RESET}")
        lines.append("")
    return "\n".join(lines)


def render_report(analyzer: TrainingLogAnalyzer, diagnostics_only: bool = False) -> str:
    """Render the full diagnostic + trend report."""
    if not analyzer.history:
        return f"{YELLOW}No __RAW_STATS__ entries found in log.{RESET}\n"

    sections: List[str] = []

    # Header
    sections.append(f"\n{BOLD}{'═' * 60}{RESET}")
    sections.append(f"{BOLD}  PPO Training Analysis{RESET}")
    sections.append(f"{BOLD}{'═' * 60}{RESET}\n")

    # Latest update summary
    sections.append(analyzer.render_summary_header())
    sections.append(analyzer.render_weights())
    sections.append("")

    # Diagnostics
    sections.append(f"{BOLD}  ── PPO Health Diagnostics ──{RESET}")
    conclusions = analyzer.run_diagnostics()
    sections.append(render_diagnostics(conclusions))

    if not diagnostics_only:
        # Trends
        sections.append(f"{BOLD}  ── Metric Trends ──{RESET}")
        sections.append(f"  {DIM}cur = latest value, win = window average, arrow = long-term slope{RESET}")
        sections.append("")
        sections.append(analyzer.render_trend_table())

    sections.append(f"{BOLD}{'═' * 60}{RESET}\n")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Universal PPO training log analyzer. Works with any experiment "
                    "that emits __RAW_STATS__ lines."
    )
    parser.add_argument("logfile", type=str, help="Path to the training log file.")
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch mode: follow the log file in real-time (like tail -f).",
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="Sliding window size for diagnostics (default: 10 updates).",
    )
    parser.add_argument(
        "--diagnostics-only", action="store_true",
        help="Show only PPO health diagnostics, skip the trend table.",
    )
    parser.add_argument(
        "--history", type=str, nargs="?", const="all", default=None,
        metavar="FILTER",
        help="Show full-history sparkline charts. Optionally filter by metric "
             "name substring (e.g. --history survived, --history ev).",
    )
    parser.add_argument(
        "--list-metrics", action="store_true",
        help="List all discovered metric names and exit.",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Refresh interval in seconds for --watch mode (default: 2.0).",
    )
    return parser.parse_args()


def _read_existing(logfile: str, analyzer: TrainingLogAnalyzer) -> int:
    """Read all existing lines from the log file. Returns number of lines read."""
    count = 0
    try:
        with open(logfile, "r", errors="replace") as f:
            for line in f:
                analyzer.feed_line(line)
                count += 1
    except FileNotFoundError:
        print(f"{RED}Error: file not found: {logfile}{RESET}", file=sys.stderr)
        sys.exit(1)
    return count


def _watch_loop(logfile: str, analyzer: TrainingLogAnalyzer,
                diagnostics_only: bool, interval: float) -> None:
    """Follow the log file and re-render on each new __RAW_STATS__ line."""
    # Open for appending
    f = open(logfile, "r", errors="replace")
    # Seek to end
    f.seek(0, os.SEEK_END)

    try:
        while True:
            line = f.readline()
            if not line:
                # No new data; wait
                time.sleep(interval)
                continue

            data = analyzer.feed_line(line)
            if data is not None:
                # Clear screen and re-render
                print("\033[2J\033[H", end="")
                print(render_report(analyzer, diagnostics_only=diagnostics_only),
                      flush=True)
    except KeyboardInterrupt:
        print(f"\n{DIM}Stopped.{RESET}")
    finally:
        f.close()


def render_history_report(analyzer: TrainingLogAnalyzer, metric_filter: Optional[str]) -> str:
    """Render the full-history sparkline report."""
    if not analyzer.history:
        return f"{YELLOW}No __RAW_STATS__ entries found in log.{RESET}\n"

    sections: List[str] = []
    sections.append(f"\n{BOLD}{'═' * 70}{RESET}")
    sections.append(f"{BOLD}  Training History — Sparkline Charts{RESET}")
    sections.append(f"{BOLD}{'═' * 70}{RESET}\n")

    filt = metric_filter if metric_filter and metric_filter != "all" else None
    if filt:
        sections.append(f"  {DIM}Filter: '{filt}'{RESET}\n")
    sections.append(analyzer.render_history(metric_filter=filt))

    sections.append(f"{BOLD}{'═' * 70}{RESET}\n")
    return "\n".join(sections)


def render_metric_list(analyzer: TrainingLogAnalyzer) -> str:
    """List all discovered metric names."""
    if not analyzer.history:
        return f"{YELLOW}No __RAW_STATS__ entries found in log.{RESET}\n"

    groups = analyzer.discover_metric_groups()
    if not groups:
        return f"{YELLOW}No metrics discovered.{RESET}\n"

    group_labels = {
        "sinfo": "Scheduler",
        "bsum":  "Rollout",
        "esum":  "Eval",
        "rsum":  "Reward/step",
        "ppo":   "PPO",
        "ev":    "Critic EV",
    }
    group_order = ["sinfo", "bsum", "esum", "rsum", "ppo", "ev"]

    lines: List[str] = [f"\n{BOLD}Discovered Metrics{RESET}\n"]
    for group in group_order:
        if group not in groups:
            continue
        label = group_labels.get(group, group)
        lines.append(f"  {BOLD}{label}:{RESET}")
        for key in groups[group]:
            # Show the dotted path for use with --history
            if group == "rsum":
                path = f"rsum.{key}_mean"
            elif group == "ev":
                path = f"stats.{key}"
            elif group in ("sinfo", "bsum", "esum"):
                path = f"{group}.{key}"
            else:
                path = f"stats.{key}"
            lines.append(f"    {key:<20} {DIM}({path}){RESET}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    analyzer = TrainingLogAnalyzer(window_size=args.window)

    # Read existing content
    _read_existing(args.logfile, analyzer)

    if args.list_metrics:
        print(render_metric_list(analyzer))
        return

    if args.history is not None:
        filt = args.history if args.history != "all" else None
        print(render_history_report(analyzer, filt))
        return

    if args.watch:
        # Print initial report, then watch
        print(render_report(analyzer, diagnostics_only=args.diagnostics_only))
        _watch_loop(args.logfile, analyzer, args.diagnostics_only, args.interval)
    else:
        # One-shot
        print(render_report(analyzer, diagnostics_only=args.diagnostics_only))


if __name__ == "__main__":
    main()
