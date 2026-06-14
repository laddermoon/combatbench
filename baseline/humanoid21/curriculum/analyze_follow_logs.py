#!/usr/bin/env python3
"""
Follow-Experiment Training Log Monitor and Diagnostic Tool.

Tracks the three key episode metrics (hold_ratio, survived, primary_ratio)
plus PPO health, critic quality, and reward-signal trends specific to the
follow-opponent curriculum.

Usage:
    # One-shot analysis
    python3 analyze_follow_logs.py follow.log

    # Real-time watch mode
    python3 analyze_follow_logs.py follow.log --watch

    # Custom window size (default 10 updates)
    python3 analyze_follow_logs.py follow.log --window 20
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


class FollowLogAnalyzer:
    """Sliding-window diagnostic engine for the follow experiment."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

    def feed_line(self, line: str) -> Optional[Dict[str, Any]]:
        if "__RAW_STATS__" in line:
            try:
                json_str = line.split("__RAW_STATS__", 1)[1].strip()
                data = json.loads(json_str)
                self.history.append(data)
                return data
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _avg(history: deque, path: str, default: float = 0.0) -> float:
        """Average a nested key path (e.g. 'stats.entropy') over history."""
        parts = path.split(".")
        vals: List[float] = []
        for d in history:
            cur: Any = d
            for p in parts:
                if isinstance(cur, dict):
                    cur = cur.get(p, default)
                else:
                    cur = default
                    break
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return sum(vals) / len(vals) if vals else default

    @staticmethod
    def _series(history: deque, path: str) -> List[float]:
        """Extract a time-series for a nested key path."""
        parts = path.split(".")
        out: List[float] = []
        for d in history:
            cur: Any = d
            for p in parts:
                if isinstance(cur, dict):
                    cur = cur.get(p)
                else:
                    cur = None
                    break
            if isinstance(cur, (int, float)):
                out.append(float(cur))
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def run_diagnostics(self) -> List[Dict[str, Any]]:
        if len(self.history) < 3:
            return []

        conclusions: List[Dict[str, Any]] = []
        u_start = self.history[0]["update"]
        u_end   = self.history[-1]["update"]

        # ---- Check A: Hold ratio stagnation (approach failure) ----
        hold_ratios = self._series(self.history, "bsum.hold_ratio")
        avg_hold = sum(hold_ratios) / len(hold_ratios)
        if avg_hold < 0.05:
            conclusions.append({
                "severity": "CRITICAL",
                "title": "Approach Failure — hold_ratio near zero",
                "conclusion": (
                    "The robot spends almost no time within 1m of the opponent. "
                    "The approach policy is not learning to move toward the target."
                ),
                "evidence": (
                    f"  Window u{u_start}–u{u_end} ({len(self.history)} updates)\n"
                    f"  avg hold_ratio = {avg_hold:.4f}\n"
                    f"  series: {[round(x, 4) for x in hold_ratios]}"
                ),
                "remedy": (
                    "1. Check that r_radial is producing meaningful gradients "
                    "(look at adv_std_r_radial — if near 0, the reward signal is too weak).\n"
                    "2. The initial checkpoint is a balance-recovery policy that only "
                    "knows how to stand. It may need more episodes or a higher radial "
                    "reward weight to discover locomotion.\n"
                    "3. Consider increasing the r_radial weight in initial_weights."
                ),
            })

        # ---- Check B: PPO early-stop every epoch ----
        epochs_dones = self._series(self.history, "stats.epochs_done")
        avg_epochs = sum(epochs_dones) / len(epochs_dones) if epochs_dones else 0
        if avg_epochs <= 1.2:
            kls = self._series(self.history, "stats.approx_kl")
            conclusions.append({
                "severity": "WARNING",
                "title": "PPO Early Stop — KL exceeds target every update",
                "conclusion": (
                    "The policy diverges so fast that PPO early-stops after epoch 0 "
                    "every update. Data efficiency is very low (1 epoch of gradient "
                    "steps before throwing away the batch)."
                ),
                "evidence": (
                    f"  avg epochs_done = {avg_epochs:.1f} (target: {4})\n"
                    f"  series: {epochs_dones}\n"
                    f"  KL values: {[round(x, 4) for x in kls]}"
                ),
                "remedy": (
                    "1. Lower the actor learning rate (e.g. halve it).\n"
                    "2. Lower the clip_eps if currently high.\n"
                    "3. Check if advantage normalization is working.\n"
                    "4. This can also happen when fine-tuning from a very different "
                    "checkpoint — the initial KL is naturally large."
                ),
            })

        # ---- Check C: Exploration collapse ----
        std_mins = self._series(self.history, "stats.std_min")
        avg_std_min = sum(std_mins) / len(std_mins) if std_mins else 1.0
        if avg_std_min <= 0.145:
            conclusions.append({
                "severity": "CRITICAL",
                "title": "Exploration Collapse — std_min locked at floor",
                "conclusion": (
                    "At least one joint's action std has hit the minimum floor. "
                    "That DOF is deterministic and can no longer explore."
                ),
                "evidence": (
                    f"  avg std_min = {avg_std_min:.4f}\n"
                    f"  series: {[round(x, 4) for x in std_mins]}"
                ),
                "remedy": (
                    "1. Raise log_std_min (e.g. from -2.7 to -2.0).\n"
                    "2. Increase entropy_coef."
                ),
            })

        # ---- Check D: Critic explained variance ----
        last = self.history[-1]
        stats = last.get("stats", {})
        for key in ("r_fall", "r_cross", "r_radial", "r_tangential"):
            ev_key = f"ev_{key}"
            if ev_key not in stats:
                continue
            evs = self._series(self.history, f"stats.{ev_key}")
            avg_ev = sum(evs) / len(evs) if evs else 0.0
            if avg_ev <= 0.0:
                conclusions.append({
                    "severity": "WARNING",
                    "title": f"Critic Blind — EV({key}) <= 0",
                    "conclusion": (
                        f"The critic for '{key}' has non-positive explained variance. "
                        "Its advantage estimates are noise, polluting the policy gradient."
                    ),
                    "evidence": (
                        f"  avg EV = {avg_ev:+.4f}\n"
                        f"  series: {[round(x, 3) for x in evs]}"
                    ),
                    "remedy": (
                        "1. Increase the critic learning rate (2–4x actor LR).\n"
                        f"2. Lower gamma for '{key}' to reduce prediction horizon.\n"
                        "3. EV can be negative early in training and recover — "
                        "check the trend before intervening."
                    ),
                })

        # ---- Check E: Survival decline ----
        surv = self._series(self.history, "bsum.survived")
        if len(surv) >= 5:
            half = len(surv) // 2
            first_avg = sum(surv[:half]) / half
            last_avg  = sum(surv[-half:]) / half
            if last_avg < first_avg - 0.15:
                conclusions.append({
                    "severity": "WARNING",
                    "title": "Survival Decline — episodes getting shorter",
                    "conclusion": (
                        "Survival rate dropped significantly in the recent window. "
                        "The policy may be forgetting how to maintain balance."
                    ),
                    "evidence": (
                        f"  first-half avg survived = {first_avg:.3f}\n"
                        f"  last-half  avg survived = {last_avg:.3f}\n"
                        f"  series: {[round(x, 3) for x in surv]}"
                    ),
                    "remedy": (
                        "1. This is common when fine-tuning — the new task gradient "
                        "can interfere with balance.\n"
                        "2. Consider lowering the learning rate temporarily.\n"
                        "3. If using MixedPolicy, ensure the gating model isn't "
                        "blocking valid balance-recovery actions."
                    ),
                })

        return conclusions

    # ------------------------------------------------------------------
    # Progress snapshot
    # ------------------------------------------------------------------

    def progress_summary(self) -> str:
        if len(self.history) < 2:
            return f"{BLUE}    collecting data...{RESET}"

        last = self.history[-1]
        u = last["update"]
        sinfo = last.get("sinfo", {})
        bsum = last.get("bsum", {})
        stats = last.get("stats", {})

        level = sinfo.get("level", 0)
        opp_speed = sinfo.get("opp_speed", 0.0)
        hold = bsum.get("hold_ratio", 0.0)
        surv = bsum.get("survived", 0.0)
        prim = bsum.get("primary_ratio", 1.0)
        ep_len = stats.get("ep_len_mean", 0.0)
        epochs = stats.get("epochs_done", 0)
        kl = stats.get("approx_kl", 0.0)
        ploss = stats.get("policy_loss", 0.0)

        lines = []
        lines.append(f"  {BOLD}Curriculum{RESET}  level={GREEN}{level}{RESET}  "
                      f"opp_speed={GREEN}{opp_speed:.2f}{RESET} m/s")
        lines.append(f"  {BOLD}Metrics   {RESET}  hold_ratio={hold:.3f}  "
                      f"survived={surv:.3f}  primary_ratio={prim:.3f}")
        lines.append(f"  {BOLD}Episode   {RESET}  mean_len={ep_len:.0f} steps")
        lines.append(f"  {BOLD}PPO       {RESET}  loss={ploss:+.4f}  "
                      f"epochs={epochs}/4  kl={kl:.4f}")

        # Trend arrows
        if len(self.history) >= 5:
            hold_series = self._series(self.history, "bsum.hold_ratio")
            half = len(hold_series) // 2
            d_hold = (sum(hold_series[-half:]) / half) - (sum(hold_series[:half]) / half)
            surv_series = self._series(self.history, "bsum.survived")
            d_surv = (sum(surv_series[-half:]) / half) - (sum(surv_series[:half]) / half)

            def _arrow(delta, thresh=0.01):
                if delta > thresh: return f"{GREEN}↑{RESET}"
                if delta < -thresh: return f"{RED}↓{RESET}"
                return f"{YELLOW}→{RESET}"

            lines.append(f"  {BOLD}Trend     {RESET}  "
                          f"hold {d_hold:+.4f} {_arrow(d_hold)}  "
                          f"surv {d_surv:+.4f} {_arrow(d_surv)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reward breakdown
    # ------------------------------------------------------------------

    def reward_breakdown(self) -> str:
        last = self.history[-1]
        rsum = last.get("rsum", {})
        stats = last.get("stats", {})
        lines = []
        for key in ("r_fall", "r_cross", "r_radial", "r_tangential", "r_gate"):
            m = rsum.get(f"{key}_mean", 0.0)
            s = rsum.get(f"{key}_std", 0.0)
            ev = stats.get(f"ev_{key}", 0.0)
            adv = stats.get(f"adv_std_{key}", 0.0)
            lines.append(
                f"    {key:<14} reward={m:+.5f}±{s:.5f}  "
                f"adv_std={adv:.3f}  EV={ev:+.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# File tailing
# ---------------------------------------------------------------------------

def tail_file(file_path):
    f = open(file_path, "r", encoding="utf-8")
    f.seek(0, os.SEEK_END)
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_update(analyzer: FollowLogAnalyzer):
    last = analyzer.history[-1]
    u = last["update"]
    print(f"\n{'='*60}")
    print(f"{BOLD}{BLUE}>>> Update {u:5d} <<<{RESET}")
    print(f"{'='*60}")
    print(analyzer.progress_summary())
    print()
    print(f"  {BOLD}Rewards{RESET}")
    print(analyzer.reward_breakdown())


def print_diagnostics(conclusions: List[Dict[str, Any]]):
    if not conclusions:
        print(f"\n{GREEN}[HEALTHY] No structural anomalies detected.{RESET}")
        return

    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC REPORT")
    print(f"{'='*60}")
    for i, c in enumerate(conclusions, 1):
        color = RED if c["severity"] == "CRITICAL" else YELLOW
        print(f"\n{color}[{i}] {c['title']} ({c['severity']}){RESET}")
        print(f"  {BOLD}Conclusion:{RESET} {c['conclusion']}")
        print(f"  {BOLD}Evidence:{RESET}\n{c['evidence']}")
        print(f"  {BOLD}Remedy:{RESET} {c['remedy']}")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Follow-experiment training log monitor."
    )
    parser.add_argument("log_file", type=str)
    parser.add_argument("--watch", action="store_true",
                        help="Watch file in real-time (tail -f).")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    analyzer = FollowLogAnalyzer(window_size=args.window)
    print(f"Analyzing '{args.log_file}' with window={args.window}...")

    try:
        with open(args.log_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: '{args.log_file}' not found.", file=sys.stderr)
        sys.exit(1)

    parsed = 0
    for line in lines:
        if analyzer.feed_line(line):
            parsed += 1

    print(f"Parsed {parsed} updates.\n")

    if parsed > 0:
        print_update(analyzer)
        print_diagnostics(analyzer.run_diagnostics())
    else:
        print(f"{YELLOW}[WARN] No __RAW_STATS__ lines found.{RESET}")

    if args.watch:
        print(f"\n{BLUE}[WATCH] Tailing '{args.log_file}'... (Ctrl+C to exit){RESET}")
        try:
            for line in tail_file(args.log_file):
                if analyzer.feed_line(line):
                    print_update(analyzer)
                    print_diagnostics(analyzer.run_diagnostics())
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
