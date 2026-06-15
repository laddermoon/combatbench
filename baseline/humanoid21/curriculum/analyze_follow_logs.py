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
        self.history: deque = deque(maxlen=100)

    def _recent_history(self) -> deque:
        """Returns a deque of the most recent self.window_size entries."""
        return deque(list(self.history)[-self.window_size:], maxlen=self.window_size)

    def _calculate_trend(self, path: str, length: int = 50) -> Optional[Dict[str, Any]]:
        """Calculate long-term trend (slope, overall change) over history."""
        series = self._series(self.history, path)
        if len(series) < 5:
            return None
        
        series = series[-length:]
        n = len(series)
        if n < 5:
            return None
        
        # Simple linear regression
        x = list(range(n))
        y = series
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        slope = num / den if den != 0 else 0.0
        
        half = max(1, n // 5)
        first_avg = sum(series[:half]) / half
        last_avg = sum(series[-half:]) / half
        overall_diff = last_avg - first_avg
        
        return {
            "slope": slope,
            "overall_diff": overall_diff,
            "first_avg": first_avg,
            "last_avg": last_avg,
            "n": n
        }

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
        recent = self._recent_history()
        u_start = recent[0]["update"]
        u_end   = recent[-1]["update"]

        # ---- Check A: Hold ratio stagnation (approach failure) ----
        hold_ratios = self._series(recent, "bsum.hold_ratio")
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
                    f"  Window u{u_start}–u{u_end} ({len(recent)} updates)\n"
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
        epochs_dones = self._series(recent, "stats.epochs_done")
        avg_epochs = sum(epochs_dones) / len(epochs_dones) if epochs_dones else 0
        if avg_epochs <= 1.2:
            kls = self._series(recent, "stats.approx_kl")
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
        std_mins = self._series(recent, "stats.std_min")
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
        last = recent[-1]
        stats = last.get("stats", {})
        for key in ("r_fall", "r_cross", "r_radial", "r_tangential"):
            ev_key = f"ev_{key}"
            if ev_key not in stats:
                continue
            evs = self._series(recent, f"stats.{ev_key}")
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
        surv = self._series(recent, "bsum.survived")
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

        # ---- Check F: Gating Network Oscillation ----
        gating_switches = self._series(recent, "bsum.gating_switches")
        if gating_switches:
            avg_switches = sum(gating_switches) / len(gating_switches)
            if avg_switches > 8.0:
                conclusions.append({
                    "severity": "WARNING",
                    "title": "Gating Network Oscillation — too many mode switches",
                    "conclusion": (
                        f"The robot switches between primary and fallback policies excessively "
                        f"({avg_switches:.1f} times per episode). This indicates control instability "
                        f"and rapid jittering near the gating threshold, which degrades performance."
                    ),
                    "evidence": (
                        f"  avg switches per episode = {avg_switches:.1f}\n"
                        f"  series: {[round(x, 1) for x in gating_switches]}"
                    ),
                    "remedy": (
                        "1. Increase release_patience in MixedPolicy (e.g. from 10 to 20 or 30) "
                        "to require a longer stable standing period before switching back to Chaser.\n"
                        "2. Increase the gap between threshold and release_threshold (e.g. set "
                        "threshold=0.6, release_threshold=0.92) to add more hysteresis.\n"
                        "3. Inspect the gating model's prediction stability."
                    ),
                })

        # ---- Check G: Chaser Speed Deficiency / Evasion Catch-up Check ----
        min_dists = self._series(recent, "bsum.min_dist")
        if min_dists:
            avg_min_dist = sum(min_dists) / len(min_dists)
            opp_speed = recent[-1].get("sinfo", {}).get("opp_speed", 0.0)
            if avg_min_dist > 1.1 and opp_speed > 0.0:
                conclusions.append({
                    "severity": "WARNING",
                    "title": "Locomotion Speed Deficiency — unable to outrun opponent evasion",
                    "conclusion": (
                        f"The average minimum distance achieved is {avg_min_dist:.2f}m. "
                        f"Since the opponent actively flees to maintain a 1.2m distance at {opp_speed:.1f} m/s, "
                        f"this indicates the chaser policy has not yet learned to sprint fast enough to "
                        f"break through the 1.2m barrier and enter the target 0.9m zone."
                    ),
                    "evidence": (
                        f"  avg min_distance = {avg_min_dist:.2f}m\n"
                        f"  opponent speed   = {opp_speed:.2f} m/s\n"
                        f"  target hold zone = 0.90m\n"
                        f"  series: {[round(x, 2) for x in min_dists]}"
                    ),
                    "remedy": (
                        "1. Ensure r_radial (approach velocity reward) is strong enough to encourage sprinting.\n"
                        "2. Inspect the chaser's gait style to verify if it falls when running fast, or if it "
                        "moves too hesitantly.\n"
                        "3. Once the chaser's speed surpasses the opponent's speed, it will break through the 1.2m "
                        "barrier and score hold_ratio successfully."
                    ),
                })

        # ---- Check H: Learning Stagnation Detection (Long-term trend analysis) ----
        if len(self.history) >= 20:
            min_dist_trend = self._calculate_trend("bsum.min_dist", length=50)
            hold_ratio_trend = self._calculate_trend("bsum.hold_ratio", length=50)
            
            if min_dist_trend and hold_ratio_trend:
                overall_d_min = min_dist_trend["overall_diff"]
                overall_d_hold = hold_ratio_trend["overall_diff"]
                avg_min = min_dist_trend["last_avg"]
                n_updates = min_dist_trend["n"]
                
                if overall_d_min >= -0.02 and overall_d_hold <= 0.005 and avg_min > 1.2:
                    conclusions.append({
                        "severity": "CRITICAL",
                        "title": "Learning Stagnation — policy is not learning to approach target",
                        "conclusion": (
                            f"Over the last {n_updates} updates, the policy has shown zero progress "
                            f"in learning to approach the opponent. The minimum achieved distance is "
                            f"stagnating at {avg_min:.2f}m (overall change = {overall_d_min:+.2f}m), "
                            f"and hold_ratio is flat (overall change = {overall_d_hold:+.3f})."
                        ),
                        "evidence": (
                            f"  Trend Window     = {n_updates} updates\n"
                            f"  Current min_dist = {avg_min:.2f}m (target: <0.9m)\n"
                            f"  min_dist overall change   = {overall_d_min:+.2f}m\n"
                            f"  hold_ratio overall change = {overall_d_hold:+.3f}"
                        ),
                        "remedy": (
                            "1. Increase r_radial (radial approach reward) weight in follow_opponent.yaml or in initial_weights "
                            "to make the chaser's directional walking gradients stronger.\n"
                            "2. Lower tangential penalty (r_tangential) or other effort penalties during early walk-discovery.\n"
                            "3. Verify if exploration has collapsed: check the 'std' values under 'Policy'. If std is near the minimum floor (0.15), "
                            "the robot is too deterministic to explore. Raise log_std_min or increase entropy_coef."
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

        # Find the latest evaluation results in history
        latest_eval = None
        eval_update = None
        for entry in reversed(self.history):
            if "esum" in entry:
                latest_eval = entry["esum"]
                eval_update = entry["update"]
                break

        lines = []
        lines.append(f"  {BOLD}Curriculum{RESET}  level={GREEN}{level}{RESET}  "
                      f"opp_speed={GREEN}{opp_speed:.2f}{RESET} m/s")
        lines.append(f"  {BOLD}Metrics   {RESET}  hold_ratio={hold:.3f}  "
                      f"survived={surv:.3f}  primary_ratio={prim:.3f}")
        
        if latest_eval is not None:
            e_hold = latest_eval.get("hold_ratio", 0.0)
            e_surv = latest_eval.get("survived", 0.0)
            e_prim = latest_eval.get("primary_ratio", 1.0)
            lines.append(f"  {BOLD}Metrics(E){RESET}  hold_ratio={e_hold:.3f}  "
                          f"survived={e_surv:.3f}  primary_ratio={e_prim:.3f} {BLUE}[u{eval_update}]{RESET}")
        
        # Display professional geometric and gating metrics if present
        if "mean_dist" in bsum or "min_dist" in bsum:
            mean_dist = bsum.get("mean_dist", 99.0)
            min_dist = bsum.get("min_dist", 99.0)
            geom_str = f"  {BOLD}Geometry  {RESET}  mean_dist={mean_dist:.2f}m  min_dist={min_dist:.2f}m"
            if latest_eval is not None and "mean_dist" in latest_eval:
                e_mean_dist = latest_eval.get("mean_dist", 99.0)
                e_min_dist = latest_eval.get("min_dist", 99.0)
                geom_str += f" | {BLUE}[Eval]{RESET} mean={e_mean_dist:.2f}m  min={e_min_dist:.2f}m"
            lines.append(geom_str)

        if "gating_switches" in bsum or "mean_p_safe" in bsum:
            switches = bsum.get("gating_switches", 0.0)
            p_safe = bsum.get("mean_p_safe", 1.0)
            gate_str = f"  {BOLD}Gating    {RESET}  switches={switches:.1f}  mean_p_safe={p_safe:.3f}"
            if latest_eval is not None and "gating_switches" in latest_eval:
                e_switches = latest_eval.get("gating_switches", 0.0)
                e_p_safe = latest_eval.get("mean_p_safe", 1.0)
                gate_str += f" | {BLUE}[Eval]{RESET} switches={e_switches:.1f}  p_safe={e_p_safe:.3f}"
            lines.append(gate_str)

        # Gating Shield details (fallback attempts, recoveries, and failure partitioning)
        if "fallback_attempts" in bsum:
            attempts = bsum.get("fallback_attempts", 0.0)
            recoveries = bsum.get("fallback_recoveries", 0.0)
            f_chaser = bsum.get("fall_on_chaser", 0.0)
            f_fallback = bsum.get("fall_on_fallback", 0.0)
            
            shield_str = f"  {BOLD}Shield    {RESET}  attempts={attempts:.1f}  recoveries={recoveries:.1f}  falls[chaser={f_chaser:.2f}, fallback={f_fallback:.2f}]"
            if latest_eval is not None and "fallback_attempts" in latest_eval:
                e_attempts = latest_eval.get("fallback_attempts", 0.0)
                e_recoveries = latest_eval.get("fallback_recoveries", 0.0)
                e_chaser = latest_eval.get("fall_on_chaser", 0.0)
                e_fallback = latest_eval.get("fall_on_fallback", 0.0)
                shield_str += f" | {BLUE}[Eval]{RESET} att={e_attempts:.1f} rec={e_recoveries:.1f} falls[ch={e_chaser:.2f}, fb={e_fallback:.2f}]"
            lines.append(shield_str)

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

            lines.append(f"  {BOLD}Trend(Short){RESET} "
                          f"hold {d_hold:+.4f} {_arrow(d_hold)}  "
                          f"surv {d_surv:+.4f} {_arrow(d_surv)}")

        # Long-term trends (e.g. over last 50 updates) to diagnose learning progress
        if len(self.history) >= 10:
            min_dist_trend = self._calculate_trend("bsum.min_dist", length=50)
            hold_ratio_trend = self._calculate_trend("bsum.hold_ratio", length=50)
            
            if min_dist_trend and hold_ratio_trend:
                d_min = min_dist_trend["overall_diff"]
                d_hold = hold_ratio_trend["overall_diff"]
                n_updates = min_dist_trend["n"]
                
                # Format descriptors
                def _geom_desc(diff):
                    if diff < -0.05: return f"{GREEN}APPROACHING 🔥 (Distance shrinking){RESET}"
                    if diff > 0.05: return f"{RED}DRIFTING FARTHER ⚠️{RESET}"
                    return f"{YELLOW}STAGNANT 🛑 (No movement){RESET}"
                
                def _hold_desc(diff):
                    if diff > 0.01: return f"{GREEN}IMPROVING 🔥 (Learning){RESET}"
                    if diff < -0.01: return f"{RED}DEGRADED ⚠️{RESET}"
                    return f"{YELLOW}STAGNANT 🛑 (No learning){RESET}"
                
                lines.append(f"  {BOLD}Trend(Long) {RESET} {BOLD}Learning Dynamics (Last {n_updates} Updates):{RESET}")
                lines.append(f"    - min_dist    : {d_min:+.2f}m ({_geom_desc(d_min)})")
                lines.append(f"    - hold_ratio  : {d_hold:+.3f} ({_hold_desc(d_hold)})")

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
