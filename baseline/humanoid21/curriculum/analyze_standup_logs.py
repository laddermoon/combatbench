#!/usr/bin/env python3
"""
Standup-Experiment Training Log Monitor and Diagnostic Tool.

Tracks the key standup metrics (success, max_stage, avg_stage, max_potential)
plus PPO health, critic quality, and reward-signal trends specific to the
stand-up potential shaping.

Usage:
    # One-shot analysis
    python3 analyze_standup_logs.py standup.log

    # Real-time watch mode
    python3 analyze_standup_logs.py standup.log --watch

    # Custom window size (default 10 updates)
    python3 analyze_standup_logs.py standup.log --window 20
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


class StandupLogAnalyzer:
    """Sliding-window diagnostic engine for the standup task."""

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

        # ---- Check A: PPO early-stop every epoch (Trust Region broken) ----
        epochs_dones = self._series(recent, "stats.epochs_done")
        avg_epochs = sum(epochs_dones) / len(epochs_dones) if epochs_dones else 0
        if avg_epochs <= 1.2:
            kls = self._series(recent, "stats.approx_kl")
            conclusions.append({
                "severity": "WARNING",
                "title": "PPO Trust Region Rupture — learning rate too high",
                "conclusion": (
                    "The policy updates are diverging so rapidly that PPO early-stops "
                    "after only 1 epoch due to KL limit violation. Experience replay data efficiency is extremely low."
                ),
                "evidence": (
                    f"  avg epochs_done = {avg_epochs:.1f} (target: 4 epochs)\n"
                    f"  series: {epochs_dones}\n"
                    f"  KL values: {[round(x, 4) for x in kls]}"
                ),
                "remedy": (
                    "1. Lower the actor learning rate by 30% to 50% (e.g. from 3e-5 to 1.5e-5).\n"
                    "2. Verify that advantage normalization is functioning correctly.\n"
                    "3. If fine-tuning from a highly specialized policy, reduce clip_eps to stabilize trust regions."
                ),
            })

        # ---- Check B: Exploration Collapse (Deterministic joints) ----
        std_mins = self._series(recent, "stats.std_min")
        avg_std_min = sum(std_mins) / len(std_mins) if std_mins else 1.0
        if avg_std_min <= 0.145:
            conclusions.append({
                "severity": "CRITICAL",
                "title": "Exploration Collapse — policy locked deterministic",
                "conclusion": (
                    "At least one major joint has hit the log_std_min boundary. "
                    "That degree of freedom is acting deterministically, completely ending exploration for that joint."
                ),
                "evidence": (
                    f"  avg std_min = {avg_std_min:.4f}\n"
                    f"  series: {[round(x, 4) for x in std_mins]}"
                ),
                "remedy": (
                    "1. Raise log_std_min in the experiment config (e.g., set to -1.5).\n"
                    "2. Increase the entropy coefficient (entropy_coef) by 2x to penalize deterministic actions.\n"
                    "3. Inject temporary noise or lower learning rate to escape local optima."
                ),
            })

        # ---- Check C: Critic Blindness (Explained Variance EV <= 0) ----
        last = recent[-1]
        stats = last.get("stats", {})
        for key in ("r_potential", "r_cross"):
            ev_key = f"ev_{key}"
            if ev_key not in stats:
                continue
            evs = self._series(recent, f"stats.{ev_key}")
            avg_ev = sum(evs) / len(evs) if evs else 0.0
            if avg_ev <= 0.0:
                conclusions.append({
                    "severity": "WARNING",
                    "title": f"Critic Blindness — EV({key}) <= 0",
                    "conclusion": (
                        f"The value function critic for '{key}' is predicting worse than simple mean variance. "
                        "The resulting advantages are noisy and distort training directions."
                    ),
                    "evidence": (
                        f"  avg EV = {avg_ev:+.4f}\n"
                        f"  series: {[round(x, 3) for x in evs]}"
                    ),
                    "remedy": (
                        "1. Increase the critic learning rate (typically 2-4x higher than actor LR).\n"
                        "2. Ensure the reward scale is not too extreme or too low.\n"
                        "3. Lower gamma for this reward key to shrink the prediction horizon."
                    ),
                })

        # ---- Check D: Stuck in Lying/Rollover Stages (Stage 0/1 block) ----
        max_stages = self._series(recent, "bsum.max_stage")
        if max_stages:
            avg_max_stage = sum(max_stages) / len(max_stages)
            if avg_max_stage < 1.5:
                conclusions.append({
                    "severity": "CRITICAL",
                    "title": "Stage Bottleneck — stuck in lying / rollover phases",
                    "conclusion": (
                        f"The robot is consistently failing to transition past Stage 1 (Double Kneeling). "
                        f"It is trapped rolling over on the floor (avg max stage: {avg_max_stage:.2f}), "
                        "unable to figure out how to push up onto its knees or feet."
                    ),
                    "evidence": (
                        f"  avg max_stage achieved = {avg_max_stage:.2f} (target: Stage 4)\n"
                        f"  series: {[round(x, 1) for x in max_stages]}"
                    ),
                    "remedy": (
                        "1. Increase the potential reward scale or design a small shaping bonus for knee contacts.\n"
                        "2. Double check that knee (shin) contacts are registering properly in simulation (forces > 1N).\n"
                        "3. Verify if torque limits are too weak to support the robot's weight when pushing up."
                    ),
                })

        # ---- Check E: High potential but no perfect stand (Stage 3 hand support trap) ----
        max_pots = self._series(recent, "bsum.max_potential")
        successes = self._series(recent, "bsum.success")
        if max_pots and successes:
            avg_max_pot = sum(max_pots) / len(max_pots)
            avg_succ = sum(successes) / len(successes)
            if avg_max_pot >= 0.65 and avg_succ < 0.05:
                conclusions.append({
                    "severity": "WARNING",
                    "title": "Standing Balance Deficiency — stuck in hands support phase",
                    "conclusion": (
                        f"The robot achieves high potential (avg max: {avg_max_pot:.3f}), meaning it gets "
                        f"to Stage 3 (Feet & Hands Support), but fails to lift hands to reach Stage 4 (Perfect Stand)."
                    ),
                    "evidence": (
                        f"  avg max_potential = {avg_max_pot:.3f}\n"
                        f"  avg success rate  = {avg_succ * 100:.1f}%\n"
                        f"  max_stages achieved: {[round(x, 1) for x in max_stages]}"
                    ),
                    "remedy": (
                        "1. Enhance the potential transition gradient between Stage 3 (max potential 0.75) and "
                        "Stage 4 (starts at 0.75) by scaling the height/stability metrics.\n"
                        "2. Increase the weight of r_cross to help the robot build a firmer stance with feet, "
                        "allowing it to confidently let go of hand support."
                    ),
                })

        # ---- Check F: Long-term Learning Stagnation ----
        if len(self.history) >= 20:
            pot_trend = self._calculate_trend("bsum.max_potential", length=50)
            succ_trend = self._calculate_trend("bsum.success", length=50)
            
            if pot_trend and succ_trend:
                d_pot = pot_trend["overall_diff"]
                d_succ = succ_trend["overall_diff"]
                avg_pot = pot_trend["last_avg"]
                n_updates = pot_trend["n"]
                
                if d_pot <= 0.01 and d_succ <= 0.005 and avg_pot < 0.4:
                    conclusions.append({
                        "severity": "CRITICAL",
                        "title": "Learning Stagnation — zero progress in stand-up learning",
                        "conclusion": (
                            f"Over the last {n_updates} updates, the policy has made zero progress in "
                            f"standing up. Potential is flatlining around {avg_pot:.3f} and success is locked at 0.0%."
                        ),
                        "evidence": (
                            f"  Trend Window     = {n_updates} updates\n"
                            f"  Current potential = {avg_pot:.3f} (target: 1.000)\n"
                            f"  potential change = {d_pot:+.4f}\n"
                            f"  success change   = {d_succ:+.3f}"
                        ),
                        "remedy": (
                            "1. Verify that contacts are detected and reward extraction is active (potential difference > 0).\n"
                            "2. Increase potential_reward_scale in exp_standup.py to generate stronger policy gradients.\n"
                            "3. Check for motor saturation. If joints are fully saturated, they cannot generate push-off force."
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

        succ = bsum.get("success", 0.0)
        max_stage = bsum.get("max_stage", 0.0)
        avg_stage = bsum.get("avg_stage", 0.0)
        max_pot = bsum.get("max_potential", 0.0)
        
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
        lines.append(f"  {BOLD}Metrics   {RESET}  success_rate={GREEN}{succ*100:5.1f}%{RESET}  "
                      f"max_stage={GREEN}{max_stage:.1f}{RESET}  avg_stage={avg_stage:.2f}  "
                      f"max_potential={BLUE}{max_pot:.3f}{RESET}")
        
        if latest_eval is not None:
            e_succ = latest_eval.get("success", 0.0)
            e_max_stage = latest_eval.get("max_stage", 0.0)
            e_avg_stage = latest_eval.get("avg_stage", 0.0)
            e_max_pot = latest_eval.get("max_potential", 0.0)
            lines.append(f"  {BOLD}Metrics(E){RESET}  success_rate={GREEN}{e_succ*100:5.1f}%{RESET}  "
                          f"max_stage={GREEN}{e_max_stage:.1f}{RESET}  avg_stage={e_avg_stage:.2f}  "
                          f"max_potential={BLUE}{e_max_pot:.3f}{RESET} {BLUE}[u{eval_update}]{RESET}")
        
        lines.append(f"  {BOLD}Episode   {RESET}  mean_len={ep_len:.0f} steps")
        lines.append(f"  {BOLD}PPO       {RESET}  loss={ploss:+.4f}  "
                      f"epochs={epochs}/4  kl={kl:.4f}")

        # Trend arrows (Short-term)
        if len(self.history) >= 5:
            pot_series = self._series(self.history, "bsum.max_potential")
            half = len(pot_series) // 2
            d_pot = (sum(pot_series[-half:]) / half) - (sum(pot_series[:half]) / half)
            
            stage_series = self._series(self.history, "bsum.max_stage")
            d_stage = (sum(stage_series[-half:]) / half) - (sum(stage_series[:half]) / half)

            def _arrow(delta, thresh=0.01):
                if delta > thresh: return f"{GREEN}↑{RESET}"
                if delta < -thresh: return f"{RED}↓{RESET}"
                return f"{YELLOW}→{RESET}"

            lines.append(f"  {BOLD}Trend(Short){RESET} "
                          f"potential {d_pot:+.4f} {_arrow(d_pot)}  "
                          f"max_stage {d_stage:+.2f} {_arrow(d_stage)}")

        # Long-term trends (e.g. over last 50 updates) to diagnose learning progress
        if len(self.history) >= 10:
            pot_trend = self._calculate_trend("bsum.max_potential", length=50)
            succ_trend = self._calculate_trend("bsum.success", length=50)
            
            if pot_trend and succ_trend:
                d_pot = pot_trend["overall_diff"]
                d_succ = succ_trend["overall_diff"]
                n_updates = pot_trend["n"]
                
                # Format descriptors
                def _pot_desc(diff):
                    if diff > 0.05: return f"{GREEN}SOARING 🔥 (Robot rising higher){RESET}"
                    if diff < -0.05: return f"{RED}REGRESSING ⚠️ (Falling lower){RESET}"
                    return f"{YELLOW}STAGNANT 🛑 (Stuck at level){RESET}"
                
                def _succ_desc(diff):
                    if diff > 0.01: return f"{GREEN}LEARNING 🔥 (Success rate rising){RESET}"
                    if diff < -0.01: return f"{RED}DEGRADED ⚠️{RESET}"
                    return f"{YELLOW}STAGNANT 🛑 (No perfect stands){RESET}"
                
                lines.append(f"  {BOLD}Trend(Long) {RESET} {BOLD}Learning Dynamics (Last {n_updates} Updates):{RESET}")
                lines.append(f"    - max_potential: {d_pot:+.3f} ({_pot_desc(d_pot)})")
                lines.append(f"    - success_rate : {d_succ:+.3f} ({_succ_desc(d_succ)})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reward breakdown
    # ------------------------------------------------------------------

    def reward_breakdown(self) -> str:
        last = self.history[-1]
        rsum = last.get("rsum", {})
        stats = last.get("stats", {})
        lines = []
        for key in ("r_potential", "r_cross"):
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

def print_update(analyzer: StandupLogAnalyzer):
    last = analyzer.history[-1]
    u = last["update"]
    print(f"\n{'='*60}")
    print(f"{BOLD}{BLUE}>>> Standup Update {u:5d} <<<{RESET}")
    print(f"{'='*60}")
    print(analyzer.progress_summary())
    print()
    print(f"  {BOLD}Rewards{RESET}")
    print(analyzer.reward_breakdown())


def print_diagnostics(conclusions: List[Dict[str, Any]]):
    if not conclusions:
        print(f"\n{GREEN}[HEALTHY] No standup training anomalies detected.{RESET}")
        return

    print(f"\n{'='*60}")
    print(f"  STANDUP DIAGNOSTIC REPORT")
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
        description="Standup-experiment training log monitor."
    )
    parser.add_argument("log_file", type=str, help="Path to training standup.log file")
    parser.add_argument("--watch", action="store_true",
                        help="Watch file in real-time (tail -f).")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    analyzer = StandupLogAnalyzer(window_size=args.window)
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
