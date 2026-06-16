#!/usr/bin/env python3
"""
Fight-Experiment Training Log Monitor and Diagnostic Tool.

Tracks the key fight episode metrics (fight_ratio, follow_ratio, recover_ratio, damage_dealt, survived)
plus PPO health, critic quality, and reward-signal trends specific to the three-way fight-mixed policy.

Usage:
    # One-shot analysis
    python3 analyze_fight_logs.py fight.log

    # Real-time watch mode
    python3 analyze_fight_logs.py fight.log --watch

    # Custom window size (default 10 updates)
    python3 analyze_fight_logs.py fight.log --window 20
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


class FightLogAnalyzer:
    """Sliding-window diagnostic engine for the fight curriculum experiment."""

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

        # ---- Check A: Exploration / Joint Std Collapse ----
        std_mins = self._series(recent, "stats.std_min")
        avg_std_min = sum(std_mins) / len(std_mins) if std_mins else 1.0
        if avg_std_min <= 0.145:
            conclusions.append({
                "severity": "CRITICAL",
                "title": "Exploration Collapse — std_min locked at floor",
                "conclusion": (
                    "At least one joint's action standard deviation has hit the minimum floor. "
                    "That DOF is deterministic and can no longer explore combat strategies."
                ),
                "evidence": (
                    f"  Window u{u_start}–u{u_end} ({len(recent)} updates)\n"
                    f"  avg std_min = {avg_std_min:.4f}\n"
                    f"  series: {[round(x, 4) for x in std_mins]}"
                ),
                "remedy": (
                    "1. Raise log_std_min in exp_fight.py (e.g. from -1.8 to -1.5).\n"
                    "2. Increase entropy_coef to encourage exploration."
                ),
            })

        # ---- Check B: PPO Update Early Stop ----
        epochs_dones = self._series(recent, "stats.epochs_done")
        avg_epochs = sum(epochs_dones) / len(epochs_dones) if epochs_dones else 0
        if avg_epochs <= 1.2:
            kls = self._series(recent, "stats.approx_kl")
            conclusions.append({
                "severity": "WARNING",
                "title": "PPO Early Stop — KL exceeds target every update",
                "conclusion": (
                    "The policy diverges so fast that PPO early-stops after epoch 0 "
                    "every update. Data efficiency is very low."
                ),
                "evidence": (
                    f"  avg epochs_done = {avg_epochs:.1f} (target: 4)\n"
                    f"  series: {epochs_dones}\n"
                    f"  KL values: {[round(x, 4) for x in kls]}"
                ),
                "remedy": (
                    "1. Lower the actor learning rate in exp_fight.py (e.g. set to 1.5e-5).\n"
                    "2. Verify advantage normalization is functioning."
                ),
            })

        # ---- Check C: Critic EV Blindness ----
        last = recent[-1]
        stats = last.get("stats", {})
        reward_keys = ("r_fall", "r_cross", "r_radial", "r_tangential", "r_damage", "r_gate", "r_follow_gate")
        for key in reward_keys:
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
                        "3. Early in training, EVs can temporarily go negative; check long-term trends."
                    ),
                })

        # ---- Check D: Combat Passivity / Low Damage (🚨 战斗消极怠工) ----
        damage_values = self._series(recent, "bsum.damage_dealt")
        if damage_values:
            avg_damage = sum(damage_values) / len(damage_values)
            if avg_damage <= -10.0:
                conclusions.append({
                    "severity": "CRITICAL",
                    "title": "Severe Combat Passivity — taking heavy damage without hitting back",
                    "conclusion": (
                        f"The agent is taking heavy net damage (avg damage_dealt: {avg_damage:.2f}). "
                        "The primary Fight policy is getting beaten by the opponent follow policy without countering."
                    ),
                    "evidence": (
                        f"  avg damage_dealt = {avg_damage:.2f} per episode\n"
                        f"  series: {[round(x, 2) for x in damage_values]}"
                    ),
                    "remedy": (
                        "1. Increase r_damage weight in initial_weights / next_weights (e.g. raise from 3.0 to 5.0).\n"
                        "2. Ensure the primary fight policy starting checkpoint (u10294/u10295) has enough baseline movement to face and hit the opponent.\n"
                        "3. Lower joint action standard deviations to make punches/attacks more coordinated."
                    ),
                })

        # ---- Check E: Gating Jitter / Mode Oscillation (🚨 频繁模式切换) ----
        switches = self._series(recent, "bsum.gating_switches")
        if switches:
            avg_switches = sum(switches) / len(switches)
            if avg_switches > 12.0:
                conclusions.append({
                    "severity": "WARNING",
                    "title": "Hyper-Oscillatory Gating — excessive fallback switching",
                    "conclusion": (
                        f"The policy switches between Fight, Follow, and Recover {avg_switches:.1f} times "
                        "per episode. Jittery switching indicates threshold boundary instability, preventing continuous walk or fight gaits."
                    ),
                    "evidence": (
                        f"  avg switches per episode = {avg_switches:.1f}\n"
                        f"  series: {[round(x, 1) for x in switches]}"
                    ),
                    "remedy": (
                        "1. Modify proximity thresholds in fight_mixed.yaml to create wider hysteresis.\n"
                        "2. Add release_patience in FightMixedPolicy to enforce a minimum staying period in each fallback mode.\n"
                        "3. Smooth p_safe predictions over time using a sliding average filter."
                    ),
                })

        # ---- Check F: Follow Fallback Stagnation (🚨 困在Follow模式无法近身) ----
        follow_ratios = self._series(recent, "bsum.follow_ratio")
        fight_ratios = self._series(recent, "bsum.fight_ratio")
        if follow_ratios and fight_ratios:
            avg_follow = sum(follow_ratios) / len(follow_ratios)
            avg_fight = sum(fight_ratios) / len(fight_ratios)
            if avg_follow > 0.65 and avg_fight < 0.20:
                conclusions.append({
                    "severity": "CRITICAL",
                    "title": "Follow Fallback Stagnation — unable to engage in Fight mode",
                    "conclusion": (
                        f"The agent is stuck in locomotion fallback (avg follow_ratio: {avg_follow:.1%}) "
                        f"and spends almost no time in primary combat mode (avg fight_ratio: {avg_fight:.1%}). "
                        "It is running after the opponent but never getting close enough (under 1.0m hysteresis threshold) to trigger Fight."
                    ),
                    "evidence": (
                        f"  avg follow_ratio = {avg_follow:.1%}\n"
                        f"  avg fight_ratio  = {avg_fight:.1%}\n"
                        f"  series(follow)   = {[round(x, 2) for x in follow_ratios]}\n"
                        f"  series(fight)    = {[round(x, 2) for x in fight_ratios]}"
                    ),
                    "remedy": (
                        "1. Verify that the pre-trained Follow fallback policy is functioning correctly and can successfully sprint to close the distance.\n"
                        "2. Lower the proximity switch threshold from 1.0m to 1.2m temporarily to help trigger Fight steps more easily.\n"
                        "3. Make sure the learning Fight policy's actions do not immediately push the opponent away upon activation."
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
        
        # Core Fight metrics
        surv = bsum.get("survived", 0.0)
        fight = bsum.get("fight_ratio", 1.0)
        follow = bsum.get("follow_ratio", 0.0)
        recover = bsum.get("recover_ratio", 0.0)
        dmg = bsum.get("damage_dealt", 0.0)
        
        ep_len = stats.get("ep_len_mean", 0.0)
        epochs = stats.get("epochs_done", 0)
        kl = stats.get("approx_kl", 0.0)
        ploss = stats.get("policy_loss", 0.0)

        # Find latest evaluation summary if present
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
        
        metrics_str = f"  {BOLD}Fight Ratios{RESET} fight={GREEN}{fight:.3f}{RESET}  " \
                      f"follow={BLUE}{follow:.3f}{RESET}  recover={YELLOW}{recover:.3f}{RESET}"
        if latest_eval is not None:
            e_fight = latest_eval.get("fight_ratio", 0.0)
            e_follow = latest_eval.get("follow_ratio", 0.0)
            e_rec = latest_eval.get("recover_ratio", 0.0)
            metrics_str += f" | {BLUE}[Eval u{eval_update}]{RESET} f={e_fight:.3f} fol={e_follow:.3f} rec={e_rec:.3f}"
        lines.append(metrics_str)

        dmg_str = f"  {BOLD}Performance{RESET}  damage_dealt={GREEN}{dmg:+.2f}{RESET}  survived={surv:.3f}"
        if latest_eval is not None:
            e_dmg = latest_eval.get("damage_dealt", 0.0)
            e_surv = latest_eval.get("survived", 0.0)
            dmg_str += f" | {BLUE}[Eval]{RESET} dmg={e_dmg:+.2f} surv={e_surv:.3f}"
        lines.append(dmg_str)

        # Geometric & Gating switch tracking
        if "mean_dist" in bsum or "min_dist" in bsum:
            mean_dist = bsum.get("mean_dist", 99.0)
            min_dist = bsum.get("min_dist", 99.0)
            geom_str = f"  {BOLD}Geometry  {RESET}  mean_dist={mean_dist:.2f}m  min_dist={min_dist:.2f}m"
            if latest_eval is not None and "mean_dist" in latest_eval:
                e_mean_dist = latest_eval.get("mean_dist", 99.0)
                e_min_dist = latest_eval.get("min_dist", 99.0)
                geom_str += f" | {BLUE}[Eval]{RESET} mean={e_mean_dist:.2f}m  min={e_min_dist:.2f}m"
            lines.append(geom_str)

        if "gating_switches" in bsum:
            switches = bsum.get("gating_switches", 0.0)
            p_safe = bsum.get("mean_p_safe", 1.0)
            gate_str = f"  {BOLD}Gating    {RESET}  switches={switches:.1f}  mean_p_safe={p_safe:.3f}"
            if latest_eval is not None and "gating_switches" in latest_eval:
                e_switches = latest_eval.get("gating_switches", 0.0)
                e_p_safe = latest_eval.get("mean_p_safe", 1.0)
                gate_str += f" | {BLUE}[Eval]{RESET} sw={e_switches:.1f} p_safe={e_p_safe:.3f}"
            lines.append(gate_str)

        # Fall allocation partition
        if "fall_on_fight" in bsum:
            f_fight = bsum.get("fall_on_fight", 0.0)
            f_follow = bsum.get("fall_on_follow", 0.0)
            f_recover = bsum.get("fall_on_recover", 0.0)
            fall_str = f"  {BOLD}Falls     {RESET}  fight_falls={RED}{f_fight:.2f}{RESET}  follow_falls={BLUE}{f_follow:.2f}{RESET}  recover_falls={YELLOW}{f_recover:.2f}{RESET}"
            if latest_eval is not None and "fall_on_fight" in latest_eval:
                e_fight = latest_eval.get("fall_on_fight", 0.0)
                e_fol = latest_eval.get("fall_on_follow", 0.0)
                e_rec = latest_eval.get("fall_on_recover", 0.0)
                fall_str += f" | {BLUE}[Eval]{RESET} fg={e_fight:.2f} fol={e_fol:.2f} rec={e_rec:.2f}"
            lines.append(fall_str)

        lines.append(f"  {BOLD}Episode   {RESET}  mean_len={ep_len:.0f} steps")
        lines.append(f"  {BOLD}PPO       {RESET}  loss={ploss:+.4f}  "
                      f"epochs={epochs}/4  kl={kl:.4f}")

        # Short-term trend summary
        if len(self.history) >= 5:
            dmg_series = self._series(self.history, "bsum.damage_dealt")
            half = len(dmg_series) // 2
            d_dmg = (sum(dmg_series[-half:]) / half) - (sum(dmg_series[:half]) / half)
            fight_series = self._series(self.history, "bsum.fight_ratio")
            d_fight = (sum(fight_series[-half:]) / half) - (sum(fight_series[:half]) / half)

            def _arrow(delta, thresh=0.01):
                if delta > thresh: return f"{GREEN}↑{RESET}"
                if delta < -thresh: return f"{RED}↓{RESET}"
                return f"{YELLOW}→{RESET}"

            lines.append(f"  {BOLD}Trend(Short){RESET} "
                          f"damage {d_dmg:+.2f} {_arrow(d_dmg, 0.5)}  "
                          f"fight_ratio {d_fight:+.3f} {_arrow(d_fight)}")

        # Long-term regression trends
        if len(self.history) >= 10:
            dmg_trend = self._calculate_trend("bsum.damage_dealt", length=50)
            fight_trend = self._calculate_trend("bsum.fight_ratio", length=50)
            
            if dmg_trend and fight_trend:
                d_dmg_long = dmg_trend["overall_diff"]
                d_fight_long = fight_trend["overall_diff"]
                avg_dmg = dmg_trend["last_avg"]
                n_updates = dmg_trend["n"]
                
                def _dmg_desc(diff, val):
                    if diff > 1.5: return f"{GREEN}ATTACK LEARNING 🔥 (Dealt damage increasing){RESET}"
                    if diff < -1.5: return f"{RED}COMBAT DEGRADATION ⚠️ (Dealt damage dropping){RESET}"
                    if val > 1.0: return f"{GREEN}STRONG COMBAT 💥 (Dealt damage holding high){RESET}"
                    return f"{YELLOW}STAGNANT 🛑 (No combat improvements){RESET}"
                
                def _fight_desc(diff):
                    if diff > 0.02: return f"{GREEN}APPROACH PROGRESS 🔥 (Spending more steps in Fight){RESET}"
                    if diff < -0.02: return f"{RED}SAFETY COLLAPSE ⚠️ (Retreating into safety Recover/Follow){RESET}"
                    return f"{YELLOW}STAGNANT 🛑 (Engagement ratio flat){RESET}"
                
                lines.append(f"  {BOLD}Trend(Long) {RESET} {BOLD}Combat Dynamics (Last {n_updates} Updates):{RESET}")
                lines.append(f"    - damage_dealt: {d_dmg_long:+.2f} ({_dmg_desc(d_dmg_long, avg_dmg)})")
                lines.append(f"    - fight_ratio : {d_fight_long:+.3f} ({_fight_desc(d_fight_long)})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reward breakdown
    # ------------------------------------------------------------------

    def reward_breakdown(self) -> str:
        last = self.history[-1]
        rsum = last.get("rsum", {})
        stats = last.get("stats", {})
        lines = []
        reward_keys = ("r_fall", "r_cross", "r_radial", "r_tangential", "r_damage", "r_gate", "r_follow_gate")
        for key in reward_keys:
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

def print_update(analyzer: FightLogAnalyzer):
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
        print(f"\n{GREEN}[HEALTHY] No structural anomalies detected in the slide-window.{RESET}")
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
        description="Fight-experiment training log monitor."
    )
    parser.add_argument("log_file", type=str)
    parser.add_argument("--watch", action="store_true",
                        help="Watch file in real-time (tail -f).")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    analyzer = FightLogAnalyzer(window_size=args.window)
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
