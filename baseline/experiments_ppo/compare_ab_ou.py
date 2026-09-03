"""A/B comparison script for OU vs white-noise exploration on basic_balance_step.

Parses __RAW_STATS__ lines from both training logs and prints a
side-by-side comparison of key metrics over training.

Usage:
    PYTHONPATH=. python3 baseline/experiments_ppo/compare_ab_ou.py \
        --ctrl baseline/runs/ab_ctrl_ou_test_ctrl/train.log \
        --ou   baseline/runs/ab_ctrl_ou_test_ou/train.log
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def parse_log(log_path: str) -> List[Dict]:
    """Extract all __RAW_STATS__ JSON entries from a training log."""
    entries = []
    pattern = re.compile(r'^__RAW_STATS__\s+(\{.*\})\s*$')
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                try:
                    entries.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    pass
    return entries


def extract_series(entries: List[Dict], key_path: str) -> np.ndarray:
    """Extract a time series from entries using dotted key path."""
    parts = key_path.split('.')
    values = []
    for e in entries:
        v = e
        try:
            for p in parts:
                v = v[p]
            values.append(float(v))
        except (KeyError, TypeError, IndexError):
            pass
    return np.array(values)


def smooth(arr: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def fmt(arr: np.ndarray, n: int = 5) -> str:
    """Format array as first/last/mean/max."""
    if len(arr) == 0:
        return "N/A"
    s = smooth(arr, min(20, len(arr)))
    return (f"first={arr[0]:.4f} last={arr[-1]:.4f} "
            f"mean={arr.mean():.4f} max={arr.max():.4f} "
            f"smoothed_last={s[-1]:.4f}")


def print_comparison(ctrl_entries, ou_entries):
    """Print side-by-side comparison."""
    metrics = [
        ("update", "update", "Update #"),
        ("ep_len_mean", "episode_stats.ep_len_mean", "Episode length"),
        ("entropy", "stats.entropy", "Entropy"),
        ("std_mean", "stats.std_mean", "Std mean"),
        ("tanh_sat_frac", "stats.tanh_sat_frac", "Tanh sat frac"),
        ("approx_kl", "stats.approx_kl", "Approx KL"),
        ("clip_frac", "stats.clip_frac", "Clip frac"),
        ("policy_loss", "stats.policy_loss", "Policy loss"),
        ("r_fall_return", "stats.ret_mean_r_fall", "r_fall return"),
        ("r_left_foot_return", "stats.ret_mean_r_left_foot", "r_left_foot return"),
        ("r_right_foot_return", "stats.ret_mean_r_right_foot", "r_right_foot return"),
        ("r_fall_ev", "stats.ev_r_fall", "r_fall EV"),
        ("r_left_foot_ev", "stats.ev_r_left_foot", "r_left_foot EV"),
        ("r_right_foot_ev", "stats.ev_r_right_foot", "r_right_foot EV"),
    ]

    print(f"\n{'='*100}")
    print(f"{'Metric':<25} {'CTRL (white noise)':<40} {'OU (temporal)':<40}")
    print(f"{'='*100}")

    for label, key_path, display in metrics:
        ctrl_arr = extract_series(ctrl_entries, key_path)
        ou_arr = extract_series(ou_entries, key_path)
        print(f"{display:<25} {fmt(ctrl_arr):<40} {fmt(ou_arr):<40}")

    print(f"{'='*100}")

    # Episode length trajectory (key indicator of stepping behavior)
    print("\nEpisode length trajectory (smoothed, every 50 updates):")
    ctrl_ep = extract_series(ctrl_entries, "episode_stats.ep_len_mean")
    ou_ep = extract_series(ou_entries, "episode_stats.ep_len_mean")
    n = min(len(ctrl_ep), len(ou_ep))
    step = max(1, n // 20)
    print(f"{'update':<10} {'CTRL':<15} {'OU':<15} {'diff':<15}")
    print(f"{'-'*55}")
    for i in range(0, n, step):
        c = ctrl_ep[i]
        o = ou_ep[i]
        print(f"{i+1:<10} {c:<15.2f} {o:<15.2f} {o-c:<+15.2f}")

    # Foot reward trajectory (key indicator of stepping)
    print("\nFoot reward (r_left_foot + r_right_foot) trajectory:")
    ctrl_lf = extract_series(ctrl_entries, "stats.ret_mean_r_left_foot")
    ou_lf = extract_series(ou_entries, "stats.ret_mean_r_left_foot")
    ctrl_rf = extract_series(ctrl_entries, "stats.ret_mean_r_right_foot")
    ou_rf = extract_series(ou_entries, "stats.ret_mean_r_right_foot")
    n = min(len(ctrl_lf), len(ou_lf))
    step = max(1, n // 20)
    print(f"{'update':<10} {'CTRL':<15} {'OU':<15} {'diff':<15}")
    print(f"{'-'*55}")
    for i in range(0, n, step):
        c = ctrl_lf[i] + ctrl_rf[i]
        o = ou_lf[i] + ou_rf[i]
        print(f"{i+1:<10} {c:<15.4f} {o:<15.4f} {o-c:<+15.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl", required=True, help="Control (white noise) log path")
    parser.add_argument("--ou", required=True, help="OU (temporal noise) log path")
    args = parser.parse_args()

    ctrl_entries = parse_log(args.ctrl)
    ou_entries = parse_log(args.ou)

    print(f"CTRL: {len(ctrl_entries)} updates parsed from {args.ctrl}")
    print(f"OU:   {len(ou_entries)} updates parsed from {args.ou}")

    if not ctrl_entries:
        print("ERROR: No CTRL stats found.")
        return
    if not ou_entries:
        print("ERROR: No OU stats found.")
        return

    print_comparison(ctrl_entries, ou_entries)


if __name__ == "__main__":
    main()
