"""Visualize impulse boundary mapping results as a heatmap.

Usage::

    python3 baseline/framework/plot_impulse_boundary.py \
        --input /tmp/boundary_gen0.csv \
        --output /tmp/boundary_gen0_heatmap.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Plot impulse boundary heatmap")
    p.add_argument("--input", required=True, help="CSV file from probe_impulse_boundary.py")
    p.add_argument("--output", default=None, help="Output PNG path. If omitted, saves next to CSV.")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".png")

    # Read CSV
    forces, durations, surv_rates, mean_lens = [], [], [], []
    with open(in_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            forces.append(float(row["force"]))
            durations.append(int(row["duration"]))
            surv_rates.append(float(row["surv_rate"]))
            mean_lens.append(float(row["mean_len"]))

    force_vals = sorted(set(forces))
    dur_vals = sorted(set(durations))

    # Build grids
    surv_grid = np.full((len(dur_vals), len(force_vals)), np.nan)
    len_grid = np.full((len(dur_vals), len(force_vals)), np.nan)
    for f, d, s, l in zip(forces, durations, surv_rates, mean_lens):
        fi = force_vals.index(f)
        di = dur_vals.index(d)
        surv_grid[di, fi] = s
        len_grid[di, fi] = l

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Heatmap 1: Survival Rate ---
    ax = axes[0]
    im = ax.imshow(surv_grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   origin="lower", extent=[-0.5, len(force_vals)-0.5, -0.5, len(dur_vals)-0.5])
    ax.set_xticks(range(len(force_vals)))
    ax.set_xticklabels([f"{int(f)}" for f in force_vals])
    ax.set_yticks(range(len(dur_vals)))
    ax.set_yticklabels(dur_vals)
    ax.set_xlabel("Force (N)")
    ax.set_ylabel("Duration (action steps)")
    ax.set_title("Survival Rate")
    # Annotate cells
    for di in range(len(dur_vals)):
        for fi in range(len(force_vals)):
            val = surv_grid[di, fi]
            if not np.isnan(val):
                color = "white" if val < 0.4 or val > 0.85 else "black"
                ax.text(fi, di, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, label="Survival Rate")

    # --- Heatmap 2: Mean Episode Length ---
    ax = axes[1]
    im2 = ax.imshow(len_grid, aspect="auto", cmap="viridis", vmin=0, vmax=600,
                    origin="lower", extent=[-0.5, len(force_vals)-0.5, -0.5, len(dur_vals)-0.5])
    ax.set_xticks(range(len(force_vals)))
    ax.set_xticklabels([f"{int(f)}" for f in force_vals])
    ax.set_yticks(range(len(dur_vals)))
    ax.set_yticklabels(dur_vals)
    ax.set_xlabel("Force (N)")
    ax.set_ylabel("Duration (action steps)")
    ax.set_title("Mean Episode Length (steps)")
    for di in range(len(dur_vals)):
        for fi in range(len(force_vals)):
            val = len_grid[di, fi]
            if not np.isnan(val):
                ax.text(fi, di, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if val < 300 else "black")
    fig.colorbar(im2, ax=ax, label="Mean Length")

    fig.suptitle("Impulse Boundary Mapping", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()
