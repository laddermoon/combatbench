#!/usr/bin/env python3
"""Generate boundary comparison visualizations across generations.

Produces:
  - boundary_comparison_static.png: 3x3 polar heatmap grid (rows=generations, cols=forces)
  - boundary_evolution.gif: animated GIF morphing between generations
  - boundary_evolution.mp4: MP4 version of the animation

Usage:
    python3 plot_boundary_comparison.py [--base-dir DIR]

    --base-dir: directory containing boundary_gen{N}.json files
                (default: baseline/humanoid21/balance_recover/)
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


def load_boundary_data(base_dir: Path, gens: list, forces: list) -> dict:
    """Load and interpolate boundary JSON data to 1-degree resolution."""
    all_data = {}
    for gen in gens:
        fpath = base_dir / f'boundary_{gen}.json'
        if not fpath.exists():
            continue
        with open(fpath) as f:
            j = json.load(f)
        raw = {}
        for r in j['results']:
            raw[(r['direction_angle'], r['force'])] = r['critical_duration']
        for f in forces:
            angles_16 = np.array(sorted(set(a for (a, ff) in raw if ff == f)))
            cds_16 = np.array([raw[(a, f)] for a in angles_16])
            angles_ext = np.concatenate([angles_16, [angles_16[0] + 360]])
            cds_ext = np.concatenate([cds_16, [cds_16[0]]])
            all_data[(gen, f)] = np.interp(np.arange(360), angles_ext, cds_ext)
    return all_data


def make_colormap():
    colors = ['#0d1b2a', '#1b4965', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    return LinearSegmentedColormap.from_list('custom', colors, N=256)


def plot_static(all_data, gens, gen_labels, forces, out_path: Path):
    """3x3 polar heatmap grid."""
    cmap = make_colormap()
    theta = np.deg2rad(np.arange(361))

    n_gens = len(gens)
    fig, axes = plt.subplots(n_gens, 3, figsize=(18, 6 * n_gens), subplot_kw={'projection': 'polar'})
    if n_gens == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('Boundary Critical Duration Comparison Across Generations', 
                 fontsize=20, fontweight='bold', y=0.98)

    for i, gen in enumerate(gens):
        for j, f in enumerate(forces):
            ax = axes[i][j]
            vals = np.append(all_data[(gen, f)], all_data[(gen, f)][0])
            for k in range(360):
                ax.fill(theta[k:k+2], vals[k:k+2], color=cmap(vals[k]/40), alpha=0.85)
            ax.plot(theta, vals, color='black', linewidth=1.5, zorder=5)
            ax.set_ylim(0, 42)
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(-1)
            ax.set_title(f'{gen_labels[i]}  F={int(f)}N\n(mean={np.mean(all_data[(gen,f)]):.1f})',
                         fontsize=13, pad=15)
            ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
            ax.set_xticklabels(['0°', '315°', '270°', '225°', '180°', '135°', '90°', '45°'], fontsize=10)
            ax.grid(color='gray', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Static saved: {out_path}")


def plot_animated(all_data, gens, gen_labels, forces, base_dir: Path):
    """Animated GIF + MP4 morphing between generations."""
    cmap = make_colormap()
    theta = np.deg2rad(np.arange(361))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), subplot_kw={'projection': 'polar'})
    n_gens = len(gens)
    n_frames_per_gen = 15
    total_frames = n_gens * n_frames_per_gen + 15

    def get_frame_data(frame):
        gen_idx = min(frame // n_frames_per_gen, n_gens - 1)
        next_idx = min(gen_idx + 1, n_gens - 1)
        frac = (frame % n_frames_per_gen) / n_frames_per_gen
        if gen_idx == n_gens - 1:
            frac, next_idx = 0, n_gens - 1
        fd = {}
        for f in forces:
            v0, v1 = all_data[(gens[gen_idx], f)], all_data[(gens[next_idx], f)]
            vals = v0 * (1 - frac) + v1 * frac
            fd[f] = np.append(vals, vals[0])
        if frac > 0:
            label = f"{gen_labels[gen_idx]} -> {gen_labels[next_idx]}  (interp {frac:.0%})"
        else:
            label = gen_labels[gen_idx]
        return fd, label

    def animate(frame):
        for ax in axes:
            ax.clear()
        fd, label = get_frame_data(frame)
        for j, f in enumerate(forces):
            ax = axes[j]
            vals = fd[f]
            for k in range(360):
                ax.fill(theta[k:k+2], vals[k:k+2], color=cmap(vals[k]/40), alpha=0.85)
            ax.plot(theta, vals, color='black', linewidth=1.5, zorder=5)
            ax.set_ylim(0, 42)
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(-1)
            ax.set_title(f'F={int(f)}N  (mean={np.mean(vals[:-1]):.1f})', fontsize=14, pad=15)
            ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
            ax.set_xticklabels(['0°', '315°', '270°', '225°', '180°', '135°', '90°', '45°'], fontsize=10)
            ax.grid(color='gray', alpha=0.3)
        fig.suptitle(f'Boundary Evolution  —  {label}', fontsize=18, fontweight='bold')

    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=200, blit=False)
    gif_path = base_dir / 'boundary_evolution.gif'
    anim.save(str(gif_path), writer='pillow', fps=10, dpi=100)
    plt.close()
    print(f"GIF saved: {gif_path}")

    anim2 = animation.FuncAnimation(fig, animate, frames=total_frames, interval=200, blit=False)
    mp4_path = base_dir / 'boundary_evolution.mp4'
    anim2.save(str(mp4_path), writer='ffmpeg', fps=15, dpi=120, bitrate=2000)
    plt.close()
    print(f"MP4 saved: {mp4_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot boundary comparison across generations')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Directory containing boundary_gen{N}.json files')
    args = parser.parse_args()

    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(__file__).resolve().parent

    # Auto-detect available generations
    gens = []
    gen_labels = []
    i = 0
    while True:
        name = f'gen{i}'
        if (base_dir / f'boundary_{name}.json').exists():
            gens.append(name)
            gen_labels.append(f'Gen {i}')
            i += 1
        else:
            break
    if not gens:
        print(f"No boundary_gen*.json files found in {base_dir}")
        return
    forces = [40.0, 100.0, 200.0]

    all_data = load_boundary_data(base_dir, gens, forces)
    if not all_data:
        print(f"No boundary data found in {base_dir}")
        return

    plot_static(all_data, gens, gen_labels, forces, base_dir / 'boundary_comparison_static.png')
    plot_animated(all_data, gens, gen_labels, forces, base_dir)


if __name__ == '__main__':
    main()
