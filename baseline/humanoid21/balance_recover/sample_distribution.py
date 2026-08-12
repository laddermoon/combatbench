"""从边界探测 JSON 生成训练采样分布 + 热力图可视化。

逻辑：
1. 读取 boundary JSON（direction, force, critical_duration）
2. 对每个 (direction, force) cell，用 sigma = k × cd 的高斯 CDF 计算区间概率
3. 每个方向均分预算，3 个 force 各初始权重 1/3，截断后归一化
4. 输出概率矩阵 NPZ + 热力图

用法::

    PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/balance_recover/sample_distribution.py \
        --input baseline/humanoid21/balance_recover/boundary_gen3.json \
        --output-dir baseline/humanoid21/balance_recover/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_boundary_json(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载 boundary JSON。

    Returns:
        angles: (N_angles,) 排序后的唯一方向
        forces: (N_forces,) 排序后的唯一力
        cds: (N_angles, N_forces) critical_duration 矩阵
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    angles = sorted(set(r["direction_angle"] for r in data["results"]))
    forces = sorted(set(r["force"] for r in data["results"]))

    angle_idx = {a: i for i, a in enumerate(angles)}
    force_idx = {f: i for i, f in enumerate(forces)}

    cds = np.zeros((len(angles), len(forces)), dtype=np.int32)
    for r in data["results"]:
        cds[angle_idx[r["direction_angle"]], force_idx[r["force"]]] = r["critical_duration"]

    return np.array(angles), np.array(forces), cds


def compute_prob_matrix(
    cds: np.ndarray,
    durations: np.ndarray,
    sigma_k: float = 0.15,
    sigma_min: float = 0.01,
) -> np.ndarray:
    """计算概率矩阵。

    对每个 (angle, force) cell:
    - center = cd + 0.5 (cd=0 时 center=0.5)
    - sigma = max(sigma_k * cd, sigma_min)
    - P(d) = CDF(d+0.5) - CDF(d-0.5) for d in durations
    - 超出 [1, max_dur] 的概率丢弃

    每个方向均分预算 1/N_angles。
    每个方向内 3 个 force 各初始权重 1/3，截断后归一化为 1。

    Returns:
        prob_matrix: (N_angles, N_forces, N_durations) 归一化概率
    """
    n_angles, n_forces = cds.shape
    n_durs = len(durations)
    dur_min = durations[0]
    dur_max = durations[-1]

    prob_matrix = np.zeros((n_angles, n_forces, n_durs), dtype=np.float64)

    for i in range(n_angles):
        retained_weights = np.zeros(n_forces, dtype=np.float64)
        cell_probs = np.zeros((n_forces, n_durs), dtype=np.float64)

        for j in range(n_forces):
            cd = cds[i, j]
            center = cd + 0.5 if cd > 0 else 0.5
            sigma = max(sigma_k * cd, sigma_min)

            for k, d in enumerate(durations):
                p = norm.cdf((d + 0.5 - center) / sigma) - norm.cdf((d - 0.5 - center) / sigma)
                cell_probs[j, k] = p

            retained = cell_probs[j].sum()
            retained_weights[j] = retained

        # 归一化 force 权重
        total_retained = retained_weights.sum()
        if total_retained > 0:
            for j in range(n_forces):
                if retained_weights[j] > 0:
                    # 方向预算 1/n_angles × force 归一化权重 × 条件概率
                    prob_matrix[i, j, :] = (
                        (1.0 / n_angles)
                        * (retained_weights[j] / total_retained)
                        * (cell_probs[j] / retained_weights[j])
                    )

    # 全局归一化
    total = prob_matrix.sum()
    if total > 0:
        prob_matrix /= total

    return prob_matrix


def plot_heatmaps(
    angles: np.ndarray,
    forces: np.ndarray,
    durations: np.ndarray,
    cds: np.ndarray,
    prob_matrix: np.ndarray,
    output_dir: Path,
) -> None:
    """生成热力图。"""
    n_angles = len(angles)
    sector_half = 360.0 / n_angles / 2.0

    # 1. critical_duration 极坐标图
    fig, axes = plt.subplots(1, len(forces), figsize=(6 * len(forces), 6),
                             subplot_kw={"projection": "polar"})
    if len(forces) == 1:
        axes = [axes]
    for j, (force, ax) in enumerate(zip(forces, axes)):
        crit = cds[:, j]
        angles_rad = np.radians(angles)
        angles_closed = np.append(angles_rad, angles_rad[0])
        crit_closed = np.append(crit, crit[0])
        ax.plot(angles_closed, crit_closed, "o-", linewidth=2, markersize=5)
        ax.fill_between(angles_closed, 0, crit_closed, alpha=0.2)
        ax.set_title(f"Critical Duration (F={int(force)}N)", pad=20)
        ax.set_thetagrids(angles, [f"{a:.0f}" for a in angles], fontsize=7)
        ax.set_rlabel_position(225)
    plt.tight_layout()
    fig.savefig(output_dir / "heatmap_critical_duration_polar.png", dpi=150)
    plt.close(fig)

    # 2. 概率热力图（每个 force 一张，direction × duration）
    for j, force in enumerate(forces):
        fig, ax = plt.subplots(figsize=(12, 6))
        data = prob_matrix[:, j, :].T  # (durations, angles)
        im = ax.imshow(data, aspect="auto", cmap="hot",
                       extent=[angles[0] - sector_half, angles[-1] + sector_half,
                               durations[-1] + 0.5, durations[0] - 0.5],
                       interpolation="nearest")
        ax.set_xlabel("Direction Angle (deg)")
        ax.set_ylabel("Duration (action steps)")
        ax.set_title(f"Sampling Probability (F={int(force)}N)")
        ax.set_xticks(angles)
        ax.set_xticklabels([f"{a:.0f}" for a in angles], fontsize=7)
        plt.colorbar(im, ax=ax, label="Probability")
        plt.tight_layout()
        fig.savefig(output_dir / f"heatmap_weight_F{int(force)}.png", dpi=150)
        plt.close(fig)

    # 3. 总概率分布（所有 force 叠加）
    fig, ax = plt.subplots(figsize=(12, 6))
    total_prob = prob_matrix.sum(axis=1).T  # (durations, angles)
    im = ax.imshow(total_prob, aspect="auto", cmap="hot",
                   extent=[angles[0] - sector_half, angles[-1] + sector_half,
                           durations[-1] + 0.5, durations[0] - 0.5],
                   interpolation="nearest")
    ax.set_xlabel("Direction Angle (deg)")
    ax.set_ylabel("Duration (action steps)")
    ax.set_title("Total Sampling Probability (all forces)")
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{a:.0f}" for a in angles], fontsize=7)
    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()
    fig.savefig(output_dir / "heatmap_weight_total.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate sampling distribution from boundary JSON")
    p.add_argument("--input", required=True,
                   help="Path to boundary JSON (from probe_boundary.py)")
    p.add_argument("--output-dir", type=str, default=".",
                   help="Output directory for plots and distribution data.")
    p.add_argument("--sigma-k", type=float, default=0.15,
                   help="Adaptive sigma factor: sigma = sigma_k * critical_duration.")
    p.add_argument("--duration-min", type=int, default=1,
                   help="Minimum duration (action steps).")
    p.add_argument("--duration-max", type=int, default=40,
                   help="Maximum duration (action steps).")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="Number of samples to generate for statistics.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 boundary JSON
    angles, forces, cds = load_boundary_json(args.input)
    durations = np.arange(args.duration_min, args.duration_max + 1)
    print(f"Loaded: {len(angles)} angles, {len(forces)} forces, {len(durations)} durations")
    print(f"  angles: {angles}")
    print(f"  forces: {forces}")
    print(f"  durations: {durations[0]}~{durations[-1]}")

    # 2. 计算概率矩阵
    prob_matrix = compute_prob_matrix(cds, durations, sigma_k=args.sigma_k)
    print(f"\n=== Probability Matrix (sigma_k={args.sigma_k}) ===")
    for j, force in enumerate(forces):
        total = prob_matrix[:, j, :].sum()
        print(f"  F={int(force):>3d}N: total_prob={total:.4f} ({100*total:.1f}%)")

    # 3. 采样统计
    flat_probs = prob_matrix.flatten()
    flat_probs = flat_probs / flat_probs.sum()
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(flat_probs), size=args.n_samples, p=flat_probs)
    n_angles = len(angles)
    n_forces = len(forces)
    n_durs = len(durations)
    sector_half = 360.0 / n_angles / 2.0

    samples = []
    for idx in indices:
        a_idx = idx // (n_forces * n_durs)
        remainder = idx % (n_forces * n_durs)
        f_idx = remainder // n_durs
        d_idx = remainder % n_durs
        angle = float(angles[a_idx]) + rng.uniform(-sector_half, sector_half)
        angle = angle % 360.0
        force = float(forces[f_idx])
        duration = int(durations[d_idx])
        samples.append((angle, force, duration))

    # 4. 保存采样分布 JSON
    dist_path = output_dir / "sample_distribution.json"
    dist_data = {
        "input_json": str(Path(args.input).resolve()),
        "sigma_k": args.sigma_k,
        "forces": forces.tolist(),
        "durations": durations.tolist(),
        "angles": angles.tolist(),
        "prob_matrix_shape": list(prob_matrix.shape),
        "samples": [{"angle": a, "force": f, "duration": d} for a, f, d in samples],
    }
    with open(dist_path, "w") as f:
        json.dump(dist_data, f, indent=2)
    print(f"\nDistribution saved to {dist_path}")

    # 保存采样 CSV
    samples_csv = output_dir / "samples.csv"
    with open(samples_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["angle", "force", "duration"])
        writer.writeheader()
        for a, force, d in samples:
            writer.writerow({"angle": f"{a:.2f}", "force": f"{force:.1f}", "duration": d})
    print(f"Samples CSV saved to {samples_csv}")

    # 5. 保存概率矩阵 NPZ
    weights_npz = output_dir / "sample_weights.npz"
    np.savez_compressed(weights_npz,
                        angles=angles,
                        forces=forces,
                        durations=durations,
                        prob_matrix=prob_matrix,
                        cds=cds,
                        sigma_k=args.sigma_k)
    print(f"Weights NPZ saved to {weights_npz}")

    # 6. 生成热力图
    print(f"\nGenerating heatmaps...")
    plot_heatmaps(angles, forces, durations, cds, prob_matrix, output_dir)
    print(f"Heatmaps saved to {output_dir}/")

    # 7. 打印采样统计
    print(f"\n=== Sample Statistics ({args.n_samples} samples) ===")
    sample_angles = np.array([s[0] for s in samples])
    sample_forces = np.array([s[1] for s in samples])
    sample_durs = np.array([s[2] for s in samples])

    print(f"  Angle: mean={sample_angles.mean():.1f}, std={sample_angles.std():.1f}")
    print(f"  Duration: mean={sample_durs.mean():.1f}, std={sample_durs.std():.1f}")
    for force in forces:
        count = (sample_forces == force).sum()
        print(f"  F={int(force):>3d}N: {count} samples ({100*count/len(samples):.1f}%)")


if __name__ == "__main__":
    main()


class ImpulseSampler:
    """从概率矩阵 NPZ 文件加载分布并采样扰动参数。

    NPZ 需包含: angles, forces, durations, prob_matrix。
    采样时在选中方向所在扇区内做均匀随机化。
    """

    def __init__(self, weight_npz_path: str, direction_jitter: float = 5.0):
        data = np.load(weight_npz_path, allow_pickle=True)
        # 支持新格式 (prob_matrix) 和旧格式 (interp_weights)
        if "prob_matrix" in data:
            self._angles = data["angles"]
            self._forces = data["forces"]
            self._durations = data["durations"]
            self._prob_matrix = data["prob_matrix"]
        else:
            # 旧格式兼容
            self._angles = data["interp_angles"]
            self._forces = data["forces"]
            self._durations = data["durations"]
            self._prob_matrix = data["interp_weights"]

        flat = self._prob_matrix.flatten().astype(np.float64)
        self._flat_probs = flat / flat.sum()
        n_angles = len(self._angles)
        self._sector_half = 360.0 / n_angles / 2.0
        # direction_jitter 参数保留但不再使用，改用扇区随机化
        self._direction_jitter = float(direction_jitter)

    def sample(self, rng: np.random.RandomState) -> dict:
        """采样一组扰动参数。

        Returns:
            {"direction_angle", "force", "duration_action_steps", "body"}
        """
        n_angles = len(self._angles)
        n_forces = len(self._forces)
        n_durs = len(self._durations)
        idx = rng.choice(len(self._flat_probs), p=self._flat_probs)
        a_idx = idx // (n_forces * n_durs)
        remainder = idx % (n_forces * n_durs)
        f_idx = remainder // n_durs
        d_idx = remainder % n_durs
        angle = float(self._angles[a_idx]) + rng.uniform(-self._sector_half, self._sector_half)
        angle = angle % 360.0
        return {
            "direction_angle": angle,
            "force": float(self._forces[f_idx]),
            "duration_action_steps": int(self._durations[d_idx]),
            "body": "torso",
        }
