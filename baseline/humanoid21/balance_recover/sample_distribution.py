"""从边界扫描结果生成训练采样分布 + 热力图可视化。

逻辑：
1. 读取全量扫描 CSV（direction, force, duration, survived）
2. 对每个 (direction, force) cell，在 duration 轴上找 1→0 跳变点
3. 以跳变点为中心分配 duration 权重（高斯衰减）
4. 在方向轴上做周期性插值，得到任意角度的权重
5. 输出采样分布 + 热力图

用法::

    PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/balance_recover/sample_distribution.py \
        --input baseline/humanoid21/balance_recover/boundary_fixaw_s42.csv \
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_scan_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载全量扫描 CSV。

    Returns:
        angles: (N_angles,) 排序后的唯一方向
        forces: (N_forces,) 排序后的唯一力
        durations: (N_durations,) 排序后的唯一 duration
        survived: (N_angles, N_forces, N_durations) 存活矩阵 (1=存活, 0=摔倒)
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "angle": float(r["direction_angle"]),
                "force": float(r["force"]),
                "duration": int(r["duration"]),
                "survived": int(r["survived"]),
            })

    angles = sorted(set(r["angle"] for r in rows))
    forces = sorted(set(r["force"] for r in rows))
    durations = sorted(set(r["duration"] for r in rows))

    angle_idx = {a: i for i, a in enumerate(angles)}
    force_idx = {f: i for i, f in enumerate(forces)}
    dur_idx = {d: i for i, d in enumerate(durations)}

    survived = np.zeros((len(angles), len(forces), len(durations)), dtype=np.float32)
    for r in rows:
        survived[angle_idx[r["angle"]], force_idx[r["force"]], dur_idx[r["duration"]]] = r["survived"]

    return np.array(angles), np.array(forces), np.array(durations), survived


def find_transitions(survived: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """找每个 (angle, force) cell 的跳变点。

    Returns:
        transitions: (N_angles, N_forces, 2) 每行 [last_surv_dur, first_fall_dur]
                     如果全存活: [max_dur, max_dur+1]（虚拟边界）
                     如果全摔倒: [0, min_dur]（虚拟边界）
    """
    n_angles, n_forces, n_durs = survived.shape
    transitions = np.zeros((n_angles, n_forces, 2), dtype=np.float32)

    for i in range(n_angles):
        for j in range(n_forces):
            s = survived[i, j, :]
            if s[0] == 0:
                # 全摔倒
                transitions[i, j] = [0, durations[0]]
            elif s[-1] == 1:
                # 全存活
                transitions[i, j] = [durations[-1], durations[-1] + 1]
            else:
                # 找最后一个 1 和第一个 0
                last_surv = durations[np.where(s == 1)[0][-1]]
                first_fall = durations[np.where(s == 0)[0][0]]
                transitions[i, j] = [last_surv, first_fall]

    return transitions


def compute_duration_weights(
    survived: np.ndarray,
    durations: np.ndarray,
    transitions: np.ndarray,
    sigma: float = 3.0,
) -> np.ndarray:
    """计算 duration 权重：以跳变点为中心的高斯衰减。

    跳变点两侧（last_surv 和 first_fall）权重最高，
    远离跳变点权重递减。

    Returns:
        weights: (N_angles, N_forces, N_durations)
    """
    n_angles, n_forces, n_durs = survived.shape
    weights = np.zeros((n_angles, n_forces, n_durs), dtype=np.float32)

    for i in range(n_angles):
        for j in range(n_forces):
            last_surv, first_fall = transitions[i, j]
            boundary_center = (last_surv + first_fall) / 2.0

            for k, d in enumerate(durations):
                dist = abs(d - boundary_center)
                weights[i, j, k] = math.exp(-0.5 * (dist / sigma) ** 2)

    return weights


def interpolate_direction_weights(
    weights: np.ndarray,
    angles: np.ndarray,
    n_interp: int = 360,
) -> Tuple[np.ndarray, np.ndarray]:
    """在方向轴上做周期性插值。

    Args:
        weights: (N_angles, N_forces, N_durations)
        angles: (N_angles,) 原始角度（度）
        n_interp: 插值后的方向数

    Returns:
        interp_angles: (n_interp,) 插值后的角度
        interp_weights: (n_interp, N_forces, N_durations)
    """
    n_angles, n_forces, n_durs = weights.shape
    angles_rad = np.radians(angles)

    # 插值目标角度（均匀分布，覆盖 0~360）
    interp_angles = np.linspace(0, 360, n_interp, endpoint=False)
    interp_angles_rad = np.radians(interp_angles)

    interp_weights = np.zeros((n_interp, n_forces, n_durs), dtype=np.float32)

    for j in range(n_forces):
        for k in range(n_durs):
            # 对每个 (force, duration) 在方向轴上做周期性插值
            # 用 sin/cos 分量做线性插值（避免 0/360 不连续）
            values = weights[:, j, k]

            # 简单方法：找最近的两个原始角度做线性插值
            for idx, target_rad in enumerate(interp_angles_rad):
                # 找到 target 在原始角度中的位置
                # 将所有角度差归一化到 [0, 2π)
                diffs = target_rad - angles_rad
                diffs = (diffs + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-π, π)

                # 找最近的两个（一个在左一个在右）
                sorted_idx = np.argsort(np.abs(diffs))
                i1 = sorted_idx[0]
                i2 = sorted_idx[1]

                d1 = diffs[i1]
                d2 = diffs[i2]

                if abs(d1) < 1e-10:
                    interp_weights[idx, j, k] = values[i1]
                else:
                    # 线性插值
                    total = abs(d1) + abs(d2)
                    if total < 1e-10:
                        interp_weights[idx, j, k] = values[i1]
                    else:
                        w1 = abs(d2) / total
                        w2 = abs(d1) / total
                        interp_weights[idx, j, k] = w1 * values[i1] + w2 * values[i2]

    return interp_angles, interp_weights


def sample_params(
    interp_angles: np.ndarray,
    interp_weights: np.ndarray,
    forces: np.ndarray,
    durations: np.ndarray,
    n_samples: int = 1000,
    direction_jitter: float = 5.0,
    rng: Optional[np.random.RandomState] = None,
) -> List[Tuple[float, float, int]]:
    """根据权重分布采样扰动参数。

    Args:
        interp_angles: (n_interp,) 插值后的角度
        interp_weights: (n_interp, N_forces, N_durations) 权重
        forces: (N_forces,) 力档位
        durations: (N_durations,) duration 值
        n_samples: 采样数
        direction_jitter: 方向抖动范围（度，±）
        rng: 随机数生成器

    Returns:
        samples: [(angle, force, duration), ...]
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # 展平权重为概率分布
    flat_weights = interp_weights.flatten()
    flat_weights = flat_weights / flat_weights.sum()

    # 采样索引
    indices = rng.choice(len(flat_weights), size=n_samples, p=flat_weights)

    n_interp = len(interp_angles)
    n_forces = len(forces)
    n_durs = len(durations)

    samples = []
    for idx in indices:
        a_idx = idx // (n_forces * n_durs)
        remainder = idx % (n_forces * n_durs)
        f_idx = remainder // n_durs
        d_idx = remainder % n_durs

        angle = float(interp_angles[a_idx]) + rng.uniform(-direction_jitter, direction_jitter)
        angle = angle % 360.0
        force = float(forces[f_idx])
        duration = int(durations[d_idx])

        samples.append((angle, force, duration))

    return samples


def plot_heatmaps(
    angles: np.ndarray,
    forces: np.ndarray,
    durations: np.ndarray,
    survived: np.ndarray,
    transitions: np.ndarray,
    weights: np.ndarray,
    interp_angles: np.ndarray,
    interp_weights: np.ndarray,
    output_dir: Path,
) -> None:
    """生成热力图。"""

    # 1. survived 热力图（每个 force 一张，direction × duration）
    for j, force in enumerate(forces):
        fig, ax = plt.subplots(figsize=(12, 6))
        data = survived[:, j, :].T  # (durations, angles)
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                       extent=[angles[0] - 11.25, angles[-1] + 11.25,
                               durations[-1] + 0.5, durations[0] - 0.5],
                       interpolation="nearest")
        ax.set_xlabel("Direction Angle (deg)")
        ax.set_ylabel("Duration (action steps)")
        ax.set_title(f"Survived Map (F={int(force)}N)")
        ax.set_xticks(angles)
        ax.set_xticklabels([f"{a:.0f}" for a in angles], fontsize=7)
        plt.colorbar(im, ax=ax, label="Survived (1=OK, 0=Fall)")
        plt.tight_layout()
        fig.savefig(output_dir / f"heatmap_survived_F{int(force)}.png", dpi=150)
        plt.close(fig)

    # 2. critical_duration 极坐标图
    fig, axes = plt.subplots(1, len(forces), figsize=(6 * len(forces), 6),
                             subplot_kw={"projection": "polar"})
    if len(forces) == 1:
        axes = [axes]
    for j, (force, ax) in enumerate(zip(forces, axes)):
        crit = transitions[:, j, 0]  # last surviving duration
        angles_rad = np.radians(angles)
        # 闭合曲线
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

    # 3. 权重热力图（每个 force 一张，插值后 direction × duration）
    for j, force in enumerate(forces):
        fig, ax = plt.subplots(figsize=(14, 6))
        data = interp_weights[:, j, :].T  # (durations, n_interp)
        im = ax.imshow(data, aspect="auto", cmap="hot",
                       extent=[0, 360, durations[-1] + 0.5, durations[0] - 0.5],
                       interpolation="bilinear")
        ax.set_xlabel("Direction Angle (deg)")
        ax.set_ylabel("Duration (action steps)")
        ax.set_title(f"Sampling Weight (F={int(force)}N)")
        ax.set_xticks(np.arange(0, 361, 45))
        plt.colorbar(im, ax=ax, label="Weight")
        # 标注原始扫描点
        for a in angles:
            ax.axvline(x=a, color="cyan", linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"heatmap_weight_F{int(force)}.png", dpi=150)
        plt.close(fig)

    # 4. 总权重分布（所有 force 叠加，插值后）
    fig, ax = plt.subplots(figsize=(14, 6))
    total_weight = interp_weights.sum(axis=1).T  # (durations, n_interp)
    im = ax.imshow(total_weight, aspect="auto", cmap="hot",
                   extent=[0, 360, durations[-1] + 0.5, durations[0] - 0.5],
                   interpolation="bilinear")
    ax.set_xlabel("Direction Angle (deg)")
    ax.set_ylabel("Duration (action steps)")
    ax.set_title("Total Sampling Weight (all forces)")
    ax.set_xticks(np.arange(0, 361, 45))
    plt.colorbar(im, ax=ax, label="Weight")
    plt.tight_layout()
    fig.savefig(output_dir / "heatmap_weight_total.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate sampling distribution from boundary scan")
    p.add_argument("--input", required=True,
                   help="Path to boundary scan CSV (from probe_boundary.py)")
    p.add_argument("--output-dir", type=str, default=".",
                   help="Output directory for plots and distribution data.")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Gaussian sigma for duration weight decay around boundary.")
    p.add_argument("--n-interp", type=int, default=360,
                   help="Number of interpolated directions.")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="Number of samples to generate.")
    p.add_argument("--direction-jitter", type=float, default=5.0,
                   help="Direction jitter in degrees (±).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    angles, forces, durations, survived = load_scan_data(args.input)
    print(f"Loaded: {len(angles)} angles, {len(forces)} forces, {len(durations)} durations")
    print(f"  angles: {angles}")
    print(f"  forces: {forces}")
    print(f"  durations: {durations[0]}~{durations[-1]}")

    # 2. 找跳变点
    transitions = find_transitions(survived, durations)
    print(f"\n=== Transitions (last_surv, first_fall) ===")
    for j, force in enumerate(forces):
        print(f"  F={int(force):>3d}N:", end="")
        for i, angle in enumerate(angles):
            ls, ff = transitions[i, j]
            print(f" {angle:.0f}:{int(ls)}→{int(ff)}", end="")
        print()

    # 3. 计算 duration 权重
    weights = compute_duration_weights(survived, durations, transitions, sigma=args.sigma)

    # 4. 方向插值
    interp_angles, interp_weights = interpolate_direction_weights(
        weights, angles, n_interp=args.n_interp,
    )
    print(f"\nInterpolated to {args.n_interp} directions")

    # 5. 采样
    rng = np.random.RandomState(args.seed)
    samples = sample_params(
        interp_angles, interp_weights, forces, durations,
        n_samples=args.n_samples, direction_jitter=args.direction_jitter,
        rng=rng,
    )

    # 6. 保存采样分布
    dist_path = output_dir / "sample_distribution.json"
    dist_data = {
        "input_csv": str(Path(args.input).resolve()),
        "sigma": args.sigma,
        "n_interp": args.n_interp,
        "direction_jitter": args.direction_jitter,
        "forces": forces.tolist(),
        "durations": durations.tolist(),
        "interp_angles": interp_angles.tolist(),
        "interp_weights_shape": list(interp_weights.shape),
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

    # 7. 保存权重矩阵（NPZ，供训练时加载）
    weights_npz = output_dir / "sample_weights.npz"
    np.savez_compressed(weights_npz,
                        interp_angles=interp_angles,
                        interp_weights=interp_weights,
                        forces=forces,
                        durations=durations,
                        transitions=transitions,
                        original_angles=angles)
    print(f"Weights NPZ saved to {weights_npz}")

    # 8. 生成热力图
    print(f"\nGenerating heatmaps...")
    plot_heatmaps(angles, forces, durations, survived, transitions,
                  weights, interp_angles, interp_weights, output_dir)
    print(f"Heatmaps saved to {output_dir}/")

    # 9. 打印采样统计
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
