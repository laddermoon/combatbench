#!/usr/bin/env python3
"""EXP-CURVED-MOTION runner — see EXP_CURVED_MOTION.md.

Rolls the converged skill policy through scripted move/turn command
schedules (temporal interleave AND simultaneous composition), records
per-step root xy/yaw/cmd + broadcast video, and summarizes whether the
trained ±1-saturated command set composes into curved locomotion.

Usage:
    PYTHONPATH=/data1/mono/things/combatbench MUJOCO_GL=egl \
      python3 -B baseline/humanoid21/hierarchy/exp_curved_motion.py \
      [--export baseline/runs/skill_train_s2b_u8000_prog/policy_exports/u09860] \
      [--out baseline/humanoid21/hierarchy/videos/<ts>] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from envs.framework.policy import PolicyBlueprint  # noqa: E402
from baseline.experiments_ppo.exp_skill_train_mbs import (  # noqa: E402
    SkillTrainMBS)
from baseline.humanoid21.hierarchy.command_simulator import (  # noqa: E402
    BODY_DIM, VX, VY, WZ, V_MAX, W_MAX)

DEFAULT_EXPORT = (REPO / "baseline/runs/skill_train_s2b_u8000_prog/"
                  "policy_exports/u09860")
T = 150  # action steps (7.5 s at 20 Hz)


def _alt(n_move: int, n_turn: int, n_pairs: int, w: float) -> list:
    rows = []
    for _ in range(n_pairs):
        rows.append([n_move, V_MAX, 0.0, 0.0])
        rows.append([n_turn, 0.0, 0.0, w])
    return rows


CONDITIONS = {
    # controls
    "ctrl_fwd":   [[T, V_MAX, 0.0, 0.0]],
    "ctrl_turn":  [[T, 0.0, 0.0, W_MAX]],
    # A: temporal interleave (saturated trained commands only)
    "alt_20_10":  _alt(20, 10, 5, W_MAX),
    "alt_10_10":  _alt(10, 10, 7, W_MAX),
    "alt_5_5":    _alt(5, 5, 15, W_MAX),
    "alt_3_3":    _alt(3, 3, 25, W_MAX),
    "alt_5_5r":   _alt(5, 5, 15, -W_MAX),
    # B: simultaneous (untrained combos/magnitudes)
    "sim_w100":   [[T, V_MAX, 0.0, W_MAX]],
    "sim_w050":   [[T, V_MAX, 0.0, 0.5 * W_MAX]],
    "sim_w025":   [[T, V_MAX, 0.0, 0.25 * W_MAX]],
    "sim_diag":   [[T, V_MAX, 0.5 * V_MAX, 0.0]],
}


def yaw_of(q) -> float:
    from scipy.spatial.transform import Rotation as _R
    return float(_R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2])


def run_condition(name: str, rows: list, policy, pol_cls: str,
                  seed: int, out_dir: Path) -> dict:
    import imageio.v2 as imageio
    env_pb = SkillTrainMBS._fam_env_pb("move").materialize(max_steps=T)
    rt = env_pb.build()
    rt.reset(seed=seed, options={
        "initial_distance": 2.0,
        "move_schedule": {"robot_a": rows, "robot_b": [[T, 0, 0, 0]]},
        "warmup_policy_cls": pol_cls,
    })
    policy.reset(seed)
    sim = rt.simulator
    xs, ys, yaws, cmds, frames, phis = [], [], [], [], [], []
    fell_step = None
    while rt.is_episode_active:
        obs_a, obs_b = rt.get_observation()
        act_a, _ = policy.act(obs_a, want_extra=False)
        act_b, _ = policy.act(obs_b, want_extra=False)
        cmds.append(np.asarray(obs_a[BODY_DIM:], dtype=np.float64).copy())
        rt.step(act_a, act_b)
        core = sim.get_core_state()["robot_a"]
        xs.append(float(core["root_pos"][0]))
        ys.append(float(core["root_pos"][1]))
        yaws.append(yaw_of(np.asarray(core["root_rot"], dtype=np.float64)))
        phis.append(float(rt.get_observer_output(
            "standing_balance_a")["potential"]))
        img = sim.get_broadcastview_image()
        if img is not None:
            frames.append(img)
        term = rt.get_agent_termination().get("robot_a")
        if term and term != "timeout":
            fell_step = int(rt.ctx.episode_step)
            break
    if frames:
        imageio.mimsave(out_dir / f"curved_{name}.mp4", frames, fps=20,
                        macro_block_size=16)
    xs, ys, yaws = map(np.asarray, (xs, ys, yaws))
    cmds = np.asarray(cmds)
    n = len(xs)
    # realized ego velocities (20 Hz control)
    dp = np.diff(np.stack([xs, ys], 1), axis=0) / 0.05
    c, s = np.cos(-yaws[1:]), np.sin(-yaws[1:])
    vx_e = c * dp[:, 0] - s * dp[:, 1]
    vy_e = s * dp[:, 0] + c * dp[:, 1]
    w_e = np.diff(np.unwrap(yaws)) / 0.05
    path_len = float(np.sum(np.linalg.norm(dp * 0.05, axis=1))) if n > 1 else 0.0
    disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) if n > 1 else 0.0
    dyaw = float(np.unwrap(yaws)[-1] - np.unwrap(yaws)[0]) if n > 1 else 0.0
    # fall events: sustained phi<0.3 runs >=10 steps (same criterion as
    # eval's fell flag); count each distinct run = one fall-and-recover
    phi = np.asarray(phis)
    low = (phi < 0.3).astype(np.int8)
    fall_events = 0
    i0 = None
    for t in range(n + 1):
        v = low[t] if t < n else 0
        if v and i0 is None:
            i0 = t
        elif not v and i0 is not None:
            if t - i0 >= 10:
                fall_events += 1
            i0 = None
    lo = min(20, n - 1)
    res = dict(
        name=name, steps=n, fell=fell_step,
        n_falls=fall_events,
        phi_min=round(float(phi.min()), 3) if n else 0.0,
        phi_mean=round(float(phi.mean()), 3) if n else 0.0,
        path_len=round(path_len, 3), disp=round(disp, 3),
        dyaw_deg=round(float(np.degrees(dyaw)), 1),
        vx_real=round(float(np.mean(vx_e[lo:])), 3) if n > lo else 0.0,
        vy_real=round(float(np.mean(vy_e[lo:])), 3) if n > lo else 0.0,
        w_real=round(float(np.mean(w_e[lo:])), 3) if n > lo else 0.0,
        vx_cmd=round(float(cmds[lo:, VX].mean() * V_MAX), 3) if n > lo else 0.0,
        vy_cmd=round(float(cmds[lo:, VY].mean() * V_MAX), 3) if n > lo else 0.0,
        w_cmd=round(float(cmds[lo:, WZ].mean() * W_MAX), 3) if n > lo else 0.0,
    )
    np.savez(out_dir / f"curved_{name}.npz",
             x=xs, y=ys, yaw=yaws, cmd=cmds, phi=phi,
             fell=fell_step or -1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conds", type=str, default=None,
                    help="comma list; default all")
    args = ap.parse_args()
    export = args.export
    manifest = json.loads((export / "MANIFEST.json").read_text())
    pol_cls = f"file:{export / 'policy.py'}:{manifest['exported_class']}"
    tmp = Path(tempfile.mkdtemp()) / "pol.yaml"
    tmp.write_text(f"version: 1\ncls: {pol_cls}\nconfig: {{}}\n")
    policy = PolicyBlueprint.load(tmp).build()

    out_dir = args.out or (Path(__file__).parent / "videos"
                           / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = (args.conds.split(",") if args.conds else list(CONDITIONS))
    print(f"[curved] export={export.name} out={out_dir} seed={args.seed}",
          flush=True)

    results = []
    for name in conds:
        r = run_condition(name, CONDITIONS[name], policy, pol_cls,
                          args.seed, out_dir)
        results.append(r)
        print(f"  {name:<10s} steps={r['steps']:>3} falls={r['n_falls']} "
              f"phi_min={r['phi_min']:.2f} "
              f"disp={r['disp']:.2f} dyaw={r['dyaw_deg']:+.0f}deg "
              f"v=({r['vx_real']:+.2f},{r['vy_real']:+.2f}) vs "
              f"cmd({r['vx_cmd']:+.2f},{r['vy_cmd']:+.2f}) "
              f"w={r['w_real']:+.2f} vs {r['w_cmd']:+.2f}", flush=True)

    # overview figure: XY paths
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 4
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.2 * nrow))
    for ax, r in zip(axes.ravel(), results):
        z = np.load(out_dir / f"curved_{r['name']}.npz")
        x, y, yaw = z["x"], z["y"], z["yaw"]
        ax.plot(x - x[0], y - y[0], "b-", lw=1.4)
        ax.scatter([0], [0], c="black", s=30, zorder=5)
        for k in range(0, len(x), 15):
            ax.annotate("", xy=(x[k] - x[0], y[k] - y[0]),
                        xytext=(x[k] - x[0] - 0.12 * np.cos(yaw[k]),
                                y[k] - y[0] - 0.12 * np.sin(yaw[k])),
                        arrowprops=dict(arrowstyle="->", color="orange",
                                        lw=1.2))
        if r["fell"]:
            ax.scatter([x[-1] - x[0]], [y[-1] - y[0]], c="red", s=60,
                       marker="x", zorder=6)
        ax.set_title(f"{r['name']}  disp={r['disp']:.1f} "
                     f"dyaw={r['dyaw_deg']:+.0f}", fontsize=8)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "curved_paths.png", dpi=110)
    (out_dir / "curved_results.json").write_text(json.dumps(
        {"export": str(export), "seed": args.seed, "results": results},
        indent=2))
    print(f"[curved] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
