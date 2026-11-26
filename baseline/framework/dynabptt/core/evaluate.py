"""Deployment evaluation: natural-init full episodes with the deterministic
policy from a core training checkpoint.

Unlike branch evals (which start from StartSet seeds), this measures the
actual deployment question: from the natural standing initial state, how
long does the robot balance?

Usage:
    python3 -m baseline.framework.dynabptt.core.evaluate \
        --ckpt runs/core_v6_frontier/ckpt_u00100.pt [--episodes 16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.dynabptt.corestate import flatten_core_state
from baseline.framework.dynabptt.core.train import phi_np
from baseline.framework.dynabptt.hybrid_ppo.standup import (
    ACTION_DIM, SELF_OBS_DIM)
from baseline.framework.dynabptt.smbptt.task import make_balance_env_bp
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy)

ZERO21 = np.zeros(21, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--phy-steps", type=int, default=10)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    actor = TruncatedNormalPolicy(
        obs_dim=SELF_OBS_DIM, action_dim=ACTION_DIM, hidden_dim=256,
        device=str(dev))
    actor.load_state_dict(ck["actor"])
    actor.eval()

    env = make_balance_env_bp(
        max_steps=args.max_steps,
        phy_steps_per_action=args.phy_steps).build()
    lens, phis_end = [], []
    for ep in range(args.episodes):
        seed = 7000 + ep
        for pl in env.plugins:
            pl.set_episode_seed(seed)
        env.reset(seed=seed, base_seed=seed)
        o = np.asarray(env.get_observation()[0], dtype=np.float32)
        L = args.max_steps
        for t in range(args.max_steps):
            with torch.no_grad():
                a = actor.deterministic_action(
                    torch.as_tensor(o[None], dtype=torch.float32,
                                    device=dev))[0].cpu().numpy()
            env.step(np.clip(a, -1, 1).astype(np.float32), ZERO21)
            o = np.asarray(env.get_observation()[0], dtype=np.float32)
            if env.is_episode_over():
                L = t + 1
                break
        cs = flatten_core_state(env.simulator.get_core_state(), "robot_a")
        lens.append(L)
        phis_end.append(float(phi_np(cs)))
        print(f"ep{ep}: len={L} ({L * 0.02:.1f}s) end_phi={phis_end[-1]:.3f}")
    lens = np.asarray(lens)
    print(f"\nmean len {lens.mean():.1f} ({lens.mean() * 0.02:.1f}s) "
          f"median {np.median(lens):.0f} full-episode rate "
          f"{(lens >= args.max_steps).mean():.2f}")


if __name__ == "__main__":
    main()
