"""Validate BalanceValueRewarder against the stable StandingCombatPolicy.

Expected behaviors (if rewarder is well-calibrated):
  1. Per-step reward is close to 1.0 almost all the time.
  2. Mean support_lateral_signed_distance across the episode sits close to
     the center of the safe white zone (i.e. near (+FRONT - BACK)/2).
"""

from __future__ import annotations

from pathlib import Path
import sys

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

import numpy as np

from baseline.humanoid21.rewards import (
    BALANCE_SAFE_BACK_MARGIN,
    BALANCE_SAFE_FRONT_MARGIN,
    BalanceValueRewarder,
)
from envs.framework import BaseFrameRecorder, EnvRuntime
from envs.humanoid21.observer_plugins import (
    Humanoid21BalanceAnalysisObserver,
    Humanoid21Observer,
)
from envs.humanoid21.simulator import MujocoCombatSimulator
from policy.load_util import load_policy_from_dir


NUM_EPISODES = 3
MAX_STEPS_PER_EPISODE = 200
POLICY_DIR = COMBATBENCH_DIR / "policy" / "standing"


def run_one_episode(seed: int, output_root: Path) -> dict:
    policy_a = load_policy_from_dir(POLICY_DIR)
    policy_b = load_policy_from_dir(POLICY_DIR)
    policy_a.reset()
    policy_b.reset()

    rewarder = BalanceValueRewarder(agent_id="robot_a")
    balance_analyzer = Humanoid21BalanceAnalysisObserver("robot_a")
    frame_recorder = BaseFrameRecorder(
        output_dir=output_root / f"seed_{int(seed):04d}",
        stride=10,
    )
    runtime = EnvRuntime(
        simulator=MujocoCombatSimulator(initial_distance=2.0),
        observer_plugins={
            "robot_a_obs": Humanoid21Observer("robot_a"),
            "robot_b_obs": Humanoid21Observer("robot_b"),
            "robot_a_reward": rewarder,
            "robot_a_balance": balance_analyzer,
        },
        plugins=[],
        recorders=[frame_recorder],
        phy_steps_per_action=1,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    runtime.reset(seed=int(seed))

    rewards: list[float] = []
    lateral_signed_distances: list[float] = []
    axis_offsets: list[float] = []
    ground_support_ok: list[bool] = []

    while runtime.is_episode_active:
        obs_a = np.asarray(runtime.get_observer_output("robot_a_obs"), dtype=np.float32)
        obs_b = np.asarray(runtime.get_observer_output("robot_b_obs"), dtype=np.float32)
        action_a = policy_a.act(obs_a)
        action_b = policy_b.act(obs_b)
        runtime.step(action_a, action_b)

        rewards.append(float(rewarder.get_output()))
        balance = balance_analyzer.get_output()
        if isinstance(balance, dict):
            signed = float(balance.get("support_lateral_signed_distance", float("nan")))
            axis_projection = float(balance.get("support_axis_projection_coordinate", float("nan")))
            span = float(balance.get("support_span", float("nan")))
            lateral_signed_distances.append(signed)
            if np.isfinite(axis_projection) and np.isfinite(span) and span > 0:
                axis_offsets.append(axis_projection - 0.5 * span)
            ground_support_ok.append(bool(balance.get("ground_support_frame_defined", False)))

        terminated, truncated = runtime.get_termination_flags()
        if terminated or truncated or not runtime.is_episode_active:
            break

    if hasattr(runtime, "close"):
        runtime.close()

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    lateral_arr = np.asarray(lateral_signed_distances, dtype=np.float64)
    axis_arr = np.asarray(axis_offsets, dtype=np.float64)
    support_ok_arr = np.asarray(ground_support_ok, dtype=bool)

    return {
        "seed": int(seed),
        "steps": int(len(rewards_arr)),
        "reward_sum": float(np.sum(rewards_arr)) if rewards_arr.size else 0.0,
        "reward_mean": float(np.mean(rewards_arr)) if rewards_arr.size else 0.0,
        "reward_min": float(np.min(rewards_arr)) if rewards_arr.size else 0.0,
        "reward_max": float(np.max(rewards_arr)) if rewards_arr.size else 0.0,
        "lateral_signed_mean": float(np.nanmean(lateral_arr)) if lateral_arr.size else float("nan"),
        "lateral_signed_std": float(np.nanstd(lateral_arr)) if lateral_arr.size else float("nan"),
        "lateral_signed_min": float(np.nanmin(lateral_arr)) if lateral_arr.size else float("nan"),
        "lateral_signed_max": float(np.nanmax(lateral_arr)) if lateral_arr.size else float("nan"),
        "axis_offset_mean": float(np.nanmean(axis_arr)) if axis_arr.size else float("nan"),
        "axis_offset_std": float(np.nanstd(axis_arr)) if axis_arr.size else float("nan"),
        "ground_support_ratio": float(np.mean(support_ok_arr)) if support_ok_arr.size else 0.0,
    }


def main() -> None:
    output_root = Path.cwd() / "debug_images_standing"
    output_root.mkdir(parents=True, exist_ok=True)
    safe_back_lower_bound = -BALANCE_SAFE_BACK_MARGIN
    white_zone_center = 0.5 * (BALANCE_SAFE_FRONT_MARGIN + safe_back_lower_bound)
    print(f"Policy dir         : {POLICY_DIR}")
    print(f"Debug images root  : {output_root}")
    print(
        f"Safe zone          : lateral signed in [{safe_back_lower_bound:+.3f}, {BALANCE_SAFE_FRONT_MARGIN:+.3f}] m, "
        f"center = {white_zone_center:+.4f} m"
    )
    print()

    results = []
    for seed in range(NUM_EPISODES):
        print(f"[seed={seed}] running...")
        result = run_one_episode(seed, output_root)
        results.append(result)

    print("\n=== Per-episode summary ===")
    for r in results:
        dist_to_center = r["lateral_signed_mean"] - white_zone_center
        print(
            f"seed={r['seed']:>3d} "
            f"steps={r['steps']:>4d} "
            f"reward: sum={r['reward_sum']:+8.3f} mean={r['reward_mean']:+.4f} "
            f"[{r['reward_min']:+.4f}, {r['reward_max']:+.4f}]   "
            f"lat_signed: mean={r['lateral_signed_mean']:+.4f} "
            f"std={r['lateral_signed_std']:.4f} "
            f"[{r['lateral_signed_min']:+.4f}, {r['lateral_signed_max']:+.4f}]   "
            f"axis_offset_mean={r['axis_offset_mean']:+.4f} "
            f"ground_ok={r['ground_support_ratio']*100:5.1f}%   "
            f"(mean-to-zone-center: {dist_to_center:+.4f} m)"
        )

    aggregate_reward_mean = float(np.mean([r["reward_mean"] for r in results]))
    aggregate_lateral_mean = float(np.nanmean([r["lateral_signed_mean"] for r in results]))
    print("\n=== Aggregate ===")
    print(f"mean per-step reward across episodes : {aggregate_reward_mean:+.4f}")
    print(f"mean lateral_signed_distance          : {aggregate_lateral_mean:+.4f} m")
    print(f"white-zone center                     : {white_zone_center:+.4f} m")
    print(f"offset from white-zone center         : {aggregate_lateral_mean - white_zone_center:+.4f} m")


if __name__ == "__main__":
    main()
