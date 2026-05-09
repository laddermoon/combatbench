"""Random-actor rollout validation script.

Runs a few short rollouts with a randomly initialized Actor, attaches
StandingPostureDeltaRewarder with debug image saving enabled, and writes
debug images into the current working directory so the full framework
(env runtime + observer + rewarder + image saving) can be eyeballed.
"""

from __future__ import annotations

from pathlib import Path
import sys

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

import numpy as np
import torch

from baseline.humanoid21.base import Actor, RolloutCollector
from baseline.humanoid21.rewards import StandingPostureDeltaRewarder, StandingPostureRewarder, BalanceValueRewarder
from envs.framework import BaseFrameRecorder, EnvRuntime
from envs.humanoid21.observer_plugins import Humanoid21Observer
from envs.humanoid21.simulator import MujocoCombatSimulator


NUM_EPISODES = 3
MAX_STEPS_PER_EPISODE = 120
DEBUG_STRIDE = 1
HIDDEN_DIM = 64


def build_runtime_builder(output_root: Path):
    rewarders: dict[int, BalanceValueRewarder] = {}

    def runtime_builder(seed: int) -> EnvRuntime:
        rewarder = BalanceValueRewarder(agent_id="robot_a")
        rewarders[int(seed)] = rewarder
        frame_recorder = BaseFrameRecorder(
            output_dir=output_root / f"seed_{int(seed):04d}",
            stride=DEBUG_STRIDE,
        )
        runtime = EnvRuntime(
            simulator=MujocoCombatSimulator(initial_distance=2.0),
            observer_plugins={
                "robot_a_obs": Humanoid21Observer("robot_a"),
                "robot_a_reward": rewarder,
            },
            plugins=[],
            recorders=[frame_recorder],
            phy_steps_per_action=1,
            max_steps=MAX_STEPS_PER_EPISODE,
        )
        return runtime

    return runtime_builder, rewarders


def main() -> None:
    output_root = Path.cwd() / "debug_images"
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Debug images will be written under: {output_root}")

    torch.manual_seed(0)
    np.random.seed(0)

    runtime_builder, rewarders = build_runtime_builder(output_root)

    actor = Actor(
        obs_dim=Humanoid21Observer.OBS_DIM,
        action_dim=Humanoid21Observer.ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(torch.device("cpu"))
    actor.eval()

    collector = RolloutCollector(runtime_builder=runtime_builder, actor=actor, max_workers=1)
    seeds = list(range(NUM_EPISODES))
    try:
        episodes = collector.collect_episodes(seeds)
    finally:
        collector.close()

    print("\n=== Rollout summary ===")
    for episode in episodes:
        seed = int(episode["seed"])
        rewards = np.asarray(episode["rewards"], dtype=np.float32)
        print(
            f"seed={seed:>3d} "
            f"controlled={episode['controlled_agent']:<7s} "
            f"steps={int(episode['steps']):>3d} "
            f"episode_reward={float(episode['episode_reward']):+.4f} "
            f"reward_mean={float(np.mean(rewards)) if rewards.size else 0.0:+.4f} "
            f"reward_min={float(np.min(rewards)) if rewards.size else 0.0:+.4f} "
            f"reward_max={float(np.max(rewards)) if rewards.size else 0.0:+.4f} "
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
