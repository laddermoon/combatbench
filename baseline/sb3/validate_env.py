from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

from .rewards import ATTACKER_REWARD_CONFIG, FIGHT_REWARD_CONFIG, STANDING_REWARD_CONFIG
from .selfplay_env import make_attacker_standing_env, make_symmetric_selfplay_env



def configure_runtime() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")



def smoke_test_phase(phase_name: str, reward_config) -> None:
    if phase_name == "fight_attacker":
        opponent_model_path = Path(__file__).resolve().parent / "runs" / "stand_env_residual_smoke_v3" / "best_model" / "best_model.zip"
        env = make_attacker_standing_env(
            opponent_model_path=str(opponent_model_path),
            render_mode=None,
            match_duration=15.0,
            control_frequency=20,
            initial_distance=2.0,
            reward_config=reward_config,
            opponent_device="cpu",
        )
    else:
        env = make_symmetric_selfplay_env(
            render_mode=None,
            match_duration=10.0 if phase_name == "stand" else 15.0,
            control_frequency=20,
            reward_config=reward_config,
        )
    obs, info = env.reset()
    print(f"[{phase_name}] reset obs shape: {obs.shape}")
    print(f"[{phase_name}] info keys: {sorted(info.keys())}")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"[{phase_name}] step={step + 1} reward={reward:.3f} terminated={terminated} truncated={truncated} winner={info.get('winner')}"
        )
        if terminated or truncated:
            break

    check_env(env, warn=True, skip_render_check=True)
    env.close()



def main() -> None:
    configure_runtime()
    smoke_test_phase("stand", STANDING_REWARD_CONFIG)
    smoke_test_phase("fight", FIGHT_REWARD_CONFIG)
    smoke_test_phase("fight_attacker", ATTACKER_REWARD_CONFIG)
    print("Environment validation completed.")


if __name__ == "__main__":
    main()
