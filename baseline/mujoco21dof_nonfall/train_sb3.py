import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from combatbench.baseline.mujoco21dof_nonfall.env_wrapper import SingleAgentAttackerEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO attacker baseline in the mujoco21dof nonfall environment.")
    parser.add_argument("--run-name", type=str, default="ppo_attacker")
    parser.add_argument("--output-dir", type=str, default="baseline/mujoco21dof_nonfall/runs")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=20000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--opponent", type=str, default="standing", choices=["standing", "random", "active", "scripted", "scripted_active"])
    parser.add_argument("--eval-opponent", type=str, default=None)
    parser.add_argument("--opponent-random-scale", type=float, default=0.1)
    parser.add_argument("--initial-distance", type=float, default=2.0)
    parser.add_argument("--control-frequency", type=int, default=20)
    parser.add_argument("--match-duration", type=float, default=10.0)
    parser.add_argument("--non-fall-pitch-limit-deg", type=float, default=15.0)
    parser.add_argument("--non-fall-roll-limit-deg", type=float, default=10.0)
    parser.add_argument("--damage-scale", type=float, default=100.0)
    parser.add_argument("--disable-non-fall-mode", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def build_run_dir(output_dir: str, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def build_env_kwargs(args: argparse.Namespace, *, eval_mode: bool = False, rank: int = 0) -> Dict[str, Any]:
    opponent = args.opponent if not eval_mode else (args.eval_opponent or args.opponent)
    opponent_seed = args.seed + 1000 + rank if eval_mode else args.seed + rank
    return {
        "render_mode": None,
        "initial_distance": args.initial_distance,
        "control_frequency": args.control_frequency,
        "match_duration": args.match_duration,
        "non_fall_mode": not args.disable_non_fall_mode,
        "non_fall_pitch_limit_deg": args.non_fall_pitch_limit_deg,
        "non_fall_roll_limit_deg": args.non_fall_roll_limit_deg,
        "damage_scale": args.damage_scale,
        "opponent": opponent,
        "opponent_seed": opponent_seed,
        "opponent_random_scale": args.opponent_random_scale,
    }


def make_env(args: argparse.Namespace, *, eval_mode: bool = False, rank: int = 0):
    env_kwargs = build_env_kwargs(args, eval_mode=eval_mode, rank=rank)

    def _factory():
        env = SingleAgentAttackerEnv(**env_kwargs)
        return Monitor(env)

    return _factory


def save_run_config(run_dir: Path, args: argparse.Namespace) -> None:
    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    config["non_fall_mode"] = not args.disable_non_fall_mode
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def build_model(args: argparse.Namespace, train_env: DummyVecEnv, tensorboard_log: str) -> PPO:
    common_kwargs = {
        "env": train_env,
        "device": args.device,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "seed": args.seed,
        "verbose": 1,
        "tensorboard_log": tensorboard_log,
        "policy_kwargs": {"net_arch": [256, 256]},
    }
    if args.resume_from:
        return PPO.load(args.resume_from, env=train_env, device=args.device)
    return PPO("MlpPolicy", **common_kwargs)


def main() -> None:
    args = parse_args()
    run_dir = build_run_dir(args.output_dir, args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    eval_log_dir = run_dir / "eval"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir, args)

    train_env = DummyVecEnv([make_env(args, eval_mode=False, rank=rank) for rank in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(args, eval_mode=True, rank=0)])

    model = build_model(args, train_env, str(tensorboard_dir))

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // max(1, args.n_envs)),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_attacker",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    print(f"Run directory: {run_dir}")
    print(f"Training opponent: {args.opponent}")
    print(f"Evaluation opponent: {args.eval_opponent or args.opponent}")
    print(f"Non-fall mode: {not args.disable_non_fall_mode}")
    print(f"Total timesteps: {args.total_timesteps}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=args.progress_bar,
            reset_num_timesteps=not bool(args.resume_from),
        )
        model.save(str(run_dir / "final_model"))
        summary = {
            "run_dir": str(run_dir),
            "final_model": str(run_dir / "final_model.zip"),
            "best_model_dir": str(best_model_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "tensorboard_dir": str(tensorboard_dir),
            "eval_log_dir": str(eval_log_dir),
            "total_timesteps": int(args.total_timesteps),
            "opponent": args.opponent,
            "eval_opponent": args.eval_opponent or args.opponent,
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("Training finished")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    main()
