from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from .rewards import FIGHT_REWARD_CONFIG, STANDING_REWARD_CONFIG, resolve_reward_config
from .selfplay_env import make_symmetric_selfplay_env


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the SB3 shared-policy combat baseline")
    parser.add_argument("--phase", choices=["stand", "fight"], default="stand")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--init-model", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--match-duration", type=float, default=None)
    parser.add_argument("--control-frequency", type=int, default=20)
    parser.add_argument("--initial-distance", type=float, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser



def configure_runtime() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")



def default_match_duration(phase: str) -> float:
    return 10.0 if phase == "stand" else 15.0



def default_initial_distance(phase: str) -> float:
    return 2.0 if phase == "stand" else 1.0



def make_run_directory(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    run_name = args.run_name or f"{args.phase}_ppo"
    return Path(__file__).resolve().parent / "runs" / run_name



def save_run_config(run_dir: Path, args: argparse.Namespace, reward_config) -> None:
    config = {
        "phase": args.phase,
        "timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "device": args.device,
        "match_duration": args.match_duration,
        "control_frequency": args.control_frequency,
        "initial_distance": args.initial_distance,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "seed": args.seed,
        "target_height": reward_config.target_height,
        "reward_config": reward_config.to_dict(),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))



def build_env(args: argparse.Namespace, phase: str):
    reward_config = resolve_reward_config(phase)
    env = make_symmetric_selfplay_env(
        render_mode=None,
        match_duration=args.match_duration,
        control_frequency=args.control_frequency,
        initial_distance=args.initial_distance,
        reward_config=reward_config,
    )
    return Monitor(env), reward_config



def build_model(args: argparse.Namespace, env, run_dir: Path):
    policy_kwargs = {
        "net_arch": [256, 256, 128],
        "activation_fn": torch.nn.Tanh,
    }
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.init_model:
        model = PPO.load(args.init_model, env=env, device=device)
        model.verbose = 1
        return model, device

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
        device=device,
        seed=args.seed,
    )
    return model, device



def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.match_duration = args.match_duration or default_match_duration(args.phase)
    if args.initial_distance is None:
        args.initial_distance = default_initial_distance(args.phase)
    configure_runtime()

    run_dir = make_run_directory(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    env, reward_config = build_env(args, args.phase)
    eval_env, _ = build_env(args, args.phase)
    save_run_config(run_dir, args, reward_config)

    model, device = build_model(args, env, run_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix=f"{args.phase}_policy",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )

    print("=" * 72)
    print(f"SB3 baseline training phase: {args.phase}")
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Match duration: {args.match_duration}s")
    print(f"Control frequency: {args.control_frequency}Hz")
    print(f"Initial distance: {args.initial_distance}m")
    print("=" * 72)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_model_path = run_dir / "model_final"
    model.save(str(final_model_path))
    print(f"Saved final model to: {final_model_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
