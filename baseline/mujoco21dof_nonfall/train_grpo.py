import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from combatbench.baseline.mujoco21dof_nonfall.env_wrapper import SingleAgentAttackerEnv
from combatbench.baseline.mujoco21dof_nonfall.grpo import (
    GRPOActor,
    GRPOModelConfig,
    GRPORolloutCollector,
    evaluate_grpo_actor,
    load_grpo_checkpoint,
    optimize_grpo,
    resolve_device,
    save_grpo_checkpoint,
)
from combatbench.baseline.mujoco21dof_nonfall.train_sb3 import (
    build_env_kwargs,
    build_run_dir,
    build_train_vec_env,
    save_run_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRPO attacker baseline in the mujoco21dof nonfall environment.")
    parser.add_argument("--run-name", type=str, default="grpo_attacker")
    parser.add_argument("--output-dir", type=str, default="baseline/mujoco21dof_nonfall/runs")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train-vec-env", type=str, default="auto", choices=["auto", "dummy", "subproc"])
    parser.add_argument("--subproc-start-method", type=str, default="spawn", choices=["spawn", "forkserver", "fork"])
    parser.add_argument("--checkpoint-freq", type=int, default=20000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--opponent", type=str, default="standing", choices=["standing", "random", "active", "scripted", "scripted_active"])
    parser.add_argument("--eval-opponent", type=str, default=None)
    parser.add_argument("--opponent-random-scale", type=float, default=0.1)
    parser.add_argument("--curriculum-stage", type=str, default="attack", choices=["attack", "distance_stage1"])
    parser.add_argument("--initial-distance", type=float, default=2.0)
    parser.add_argument("--control-frequency", type=int, default=20)
    parser.add_argument("--match-duration", type=float, default=10.0)
    parser.add_argument("--non-fall-pitch-limit-deg", type=float, default=5.0)
    parser.add_argument("--non-fall-roll-limit-deg", type=float, default=5.0)
    parser.add_argument("--damage-scale", type=float, default=100.0)
    parser.add_argument("--distance-stage-target-distance", type=float, default=0.55)
    parser.add_argument("--distance-stage-reward-mode", type=str, default="step_delta", choices=["step_delta", "episode_uniform"])
    parser.add_argument("--distance-stage-reward-power", type=float, default=2.0)
    parser.add_argument("--distance-stage-clamp-penalty-scale", type=float, default=0.002)
    parser.add_argument("--distance-stage-prioritize-no-clamp", action="store_true")
    parser.add_argument("--disable-non-fall-mode", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.episodes_per_update <= 0:
        raise ValueError("--episodes-per-update must be positive")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if args.episodes_per_update < args.group_size:
        raise ValueError("--episodes-per-update must be greater than or equal to --group-size")
    if args.episodes_per_update % args.group_size != 0:
        raise ValueError("--episodes-per-update must be divisible by --group-size")
    if args.minibatch_size <= 0:
        raise ValueError("--minibatch-size must be positive")
    if args.update_epochs <= 0:
        raise ValueError("--update-epochs must be positive")
    if len(args.hidden_sizes) <= 0:
        raise ValueError("--hidden-sizes must contain at least one layer size")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be positive")
    if args.checkpoint_freq <= 0:
        raise ValueError("--checkpoint-freq must be positive")


def build_eval_env(args: argparse.Namespace) -> SingleAgentAttackerEnv:
    env_kwargs = build_env_kwargs(args, eval_mode=True, rank=0)
    return SingleAgentAttackerEnv(**env_kwargs)


def choose_target_episodes(args: argparse.Namespace, completed_timesteps: int) -> int:
    remaining_timesteps = max(0, int(args.total_timesteps) - int(completed_timesteps))
    estimated_episode_length = max(1, int(args.match_duration * args.control_frequency))
    estimated_episodes = remaining_timesteps // estimated_episode_length
    if estimated_episodes <= 0:
        return int(args.group_size)
    grouped_episodes = (estimated_episodes // int(args.group_size)) * int(args.group_size)
    if grouped_episodes <= 0:
        return int(args.group_size)
    return min(int(args.episodes_per_update), grouped_episodes)


def make_metadata(args: argparse.Namespace, run_dir: Path, total_timesteps: int, update_index: int, best_eval_reward: float) -> Dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "algorithm": "grpo",
        "total_timesteps": int(total_timesteps),
        "update_index": int(update_index),
        "best_eval_reward": float(best_eval_reward),
        "args": vars(args).copy(),
        "saved_at": datetime.now().isoformat(),
    }


def save_eval_history(eval_path: Path, eval_history: List[Dict[str, Any]]) -> None:
    if not eval_history:
        return
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        eval_path,
        timesteps=np.asarray([item["timesteps"] for item in eval_history], dtype=np.int64),
        results=np.asarray([item["episode_returns"] for item in eval_history], dtype=np.float32),
        episode_lengths=np.asarray([item["episode_lengths"] for item in eval_history], dtype=np.float32),
        clamp_counts=np.asarray([item["episode_clamp_counts"] for item in eval_history], dtype=np.float32),
        final_distances=np.asarray([item["final_distances"] for item in eval_history], dtype=np.float32),
    )


def maybe_eval_and_checkpoint(
    *,
    actor: GRPOActor,
    optimizer: torch.optim.Optimizer,
    eval_env: SingleAgentAttackerEnv,
    device: torch.device,
    args: argparse.Namespace,
    run_dir: Path,
    best_model_dir: Path,
    checkpoint_dir: Path,
    eval_log_dir: Path,
    eval_history: List[Dict[str, Any]],
    total_timesteps: int,
    update_index: int,
    best_eval_reward: float,
    force_eval: bool,
    next_eval_at: int,
    next_checkpoint_at: int,
) -> tuple[float, int, int, bool]:
    if total_timesteps >= next_checkpoint_at:
        checkpoint_path = checkpoint_dir / f"grpo_attacker_{total_timesteps}.pt"
        save_grpo_checkpoint(
            checkpoint_path,
            actor,
            optimizer,
            make_metadata(args, run_dir, total_timesteps, update_index, best_eval_reward),
        )
        while total_timesteps >= next_checkpoint_at:
            next_checkpoint_at += int(args.checkpoint_freq)

    should_eval = force_eval or total_timesteps >= next_eval_at
    if not should_eval:
        return best_eval_reward, next_eval_at, next_checkpoint_at, False

    actor.eval()
    eval_metrics = evaluate_grpo_actor(
        actor,
        eval_env,
        device=device,
        episodes=args.eval_episodes,
        deterministic=True,
        seed=args.seed,
    )
    actor.train()

    eval_history.append(
        {
            "timesteps": int(total_timesteps),
            "episode_returns": eval_metrics["episode_returns"].tolist(),
            "episode_lengths": eval_metrics["episode_lengths"].tolist(),
            "episode_clamp_counts": eval_metrics["episode_clamp_counts"].tolist(),
            "final_distances": eval_metrics["final_distances"].tolist(),
        }
    )
    save_eval_history(eval_log_dir / "evaluations.npz", eval_history)

    summary_payload = {
        "timesteps": int(total_timesteps),
        "mean_reward": float(eval_metrics["mean_reward"]),
        "std_reward": float(eval_metrics["std_reward"]),
        "mean_episode_length": float(eval_metrics["mean_episode_length"]),
        "mean_episode_clamp_count": float(eval_metrics["mean_episode_clamp_count"]),
        "mean_final_distance": float(eval_metrics["mean_final_distance"]),
        "episode_returns": eval_metrics["episode_returns"].tolist(),
        "episode_lengths": eval_metrics["episode_lengths"].tolist(),
        "episode_clamp_counts": eval_metrics["episode_clamp_counts"].tolist(),
        "final_distances": eval_metrics["final_distances"].tolist(),
    }
    with open(eval_log_dir / f"eval_{total_timesteps}.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)

    print(f"Eval num_timesteps={total_timesteps}, episode_reward={eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}")
    print(f"Episode length: {eval_metrics['mean_episode_length']:.2f}")
    print(f"Mean episode clamp count: {eval_metrics['mean_episode_clamp_count']:.2f}")
    print(f"Mean final distance: {eval_metrics['mean_final_distance']:.3f}")

    if eval_metrics["mean_reward"] > best_eval_reward:
        best_eval_reward = float(eval_metrics["mean_reward"])
        save_grpo_checkpoint(
            best_model_dir / "best_model.pt",
            actor,
            optimizer,
            make_metadata(args, run_dir, total_timesteps, update_index, best_eval_reward),
        )
        print("New best mean reward!")

    while total_timesteps >= next_eval_at:
        next_eval_at += int(args.eval_freq)
    return best_eval_reward, next_eval_at, next_checkpoint_at, True


def maybe_resume_actor(
    args: argparse.Namespace,
    actor: GRPOActor,
    optimizer: torch.optim.Optimizer,
) -> None:
    if not args.resume_from:
        return
    resumed_actor, checkpoint = load_grpo_checkpoint(args.resume_from, device=resolve_device(args.device))
    actor.load_state_dict(resumed_actor.state_dict())
    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)


def main() -> None:
    args = parse_args()
    validate_args(args)
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

    device = resolve_device(args.device)
    train_env = build_train_vec_env(args)
    eval_env = build_eval_env(args)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    obs_dim = int(train_env.observation_space.shape[0])
    action_dim = int(train_env.action_space.shape[0])
    actor = GRPOActor(
        GRPOModelConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(int(size) for size in args.hidden_sizes),
            log_std_init=float(args.log_std_init),
        )
    ).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    maybe_resume_actor(args, actor, optimizer)
    actor.train()

    collector = GRPORolloutCollector(train_env)
    total_timesteps = 0
    update_index = 0
    best_eval_reward = float("-inf")
    eval_history: List[Dict[str, Any]] = []
    next_eval_at = int(args.eval_freq)
    next_checkpoint_at = int(args.checkpoint_freq)
    last_eval_timestep = -1

    print(f"Run directory: {run_dir}")
    print("Algorithm: grpo")
    print(f"Curriculum stage: {args.curriculum_stage}")
    print(f"Distance-stage reward mode: {args.distance_stage_reward_mode}")
    print(f"Distance-stage target distance: {args.distance_stage_target_distance}")
    print(f"Distance-stage clamp penalty scale: {args.distance_stage_clamp_penalty_scale}")
    print(f"Training vec env: {args.train_vec_env}")
    print(f"Subproc start method: {args.subproc_start_method}")
    print(f"Training opponent: {args.opponent}")
    print(f"Evaluation opponent: {args.eval_opponent or args.opponent}")
    print(f"Non-fall mode: {not args.disable_non_fall_mode}")
    print(f"Device: {device}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Episodes per update: {args.episodes_per_update}")
    print(f"Group size: {args.group_size}")

    try:
        while total_timesteps < args.total_timesteps:
            target_episodes = choose_target_episodes(args, total_timesteps)
            batch, rollout_stats = collector.collect(
                actor,
                device=device,
                target_episodes=target_episodes,
                group_size=args.group_size,
            )
            update_stats = optimize_grpo(
                actor,
                optimizer,
                batch,
                device=device,
                minibatch_size=args.minibatch_size,
                update_epochs=args.update_epochs,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
            )

            total_timesteps += int(rollout_stats["env_steps"])
            update_index += 1

            writer.add_scalar("rollout/mean_episode_return", rollout_stats["mean_episode_return"], total_timesteps)
            writer.add_scalar("rollout/std_episode_return", rollout_stats["std_episode_return"], total_timesteps)
            writer.add_scalar("rollout/mean_episode_length", rollout_stats["mean_episode_length"], total_timesteps)
            writer.add_scalar("rollout/mean_episode_clamp_count", rollout_stats["mean_episode_clamp_count"], total_timesteps)
            writer.add_scalar("rollout/mean_final_distance", rollout_stats["mean_final_distance"], total_timesteps)
            writer.add_scalar("rollout/samples_collected", rollout_stats["samples_collected"], total_timesteps)
            writer.add_scalar("train/policy_loss", update_stats["policy_loss"], total_timesteps)
            writer.add_scalar("train/entropy", update_stats["entropy"], total_timesteps)
            writer.add_scalar("train/approx_kl", update_stats["approx_kl"], total_timesteps)
            writer.add_scalar("train/clip_fraction", update_stats["clip_fraction"], total_timesteps)
            writer.add_scalar("train/grad_norm", update_stats["grad_norm"], total_timesteps)
            writer.add_scalar("train/updates", update_stats["updates"], total_timesteps)

            print("---------------------------------------------------")
            print(f"Update: {update_index}")
            print(f"Total timesteps: {total_timesteps}")
            print(f"Episodes this update: {target_episodes}")
            print(f"Mean episode return: {rollout_stats['mean_episode_return']:.3f}")
            print(f"Mean episode clamp count: {rollout_stats['mean_episode_clamp_count']:.3f}")
            print(f"Mean final distance: {rollout_stats['mean_final_distance']:.3f}")
            print(f"Policy loss: {update_stats['policy_loss']:.6f}")
            print(f"Entropy: {update_stats['entropy']:.6f}")
            print(f"Approx KL: {update_stats['approx_kl']:.6f}")
            print(f"Clip fraction: {update_stats['clip_fraction']:.6f}")

            best_eval_reward, next_eval_at, next_checkpoint_at, did_eval = maybe_eval_and_checkpoint(
                actor=actor,
                optimizer=optimizer,
                eval_env=eval_env,
                device=device,
                args=args,
                run_dir=run_dir,
                best_model_dir=best_model_dir,
                checkpoint_dir=checkpoint_dir,
                eval_log_dir=eval_log_dir,
                eval_history=eval_history,
                total_timesteps=total_timesteps,
                update_index=update_index,
                best_eval_reward=best_eval_reward,
                force_eval=False,
                next_eval_at=next_eval_at,
                next_checkpoint_at=next_checkpoint_at,
            )
            if did_eval:
                last_eval_timestep = total_timesteps

        if last_eval_timestep != total_timesteps:
            best_eval_reward, next_eval_at, next_checkpoint_at, did_eval = maybe_eval_and_checkpoint(
                actor=actor,
                optimizer=optimizer,
                eval_env=eval_env,
                device=device,
                args=args,
                run_dir=run_dir,
                best_model_dir=best_model_dir,
                checkpoint_dir=checkpoint_dir,
                eval_log_dir=eval_log_dir,
                eval_history=eval_history,
                total_timesteps=total_timesteps,
                update_index=update_index,
                best_eval_reward=best_eval_reward,
                force_eval=True,
                next_eval_at=next_eval_at,
                next_checkpoint_at=next_checkpoint_at,
            )
            if did_eval:
                last_eval_timestep = total_timesteps

        final_model_path = run_dir / "final_model.pt"
        save_grpo_checkpoint(
            final_model_path,
            actor,
            optimizer,
            make_metadata(args, run_dir, total_timesteps, update_index, best_eval_reward),
        )
        summary = {
            "algorithm": "grpo",
            "run_dir": str(run_dir),
            "final_model": str(final_model_path),
            "best_model": str(best_model_dir / "best_model.pt"),
            "checkpoint_dir": str(checkpoint_dir),
            "tensorboard_dir": str(tensorboard_dir),
            "eval_log_dir": str(eval_log_dir),
            "total_timesteps": int(total_timesteps),
            "updates": int(update_index),
            "best_eval_reward": float(best_eval_reward),
            "opponent": args.opponent,
            "eval_opponent": args.eval_opponent or args.opponent,
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("Training finished")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        writer.close()
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    main()
