"""
GRPO 训练入口

基于新框架 (envs/framework + envs/humanoid21) 的 GRPO 训练脚本
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from combatbench.baseline.humanoid21_nonfall.gym_adapter import SingleAgentAttackerEnv
from combatbench.baseline.humanoid21_nonfall.grpo import (
    GRPOActor,
    GRPOActionPenaltyConfig,
    GRPOModelConfig,
    GRPORolloutCollector,
    evaluate_grpo_actor,
    load_grpo_checkpoint,
    optimize_grpo,
    resolve_device,
    save_grpo_checkpoint,
)
from combatbench.baseline.humanoid21_nonfall.reward_config import (
    AttackerRewardConfig,
    DistanceStageRewardConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GRPO attacker with new framework"
    )
    # 运行配置
    parser.add_argument("--run-name", type=str, default="grpo_humanoid21")
    parser.add_argument("--output-dir", type=str, default="baseline/humanoid21_nonfall/runs")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    # 训练参数
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

    # 网络参数
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--log-std-init", type=float, default=-0.5)

    # 检查点和评估
    parser.add_argument("--checkpoint-freq", type=int, default=20000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=3)

    # VecEnv 配置
    parser.add_argument("--train-vec-env", type=str, default="auto", choices=["auto", "dummy", "subproc"])
    parser.add_argument("--subproc-start-method", type=str, default="spawn", choices=["spawn", "forkserver", "fork"])

    # 环境参数
    parser.add_argument("--arena-xml", type=str, default=None)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--control-frequency", type=int, default=20)
    parser.add_argument("--match-duration", type=float, default=10.0)
    parser.add_argument("--initial-distance", type=float, default=2.0)
    parser.add_argument("--initial-health", type=float, default=100.0)
    parser.add_argument("--damage-scale", type=float, default=100.0)

    # 约束参数
    parser.add_argument("--disable-non-fall-mode", action="store_true")
    parser.add_argument("--non-fall-pitch-limit-deg", type=float, default=5.0)
    parser.add_argument("--non-fall-roll-limit-deg", type=float, default=5.0)

    # 对手参数
    parser.add_argument("--opponent", type=str, default="standing", choices=["standing", "random", "active", "scripted", "scripted_active"])
    parser.add_argument("--eval-opponent", type=str, default=None)
    parser.add_argument("--opponent-random-scale", type=float, default=0.1)

    # 奖励参数
    parser.add_argument("--curriculum-stage", type=str, default="attack", choices=["attack", "distance_stage1"])

    # 攻击模式奖励参数
    parser.add_argument("--damage-reward-scale", type=float, default=1.0)
    parser.add_argument("--hit-reward-scale", type=float, default=0.35)
    parser.add_argument("--approach-reward-scale", type=float, default=0.8)
    parser.add_argument("--close-distance-reward-scale", type=float, default=0.08)
    parser.add_argument("--retreat-penalty-scale", type=float, default=0.35)
    parser.add_argument("--facing-reward-scale", type=float, default=0.05)
    parser.add_argument("--upright-reward-scale", type=float, default=0.03)
    parser.add_argument("--win-bonus", type=float, default=2.0)
    parser.add_argument("--loss-penalty", type=float, default=0.5)

    # 距离阶段奖励参数
    parser.add_argument("--distance-stage-target-distance", type=float, default=0.55)
    parser.add_argument("--distance-stage-reward-mode", type=str, default="step_delta", choices=["step_delta", "episode_uniform", "episode_curriculum"])
    parser.add_argument("--distance-stage-reward-power", type=float, default=2.0)
    parser.add_argument("--distance-stage-clamp-penalty-scale", type=float, default=0.002)
    parser.add_argument("--distance-stage-prioritize-no-clamp", action="store_true")
    parser.add_argument("--distance-stage-close-enough-distance", type=float, default=0.6)
    parser.add_argument("--distance-stage-attack-damage-reward-scale", type=float, default=1000.0)

    # Action penalty 参数
    parser.add_argument("--action-magnitude-loss-coef", type=float, default=1.0)
    parser.add_argument("--action-delta-loss-coef", type=float, default=1.0)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """验证参数"""
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
    if args.action_magnitude_loss_coef < 1.0:
        raise ValueError("--action-magnitude-loss-coef must be greater than or equal to 1.0")
    if args.action_delta_loss_coef < 1.0:
        raise ValueError("--action-delta-loss-coef must be greater than or equal to 1.0")


def build_run_dir(output_dir: str, run_name: str) -> Path:
    """构建运行目录"""
    base_dir = Path(output_dir)
    existing_runs = list(base_dir.glob(f"{run_name}_*"))
    if existing_runs:
        run_numbers = []
        for path in existing_runs:
            try:
                num = int(path.name.split("_")[-1])
                run_numbers.append(num)
            except ValueError:
                pass
        next_num = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_num = 1
    run_dir = base_dir / f"{run_name}_{next_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_arena_xml_path(args: argparse.Namespace) -> str:
    """获取场景 XML 路径"""
    if args.arena_xml is not None:
        return args.arena_xml
    # 尝试多个可能的路径
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "assets" / "battle_v1.xml",
        Path(__file__).parent.parent.parent / "assets" / "battle_v1.xml",
        Path("assets/battle_v1.xml"),
    ]
    for default_path in possible_paths:
        if default_path.exists():
            return str(default_path)
    raise FileNotFoundError(f"Arena XML not found at any of {possible_paths}, please specify --arena-xml")


def build_attacker_reward_config(args: argparse.Namespace) -> AttackerRewardConfig:
    """构建攻击者奖励配置"""
    return AttackerRewardConfig(
        damage_reward_scale=args.damage_reward_scale,
        hit_reward_scale=args.hit_reward_scale,
        approach_reward_scale=args.approach_reward_scale,
        close_distance_reward_scale=args.close_distance_reward_scale,
        retreat_penalty_scale=args.retreat_penalty_scale,
        facing_reward_scale=args.facing_reward_scale,
        upright_reward_scale=args.upright_reward_scale,
        win_bonus=args.win_bonus,
        loss_penalty=args.loss_penalty,
    )


def build_distance_stage_reward_config(args: argparse.Namespace) -> DistanceStageRewardConfig:
    """构建距离阶段奖励配置"""
    return DistanceStageRewardConfig(
        target_distance=args.distance_stage_target_distance,
        reward_mode=args.distance_stage_reward_mode,
        distance_reward_power=args.distance_stage_reward_power,
        clamp_penalty_scale=args.distance_stage_clamp_penalty_scale,
        prioritize_no_clamp=args.distance_stage_prioritize_no_clamp,
        close_enough_distance=args.distance_stage_close_enough_distance,
        attack_damage_reward_scale=args.distance_stage_attack_damage_reward_scale,
    )


def make_env_factory(
    args: argparse.Namespace,
    rank: int = 0,
    eval_mode: bool = False,
) -> callable:
    """创建环境工厂函数"""

    def _env_factory() -> SingleAgentAttackerEnv:
        opponent = args.eval_opponent if eval_mode else args.opponent
        opponent_seed = args.seed + rank if not eval_mode else args.seed

        return SingleAgentAttackerEnv(
            arena_xml=get_arena_xml_path(args),
            dt=args.dt,
            control_frequency=args.control_frequency,
            match_duration=args.match_duration,
            initial_distance=args.initial_distance,
            initial_health=args.initial_health,
            damage_scale=args.damage_scale,
            opponent=opponent,
            opponent_seed=opponent_seed,
            opponent_random_scale=args.opponent_random_scale,
            curriculum_stage=args.curriculum_stage,
            reward_config=build_attacker_reward_config(args),
            distance_stage_config=build_distance_stage_reward_config(args),
            non_fall_mode=not args.disable_non_fall_mode,
            non_fall_pitch_limit_deg=args.non_fall_pitch_limit_deg,
            non_fall_roll_limit_deg=args.non_fall_roll_limit_deg,
        )

    return _env_factory


def build_train_vec_env(args: argparse.Namespace):
    """构建训练 VecEnv"""
    env_fns = [make_env_factory(args, rank=i) for i in range(args.n_envs)]

    vec_env_type = args.train_vec_env
    if vec_env_type == "auto":
        vec_env_type = "subproc" if args.n_envs > 1 else "dummy"

    if vec_env_type == "subproc":
        return SubprocVecEnv(env_fns, start_method=args.subproc_start_method)
    else:
        return DummyVecEnv(env_fns)


def build_eval_env(args: argparse.Namespace) -> SingleAgentAttackerEnv:
    """构建评估环境"""
    return make_env_factory(args, eval_mode=True)()


def save_run_config(run_dir: Path, args: argparse.Namespace) -> None:
    """保存运行配置"""
    config_path = run_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def choose_target_episodes(args: argparse.Namespace, completed_timesteps: int) -> int:
    """选择目标回合数"""
    remaining_timesteps = max(0, int(args.total_timesteps) - int(completed_timesteps))
    estimated_episode_length = max(1, int(args.match_duration * args.control_frequency))
    estimated_episodes = remaining_timesteps // estimated_episode_length
    if estimated_episodes <= 0:
        return int(args.group_size)
    grouped_episodes = (estimated_episodes // int(args.group_size)) * int(args.group_size)
    if grouped_episodes <= 0:
        return int(args.group_size)
    return min(int(args.episodes_per_update), grouped_episodes)


def make_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    total_timesteps: int,
    update_index: int,
    best_eval_reward: float,
) -> Dict[str, Any]:
    """创建元数据"""
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
    """保存评估历史"""
    if not eval_history:
        return
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        eval_path,
        timesteps=np.asarray([item["timesteps"] for item in eval_history], dtype=np.int64),
        results=np.asarray([item["episode_returns"] for item in eval_history], dtype=np.float32),
        episode_lengths=np.asarray([item["episode_lengths"] for item in eval_history], dtype=np.float32),
        clamp_counts=np.asarray([item["episode_clamp_counts"] for item in eval_history], dtype=np.float32),
        episode_damage_dealt=np.asarray([item["episode_damage_dealt"] for item in eval_history], dtype=np.float32),
        episode_min_horizontal_distances=np.asarray(
            [item["episode_min_horizontal_distances"] for item in eval_history], dtype=np.float32
        ),
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
    """执行评估和检查点保存"""
    if total_timesteps >= next_checkpoint_at:
        checkpoint_path = checkpoint_dir / f"grpo_actor_{total_timesteps}.pt"
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
            "episode_damage_dealt": eval_metrics["episode_damage_dealt"].tolist(),
            "episode_min_horizontal_distances": eval_metrics["episode_min_horizontal_distances"].tolist(),
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
        "mean_episode_damage_dealt": float(eval_metrics["mean_episode_damage_dealt"]),
        "mean_episode_min_horizontal_distance": float(eval_metrics["mean_episode_min_horizontal_distance"]),
        "mean_final_distance": float(eval_metrics["mean_final_distance"]),
        "episode_returns": eval_metrics["episode_returns"].tolist(),
        "episode_lengths": eval_metrics["episode_lengths"].tolist(),
        "episode_clamp_counts": eval_metrics["episode_clamp_counts"].tolist(),
        "episode_damage_dealt": eval_metrics["episode_damage_dealt"].tolist(),
        "episode_min_horizontal_distances": eval_metrics["episode_min_horizontal_distances"].tolist(),
        "final_distances": eval_metrics["final_distances"].tolist(),
    }
    with open(eval_log_dir / f"eval_{total_timesteps}.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)

    print(f"Eval timesteps={total_timesteps}, reward={eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}")
    print(f"Episode length: {eval_metrics['mean_episode_length']:.2f}")
    print(f"Mean clamp count: {eval_metrics['mean_episode_clamp_count']:.2f}")
    print(f"Mean damage dealt: {eval_metrics['mean_episode_damage_dealt']:.4f}")
    print(f"Mean min distance: {eval_metrics['mean_episode_min_horizontal_distance']:.3f}")
    print(f"Mean final distance: {eval_metrics['mean_final_distance']:.3f}")

    if eval_metrics["mean_reward"] > best_eval_reward:
        best_eval_reward = float(eval_metrics["mean_reward"])
        save_grpo_checkpoint(
            best_model_dir / "best_model.pt",
            actor,
            optimizer,
            make_metadata(args, run_dir, total_timesteps, update_index, best_eval_reward),
        )
        print("New best model!")

    while total_timesteps >= next_eval_at:
        next_eval_at += int(args.eval_freq)
    return best_eval_reward, next_eval_at, next_checkpoint_at, True


def maybe_resume_actor(
    args: argparse.Namespace,
    actor: GRPOActor,
    optimizer: torch.optim.Optimizer,
) -> None:
    """恢复训练"""
    if not args.resume_from:
        return
    resumed_actor, checkpoint = load_grpo_checkpoint(args.resume_from, device=resolve_device(args.device))
    actor.load_state_dict(resumed_actor.state_dict())
    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    print(f"Resumed from {args.resume_from}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    # 创建运行目录
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

    # 设置设备
    device = resolve_device(args.device)

    # 创建环境
    train_env = build_train_vec_env(args)
    eval_env = build_eval_env(args)

    # 创建策略网络
    obs_dim = int(train_env.observation_space.shape[-1])
    action_dim = int(train_env.action_space.shape[-1])
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

    # 创建收集器
    collector = GRPORolloutCollector(
        train_env,
        curriculum_stage=args.curriculum_stage,
    )

    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # 训练状态
    total_timesteps = 0
    update_index = 0
    best_eval_reward = float("-inf")
    eval_history: List[Dict[str, Any]] = []
    next_eval_at = int(args.eval_freq)
    next_checkpoint_at = int(args.checkpoint_freq)
    last_eval_timestep = -1

    # Action penalty 配置
    action_penalty_config = GRPOActionPenaltyConfig(
        action_magnitude_coef=float(args.action_magnitude_loss_coef),
        action_delta_coef=float(args.action_delta_loss_coef),
    )

    # 打印配置
    print(f"Run directory: {run_dir}")
    print(f"Algorithm: GRPO")
    print(f"Curriculum stage: {args.curriculum_stage}")
    print(f"Device: {device}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Environments: {args.n_envs}")
    print(f"Episodes per update: {args.episodes_per_update}")
    print(f"Group size: {args.group_size}")
    print(f"Opponent: {args.opponent}")
    print(f"Eval opponent: {args.eval_opponent or args.opponent}")
    print(f"Non-fall mode: {not args.disable_non_fall_mode}")

    try:
        while total_timesteps < args.total_timesteps:
            target_episodes = choose_target_episodes(args, total_timesteps)

            # 收集经验
            batch, rollout_stats = collector.collect(
                actor,
                device=device,
                target_episodes=target_episodes,
                group_size=args.group_size,
            )

            # 优化策略
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
                action_penalty_config=action_penalty_config,
            )

            total_timesteps += int(rollout_stats["env_steps"])
            update_index += 1

            # 记录到 TensorBoard
            writer.add_scalar("rollout/mean_episode_return", rollout_stats["mean_episode_return"], total_timesteps)
            writer.add_scalar("rollout/std_episode_return", rollout_stats["std_episode_return"], total_timesteps)
            writer.add_scalar("rollout/mean_episode_length", rollout_stats["mean_episode_length"], total_timesteps)
            writer.add_scalar("rollout/mean_episode_clamp_count", rollout_stats["mean_episode_clamp_count"], total_timesteps)
            writer.add_scalar("rollout/mean_episode_damage_dealt", rollout_stats["mean_episode_damage_dealt"], total_timesteps)
            writer.add_scalar("rollout/mean_episode_min_horizontal_distance", rollout_stats["mean_episode_min_horizontal_distance"], total_timesteps)
            writer.add_scalar("rollout/mean_final_distance", rollout_stats["mean_final_distance"], total_timesteps)
            writer.add_scalar("rollout/mean_action_magnitude", rollout_stats["mean_action_magnitude"], total_timesteps)
            writer.add_scalar("rollout/mean_action_delta", rollout_stats["mean_action_delta"], total_timesteps)
            writer.add_scalar("train/policy_loss", update_stats["policy_loss"], total_timesteps)
            writer.add_scalar("train/base_policy_loss", update_stats["base_policy_loss"], total_timesteps)
            writer.add_scalar("train/entropy", update_stats["entropy"], total_timesteps)
            writer.add_scalar("train/approx_kl", update_stats["approx_kl"], total_timesteps)
            writer.add_scalar("train/clip_fraction", update_stats["clip_fraction"], total_timesteps)
            writer.add_scalar("train/grad_norm", update_stats["grad_norm"], total_timesteps)

            # 打印进度
            print("=" * 60)
            print(f"Update: {update_index} | Timesteps: {total_timesteps}")
            print(f"Episodes: {target_episodes} | Mean return: {rollout_stats['mean_episode_return']:.3f}")
            print(f"Clamp: {rollout_stats['mean_episode_clamp_count']:.2f} | Damage: {rollout_stats['mean_episode_damage_dealt']:.4f}")
            print(f"Min distance: {rollout_stats['mean_episode_min_horizontal_distance']:.3f}")
            print(f"Policy loss: {update_stats['policy_loss']:.6f} | Entropy: {update_stats['entropy']:.6f}")
            print(f"KL: {update_stats['approx_kl']:.6f} | Clip frac: {update_stats['clip_fraction']:.6f}")

            # 评估和检查点
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

        # 最终评估
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

        # 保存最终模型
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
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        print("\n" + "=" * 60)
        print("Training completed!")
        print(json.dumps(summary, indent=2))

    finally:
        writer.close()
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    main()
