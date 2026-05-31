"""孪生对抗自博弈 PPO 训练器（独立脚本，复用现有 curriculum 组件）。

设计动机
========
现状：单一共享策略 π 同时控制 robot_a / robot_b（镜像自博弈）。在对称
博弈里这天然收敛到**对称均衡**——当"接近=失衡风险"时，"双方都站桩"
是一个稳定吸引子，谁也不施压，于是机器人互相不靠近，且无法角色分化。

本脚本用**两个独立网络**（孪生）打破对称：

  * ``actor_a`` 控制 robot_a，``actor_b`` 控制 robot_b。
  * 每个 update 只训练其中一方（learner），另一方（opponent）当回合冻结，
    仅用于 rollout —— 即 iterated best response / 交替最佳响应。
  * 可选：加一个**行为多样性损失**，把 learner 的均值动作从 opponent 推开，
    强制两网行为不一致（默认关闭，``--diversity-coef 0``）。

与现有训练框架的关系
====================
完全复用 ``train_curriculum`` 的数据/PPO 内核：
  * :class:`_PPOBuffer` —— 从 Episode 构建 PPO buffer（按 learner 的 agent_id 抽取）。
  * :func:`_ppo_update` —— 多 critic GAE + PPO 更新。
  * :func:`_batch_summary` / :func:`_reward_summary` —— 日志。
唯一的新增是：
  * :func:`_build_twin_rollout_jobs` —— 让对手槽位用**另一个网络**的 blueprint。
  * 交替 learner 的主循环 + 两套 actor/critic。

关键事实（rollout 契约）
========================
``ParallelRollouter`` 的每个 job 是 ``(policy_a_bp, policy_b_bp, env_bp, seed, options)``，
其中 ``policy_a_bp`` 控制 robot_a、``policy_b_bp`` 控制 robot_b。原 curriculum
两个槽位用同一 blueprint（共享策略）；这里让 learner 和 opponent 用不同的 blueprint。

注意 / 局限
===========
  * 交替最佳响应在非传递性策略下可能**循环（cycling）**。要稳，可扩展为
    "对手从历史快照池采样"（见文末 TODO）——本简单版只用当前的另一网络。
  * 为保持简单，本脚本用**固定 stage 权重**（``FIXED_STAGE_WEIGHTS``），
    不接 ``CurriculumStageGate``；需要课程门控可自行接回。
  * 2x 网络 => 2x 显存/计算。
"""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from baseline.common.policies import CriticMLP, TanhGaussianMLPPolicy
from baseline.common.rollout import Episode, ParallelRollouter
from baseline.humanoid21.curriculum.common import (
    CurriculumConfig,
    ROLLOUT_INITIAL_DISTANCE_MAX,
    ROLLOUT_INITIAL_DISTANCE_MIN,
    set_seed,
)
from baseline.humanoid21.curriculum.train_curriculum import (
    REWARD_KEYS,
    _batch_summary,
    _ppo_update,
    _PPOBuffer,
    _reward_summary,
)
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

# 固定的奖励组合权重（r_fall, r_cross, r_relation, r_damage）。孪生版不接
# 课程门控，直接给一个"既要平衡又要接近也要打击"的组合；按需调整。
FIXED_STAGE_WEIGHTS: Tuple[float, float, float, float] = (0.4, 0.2, 0.3, 0.1)

AGENTS: Tuple[str, str] = ("robot_a", "robot_b")


# ---------------------------------------------------------------------------
# Rollout jobs —— 让对手槽位使用另一个网络
# ---------------------------------------------------------------------------

def _build_twin_rollout_jobs(
    env_pb: ParameterizedEnvBlueprint,
    learner_bp: PolicyBlueprint,
    opponent_bp: PolicyBlueprint,
    learner_agent: str,
    base_seed: int,
    n_episodes: int,
    max_steps: int,
) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
    """构造 n 个 job：learner 控制 ``learner_agent`` 一侧，opponent 控制另一侧。

    所有 job 的 ``agent_id`` 都设为 ``learner_agent``，于是 :class:`_PPOBuffer`
    只抽取 learner 一侧的轨迹与奖励来训练。
    """
    opponent_agent = "robot_b" if learner_agent == "robot_a" else "robot_a"
    rng = np.random.default_rng(base_seed)
    # 奖励观察者按 agent_id 挂载，所以 env 以 learner 视角 materialize。
    env_bp = env_pb.materialize(max_steps=max_steps, agent_id=learner_agent)

    jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
    for i in range(n_episodes):
        seed = int(base_seed + i)
        initial_distance = float(
            rng.uniform(ROLLOUT_INITIAL_DISTANCE_MIN, ROLLOUT_INITIAL_DISTANCE_MAX)
        )
        # 槽位 0 -> robot_a，槽位 1 -> robot_b。
        if learner_agent == "robot_a":
            policy_a, policy_b = learner_bp, opponent_bp
        else:
            policy_a, policy_b = opponent_bp, learner_bp
        jobs.append((
            policy_a, policy_b, env_bp, seed,
            {"agent_id": learner_agent, "initial_distance": initial_distance},
        ))
    return jobs


# ---------------------------------------------------------------------------
# 行为多样性损失（可选）—— 把 learner 的均值动作从 opponent 推开
# ---------------------------------------------------------------------------

def _diversity_step(
    learner: TanhGaussianMLPPolicy,
    opponent: TanhGaussianMLPPolicy,
    obs: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    coef: float,
    grad_clip_norm: float,
) -> float:
    """对 learner 做一步梯度，最大化其与 opponent 在相同 obs 上的均值动作差异。

    diversity_loss = -coef * mean((mu_learner - mu_opponent.detach())^2)
    （最小化它 = 增大两者动作均值的距离）。``coef=0`` 时直接跳过。
    返回本步的多样性距离（均方），用于日志。
    """
    if coef <= 0.0 or obs.shape[0] == 0:
        return 0.0
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu_opp, _ = opponent.forward(obs_t)
    mu_learner, _ = learner.forward(obs_t)
    sq_dist = (mu_learner - mu_opp).pow(2).mean()
    loss = -coef * sq_dist
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(learner.parameters(), grad_clip_norm)
    optimizer.step()
    return float(sq_dist.detach().item())


# ---------------------------------------------------------------------------
# Checkpoint（最小实现，存两套网络）
# ---------------------------------------------------------------------------

def _save_twin_checkpoint(
    path: Path,
    actors: Dict[str, TanhGaussianMLPPolicy],
    critics: Dict[str, Dict[str, CriticMLP]],
    actor_opts: Dict[str, torch.optim.Optimizer],
    critic_opts: Dict[str, Dict[str, torch.optim.Optimizer]],
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"update": int(update)}
    for ag in AGENTS:
        payload[f"actor_{ag}"] = actors[ag].state_dict()
        payload[f"actor_opt_{ag}"] = actor_opts[ag].state_dict()
        payload[f"critics_{ag}"] = {k: c.state_dict() for k, c in critics[ag].items()}
        payload[f"critic_opts_{ag}"] = {
            k: o.state_dict() for k, o in critic_opts[ag].items()
        }
    torch.save(payload, path)


def _load_twin_checkpoint(
    path: Path,
    actors: Dict[str, TanhGaussianMLPPolicy],
    critics: Dict[str, Dict[str, CriticMLP]],
    actor_opts: Dict[str, torch.optim.Optimizer],
    critic_opts: Dict[str, Dict[str, torch.optim.Optimizer]],
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    for ag in AGENTS:
        actors[ag].load_state_dict(ckpt[f"actor_{ag}"])
        actor_opts[ag].load_state_dict(ckpt[f"actor_opt_{ag}"])
        for k, sd in ckpt[f"critics_{ag}"].items():
            critics[ag][k].load_state_dict(sd)
        for k, sd in ckpt[f"critic_opts_{ag}"].items():
            critic_opts[ag][k].load_state_dict(sd)
    return int(ckpt.get("update", 0)) + 1


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------

def train_twin(
    cfg: CurriculumConfig,
    *,
    run_dir: Path,
    diversity_coef: float = 0.0,
    opponent_stochastic: bool = True,
    resume_from: Optional[Path] = None,
) -> None:
    def _shutdown_handler(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blueprint_dir = Path(__file__).resolve().parent.parent / "blueprints"
    env_pb = ParameterizedEnvBlueprint.load(blueprint_dir / "curriculum_env.yaml")
    init_policy_bp = PolicyBlueprint.load(blueprint_dir / "init_policy.yaml")

    # 两套 actor（孪生）。从同一初始 blueprint 构建，但用不同随机种子让初始
    # 权重不同，给"行为分化"一个起点。
    actors: Dict[str, TanhGaussianMLPPolicy] = {}
    for idx, ag in enumerate(AGENTS):
        torch.manual_seed(cfg.seed + 1000 * (idx + 1))
        actors[ag] = init_policy_bp.build().to(device)

    # 每个 actor 一套多 critic。
    critics: Dict[str, Dict[str, CriticMLP]] = {
        ag: {
            key: CriticMLP(obs_dim=cfg.obs_dim, hidden_dim=cfg.critic_hidden_dim).to(device)
            for key in REWARD_KEYS
        }
        for ag in AGENTS
    }
    actor_opts: Dict[str, torch.optim.Optimizer] = {
        ag: torch.optim.Adam(actors[ag].parameters(), lr=cfg.learning_rate)
        for ag in AGENTS
    }
    critic_opts: Dict[str, Dict[str, torch.optim.Optimizer]] = {
        ag: {
            key: torch.optim.Adam(critics[ag][key].parameters(), lr=cfg.critic_learning_rate)
            for key in REWARD_KEYS
        }
        for ag in AGENTS
    }

    start_update = 1
    if resume_from is not None:
        start_update = _load_twin_checkpoint(
            Path(resume_from), actors, critics, actor_opts, critic_opts,
        )
        print(f"[resume] loaded from {resume_from}, starting at update={start_update}", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    print(f"run_dir={run_dir} diversity_coef={diversity_coef} "
          f"opponent_stochastic={opponent_stochastic}", flush=True)
    print(f"[DEBUG] rollout_workers={cfg.rollout_workers} "
          f"episodes_per_update={cfg.episodes_per_update} max_steps={cfg.max_steps}", flush=True)

    with ParallelRollouter(num_workers=cfg.rollout_workers) as rollouter:
        for u in range(start_update, cfg.max_updates + 1):
            t_update_start = time.perf_counter()

            # 交替 learner：偶数 update 训 robot_a，奇数训 robot_b。
            learner_agent = AGENTS[u % 2]
            opponent_agent = "robot_b" if learner_agent == "robot_a" else "robot_a"

            # 导出两网 blueprint：learner 随机采样（探索），opponent 可选随机/确定。
            export_dir = run_dir / "policy_exports" / f"u{u:05d}"
            learner_bp = actors[learner_agent].to_blueprint(
                dest_path=str(export_dir / "learner"))
            learner_bp.config["stochastic"] = True
            opponent_bp = actors[opponent_agent].to_blueprint(
                dest_path=str(export_dir / "opponent"))
            opponent_bp.config["stochastic"] = bool(opponent_stochastic)

            # Rollout
            t0 = time.perf_counter()
            rollout_seed = cfg.seed + u * cfg.episodes_per_update
            jobs = _build_twin_rollout_jobs(
                env_pb, learner_bp, opponent_bp, learner_agent,
                rollout_seed, cfg.episodes_per_update, max_steps=cfg.max_steps,
            )
            episodes: List[Episode] = rollouter.collect(jobs)
            t_rollout = time.perf_counter() - t0

            # 构建 buffer（只抽 learner 一侧），并更新 learner。
            buf = _PPOBuffer(
                episodes=episodes,
                stage_weights=FIXED_STAGE_WEIGHTS,
                actor=actors[learner_agent],
                device=device,
                terminal_fall_penalty=cfg.terminal_fall_penalty,
            )
            if buf.is_empty():
                print(f"update={u:4d} learner={learner_agent} [empty buffer, skip]", flush=True)
                continue

            t0 = time.perf_counter()
            stats = _ppo_update(
                actors[learner_agent], critics[learner_agent],
                actor_opts[learner_agent], critic_opts[learner_agent],
                buf, cfg, device, FIXED_STAGE_WEIGHTS,
            )
            t_ppo = time.perf_counter() - t0

            # 可选：行为多样性梯度步（把 learner 从 opponent 推开）。
            div_dist = _diversity_step(
                actors[learner_agent], actors[opponent_agent],
                buf.obs, actor_opts[learner_agent], device,
                coef=diversity_coef, grad_clip_norm=cfg.grad_clip_norm,
            )

            bsum = _batch_summary(buf, cfg.max_steps)
            rsum = _reward_summary(buf)
            t_total = time.perf_counter() - t_update_start
            print(
                f"update={u:4d} learner={learner_agent} "
                f"len={bsum['mean_length']:6.2f} term={bsum['term_rate']:.3f} "
                f"final_in_zone={bsum['final_in_zone_ratio']:.3f} "
                f"r_relation={rsum['r_relation_mean']:+.3f}±{rsum['r_relation_std']:.3f} "
                f"policy_loss={stats['policy_loss']:+.5f} kl={stats['approx_kl']:.4f} "
                f"div_dist={div_dist:.4f} "
                f"| time: total={t_total:.1f}s rollout={t_rollout:.1f}s ppo={t_ppo:.2f}s",
                flush=True,
            )

            if u % cfg.eval_interval == 0 or u == 1:
                _save_twin_checkpoint(
                    ckpt_dir / f"checkpoint_u{u:05d}.pt",
                    actors, critics, actor_opts, critic_opts, update=u,
                )

    # TODO(对手池 / league)：要更稳、避免交替最佳响应循环，可把 opponent_bp
    # 从"历史 checkpoint 池"里按概率采样，而不是只用当前的另一网络。这是
    # AlphaStar / Bansal-2018 人形对抗里防止策略循环与坍缩的标准做法。


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Twin adversarial self-play PPO (humanoid21).")
    p.add_argument("--run-dir", type=str, default=None,
                   help="输出目录；默认 runs/twin_<时间戳>")
    p.add_argument("--max-updates", type=int, default=None)
    p.add_argument("--diversity-coef", type=float, default=0.0,
                   help=">0 时启用行为多样性损失（把两网动作均值推开）")
    p.add_argument("--opponent-deterministic", action="store_true",
                   help="对手用确定性动作（默认随机）")
    p.add_argument("--resume-from", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = CurriculumConfig()
    if args.max_updates is not None:
        cfg.max_updates = int(args.max_updates)

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = (
            Path(__file__).resolve().parent.parent
            / "runs" / f"twin_{time.strftime('%Y%m%d_%H%M%S')}"
        )

    train_twin(
        cfg,
        run_dir=run_dir,
        diversity_coef=float(args.diversity_coef),
        opponent_stochastic=not args.opponent_deterministic,
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )


if __name__ == "__main__":
    main()
