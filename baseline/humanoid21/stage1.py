"""Stage-1 PPO trainer: terminal fall penalty as primary signal.

Implementation style follows ``simple_standing.py`` (composition over
framework primitives), while algorithm semantics follow
``obsolete/standing_turbulence_ppo.py``:

  * actor + critic PPO with GAE;
  * per-step cross-support reward (scaled) + terminal fall penalty;
  * fall detection / termination comes from
    :class:`baseline.humanoid21.common.ImbalanceTerminationPlugin`.
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

import numpy as np
import torch

from envs.framework import BaseObserverPlugin, EnvRuntime, ReadOnlySimContext

from baseline.common.algos import compute_gae, ppo_loss
from baseline.common.policies import (
    CriticMLP,
    TanhGaussianMLPPolicy,
    export_actor_policy_artifacts,
)
from baseline.common.rollout import RolloutBatch, RolloutCollector
from baseline.humanoid21.common import (
    CONTROL_FREQUENCY,
    CrossSupportBalanceRewarder,
    Humanoid21Observer,
    ImbalanceTerminationPlugin,
    INITIAL_DISTANCE,
    MujocoCombatSimulator,
    make_standing_adapter,
    make_standing_options_fn,
    set_seed,
)

STAGE1_MATCH_DURATION_SECONDS = 10.0
STAGE1_MAX_STEPS = int(CONTROL_FREQUENCY * STAGE1_MATCH_DURATION_SECONDS)


@dataclass
class Stage1Config:
    # Network shape.
    obs_dim: int = Humanoid21Observer.OBS_DIM
    action_dim: int = Humanoid21Observer.ACTION_DIM
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    log_std_min: float = -4.0
    log_std_max: float = 1.0

    # PPO knobs.
    learning_rate: float = 3e-4
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 1e-3
    grad_clip_norm: float = 1.0
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 8

    # GAE.
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Sparse terminal penalty.
    terminal_fall_penalty: float = 1.0
    # 第一版刻意把交叉支撑奖励权重设小，让失衡终止信号主导早期学习。
    cross_support_reward_scale: float = 0.02

    # Rollout / eval schedule.
    episodes_per_update: int = 256 * 8
    max_updates: int = 10000
    eval_interval: int = 5
    eval_episodes: int = 16

    # Runtime horizon.
    max_steps: int = STAGE1_MAX_STEPS

    # Parallelism.
    rollout_workers: int = max(1, min(64, max(1, (torch.get_num_threads() or 1) // 2)))
    eval_workers: int = max(1, min(16, max(1, (torch.get_num_threads() or 1) // 4)))

    seed: int = 42


def _agent_from_rollout_seed(seed: int) -> str:
    rng = np.random.default_rng(int(seed) + 937)
    return "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"


def _select_target_trajectories(
    trajectories: Sequence[RolloutBatch],
    *,
    target_agent: str,
) -> List[RolloutBatch]:
    return [t for t in trajectories if t.agent_id == target_agent]


class ScaledCrossSupportRewarder(BaseObserverPlugin):
    """给 CrossSupport 奖励加缩放系数。"""

    def __init__(self, agent_id: str, scale: float) -> None:
        self._inner = CrossSupportBalanceRewarder(agent_id)
        self._scale = float(scale)
        self._output = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._inner.on_pre_episode(ctx)
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._inner.on_post_action_step(ctx)
        self._output = self._scale * float(self._inner.get_output())

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self._inner.agent_id, "scale": self._scale}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ScaledCrossSupportRewarder":
        return cls(**config)


def make_stage1_runtime_for(
    agent_id: str,
    *,
    cross_support_reward_scale: float,
    max_steps: int,
) -> EnvRuntime:
    """Runtime with cross-support reward + one-sided fall termination."""
    target = str(agent_id)
    if target not in ("robot_a", "robot_b"):
        raise ValueError(f"Unsupported agent_id: {agent_id!r}")
    simulator = MujocoCombatSimulator(initial_distance=INITIAL_DISTANCE)
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / CONTROL_FREQUENCY)))

    runtime = EnvRuntime(
        simulator=simulator,
        observer_plugins={
            "robot_a_obs": Humanoid21Observer("robot_a"),
            "robot_b_obs": Humanoid21Observer("robot_b"),
            "robot_a_reward": ScaledCrossSupportRewarder(
                "robot_a", scale=cross_support_reward_scale
            ),
            "robot_b_reward": ScaledCrossSupportRewarder(
                "robot_b", scale=cross_support_reward_scale
            ),
        },
        plugins=[
            ImbalanceTerminationPlugin(target),
        ],
        phy_steps_per_action=phy_steps_per_action,
        max_steps=int(max_steps),
    )
    runtime.observation_space = Humanoid21Observer.get_observation_space()
    runtime.action_space = Humanoid21Observer.get_action_space()
    return runtime


def _critic_values_and_bootstraps(
    critic: torch.nn.Module,
    trajectories: Sequence[RolloutBatch],
    device: torch.device,
) -> tuple[list[np.ndarray], list[float]]:
    steps = [int(t.num_steps) for t in trajectories]
    obs_flat = np.concatenate([t.obs[:-1] for t in trajectories], axis=0)
    obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
    with torch.no_grad():
        values_flat = critic(obs_t).cpu().numpy().astype(np.float32)
    offsets = np.cumsum([0] + steps).tolist()
    per_traj_values = [
        values_flat[offsets[i]: offsets[i + 1]] for i in range(len(trajectories))
    ]

    trunc_indices: list[int] = []
    trunc_final_obs: list[np.ndarray] = []
    for i, t in enumerate(trajectories):
        if t.truncated and not t.terminated:
            trunc_indices.append(i)
            trunc_final_obs.append(np.asarray(t.final_obs, dtype=np.float32))
    bootstraps = [0.0] * len(trajectories)
    if trunc_final_obs:
        boot_t = torch.as_tensor(
            np.stack(trunc_final_obs), dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            boot_vals = critic(boot_t).cpu().numpy().astype(np.float32)
        for idx, v in zip(trunc_indices, boot_vals):
            bootstraps[idx] = float(v)
    return per_traj_values, bootstraps


def _inject_terminal_fall_penalty(
    trajectories: Sequence[RolloutBatch],
    *,
    terminal_fall_penalty: float,
) -> tuple[int, float]:
    """Apply sparse terminal reward: only terminated episodes get a final penalty."""
    penalty = float(terminal_fall_penalty)
    if penalty <= 0.0:
        return 0, 0.0
    terminated_count = 0
    total_penalty = 0.0
    for t in trajectories:
        if t.terminated and t.rewards.size > 0:
            t.rewards[-1] = float(t.rewards[-1] - penalty)
            terminated_count += 1
            total_penalty += penalty
    return terminated_count, total_penalty


def _ppo_update(
    actor: TanhGaussianMLPPolicy,
    critic: CriticMLP,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[RolloutBatch],
    cfg: Stage1Config,
    device: torch.device,
) -> Dict[str, float]:
    valid_trajectories: List[RolloutBatch] = []
    for t in trajectories:
        if t.log_probs is None:
            continue
        lp = np.asarray(t.log_probs, dtype=np.float32).reshape(-1)
        steps = int(t.num_steps)
        if (
            lp.shape[0] != steps
            or t.rewards.shape[0] != steps
            or t.actions.shape[0] != steps
            or t.obs.shape[0] != steps + 1
        ):
            continue
        valid_trajectories.append(t)
    if not valid_trajectories:
        raise ValueError("No valid trajectories with 1-D log_probs were collected.")

    values_per_traj, bootstraps = _critic_values_and_bootstraps(
        critic, valid_trajectories, device,
    )

    advs_list: List[np.ndarray] = []
    rets_list: List[np.ndarray] = []
    for t, values, last_value in zip(valid_trajectories, values_per_traj, bootstraps):
        adv, ret = compute_gae(
            rewards=t.rewards,
            values=values,
            last_value=float(last_value),
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
        )
        advs_list.append(adv)
        rets_list.append(ret)

    advs = np.concatenate(advs_list, axis=0)
    advs = (advs - advs.mean()) / (advs.std() + 1e-6)
    rets = np.concatenate(rets_list, axis=0)
    values_flat = np.concatenate(values_per_traj, axis=0)
    obs = np.concatenate([t.obs[:-1] for t in valid_trajectories], axis=0)
    actions = np.concatenate([t.actions for t in valid_trajectories], axis=0)
    old_lp = np.concatenate(
        [np.asarray(t.log_probs, dtype=np.float32).reshape(-1) for t in valid_trajectories],
        axis=0,
    )

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    old_lp_t = torch.as_tensor(old_lp, dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(advs, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(rets, dtype=torch.float32, device=device)
    val_t = torch.as_tensor(values_flat, dtype=torch.float32, device=device)

    n = obs_t.shape[0]
    pol_losses: List[float] = []
    val_losses: List[float] = []
    kls: List[float] = []
    ratios: List[float] = []
    early_stop_kl = 0.0
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n, device=device)
        early_stop = False
        for s in range(0, n, cfg.minibatch_size):
            idx = perm[s: s + cfg.minibatch_size]
            new_lp, entropy = actor.evaluate_actions(obs_t[idx], act_t[idx])
            new_val = critic(obs_t[idx])
            with torch.no_grad():
                approx_kl = float((old_lp_t[idx] - new_lp).mean().item())
            kls.append(approx_kl)
            if cfg.target_kl > 0.0 and approx_kl > cfg.target_kl:
                early_stop_kl = approx_kl
                early_stop = True
                break
            out = ppo_loss(
                log_probs_old=old_lp_t[idx],
                log_probs_new=new_lp,
                advantages=adv_t[idx],
                values_old=val_t[idx],
                values_new=new_val,
                returns=ret_t[idx],
                entropy=entropy,
                clip_range=cfg.clip_eps,
                value_coef=cfg.value_loss_coef,
                entropy_coef=cfg.entropy_coef,
                value_clip=None,
                normalize_advantages=False,
            )
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                cfg.grad_clip_norm,
            )
            optimizer.step()
            pol_losses.append(float(out.policy_loss))
            val_losses.append(float(out.value_loss))
            with torch.no_grad():
                ratios.append(float(torch.exp(new_lp - old_lp_t[idx]).mean().item()))
        if early_stop:
            break

    return {
        "policy_loss": float(np.mean(pol_losses)) if pol_losses else 0.0,
        "value_loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "ratio": float(np.mean(ratios)) if ratios else 1.0,
        "early_stop_kl": early_stop_kl,
    }


def _snapshot(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def train(cfg: Stage1Config, *, run_dir: Path) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = TanhGaussianMLPPolicy(
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.actor_hidden_dim,
        log_std_min=cfg.log_std_min,
        log_std_max=cfg.log_std_max,
    ).to(device)
    critic = CriticMLP(obs_dim=cfg.obs_dim, hidden_dim=cfg.critic_hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=cfg.learning_rate,
    )
    options_fn = make_standing_options_fn()

    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policy"
    print(f"run_dir={run_dir}", flush=True)
    print(
        "reward_setup: "
        f"cross_support_reward_scale={cfg.cross_support_reward_scale:.4f}, "
        f"terminal_fall_penalty={cfg.terminal_fall_penalty:.4f}, "
        f"match_duration={cfg.max_steps / CONTROL_FREQUENCY:.1f}s",
        flush=True,
    )

    base_factory_kwargs = dict(
        policy_factories={
            "robot_a": make_standing_adapter,
            "robot_b": make_standing_adapter,
        },
        capture_agents=("robot_a", "robot_b"),
    )
    with RolloutCollector(
        runtime_factory=functools.partial(
            make_stage1_runtime_for,
            "robot_a",
            cross_support_reward_scale=cfg.cross_support_reward_scale,
            max_steps=cfg.max_steps,
        ),
        max_workers=cfg.rollout_workers,
        **base_factory_kwargs,
    ) as collector_a, RolloutCollector(
        runtime_factory=functools.partial(
            make_stage1_runtime_for,
            "robot_b",
            cross_support_reward_scale=cfg.cross_support_reward_scale,
            max_steps=cfg.max_steps,
        ),
        max_workers=cfg.rollout_workers,
        **base_factory_kwargs,
    ) as collector_b:
        best_eval = -float("inf")
        for u in range(1, cfg.max_updates + 1):
            actor_sd = _snapshot(actor)
            rollout_seed = cfg.seed + u * cfg.episodes_per_update
            target_agent = _agent_from_rollout_seed(rollout_seed)
            collector = collector_a if target_agent == "robot_a" else collector_b
            batches = collector.collect(
                n=cfg.episodes_per_update,
                base_seed=rollout_seed,
                options_fn=options_fn,
                deterministic=False,
                state_dicts={"robot_a": actor_sd, "robot_b": actor_sd},
            )
            trajectories = batches.get("robot_a", []) + batches.get("robot_b", [])
            trajectories = _select_target_trajectories(
                trajectories, target_agent=target_agent,
            )
            if not trajectories:
                print(f"update={u} | no valid trajectories", flush=True)
                continue
            cross_support_returns = np.asarray(
                [float(t.rewards.sum()) for t in trajectories], dtype=np.float32
            )
            term_count, total_term_penalty = _inject_terminal_fall_penalty(
                trajectories, terminal_fall_penalty=cfg.terminal_fall_penalty,
            )

            stats = _ppo_update(actor, critic, optimizer, trajectories, cfg, device)
            mean_reward = float(np.mean([float(t.rewards.sum()) for t in trajectories]))
            mean_cross_support_reward = float(cross_support_returns.mean())
            mean_length = float(np.mean([int(t.num_steps) for t in trajectories]))
            term_rate = float(np.mean([1.0 if t.terminated else 0.0 for t in trajectories]))
            mean_terminal_penalty = float(total_term_penalty / max(1, len(trajectories)))
            term_vs_cross_ratio = float(
                mean_terminal_penalty / (abs(mean_cross_support_reward) + 1e-6)
            )
            line = (
                f"update={u:4d} target={target_agent} mean_reward={mean_reward:+.4f} "
                f"mean_length={mean_length:6.2f} term_rate={term_rate:.3f} "
                f"cross_r={mean_cross_support_reward:+.4f} "
                f"term_pen={mean_terminal_penalty:+.4f} "
                f"pen/cross={term_vs_cross_ratio:.2f} "
                f"term_cnt={term_count:d} "
                f"policy_loss={stats['policy_loss']:+.5f} "
                f"value_loss={stats['value_loss']:+.5f} "
                f"ratio={stats['ratio']:.4f} kl={stats['approx_kl']:.4f}"
            )
            if u % cfg.eval_interval == 0:
                eval_seed = cfg.seed + 100_000 + u * 97
                eval_target_agent = _agent_from_rollout_seed(eval_seed)
                eval_collector = (
                    collector_a if eval_target_agent == "robot_a" else collector_b
                )
                eval_batches = eval_collector.collect(
                    n=cfg.eval_episodes,
                    base_seed=eval_seed,
                    options_fn=options_fn,
                    deterministic=True,
                    state_dicts={"robot_a": actor_sd, "robot_b": actor_sd},
                )
                eval_trajectories = (
                    eval_batches.get("robot_a", []) + eval_batches.get("robot_b", [])
                )
                eval_trajectories = _select_target_trajectories(
                    eval_trajectories, target_agent=eval_target_agent,
                )
                if not eval_trajectories:
                    eval_reward = 0.0
                    eval_length = 0.0
                else:
                    eval_reward = float(np.mean([float(t.rewards.sum()) for t in eval_trajectories]))
                    eval_length = float(np.mean([int(t.num_steps) for t in eval_trajectories]))
                line += (
                    f" | eval_target={eval_target_agent} eval_reward={eval_reward:+.4f} "
                    f"eval_length={eval_length:6.2f}"
                )
                if eval_length > best_eval:
                    best_eval = eval_length
                    export_actor_policy_artifacts(
                        actor=actor,
                        policy_dir=policy_dir,
                        extra_payload={
                            "algorithm": "ppo_stage1_terminal_fall_penalty",
                            "update": u,
                            "best_eval_length": eval_length,
                            "best_eval_reward": eval_reward,
                            "terminal_fall_penalty": cfg.terminal_fall_penalty,
                        },
                    )
                    line += "  [new_best]"
            print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--terminal-fall-penalty", type=float, default=None)
    parser.add_argument("--cross-support-reward-scale", type=float, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short smoke run (max_updates=2, episodes_per_update=16, eval_episodes=4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Stage1Config()
    if args.smoke:
        cfg.max_updates = 2
        cfg.episodes_per_update = 16
        cfg.eval_episodes = 4
        cfg.eval_interval = 1
        cfg.rollout_workers = 2
        cfg.eval_workers = 2
        cfg.minibatch_size = 256
    if args.max_updates is not None:
        cfg.max_updates = int(args.max_updates)
    if args.episodes_per_update is not None:
        cfg.episodes_per_update = int(args.episodes_per_update)
    if args.rollout_workers is not None:
        cfg.rollout_workers = int(args.rollout_workers)
    if args.eval_workers is not None:
        cfg.eval_workers = int(args.eval_workers)
    if args.terminal_fall_penalty is not None:
        cfg.terminal_fall_penalty = float(args.terminal_fall_penalty)
    if args.cross_support_reward_scale is not None:
        cfg.cross_support_reward_scale = float(args.cross_support_reward_scale)

    run_dir = (
        Path(__file__).resolve().parent / "runs"
        / f"stage1_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    train(cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()

