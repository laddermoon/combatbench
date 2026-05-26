"""Minimal PPO trainer for perturbed standing with balance reward.

Same recipe as ``simple_standing.py`` but for the PPO-with-critic
variant of ``standing_balance_ppo.py``:

  * **Reward** — balance score from the support-polygon analysis
    (:class:`BalanceValueRewarder`), bounded roughly in ``[-4, 1]``.
  * **Termination** — persistent low balance score
    (:class:`BalanceScoreTerminationPlugin`), not height/uprightness.
  * **Initial-state perturbation** — every episode starts with a
    random nudge of joints / root pose / velocities, so the policy
    has to actively stabilize rather than coast on a symmetric reset.
  * **Algorithm** — full PPO: actor + critic, GAE(λ), value-clip off,
    advantage normalization, target-KL early stop.

Env wiring lives in :func:`baseline.humanoid21.common.make_perturbed_balance_runtime`
and is reused unchanged. Training primitives come from
``baseline.common`` (``RolloutCollector``, ``PolicyEvaluator``,
``compute_gae``, ``ppo_loss``, ``export_actor_policy_artifacts``). This
script only contains the algorithm glue: rollout → critic forward on
main process → GAE → PPO update → eval.

Critic lives only on the main process
-------------------------------------
Workers run an actor-only :class:`TanhGaussianMLPPolicy`. We deliberately
do NOT broadcast critic weights — ``TanhGaussianMLPPolicy`` (an
``nn.Module``) receives the full ``state_dict`` via ``load_state_dict``,
so a critic on workers would be stale from iteration 2 onwards. Instead
we compute per-step values with **one batched forward pass** on the
main process after rollout, plus a second batched pass on the
per-episode :attr:`final_obs` for the truncation bootstrap value. This
matches the v1 script's behavior exactly and is strictly faster than
one-value-at-a-time per worker.

CLI
---

    # Default (10000 updates, parallel rollout/eval pools).
    python perturbed_standing.py

    # End-to-end smoke run (~1 min on GPU server).
    python perturbed_standing.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

# Let ``python perturbed_standing.py`` work without installing the package.
COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

import numpy as np
import torch

from baseline.common.algos import compute_gae, ppo_loss
from baseline.common.eval import PolicyEvaluator
from baseline.common.policies import (
    CriticMLP,
    TanhGaussianMLPPolicy,
    export_actor_policy_artifacts,
)
from baseline.common.rollout import RolloutBatch, RolloutCollector

from baseline.humanoid21.common import (
    PerturbedBalanceConfig,
    make_perturbed_balance_runtime,
    make_standing_policy,
    make_standing_options_fn,
    set_seed,
)


# ---------------------------------------------------------------------------
# Main-process critic pass: produces (values, bootstrap_values) for GAE.
# ---------------------------------------------------------------------------
def _critic_values_and_bootstraps(
    critic: torch.nn.Module,
    trajectories: Sequence[RolloutBatch],
    device: torch.device,
) -> tuple[list[np.ndarray], list[float]]:
    """One batched forward for values, one for bootstraps.

    Values are evaluated at ``obs[:-1]`` (the states that actually
    produced an action). Bootstrap values are evaluated at
    ``final_obs`` for *truncated* episodes only; for *terminated*
    episodes the bootstrap is 0.0 (MDP-terminal semantics; see the
    docstring of :func:`baseline.common.algos.compute_gae`).
    """
    # Concatenate obs[:-1] across all trajectories for one forward.
    steps = [int(t.num_steps) for t in trajectories]
    obs_flat = np.concatenate(
        [t.obs[:-1] for t in trajectories], axis=0,
    )
    obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
    with torch.no_grad():
        values_flat = critic(obs_t).cpu().numpy().astype(np.float32)
    offsets = np.cumsum([0] + steps).tolist()
    per_traj_values = [
        values_flat[offsets[i]: offsets[i + 1]] for i in range(len(trajectories))
    ]

    # Bootstrap: only truncated episodes need V(final_obs). A single
    # batched forward over every truncated final_obs; terminated
    # episodes get 0.0 (MDP-terminal semantics).
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


# ---------------------------------------------------------------------------
# One PPO update over the on-policy batch.
# ---------------------------------------------------------------------------
def _ppo_update(
    actor: TanhGaussianMLPPolicy,
    critic: CriticMLP,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[RolloutBatch],
    cfg: PerturbedBalanceConfig,
    device: torch.device,
) -> Dict[str, float]:
    # 1) Values + bootstraps (one main-process critic pass).
    values_per_traj, bootstraps = _critic_values_and_bootstraps(
        critic, trajectories, device,
    )

    # 2) Per-episode GAE → concatenate → flatten-normalize advantages.
    advs_list: List[np.ndarray] = []
    rets_list: List[np.ndarray] = []
    for t, v, last_v in zip(trajectories, values_per_traj, bootstraps):
        adv, ret = compute_gae(
            rewards=t.rewards, values=v,
            last_value=float(last_v),
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )
        advs_list.append(adv)
        rets_list.append(ret)
    advs = np.concatenate(advs_list, axis=0)
    advs = (advs - advs.mean()) / (advs.std() + 1e-6)
    rets = np.concatenate(rets_list, axis=0)
    values_flat = np.concatenate(values_per_traj, axis=0)

    # 3) Pack the rest of the batch.
    obs = np.concatenate([t.obs[:-1] for t in trajectories], axis=0)
    actions = np.concatenate([t.actions for t in trajectories], axis=0)
    old_lp = np.concatenate([t.log_probs for t in trajectories], axis=0)

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
                value_clip=None,              # no value clipping
                normalize_advantages=False,   # already normalized above
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
                ratios.append(
                    float(torch.exp(new_lp - old_lp_t[idx]).mean().item())
                )
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: PerturbedBalanceConfig, *, run_dir: Path) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = TanhGaussianMLPPolicy(
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.actor_hidden_dim,
        log_std_min=cfg.log_std_min,
        log_std_max=cfg.log_std_max,
    ).to(device)
    critic = CriticMLP(
        obs_dim=cfg.obs_dim, hidden_dim=cfg.critic_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=cfg.learning_rate,
    )
    options_fn = make_standing_options_fn()
    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policy"
    print(f"run_dir={run_dir}", flush=True)

    factory_kwargs = dict(
        runtime_factory=make_perturbed_balance_runtime,
        policy_factories={
            "robot_a": make_standing_policy,
            "robot_b": make_standing_policy,
        },
        capture_agents=("robot_a", "robot_b"),
    )
    with RolloutCollector(
        max_workers=cfg.rollout_workers, **factory_kwargs,
    ) as collector, PolicyEvaluator(
        max_workers=cfg.eval_workers,
        deterministic=True,
        **{**factory_kwargs, "capture_agents": ("robot_a",)},
    ) as evaluator:
        best_eval = -float("inf")
        for u in range(1, cfg.max_updates + 1):
            actor_sd = _snapshot(actor)
            # Only the actor is pushed to workers; critic stays
            # main-process only (see module docstring for why).
            batches = collector.collect(
                n=cfg.episodes_per_update,
                base_seed=cfg.seed + u * cfg.episodes_per_update,
                options_fn=options_fn,
                state_dicts={"robot_a": actor_sd, "robot_b": actor_sd},
            )
            trajectories = batches.get("robot_a", []) + batches.get("robot_b", [])
            if not trajectories:
                print(f"update={u} | no valid trajectories", flush=True)
                continue

            stats = _ppo_update(actor, critic, optimizer, trajectories, cfg, device)
            mean_reward = float(np.mean(
                [float(t.rewards.sum()) for t in trajectories]
            ))
            mean_length = float(np.mean(
                [int(t.num_steps) for t in trajectories]
            ))
            line = (
                f"update={u:4d} mean_reward={mean_reward:+.4f} "
                f"mean_length={mean_length:6.2f} "
                f"policy_loss={stats['policy_loss']:+.5f} "
                f"value_loss={stats['value_loss']:+.5f} "
                f"ratio={stats['ratio']:.4f} kl={stats['approx_kl']:.4f}"
            )
            if u % cfg.eval_interval == 0:
                report = evaluator.evaluate(
                    n=cfg.eval_episodes,
                    base_seed=cfg.seed + 100_000,
                    options_fn=options_fn,
                    state_dicts={"robot_a": actor_sd, "robot_b": actor_sd},
                )
                stats_a = report.per_agent["robot_a"]
                eval_reward = float(stats_a["return"].mean)
                eval_length = float(stats_a["length"].mean)
                line += (
                    f" | eval_reward={eval_reward:+.4f} "
                    f"eval_length={eval_length:6.2f}"
                )
                # Rank by survival length first, reward second —
                # matches v1 (best_eval_score tuple ordering).
                current = (eval_length, eval_reward)
                if current > (best_eval if isinstance(best_eval, tuple)
                              else (-float("inf"), -float("inf"))):
                    best_eval = current
                    export_actor_policy_artifacts(
                        actor=actor,
                        policy_dir=policy_dir,
                        extra_payload={
                            "algorithm": "ppo_perturbed_balance",
                            "update": u,
                            "best_eval_length": eval_length,
                            "best_eval_reward": eval_reward,
                        },
                    )
                    line += "  [new_best]"
            print(line, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Short smoke run (max_updates=2, episodes_per_update=16, "
             "eval_episodes=4, workers=2). Exercises every path "
             "including spawn-mode parallel rollout + parallel eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PerturbedBalanceConfig()
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

    run_dir = (
        Path(__file__).resolve().parent / "runs"
        / f"perturbed_standing_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    train(cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()
