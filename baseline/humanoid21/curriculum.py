"""Unified curriculum-learning PPO trainer for humanoid21 combat.

Three stages, *one* training script, *one* model — the curriculum
state is decided by data, not by an explicit schedule:

  stage 1 — perturbed cross-support balance (weights = (1, 0, 0))
  stage 2 — stage 1 + approach within tolerance      ((1, 1, 0))
  stage 3 — stage 1 + stage 2 + net damage           ((1, 1, 1))

The :class:`MultiSignalRewardObserver` collects all three signals every
step and emits the *weighted* sum as the reward; the weights flow
in via ``options_fn`` from a :class:`CurriculumStageGate` that watches
imbalance termination rate, mean episode length, and time-in-range
ratio. Lower-stage rewards never get fully turned off — the gate's
hysteresis demotes the policy back if balance regresses, which is the
catastrophic-forgetting safeguard the user explicitly asked for.

Implementation style mirrors ``stage1.py``: actor + critic PPO with
GAE, single train(cfg, run_dir) function, RolloutCollector for
parallel rollout, no Trainer class. The only new piece relative to
``stage1.py`` is the per-update gate.update() + per-episode
options_fn closure that publishes the latest weights.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

import numpy as np
import torch

from baseline.common.algos import compute_gae, ppo_loss
from baseline.common.policies import (
    CriticMLP,
    TanhGaussianMLPPolicy,
    export_actor_policy_artifacts,
)
from baseline.common.rollout import RolloutBatch, RolloutCollector
from baseline.humanoid21.common import (
    CONTROL_FREQUENCY,
    CurriculumConfig,
    CurriculumStageGate,
    Humanoid21Observer,
    make_curriculum_runtime,
    make_standing_adapter,
    make_standing_options_fn,
    set_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _critic_values_and_bootstraps(
    critic: torch.nn.Module,
    trajectories: Sequence[RolloutBatch],
    device: torch.device,
) -> tuple[list[np.ndarray], list[float]]:
    """Identical to ``stage1.py``: one batched forward for values, one
    for bootstraps. Critic stays main-process only — see docstring of
    ``perturbed_standing.py`` for why."""
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


def _ppo_update(
    actor: TanhGaussianMLPPolicy,
    critic: CriticMLP,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[RolloutBatch],
    cfg: CurriculumConfig,
    device: torch.device,
) -> Dict[str, float]:
    """One PPO epoch over the on-policy batch. Mirrors ``stage1.py``."""
    valid: List[RolloutBatch] = []
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
        valid.append(t)
    if not valid:
        raise ValueError("No valid trajectories with 1-D log_probs were collected.")

    values_per_traj, bootstraps = _critic_values_and_bootstraps(critic, valid, device)

    advs_list: List[np.ndarray] = []
    rets_list: List[np.ndarray] = []
    for t, values, last_value in zip(valid, values_per_traj, bootstraps):
        adv, ret = compute_gae(
            rewards=t.rewards, values=values,
            last_value=float(last_value),
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )
        advs_list.append(adv)
        rets_list.append(ret)

    advs = np.concatenate(advs_list, axis=0)
    advs = (advs - advs.mean()) / (advs.std() + 1e-6)
    rets = np.concatenate(rets_list, axis=0)
    values_flat = np.concatenate(values_per_traj, axis=0)
    obs = np.concatenate([t.obs[:-1] for t in valid], axis=0)
    actions = np.concatenate([t.actions for t in valid], axis=0)
    old_lp = np.concatenate(
        [np.asarray(t.log_probs, dtype=np.float32).reshape(-1) for t in valid],
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
        if early_stop:
            break

    return {
        "policy_loss": float(np.mean(pol_losses)) if pol_losses else 0.0,
        "value_loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "early_stop_kl": early_stop_kl,
    }


def _snapshot(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _summarize_batch(
    trajectories: Sequence[RolloutBatch],
    *,
    max_steps: int,
) -> Dict[str, float]:
    """Collapse a rollout batch into the metrics consumed by the gate.

    ``term_rate``: fraction of episodes terminated by the imbalance
    plugin (== ``RolloutBatch.terminated`` since that's the only
    termination cause in ``make_curriculum_runtime``).
    ``mean_length``: average ``num_steps``.
    ``in_range_ratio``: average per-episode ``in_range_steps / num_steps``,
    pulled from :attr:`RolloutBatch.info`.
    """
    if not trajectories:
        return {"term_rate": 0.0, "mean_length": 0.0, "in_range_ratio": 0.0}
    n = len(trajectories)
    term_rate = float(sum(1 for t in trajectories if t.terminated) / n)
    mean_len = float(np.mean([int(t.num_steps) for t in trajectories]))
    in_range_ratio_values: List[float] = []
    for t in trajectories:
        steps = max(1, int(t.num_steps))
        in_range = int((t.info or {}).get("in_range_steps", 0))
        in_range_ratio_values.append(in_range / steps)
    in_range_ratio = float(np.mean(in_range_ratio_values))
    return {
        "term_rate": term_rate,
        "mean_length": mean_len,
        "in_range_ratio": in_range_ratio,
        "max_steps": float(max_steps),
        "len_ratio": mean_len / float(max_steps),
    }


def _component_summary(trajectories: Sequence[RolloutBatch]) -> Dict[str, float]:
    """Mean per-episode r1/r2/r3 sums (raw, pre-scale)."""
    if not trajectories:
        return {"r1_mean": 0.0, "r2_mean": 0.0, "r3_mean": 0.0}
    r1 = [float((t.info or {}).get("r1_sum", 0.0)) for t in trajectories]
    r2 = [float((t.info or {}).get("r2_sum", 0.0)) for t in trajectories]
    r3 = [float((t.info or {}).get("r3_sum", 0.0)) for t in trajectories]
    return {
        "r1_mean": float(np.mean(r1)),
        "r2_mean": float(np.mean(r2)),
        "r3_mean": float(np.mean(r3)),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(cfg: CurriculumConfig, *, run_dir: Path) -> None:
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

    gate = CurriculumStageGate(
        max_steps=cfg.max_steps,
        pass_term_rate=cfg.stage1_pass_term_rate,
        fail_term_rate=cfg.stage1_fail_term_rate,
        pass_len_ratio=cfg.stage1_pass_len_ratio,
        fail_len_ratio=cfg.stage1_fail_len_ratio,
        pass_in_range=cfg.stage2_pass_in_range,
        fail_in_range=cfg.stage2_fail_in_range,
        window=cfg.gate_window,
        min_dwell=cfg.gate_min_dwell,
    )

    distance_options_fn = make_standing_options_fn()

    def options_fn(episode_index: int) -> Dict[str, object]:
        # Distance options + the *current* curriculum weights. The
        # main-process gate is mutated outside this closure between
        # calls to ``RolloutCollector.collect``, so each rollout sees a
        # consistent weight tuple within itself but reflects the latest
        # gate decision at the next collect() boundary.
        opts = dict(distance_options_fn(episode_index))
        opts["reward_weights"] = tuple(gate.weights)
        return opts

    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policy"
    print(f"run_dir={run_dir}", flush=True)
    print(
        f"curriculum: max_steps={cfg.max_steps} "
        f"r1_scale={cfg.r1_scale} r2_scale={cfg.r2_scale} r3_scale={cfg.r3_scale} "
        f"window={cfg.gate_window} dwell={cfg.gate_min_dwell}",
        flush=True,
    )

    base_factory_kwargs = dict(
        runtime_factory=make_curriculum_runtime,
        policy_factories={
            "robot_a": make_standing_adapter,
            "robot_b": make_standing_adapter,
        },
        capture_agents=("robot_a", "robot_b"),
    )

    with RolloutCollector(
        max_workers=cfg.rollout_workers, **base_factory_kwargs,
    ) as collector:
        best_eval = -float("inf")
        for u in range(1, cfg.max_updates + 1):
            actor_sd = _snapshot(actor)
            batches = collector.collect(
                n=cfg.episodes_per_update,
                base_seed=cfg.seed + u * cfg.episodes_per_update,
                options_fn=options_fn,
                deterministic=False,
                state_dicts={"robot_a": actor_sd, "robot_b": actor_sd},
            )
            trajectories = batches.get("robot_a", []) + batches.get("robot_b", [])
            if not trajectories:
                print(f"update={u} | no trajectories", flush=True)
                continue

            batch_summary = _summarize_batch(trajectories, max_steps=cfg.max_steps)
            comp_summary = _component_summary(trajectories)
            stats = _ppo_update(actor, critic, optimizer, trajectories, cfg, device)
            gate_info = gate.update(batch_summary)

            mean_reward = float(np.mean([float(t.rewards.sum()) for t in trajectories]))
            line = (
                f"update={u:4d} stage={gate_info['stage']} "
                f"weights={tuple(round(w, 2) for w in gate_info['weights'])} "
                f"reward={mean_reward:+.4f} "
                f"len={batch_summary['mean_length']:6.2f} "
                f"term={batch_summary['term_rate']:.3f} "
                f"in_range={batch_summary['in_range_ratio']:.3f} "
                f"r1={comp_summary['r1_mean']:+.3f} "
                f"r2={comp_summary['r2_mean']:+.3f} "
                f"r3={comp_summary['r3_mean']:+.3f} "
                f"policy_loss={stats['policy_loss']:+.5f} "
                f"value_loss={stats['value_loss']:+.5f} "
                f"kl={stats['approx_kl']:.4f} "
                f"gate_reason={gate_info['reason']!r}"
            )

            if u % cfg.eval_interval == 0:
                eval_seed = cfg.seed + 100_000 + u * 97
                eval_batches = collector.collect(
                    n=cfg.eval_episodes,
                    base_seed=eval_seed,
                    options_fn=options_fn,
                    deterministic=True,
                    state_dicts={"robot_a": actor_sd, "robot_b": actor_sd},
                )
                eval_trajectories = (
                    eval_batches.get("robot_a", []) + eval_batches.get("robot_b", [])
                )
                if eval_trajectories:
                    eval_reward = float(
                        np.mean([float(t.rewards.sum()) for t in eval_trajectories])
                    )
                    eval_length = float(
                        np.mean([int(t.num_steps) for t in eval_trajectories])
                    )
                    line += f" | eval_reward={eval_reward:+.4f} eval_length={eval_length:6.2f}"
                    score = eval_length  # primary criterion: survival under curriculum
                    if score > best_eval:
                        best_eval = score
                        export_actor_policy_artifacts(
                            actor=actor,
                            policy_dir=policy_dir,
                            extra_payload={
                                "algorithm": "ppo_curriculum",
                                "update": u,
                                "stage": gate_info["stage"],
                                "weights": list(gate_info["weights"]),
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
    parser.add_argument(
        "--smoke", action="store_true",
        help="Short smoke run (max_updates=2, episodes_per_update=8, eval_episodes=4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CurriculumConfig()
    if args.smoke:
        cfg.max_updates = 2
        cfg.episodes_per_update = 8
        cfg.eval_episodes = 4
        cfg.eval_interval = 1
        cfg.rollout_workers = 2
        cfg.minibatch_size = 256
        cfg.gate_min_dwell = 1
        cfg.gate_window = 1
    if args.max_updates is not None:
        cfg.max_updates = int(args.max_updates)
    if args.episodes_per_update is not None:
        cfg.episodes_per_update = int(args.episodes_per_update)
    if args.rollout_workers is not None:
        cfg.rollout_workers = int(args.rollout_workers)

    run_dir = (
        Path(__file__).resolve().parent / "runs"
        / f"curriculum_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    train(cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()
