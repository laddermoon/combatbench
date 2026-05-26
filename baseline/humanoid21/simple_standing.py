"""Minimal GRPO-RTG standing trainer — recommended composition style.

This is a from-scratch, *idiomatic* rewrite of
``standing_grpo_rtg_tune_v2.py``. Same algorithm (GRPO-RTG: per-episode
reward-to-go, group-flatten normalization, PPO-clip surrogate, no
critic), same env (humanoid21 self-play, posture-delta reward, fall
termination), but the script itself is just glue:

  * Env wiring lives in :mod:`baseline.humanoid21.common` —
    :func:`make_standing_runtime` / :func:`make_standing_policy` are
    top-level picklable factories that ``RolloutCollector`` ships to
    its worker pool unchanged.
  * Training-loop primitives come from :mod:`baseline.common` —
    :class:`RolloutCollector`, :class:`PolicyEvaluator`,
    :func:`compute_returns_to_go`, :func:`ppo_loss`,
    :func:`export_actor_policy_artifacts`. We do not reimplement any
    of these.
  * Hyperparameters live on a single :class:`StandingConfig` dataclass
    and are overridable by CLI flags.

Compared to ``standing_grpo_rtg_tune_v2.py`` the algorithm and
hyperparameters are identical; what's stripped out is ops-side noise
(checkpoint resume, json history dumps, run-config snapshots,
intermediate checkpoint cadence). When you actually need those for a
production run, copy this file and add them — the point of *this*
script is to show what the bare composition looks like.

CLI
---

    # Default 10000 updates, parallel rollout/eval pools.
    python simple_standing.py

    # End-to-end smoke run (~30 s on the GPU server, exercises every
    # framework path including spawn-mode parallel rollout).
    python simple_standing.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

# Make ``python simple_standing.py`` work without installing the package.
COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

import numpy as np
import torch

from baseline.common.algos import compute_returns_to_go, ppo_loss
from baseline.common.eval import PolicyEvaluator
from baseline.common.policies import (
    TanhGaussianMLPPolicy,
    export_actor_policy_artifacts,
)
from baseline.common.rollout import RolloutBatch, RolloutCollector

from baseline.humanoid21.common import (
    StandingConfig,
    make_standing_policy,
    make_standing_options_fn,
    make_standing_runtime,
    set_seed,
)


# ---------------------------------------------------------------------------
# GRPO-RTG advantage: per-episode RTG → group-flatten normalize.
# ---------------------------------------------------------------------------
def _group_normalized_rtg(
    trajectories: Sequence[RolloutBatch],
    *,
    group_size: int,
    gamma: float,
) -> np.ndarray:
    """Concatenate per-trajectory advantages into one flat ``(sum_T,)`` buffer.

    For every contiguous group of ``group_size`` trajectories:

      1. compute per-step reward-to-go with discount ``gamma``;
      2. normalize each step by the *flattened* group mean / std.

    Bit-identical to v2's ``_build_group_normalized_rtg``.
    """
    advantages: List[np.ndarray] = []
    for start in range(0, len(trajectories), group_size):
        group = trajectories[start: start + group_size]
        if not group:
            continue
        rtgs = [
            compute_returns_to_go(t.rewards, gamma=gamma, last_value=0.0)
            for t in group
        ]
        flat = np.concatenate(rtgs, axis=0)
        denom = float(flat.std()) + 1e-6
        mean = float(flat.mean())
        for rtg in rtgs:
            advantages.append(((rtg - mean) / denom).astype(np.float32))
    return np.concatenate(advantages, axis=0)


# ---------------------------------------------------------------------------
# One PPO-clip update over the on-policy batch (no critic).
# ---------------------------------------------------------------------------
def _grpo_update(
    actor: TanhGaussianMLPPolicy,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[RolloutBatch],
    cfg: StandingConfig,
    device: torch.device,
) -> Dict[str, float]:
    """One GRPO-RTG update; returns scalar metrics for logging."""
    advs = _group_normalized_rtg(
        trajectories, group_size=cfg.group_size, gamma=cfg.rtg_gamma,
    )
    obs = np.concatenate([t.obs[:-1] for t in trajectories], axis=0)
    actions = np.concatenate([t.actions for t in trajectories], axis=0)
    old_lp = np.concatenate(
        [t.log_probs for t in trajectories],   # store_extras=True ⇒ never None
        axis=0,
    )

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    old_lp_t = torch.as_tensor(old_lp, dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(advs, dtype=torch.float32, device=device)
    n = obs_t.shape[0]

    losses: List[float] = []
    kls: List[float] = []
    ratios: List[float] = []
    early_stop_kl = 0.0
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n, device=device)
        early_stop = False
        for s in range(0, n, cfg.minibatch_size):
            idx = perm[s: s + cfg.minibatch_size]
            new_lp, entropy = actor.evaluate_actions(obs_t[idx], act_t[idx])
            with torch.no_grad():
                approx_kl = float((old_lp_t[idx] - new_lp).mean().item())
            kls.append(approx_kl)
            if cfg.target_kl > 0.0 and approx_kl > cfg.target_kl:
                early_stop_kl = approx_kl
                early_stop = True
                break
            # No critic in GRPO — feed zeros and zero out the value coef.
            zeros = torch.zeros_like(new_lp)
            out = ppo_loss(
                log_probs_old=old_lp_t[idx],
                log_probs_new=new_lp,
                advantages=adv_t[idx],
                values_old=zeros,
                values_new=zeros,
                returns=zeros,
                entropy=entropy,
                clip_range=cfg.clip_eps,
                value_coef=0.0,
                entropy_coef=cfg.entropy_coef,
                value_clip=None,
                normalize_advantages=False,  # already normalized above
            )
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            losses.append(float(out.policy_loss))
            with torch.no_grad():
                ratios.append(
                    float(torch.exp(new_lp - old_lp_t[idx]).mean().item())
                )
        if early_stop:
            break
    return {
        "policy_loss": float(np.mean(losses)) if losses else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "ratio": float(np.mean(ratios)) if ratios else 1.0,
        "early_stop_kl": early_stop_kl,
    }


def _snapshot(actor: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """CPU-detached state_dict — what RolloutCollector ships to workers."""
    return {k: v.detach().cpu() for k, v in actor.state_dict().items()}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def train(cfg: StandingConfig, *, run_dir: Path) -> None:
    """End-to-end GRPO-RTG training loop. ~50 lines of actual logic."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = TanhGaussianMLPPolicy(
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.actor_hidden_dim,
        log_std_min=cfg.log_std_min,
        log_std_max=cfg.log_std_max,
    ).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    options_fn = make_standing_options_fn()

    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policy"
    print(f"run_dir={run_dir}", flush=True)

    # One RolloutCollector + one PolicyEvaluator, kept alive for the full
    # training run — both own persistent worker pools, so spinning up a
    # fresh pool every iteration would dominate wall time.
    collect_kwargs = dict(
        runtime_factory=make_standing_runtime,
        policy_factories={
            "robot_a": make_standing_policy,
            "robot_b": make_standing_policy,
        },
        capture_agents=("robot_a", "robot_b"),
    )
    with RolloutCollector(
        max_workers=cfg.rollout_workers, **collect_kwargs,
    ) as collector, PolicyEvaluator(
        max_workers=cfg.eval_workers,
        deterministic=True,
        capture_agents=("robot_a",),
        runtime_factory=make_standing_runtime,
        policy_factories={
            "robot_a": make_standing_policy,
            "robot_b": make_standing_policy,
        },
    ) as evaluator:
        best_eval = -float("inf")
        for u in range(1, cfg.max_updates + 1):
            sd = _snapshot(actor)
            batches = collector.collect(
                n=cfg.episodes_per_update,
                base_seed=cfg.seed + u * cfg.episodes_per_update,
                options_fn=options_fn,
                state_dicts={"robot_a": sd, "robot_b": sd},
            )
            # Capture both sides (self-play symmetry) → 2× the data per update.
            trajectories = batches.get("robot_a", []) + batches.get("robot_b", [])
            stats = _grpo_update(actor, optimizer, trajectories, cfg, device)
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
                f"ratio={stats['ratio']:.4f} kl={stats['approx_kl']:.4f}"
            )
            if u % cfg.eval_interval == 0:
                report = evaluator.evaluate(
                    n=cfg.eval_episodes,
                    base_seed=cfg.seed + 100_000,
                    options_fn=options_fn,
                    state_dicts={"robot_a": sd, "robot_b": sd},
                )
                stats_a = report.per_agent["robot_a"]
                eval_reward = float(stats_a["return"].mean)
                line += (
                    f" | eval_reward={eval_reward:+.4f} "
                    f"eval_length={float(stats_a['length'].mean):6.2f}"
                )
                if eval_reward > best_eval:
                    best_eval = eval_reward
                    export_actor_policy_artifacts(
                        actor=actor,
                        policy_dir=policy_dir,
                        extra_payload={
                            "algorithm": "grpo_rtg_simple",
                            "update": u,
                            "best_eval_reward": best_eval,
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
        help="Short end-to-end smoke run (max_updates=2, "
             "episodes_per_update=16, eval_episodes=4, parallel=2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StandingConfig()
    if args.smoke:
        cfg.max_updates = 2
        cfg.episodes_per_update = 16
        cfg.eval_episodes = 4
        cfg.eval_interval = 1
        cfg.rollout_workers = 2
        cfg.eval_workers = 2
    # CLI overrides win over --smoke defaults so callers can mix.
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
        / f"simple_standing_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    train(cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()
