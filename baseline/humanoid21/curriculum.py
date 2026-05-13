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
``stage1.py`` is the eval-driven CurriculumStageGate: every
``cfg.eval_interval`` updates we run a deterministic eval batch and
feed its summary to ``gate.assign_from_eval()``, which classifies
the NEXT stage purely from that single eval (no hysteresis, no
dwell, no fixed transition graph). Between evals the gate state is
static and read via ``gate.current_state()``.
"""
from __future__ import annotations

import argparse
import functools
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
    make_curriculum_runtime_for,
    make_standing_adapter,
    make_standing_options_fn,
    set_seed,
)


def _agent_from_rollout_seed(seed: int) -> str:
    """Pick a training-target agent deterministically from the rollout seed.

    Identical to ``stage1.py``'s function — alternates roughly 50/50
    between robot_a and robot_b based on the seed, so over the run
    each agent gets equal training-target time.
    """
    rng = np.random.default_rng(int(seed) + 937)
    return "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"


def _select_target_trajectories(
    trajectories: Sequence[RolloutBatch],
    *,
    target_agent: str,
) -> List[RolloutBatch]:
    """Discard non-target trajectories — same pattern as ``stage1.py``."""
    return [t for t in trajectories if t.agent_id == target_agent]


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


def _apply_discounted_damage_shaping(
    trajectories: Sequence[RolloutBatch],
    *,
    gamma: float,
    r3_scale: float,
    r3_weight: float,
) -> Dict[str, float]:
    """Densify the sparse r3 (net damage) signal via discounted-future shaping.

    For each trajectory with per-step raw net damage ``d[t]`` (length
    ``T``) we compute

        shaped[t] = sum_{k>=0}  gamma^k * d[t+k]
                  = d[t] + gamma * shaped[t+1]      (right-to-left scan)

    and adjust the in-stream rewards in place::

        rewards[t] += r3_weight * r3_scale * (shaped[t] - d[t])

    The observer already emitted ``r3_weight * r3_scale * d[t]`` per step
    during rollout (see :class:`MultiSignalRewardObserver`), so adding
    ``(shaped - d)`` makes the final per-step r3 contribution equal to
    ``r3_weight * r3_scale * shaped[t]``. The reward at a hit frame is
    unchanged; what changes is that every step in the ``~ -log(eps) /
    log(1/gamma)`` window BEFORE the hit now receives a back-propagated
    fraction of that hit's credit.

    No-op when:
      * ``gamma <= 0`` (shaping disabled),
      * ``r3_weight == 0`` (Stage 1 / 2 — r3 not active),
      * ``r3_scale == 0``,
      * a trajectory is missing the ``r3_per_step`` info key (defensive).

    Returns aggregate diagnostics for logging:
      * ``raw_r3_mean``    mean of sum of raw d[t] across trajectories
      * ``shaped_r3_mean`` mean of sum of shaped[t] across trajectories
      * ``delta_sum_mean`` mean of (shaped - raw) sum * weight * scale
                           (i.e. the total per-trajectory reward shift)
    """
    if gamma <= 0.0 or r3_weight == 0.0 or r3_scale == 0.0:
        return {"raw_r3_mean": 0.0, "shaped_r3_mean": 0.0, "delta_sum_mean": 0.0}
    if not trajectories:
        return {"raw_r3_mean": 0.0, "shaped_r3_mean": 0.0, "delta_sum_mean": 0.0}

    coef = float(r3_weight) * float(r3_scale)
    g = float(gamma)
    raw_sums: List[float] = []
    shaped_sums: List[float] = []
    delta_sums: List[float] = []
    for traj in trajectories:
        info = traj.info or {}
        raw = info.get("r3_per_step")
        if raw is None:
            continue
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        T = int(traj.num_steps)
        if raw.shape[0] != T:
            # Length mismatch — be defensive, skip rather than corrupt.
            continue
        if T == 0:
            continue
        # Right-to-left discounted future sum.
        shaped = np.empty(T, dtype=np.float64)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = raw[t] + g * running
            shaped[t] = running
        delta = (shaped - raw) * coef
        traj.rewards = (traj.rewards.astype(np.float64) + delta).astype(np.float32)
        raw_sums.append(float(raw.sum()))
        shaped_sums.append(float(shaped.sum()))
        delta_sums.append(float(delta.sum()))

    if not raw_sums:
        return {"raw_r3_mean": 0.0, "shaped_r3_mean": 0.0, "delta_sum_mean": 0.0}
    return {
        "raw_r3_mean": float(np.mean(raw_sums)),
        "shaped_r3_mean": float(np.mean(shaped_sums)),
        "delta_sum_mean": float(np.mean(delta_sums)),
    }


def _inject_terminal_fall_penalty(
    trajectories: Sequence[RolloutBatch],
    *,
    terminal_fall_penalty: float,
) -> tuple[int, float]:
    """Apply sparse terminal fall penalty (literally copied from stage1.py).

    Subtracts ``terminal_fall_penalty`` from the LAST step reward of every
    trajectory whose ``terminated`` flag is True (i.e. died via
    :class:`ImbalanceTerminationPlugin`). Truncated-only trajectories
    are untouched.

    This mirrors stage1.py's recipe exactly. Together with
    ``r1_scale=0.02`` (matching stage1's ``cross_support_reward_scale``)
    the per-step reward in stage 1 is identical to stage1.py:

        per-step:  0.02 * r1_cross_support
        terminal:  -1.0 if terminated else 0.0

    Higher curriculum stages also receive the penalty — falling over is
    bad regardless of stage. The shaping rewards r2 / r3 add positively
    on top.

    Returns ``(n_terminated, total_penalty)`` for logging.
    """
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
    termination cause in ``make_curriculum_runtime_for``).
    ``mean_length``: average ``num_steps``.
    ``in_range_ratio``: average per-episode ``in_range_steps / num_steps``
    (still useful for diagnostics).
    ``final_in_zone_ratio``: fraction of episodes whose LAST step has
    both ``in_range`` and heading-in-tolerance — the curriculum gate's
    Stage 3 admission criterion.
    """
    if not trajectories:
        return {
            "term_rate": 0.0, "mean_length": 0.0,
            "in_range_ratio": 0.0, "final_in_zone_ratio": 0.0,
            "max_steps": float(max_steps), "len_ratio": 0.0,
        }
    n = len(trajectories)
    term_rate = float(sum(1 for t in trajectories if t.terminated) / n)
    mean_len = float(np.mean([int(t.num_steps) for t in trajectories]))
    in_range_ratio_values: List[float] = []
    final_in_zone_flags: List[float] = []
    for t in trajectories:
        steps = max(1, int(t.num_steps))
        info = t.info or {}
        in_range_ratio_values.append(int(info.get("in_range_steps", 0)) / steps)
        final_in_zone_flags.append(float(int(info.get("final_in_non_penalty_zone", 0))))
    in_range_ratio = float(np.mean(in_range_ratio_values))
    final_in_zone_ratio = float(np.mean(final_in_zone_flags))
    return {
        "term_rate": term_rate,
        "mean_length": mean_len,
        "in_range_ratio": in_range_ratio,
        "final_in_zone_ratio": final_in_zone_ratio,
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
def _load_actor_checkpoint(actor: torch.nn.Module, ckpt_path: Path) -> Dict[str, object]:
    """Load actor weights from a stage1 / curriculum checkpoint.

    Accepts both the wrapper dict produced by
    ``export_actor_policy_artifacts`` (``{"state_dict": <sd>, ...}``)
    and a bare state_dict. Returns the loaded payload (or empty dict)
    so the caller can log the originating ``algorithm`` / ``update``.

    Critic is NOT loaded — different reward = different value
    function, so we re-learn it from scratch. This is the safest
    option (loading a stale critic is the #1 cause of "resume looks
    fine for 5 updates and then collapses" in PPO).
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        sd = payload["state_dict"]
        meta = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        sd = payload
        meta = {}
    missing, unexpected = actor.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"[resume] partial load: missing={list(missing)} unexpected={list(unexpected)}",
            flush=True,
        )
    return meta


def train(
    cfg: CurriculumConfig,
    *,
    run_dir: Path,
    resume_from: Path | None = None,
) -> None:
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

    if resume_from is not None:
        meta = _load_actor_checkpoint(actor, Path(resume_from))
        print(
            f"[resume] loaded actor from {resume_from} "
            f"(algorithm={meta.get('algorithm')!r} "
            f"update={meta.get('update')} "
            f"best_eval_length={meta.get('best_eval_length')})",
            flush=True,
        )

    gate = CurriculumStageGate(
        max_steps=cfg.max_steps,
        pass_len_ratio=cfg.stage1_pass_len_ratio,
        pass_final_in_zone=cfg.stage2_pass_final_in_zone,
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
        f"damage_shaping_gamma={cfg.damage_shaping_gamma} "
        f"terminal_fall_penalty={cfg.terminal_fall_penalty} "
        f"log_std_max={cfg.log_std_max} "
        f"gate=eval-driven(pass_len={cfg.stage1_pass_len_ratio:.2f},"
        f"pass_final_in_zone={cfg.stage2_pass_final_in_zone:.2f},"
        f"eval_every={cfg.eval_interval})",
        flush=True,
    )

    base_factory_kwargs = dict(
        policy_factories={
            "robot_a": make_standing_adapter,
            "robot_b": make_standing_adapter,
        },
        capture_agents=("robot_a", "robot_b"),
    )

    # Two collectors — one per target. Mirror stage1.py: the runtime
    # only terminates on the TARGET agent's imbalance, so the target's
    # ``terminated`` flag is the honest "this agent fell" signal that
    # ``_inject_terminal_fall_penalty`` requires for stage-1 alignment.
    with RolloutCollector(
        runtime_factory=functools.partial(make_curriculum_runtime_for, "robot_a"),
        max_workers=cfg.rollout_workers, **base_factory_kwargs,
    ) as collector_a, RolloutCollector(
        runtime_factory=functools.partial(make_curriculum_runtime_for, "robot_b"),
        max_workers=cfg.rollout_workers, **base_factory_kwargs,
    ) as collector_b:
        # Best-eval score is a tuple (stage, eval_length, eval_reward)
        # compared lexicographically:
        #   * Stage 3 saves beat all Stage 2 saves beat all Stage 1 saves.
        #     This is essential: with the early-resume run, Stage 1 hit
        #     eval_length=200 very quickly. Without stage-ranked scoring,
        #     every later Stage 3 success (eval_length=200, eval_reward
        #     jumping from ~0 to +3.7) ties on length and never triggers
        #     a save, so the exported policy stays balance-only.
        #   * Within a stage, longer surviving evals win.
        #   * Ties broken by higher eval_reward — so Stage 3 saves
        #     prefer the highest-combat-score actor.
        best_eval: tuple = (-1, -float("inf"), -float("inf"))
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
            all_trajectories = batches.get("robot_a", []) + batches.get("robot_b", [])
            trajectories = _select_target_trajectories(
                all_trajectories, target_agent=target_agent,
            )
            if not trajectories:
                print(f"update={u} | no target trajectories (target={target_agent})", flush=True)
                continue

            batch_summary = _summarize_batch(trajectories, max_steps=cfg.max_steps)
            comp_summary = _component_summary(trajectories)
            # Mirror stage1.py exactly: post-rollout, BEFORE PPO update,
            # subtract a sparse terminal penalty from the last step of
            # every imbalance-terminated trajectory. This is what gives
            # PPO a non-zero gradient on episode length once r1 has
            # converged.
            term_count, total_term_penalty = _inject_terminal_fall_penalty(
                trajectories, terminal_fall_penalty=cfg.terminal_fall_penalty,
            )
            # Densify the sparse r3 (net damage) signal: each step now
            # gets credit for future damage events, decayed by
            # ``damage_shaping_gamma`` per step. Pure no-op outside
            # Stage 3 (where ``gate.weights[2] == 0``) or when the
            # shaping discount is set to 0. The reward at hit frames
            # is unchanged; the prior 1-3 s window gains positive
            # gradient where it previously had ~zero.
            shaping_stats = _apply_discounted_damage_shaping(
                trajectories,
                gamma=cfg.damage_shaping_gamma,
                r3_scale=cfg.r3_scale,
                r3_weight=float(gate.weights[2]),
            )
            stats = _ppo_update(actor, critic, optimizer, trajectories, cfg, device)
            # Gate state is fixed between evals; we only read it here so
            # the per-update log line carries the current stage/weights.
            gate_info = gate.current_state()

            mean_reward = float(np.mean([float(t.rewards.sum()) for t in trajectories]))
            mean_term_penalty = float(total_term_penalty / max(1, len(trajectories)))
            line = (
                f"update={u:4d} target={target_agent} stage={gate_info['stage']} "
                f"weights={tuple(round(w, 2) for w in gate_info['weights'])} "
                f"reward={mean_reward:+.4f} "
                f"len={batch_summary['mean_length']:6.2f} "
                f"term={batch_summary['term_rate']:.3f} "
                f"in_range={batch_summary['in_range_ratio']:.3f} "
                f"final_in_zone={batch_summary['final_in_zone_ratio']:.3f} "
                f"r1={comp_summary['r1_mean']:+.3f} "
                f"r2={comp_summary['r2_mean']:+.3f} "
                f"r3={comp_summary['r3_mean']:+.3f} "
                f"shaped_r3={shaping_stats['shaped_r3_mean']:+.3f} "
                f"term_pen={mean_term_penalty:+.3f} "
                f"policy_loss={stats['policy_loss']:+.5f} "
                f"value_loss={stats['value_loss']:+.5f} "
                f"kl={stats['approx_kl']:.4f} "
                f"gate_reason={gate_info['reason']!r}"
            )

            if u % cfg.eval_interval == 0:
                # Eval mirrors training: pick a target, use its collector,
                # filter to target trajectories, summarize — so the gate
                # decision is grounded in the same one-sided-termination
                # signal the trainer optimizes against.
                eval_seed = cfg.seed + 100_000 + u * 97
                eval_target = _agent_from_rollout_seed(eval_seed)
                eval_collector = (
                    collector_a if eval_target == "robot_a" else collector_b
                )
                eval_batches = eval_collector.collect(
                    n=cfg.eval_episodes,
                    base_seed=eval_seed,
                    options_fn=options_fn,
                    deterministic=True,
                    state_dicts={"robot_a": actor_sd, "robot_b": actor_sd},
                )
                eval_trajectories = _select_target_trajectories(
                    eval_batches.get("robot_a", []) + eval_batches.get("robot_b", []),
                    target_agent=eval_target,
                )
                if eval_trajectories:
                    eval_summary = _summarize_batch(
                        eval_trajectories, max_steps=cfg.max_steps,
                    )
                    eval_reward = float(
                        np.mean([float(t.rewards.sum()) for t in eval_trajectories])
                    )
                    eval_length = eval_summary["mean_length"]
                    eval_in_range = eval_summary["in_range_ratio"]
                    eval_final_in_zone = eval_summary["final_in_zone_ratio"]
                    line += (
                        f" | eval_target={eval_target}"
                        f" eval_reward={eval_reward:+.4f}"
                        f" eval_length={eval_length:6.2f}"
                        f" eval_in_range={eval_in_range:.3f}"
                        f" eval_final_in_zone={eval_final_in_zone:.3f}"
                    )
                    # ----- eval-driven stage classification --------------
                    prev_stage = gate.stage
                    gate_info = gate.assign_from_eval(eval_summary)
                    if gate_info["stage"] != prev_stage:
                        line += (
                            f"  [stage {prev_stage}->{gate_info['stage']}"
                            f" {gate_info['reason']}]"
                        )
                    # Per-eval snapshot — ALWAYS save, regardless of
                    # "best" status. The (stage, eval_length, eval_reward)
                    # ranking is a one-dimensional projection of a multi-
                    # objective fitness, and what looks suboptimal on the
                    # ranking may still be the actor a downstream user
                    # wants to inspect (e.g. the brief Stage 3 visits
                    # before a sticky-stage demotion happen). Disk cost
                    # is ~390 kB per snapshot * (max_updates / eval_interval)
                    # = ~780 MB worst case, which is acceptable.
                    snapshot_dir = (
                        policy_dir.parent
                        / "eval_snapshots"
                        / (
                            f"u{u:05d}_s{gate_info['stage']}"
                            f"_l{eval_length:03.0f}"
                            f"_r{eval_reward:+.2f}"
                            f"_fiz{eval_final_in_zone:.2f}"
                        )
                    )
                    snapshot_payload = {
                        "algorithm": "ppo_curriculum",
                        "update": u,
                        "stage": gate_info["stage"],
                        "weights": list(gate_info["weights"]),
                        "best_eval_length": eval_length,
                        "best_eval_reward": eval_reward,
                        "best_eval_final_in_zone": eval_final_in_zone,
                    }
                    export_actor_policy_artifacts(
                        actor=actor,
                        policy_dir=snapshot_dir,
                        extra_payload=snapshot_payload,
                    )

                    # Stage-ranked best-of-run also mirrored into the
                    # canonical ``policy/`` dir so downstream tools can
                    # keep pointing at a single path. See comment above
                    # on ``best_eval`` for the ranking rationale.
                    score = (gate_info["stage"], eval_length, eval_reward)
                    if score > best_eval:
                        best_eval = score
                        export_actor_policy_artifacts(
                            actor=actor,
                            policy_dir=policy_dir,
                            extra_payload=snapshot_payload,
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
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to a stage1 (or earlier curriculum) checkpoint .pt "
             "file. Loads actor only; critic is re-initialized.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Override the auto-generated run directory name.",
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
    if args.max_updates is not None:
        cfg.max_updates = int(args.max_updates)
    if args.episodes_per_update is not None:
        cfg.episodes_per_update = int(args.episodes_per_update)
    if args.rollout_workers is not None:
        cfg.rollout_workers = int(args.rollout_workers)

    name = args.run_name or f"curriculum_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent / "runs" / name
    resume = Path(args.resume_from) if args.resume_from else None
    train(cfg, run_dir=run_dir, resume_from=resume)


if __name__ == "__main__":
    main()
