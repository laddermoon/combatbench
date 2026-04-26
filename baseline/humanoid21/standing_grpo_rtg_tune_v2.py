"""GRPO-RTG standing trainer, reimplemented on ``baseline/common/``.

This is a faithful reproduction of
:mod:`baseline.humanoid21.standing_grpo_rtg_tune` using the training-side
building blocks landed in ``baseline/common/`` (PRs 1-9 + parallel
``RolloutCollector``). Same hyperparameters, same env wiring, same
per-step reward (delta of posture score) + per-episode group-normalized
reward-to-go advantage. The v1 script is known to reliably train a very
stable standing policy from scratch — the point of this v2 is to show
that the new framework can reproduce that result end-to-end, and in
the process check that the framework's contracts and picklability
assumptions hold for a real training loop.

What changed vs. v1
-------------------
* **Rollout**: a ``RolloutCollector(max_workers=N)`` with per-iteration
  ``state_dicts=`` broadcast replaces the bespoke
  ``ProcessPoolExecutor`` + ``_collect_episode_chunk`` worker layer in
  v1. Same multi-process topology, same amortized worker cost, but all
  episode-loop / observer routing / seeding lives in
  :class:`envs.framework.episode_runner.EpisodeRunner`.
* **Actor**: :class:`baseline.common.policies.TanhGaussianMLPPolicy`
  (identical architecture to v1's ``Actor``) wrapped in a
  :class:`TorchPolicyAdapter`.
* **Self-play symmetry**: both ``robot_a`` and ``robot_b`` get the same
  shared actor via parallel state_dict broadcast; we capture *both*
  sides' trajectories (2 × ``EPISODES_PER_UPDATE`` learning trajectories
  per iteration). v1 sampled which side is controlled per-episode; this
  is strictly more data-efficient with the same policy.
* **Advantage**: per-episode reward-to-go with ``gamma=0.9`` followed
  by group-normalization (flatten RTGs across each group of
  ``GROUP_SIZE`` trajectories, normalize by flattened mean/std) — bit
  identical to v1's :func:`build_group_normalized_reward_to_go`.
* **PPO loss**: :func:`baseline.common.algos.ppo_loss` with
  ``value_clip=None`` and ``normalize_advantages=False`` (we already
  normalized above) — recovers v1's clipped-surrogate policy loss.
* **Evaluation**: :class:`PolicyEvaluator` with ``max_workers=N_eval``.
* **Plugins / observers / runtime**: imported unchanged from
  ``standing_grpo_rtg_tune`` so we are bit-compatible on the env side.

The v1 script remains untouched (the user's "legacy scripts left alone"
rule from DESIGN.md §4). Runs produced by v2 live under
``runs/standing_grpo_rtg_tune_v2_<timestamp>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

from baseline.common.algos import ppo_loss, compute_returns_to_go
from baseline.common.eval import PolicyEvaluator
from baseline.common.policies import (
    TanhGaussianMLPPolicy,
    TorchPolicyAdapter,
    export_actor_policy_artifacts,
)
from baseline.common.rollout import RolloutBatch, RolloutCollector

from envs.framework import (
    BaseObserverPlugin,
    EnvRuntime,
    ReadOnlySimContext,
)
from envs.humanoid21 import Humanoid21Observer, MujocoCombatSimulator

# Reuse hyperparameters + the (correct) termination plugin from v1.
# We INTENTIONALLY skip ``build_runtime`` and ``StandingRewardObserver``
# from v1 — see the bug-fix block below.
from baseline.humanoid21.standing_grpo_rtg_tune import (  # noqa: E402
    ACTION_DIM,
    ACTOR_HIDDEN_DIM,
    CLIP_EPS,
    CONTROL_FREQUENCY,
    ENTROPY_COEF,
    EPISODES_PER_UPDATE,
    EVAL_EPISODES,
    EVAL_INTERVAL,
    FALL_GRACE_STEPS,
    FALL_HEIGHT_THRESHOLD,
    FALL_UPRIGHT_THRESHOLD,
    GRAD_CLIP_NORM,
    GROUP_SIZE,
    HEIGHT_FULL_PENALTY_DELTA,
    INITIAL_DISTANCE,
    JOINT_POSE_FULL_PENALTY_MEAN_ABS,
    JOINT_VEL_FULL_PENALTY_MEAN_ABS,
    LEARNING_RATE,
    LOG_STD_MAX,
    LOG_STD_MIN,
    MAX_STEPS,
    MAX_UPDATES,
    MATCH_DURATION_SECONDS,
    MINIBATCH_SIZE,
    OBS_DIM,
    POSTURE_SCORE_VERBOSE,
    POSTURE_SCORE_VERBOSE_AGENT,
    POSTURE_SCORE_VERBOSE_STRIDE,
    ROLLOUT_INITIAL_DISTANCE_MAX,
    ROLLOUT_INITIAL_DISTANCE_MIN,
    ROOT_XY_FULL_PENALTY_DISTANCE,
    RTG_GAMMA,
    SEED,
    STANDING_SCORE_MAX,
    TARGET_HEIGHT,
    TARGET_KL,
    UPDATE_EPOCHS,
    UPRIGHT_FULL_PENALTY_COSINE,
    UPRIGHT_TILT_FULL_PENALTY_DEGREES,
    StandingTerminationPlugin,
    set_seed,
)


# ---------------------------------------------------------------------------
# BUG FIX (critical!): StandingRewardObserver vs framework hook names
# ---------------------------------------------------------------------------
# ``baseline/humanoid21/standing_*.StandingRewardObserver`` overrides
# ``on_reset`` and ``on_post_step`` to populate its output. These are NOT
# framework hook names — ``_ObserverDispatcherPlugin`` only dispatches
# ``on_pre_episode`` / ``on_post_action_step`` / ``on_post_episode``
# (see ``envs/framework/observer_plugin.py``). So the v1 reward observer
# silently returns its initial ``self._output = 0.0`` *for every step of
# every episode*. Rewards ≡ 0 → RTGs ≡ 0 → group-normalized advantages
# are 0/0 → ``ppo_loss`` returns exactly zero policy_loss → only the
# entropy bonus moves any weights. This matches the pathology observed
# at update=34: policy_loss=0, ratio=1, approx_kl=0, entropy growing.
#
# Per the "leave ``standing_*.py`` alone" rule (DESIGN.md §4 + user
# directive), we do NOT patch v1. v2 ships its own observer below.
# ---------------------------------------------------------------------------
class StandingRewardObserverV2(BaseObserverPlugin):
    """Per-step standing reward = delta of a posture-score function.

    Bit-identical math to v1's ``StandingRewardObserver._compute_reward_terms``
    but wired to the framework's actual dispatch hooks
    (``on_pre_episode`` / ``on_post_action_step``).
    """

    def __init__(
        self,
        agent_id: str,
        verbose: bool = False,
        verbose_stride: int = 1,
    ) -> None:
        self.agent_id = str(agent_id)
        self.verbose = bool(verbose)
        self.verbose_stride = max(1, int(verbose_stride))
        self._output: float = 0.0
        self._reference_root_xy: Optional[np.ndarray] = None
        self._reference_joint_pos: Optional[np.ndarray] = None
        self._previous_total_score: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )
        self._reference_root_xy = np.asarray(
            core_state["root_pos"][:2], dtype=np.float32
        ).copy()
        self._reference_joint_pos = np.asarray(
            core_state["joint_pos_norm"], dtype=np.float32
        ).copy()
        self._previous_total_score = self._compute_posture_score(
            ctx, height=height, uprightness=uprightness,
        )
        # First step has no prior delta — reward at t=0 is 0 (canonical).
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )
        total_score = self._compute_posture_score(
            ctx, height=height, uprightness=uprightness,
        )
        reward = total_score - self._previous_total_score
        self._previous_total_score = total_score
        self._output = float(reward)

    def get_output(self) -> float:
        return float(self._output)

    def _compute_posture_score(
        self,
        ctx: ReadOnlySimContext,
        *,
        height: float,
        uprightness: float,
    ) -> float:
        """Bit-identical to v1's ``_compute_posture_terms['total_score']``."""
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        root_xy = np.asarray(core_state["root_pos"][:2], dtype=np.float32)
        joint_pos = np.asarray(core_state["joint_pos_norm"], dtype=np.float32)
        ref_root_xy = (
            root_xy if self._reference_root_xy is None else self._reference_root_xy
        )
        ref_joint_pos = (
            joint_pos if self._reference_joint_pos is None else self._reference_joint_pos
        )
        root_xy_distance = float(np.linalg.norm(root_xy - ref_root_xy))
        joint_pose_mean_abs = float(np.mean(np.abs(joint_pos - ref_joint_pos)))
        joint_velocity_mean_abs = float(
            np.mean(np.abs(np.asarray(core_state["joint_vel_norm"], dtype=np.float32)))
        )
        height_deficit = max(0.0, TARGET_HEIGHT - height)
        tilt_angle_degrees = float(
            np.degrees(float(np.arccos(np.clip(uprightness, -1.0, 1.0))))
        )
        height_penalty = (height_deficit / HEIGHT_FULL_PENALTY_DELTA) ** 2
        uprightness_penalty = (
            tilt_angle_degrees / UPRIGHT_TILT_FULL_PENALTY_DEGREES
        ) ** 2
        root_xy_penalty = (root_xy_distance / ROOT_XY_FULL_PENALTY_DISTANCE) ** 2
        joint_pose_penalty = (
            joint_pose_mean_abs / JOINT_POSE_FULL_PENALTY_MEAN_ABS
        ) ** 2
        joint_velocity_penalty = (
            joint_velocity_mean_abs / JOINT_VEL_FULL_PENALTY_MEAN_ABS
        ) ** 2
        total_penalty = (
            height_penalty + uprightness_penalty + root_xy_penalty
            + joint_pose_penalty + joint_velocity_penalty
        )
        return float(STANDING_SCORE_MAX - total_penalty)


def build_runtime_v2() -> EnvRuntime:
    """Same humanoid21 runtime as v1's ``build_runtime`` but with the
    corrected :class:`StandingRewardObserverV2` wired into the reward
    slots. Everything else is identical."""
    simulator = MujocoCombatSimulator(initial_distance=INITIAL_DISTANCE)
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / CONTROL_FREQUENCY)))
    observer_plugins = {
        "robot_a_obs": Humanoid21Observer("robot_a"),
        "robot_b_obs": Humanoid21Observer("robot_b"),
        "robot_a_reward": StandingRewardObserverV2(
            agent_id="robot_a",
            verbose=POSTURE_SCORE_VERBOSE and POSTURE_SCORE_VERBOSE_AGENT in {"robot_a", "all"},
            verbose_stride=POSTURE_SCORE_VERBOSE_STRIDE,
        ),
        "robot_b_reward": StandingRewardObserverV2(
            agent_id="robot_b",
            verbose=POSTURE_SCORE_VERBOSE and POSTURE_SCORE_VERBOSE_AGENT in {"robot_b", "all"},
            verbose_stride=POSTURE_SCORE_VERBOSE_STRIDE,
        ),
    }
    runtime = EnvRuntime(
        simulator=simulator,
        observer_plugins=observer_plugins,
        plugins=[
            StandingTerminationPlugin(
                agent_id="robot_a",
                fall_height_threshold=FALL_HEIGHT_THRESHOLD,
                fall_upright_threshold=FALL_UPRIGHT_THRESHOLD,
                fall_grace_steps=FALL_GRACE_STEPS,
            ),
            StandingTerminationPlugin(
                agent_id="robot_b",
                fall_height_threshold=FALL_HEIGHT_THRESHOLD,
                fall_upright_threshold=FALL_UPRIGHT_THRESHOLD,
                fall_grace_steps=FALL_GRACE_STEPS,
            ),
        ],
        phy_steps_per_action=phy_steps_per_action,
        max_steps=MAX_STEPS,
    )
    runtime.observation_space = Humanoid21Observer.get_observation_space()
    runtime.action_space = Humanoid21Observer.get_action_space()
    return runtime

RUNS_DIR = Path(__file__).resolve().parent / "runs"
ROLLOUT_WORKERS = max(1, int(os.environ.get(
    "STANDING_ROLLOUT_WORKERS",
    str(min(64, max(1, (os.cpu_count() or 1) // 2))),
)))
EVAL_WORKERS = max(1, int(os.environ.get(
    "STANDING_EVAL_WORKERS", str(min(ROLLOUT_WORKERS, EVAL_EPISODES))
)))


# ---------------------------------------------------------------------------
# Picklable top-level factories (for RolloutCollector workers).
# ---------------------------------------------------------------------------
def _make_runtime():
    """Top-level runtime factory — uses ``build_runtime_v2`` with the
    corrected reward observer (see class-level note on
    :class:`StandingRewardObserverV2`)."""
    return build_runtime_v2()


def _make_adapter() -> TorchPolicyAdapter:
    """Top-level policy factory.

    Each collector worker gets its own adapter built on a fresh
    ``TanhGaussianMLPPolicy``. The main process pushes the live actor
    state_dict into each ``collect()`` call via ``state_dicts=...``,
    which the worker applies before running its episode chunk.
    """
    actor = TanhGaussianMLPPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=ACTOR_HIDDEN_DIM,
        log_std_min=LOG_STD_MIN,
        log_std_max=LOG_STD_MAX,
    )
    return TorchPolicyAdapter(actor=actor, device="cpu", deterministic=False)


def _options_fn(episode_index: int) -> Dict[str, Any]:
    """Per-episode initial-distance sampling (threaded through ``options``).

    Deterministic per ``episode_index`` so sequential / parallel collects
    produce the same per-seed trajectories. Matches v1's
    :func:`_sample_rollout_setup` in distribution.
    """
    rng = np.random.default_rng(int(episode_index) + 2024)
    return {
        "initial_distance": float(
            rng.uniform(ROLLOUT_INITIAL_DISTANCE_MIN, ROLLOUT_INITIAL_DISTANCE_MAX)
        ),
    }


# ---------------------------------------------------------------------------
# Advantage: per-episode RTG + group-flatten normalization.
# ---------------------------------------------------------------------------
def _build_group_normalized_rtg(
    trajectories: Sequence[RolloutBatch],
    group_size: int,
    gamma: float,
) -> List[np.ndarray]:
    """Reproduce v1's ``build_group_normalized_reward_to_go`` exactly.

    For each contiguous group of ``group_size`` trajectories:
      1. compute per-step RTG with discount ``gamma`` (bootstrap = 0;
         episodes that terminated *or* truncated are both treated as
         complete here — matches v1 behavior);
      2. concatenate the per-step RTGs across the whole group into a
         single 1-D buffer;
      3. normalize each step by the *group-flatten* mean and std:
         ``adv_t = (rtg_t - group_mean) / (group_std + 1e-6)``.

    Returns per-trajectory advantage arrays aligned 1:1 with
    ``trajectories``.
    """
    advantages: List[np.ndarray] = []
    for start in range(0, len(trajectories), group_size):
        group = trajectories[start:start + group_size]
        if not group:
            continue
        group_rtgs = [
            compute_returns_to_go(t.rewards, gamma=gamma, last_value=0.0)
            for t in group
        ]
        flat = np.concatenate(group_rtgs, axis=0)
        mean = float(flat.mean())
        std = float(flat.std())
        denom = std + 1e-6
        for rtg in group_rtgs:
            advantages.append(((rtg - mean) / denom).astype(np.float32))
    return advantages


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
@dataclass
class _UpdateStats:
    policy_loss: float
    entropy: float
    ratio: float
    approx_kl: float
    optimizer_steps: int
    early_stop: int
    early_stop_kl: float


class StandingGRPOTrainerV2:
    """GRPO-RTG standing trainer on the ``baseline/common/`` stack."""

    def __init__(
        self,
        *,
        device: torch.device,
        resume_from: Optional[Path] = None,
        max_updates: int = MAX_UPDATES,
        rollout_workers: int = ROLLOUT_WORKERS,
        eval_workers: int = EVAL_WORKERS,
        episodes_per_update: int = EPISODES_PER_UPDATE,
        eval_episodes: int = EVAL_EPISODES,
    ) -> None:
        self.device = device
        self.max_updates = int(max_updates)
        self.episodes_per_update = int(episodes_per_update)
        self.eval_episodes = int(eval_episodes)
        self.rollout_workers = max(1, int(rollout_workers))
        self.eval_workers = max(1, int(eval_workers))

        # Main-process actor: the single source of truth. Optimizer
        # lives here; workers receive snapshots via state_dict broadcast.
        self.actor = TanhGaussianMLPPolicy(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=ACTOR_HIDDEN_DIM,
            log_std_min=LOG_STD_MIN,
            log_std_max=LOG_STD_MAX,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        self.best_eval_reward = -float("inf")
        self.history: List[Dict[str, Any]] = []
        self.run_dir = RUNS_DIR / f"standing_grpo_rtg_tune_v2_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.policy_dir = self.run_dir / "policy"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.resume_from = resume_from.resolve() if resume_from is not None else None
        if self.resume_from is not None:
            self._load_checkpoint(self.resume_from)

        # Lazy — built on first train()/evaluate() so CLI --help stays cheap.
        self._collector: Optional[RolloutCollector] = None
        self._evaluator: Optional[PolicyEvaluator] = None
        self._save_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self) -> None:
        try:
            self._build_collector()
            for update_index in range(1, self.max_updates + 1):
                trajectories = self._collect(update_index=update_index, deterministic=False)
                stats = self._update_actor(trajectories)
                mean_reward = float(np.mean([float(t.rewards.sum()) for t in trajectories]))
                mean_length = float(np.mean([int(t.num_steps) for t in trajectories]))
                record: Dict[str, Any] = {
                    "update": update_index,
                    "train_mean_reward": mean_reward,
                    "train_mean_length": mean_length,
                    "policy_loss": stats.policy_loss,
                    "entropy": stats.entropy,
                    "ratio": stats.ratio,
                    "approx_kl": stats.approx_kl,
                    "optimizer_steps": stats.optimizer_steps,
                    "early_stop": stats.early_stop,
                    "early_stop_kl": stats.early_stop_kl,
                }
                if update_index % EVAL_INTERVAL == 0:
                    eval_stats = self._evaluate()
                    record.update({f"eval_{k}": v for k, v in eval_stats.items()})
                    if eval_stats["mean_reward"] > self.best_eval_reward:
                        self.best_eval_reward = float(eval_stats["mean_reward"])
                        self._save_checkpoint(self.run_dir / "best_model.pt")
                        self._export_policy(self.policy_dir, self.run_dir / "best_model.pt")
                self.history.append(record)
                self._print_record(record)
                if update_index % EVAL_INTERVAL == 0:
                    self._write_history()
                if update_index % 25 == 0:
                    self._save_checkpoint(self.checkpoint_dir / f"update_{update_index}.pt")
            final_path = self.run_dir / "final_model.pt"
            self._save_checkpoint(final_path)
            if not self.policy_dir.exists():
                self._export_policy(self.policy_dir, final_path)
            self._write_history()
        finally:
            self.close()

    def close(self) -> None:
        if self._collector is not None:
            self._collector.close()
            self._collector = None
        if self._evaluator is not None:
            self._evaluator.close()
            self._evaluator = None

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def _build_collector(self) -> None:
        if self._collector is not None:
            return
        self._collector = RolloutCollector(
            runtime_factory=_make_runtime,
            policy_factories={
                "robot_a": _make_adapter,
                "robot_b": _make_adapter,
            },
            capture_agents=("robot_a", "robot_b"),
            max_workers=self.rollout_workers,
        )

    def _snapshot_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}

    def _collect(self, *, update_index: int, deterministic: bool) -> List[RolloutBatch]:
        """Collect ``episodes_per_update`` episodes; return flattened
        list of per-agent trajectories (2 × episodes — both sides captured).
        """
        assert self._collector is not None
        sd = self._snapshot_state_dict()
        base_seed = SEED + update_index * self.episodes_per_update
        batches = self._collector.collect(
            n=self.episodes_per_update,
            base_seed=base_seed,
            options_fn=_options_fn,
            deterministic=deterministic,
            state_dicts={"robot_a": sd, "robot_b": sd},
        )
        # Flatten {agent: [batch, ...]} → [batch, ...] preserving per-agent
        # order then cross-agent order. Group-normalization downstream
        # treats contiguous GROUP_SIZE slices as one group, which matches
        # v1 (v1 groups the 256 controlled-side episodes directly).
        merged: List[RolloutBatch] = []
        for agent in ("robot_a", "robot_b"):
            merged.extend(batches.get(agent, []))
        return merged

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _build_evaluator(self) -> None:
        if self._evaluator is not None:
            return
        self._evaluator = PolicyEvaluator(
            runtime_factory=_make_runtime,
            policy_factories={
                "robot_a": _make_adapter,
                "robot_b": _make_adapter,
            },
            capture_agents=("robot_a",),
            deterministic=True,
        )

    def _evaluate(self) -> Dict[str, float]:
        self._build_evaluator()
        assert self._evaluator is not None
        sd = self._snapshot_state_dict()
        report = self._evaluator.evaluate(
            n=self.eval_episodes,
            base_seed=SEED + 100000,
            state_dicts={"robot_a": sd, "robot_b": sd},
            options_fn=_options_fn,
        )
        stats_a = report.per_agent["robot_a"]
        return {
            "mean_reward": float(stats_a["return"].mean),
            "mean_length": float(stats_a["length"].mean),
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def _update_actor(self, trajectories: Sequence[RolloutBatch]) -> _UpdateStats:
        """GRPO-RTG update: per-episode RTG → group-flatten normalize →
        PPO clipped surrogate. No value head (advantage is already a
        normalized return target, not an advantage over a critic)."""
        adv_sequences = _build_group_normalized_rtg(trajectories, GROUP_SIZE, RTG_GAMMA)

        obs_batch = np.concatenate([t.obs[:-1] for t in trajectories], axis=0)
        action_batch = np.concatenate([t.actions for t in trajectories], axis=0)
        log_prob_batch = np.concatenate([
            t.log_probs if t.log_probs is not None
            else np.zeros(t.num_steps, dtype=np.float32)
            for t in trajectories
        ], axis=0)
        adv_batch = np.concatenate(adv_sequences, axis=0)

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(action_batch, dtype=torch.float32, device=self.device)
        old_lp_t = torch.as_tensor(log_prob_batch, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv_batch, dtype=torch.float32, device=self.device)

        total_steps = obs_t.shape[0]
        policy_losses: List[float] = []
        entropies: List[float] = []
        ratios: List[float] = []
        approx_kls: List[float] = []
        optimizer_steps = 0
        early_stop = False
        early_stop_kl = 0.0

        for _epoch in range(UPDATE_EPOCHS):
            perm = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, MINIBATCH_SIZE):
                idx = perm[start:start + MINIBATCH_SIZE]
                mb_obs = obs_t[idx]
                mb_act = act_t[idx]
                mb_old_lp = old_lp_t[idx]
                mb_adv = adv_t[idx]

                new_lp, entropy = self.actor.evaluate_actions(mb_obs, mb_act)
                # Preview KL to decide early-stop WITHOUT taking the step.
                with torch.no_grad():
                    approx_kl = float((mb_old_lp - new_lp).mean().item())
                approx_kls.append(approx_kl)
                if TARGET_KL > 0.0 and approx_kl > TARGET_KL:
                    early_stop = True
                    early_stop_kl = approx_kl
                    break

                # GRPO has no critic — feed zero value / return tensors
                # and value_coef=0 so the value-loss term is wired in
                # but contributes nothing to gradients. (ppo_loss's
                # shape-validator treats None as "absent", but
                # value_loss computation itself requires concrete
                # tensors; see ppo_loss value_loss branch.)
                zeros = torch.zeros_like(new_lp)
                out = ppo_loss(
                    log_probs_old=mb_old_lp,
                    log_probs_new=new_lp,
                    advantages=mb_adv,
                    values_old=zeros,
                    values_new=zeros,
                    returns=zeros,
                    entropy=entropy,
                    clip_range=CLIP_EPS,
                    value_coef=0.0,
                    entropy_coef=ENTROPY_COEF,
                    value_clip=None,
                    normalize_advantages=False,
                )
                self.optimizer.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
                self.optimizer.step()

                optimizer_steps += 1
                policy_losses.append(float(out.policy_loss))
                entropies.append(float(entropy.mean().item()))
                with torch.no_grad():
                    ratios.append(float(torch.exp(new_lp - mb_old_lp).mean().item()))
            if early_stop:
                break

        return _UpdateStats(
            policy_loss=float(np.mean(policy_losses)) if policy_losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            ratio=float(np.mean(ratios)) if ratios else 0.0,
            approx_kl=float(np.mean(approx_kls)) if approx_kls else 0.0,
            optimizer_steps=int(optimizer_steps),
            early_stop=int(early_stop),
            early_stop_kl=float(early_stop_kl),
        )

    # ------------------------------------------------------------------
    # Checkpoint / export / logging
    # ------------------------------------------------------------------
    def _save_checkpoint(self, path: Path) -> None:
        payload = {
            "algorithm": "grpo_rtg_v2",
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": ACTOR_HIDDEN_DIM,
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, path)

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device)
        if int(payload.get("obs_dim", OBS_DIM)) != OBS_DIM:
            raise ValueError(
                f"obs_dim mismatch: expected {OBS_DIM}, got {payload.get('obs_dim')}"
            )
        if int(payload.get("action_dim", ACTION_DIM)) != ACTION_DIM:
            raise ValueError(
                f"action_dim mismatch: expected {ACTION_DIM}, got {payload.get('action_dim')}"
            )
        self.actor.load_state_dict(payload["state_dict"])
        opt_state = payload.get("optimizer_state_dict")
        if opt_state is not None:
            with suppress(ValueError):
                self.optimizer.load_state_dict(opt_state)
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(self.device)

    def _export_policy(self, policy_dir: Path, model_path: Path) -> None:
        # Use the framework's canonical actor-export path — same artifact
        # layout as v1 (model.pt + policy.py) so downstream eval tooling
        # is unchanged.
        export_actor_policy_artifacts(
            actor=self.actor,
            policy_dir=policy_dir,
            extra_payload={"algorithm": "grpo_rtg_v2"},
        )

    def _save_config(self) -> None:
        cfg = {
            "algorithm": "grpo_rtg_v2",
            "framework": "baseline.common (PR1-10)",
            "control_frequency": CONTROL_FREQUENCY,
            "match_duration_seconds": MATCH_DURATION_SECONDS,
            "max_steps": MAX_STEPS,
            "initial_distance": INITIAL_DISTANCE,
            "rollout_initial_distance_min": ROLLOUT_INITIAL_DISTANCE_MIN,
            "rollout_initial_distance_max": ROLLOUT_INITIAL_DISTANCE_MAX,
            "group_size": GROUP_SIZE,
            "episodes_per_update": self.episodes_per_update,
            "update_epochs": UPDATE_EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "max_updates": self.max_updates,
            "eval_interval": EVAL_INTERVAL,
            "eval_episodes": self.eval_episodes,
            "learning_rate": LEARNING_RATE,
            "clip_eps": CLIP_EPS,
            "entropy_coef": ENTROPY_COEF,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "target_kl": TARGET_KL,
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "rtg_gamma": RTG_GAMMA,
            "advantage_mode": "reward_to_go_group_normalized",
            "rollout_workers": self.rollout_workers,
            "eval_workers": self.eval_workers,
            "seed": SEED,
            "resume_from": str(self.resume_from) if self.resume_from else None,
        }
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(cfg, handle, ensure_ascii=False, indent=2)

    def _write_history(self) -> None:
        with (self.run_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _print_record(record: Dict[str, Any]) -> None:
        keys = [
            "update", "train_mean_reward", "train_mean_length",
            "policy_loss", "entropy", "ratio", "approx_kl", "optimizer_steps",
        ]
        if record.get("early_stop"):
            keys.extend(["early_stop", "early_stop_kl"])
        if "eval_mean_reward" in record:
            keys.extend(["eval_mean_reward", "eval_mean_length"])
        msg = " | ".join(
            f"{k}={record[k]:.4f}" if isinstance(record[k], float) else f"{k}={record[k]}"
            for k in keys if k in record
        )
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--max-updates", type=int, default=MAX_UPDATES,
        help="Override MAX_UPDATES (default %(default)s).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Short smoke run: max_updates=2, episodes_per_update=16, "
             "eval_episodes=4, rollout_workers=2. Useful for end-to-end "
             "verification of the v2 pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs: Dict[str, Any] = {
        "device": device,
        "resume_from": args.resume_from,
        "max_updates": args.max_updates,
    }
    if args.smoke:
        kwargs.update(
            max_updates=2,
            episodes_per_update=16,
            eval_episodes=4,
            rollout_workers=2,
            eval_workers=2,
        )

    trainer = StandingGRPOTrainerV2(**kwargs)
    trainer.train()
    print(f"run_dir={trainer.run_dir}", flush=True)
    print(f"policy_dir={trainer.policy_dir}", flush=True)


if __name__ == "__main__":
    main()
