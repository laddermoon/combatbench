"""PPO buffer and update for ExperimentV2.

Clean rewrite of ppo_trainer.py for the V2 experiment interface.

Design intent
-------------
This module implements the *mechanics* of PPO with multi-channel
advantages. It is deliberately experiment-agnostic: it knows nothing
about reward semantics, phase segmentation, or curriculum scheduling.

The core idea is **per-channel critics + confidence-weighted advantage
combination**. Each reward channel has its own critic (V_c(s)), its own
GAE computation, and its own z-score normalized advantage. The combined
advantage that drives the actor is:

    combined_adv = Σ_c  actor_weight_c × confidence_c × norm_adv_c

where:
- ``actor_weight_c`` is set by the experiment (curriculum control)
- ``confidence_c = clip(EV_c, 0, 1)^0.5`` down-weights channels whose
  critic is inaccurate (low explained variance)
- ``norm_adv_c`` is z-score normalized on active frames only

This design lets experiments decompose complex reward structures into
independent channels (e.g. survival, posture, height) without worrying
about scale mismatches — normalization handles that. The experiment's
job is to decide *which channels exist* and *how much each matters*
(actor_weight); the framework handles the rest.

Key differences from v1
-----------------------
- Buffer consumes ``List[Trajectory]`` directly — no v1 segment/legacy adapters.
- ``reward_keys`` passed explicitly by the training loop (from
  ``experiment.reward_channels()``), ensuring consistency between
  buffer and update.
- No ``episode_metrics`` / ``episode_lengths`` — eval is ``on_eval()``'s job.
- No ``batch_summary()`` / ``reward_summary()`` — logging is the loop's job.
- ``ppo_update_v2`` takes ``RewardChannel`` tuple + ``PPOParams`` directly.
- No ``experiment`` parameter — no optional overrides for normalization,
  combination, or sample weight scaling.  All defaults are fixed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from baseline.common.algos import compute_gae

from .experiment_v2 import PPOParams, TrainablePolicy
from .trajectory import RewardChannel, Trajectory


# ---------------------------------------------------------------------------
# Seeding — single entry point for reproducible training.
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set numpy + torch (CPU/CUDA) seeds in one call."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# PPO buffer — flat numpy arrays from List[Trajectory]
#
# The buffer flattens a list of variable-length trajectories into
# contiguous numpy arrays for batched GAE and minibatch updates.
#
# Key design: a single batched ``actor.evaluate_actions`` call computes
# log_probs for ALL frames across ALL trajectories at once, then slices
# the results back. This is far more efficient than per-trajectory calls.
#
# Per-channel data is tracked at the *segment* level (one entry per
# trajectory per channel): active flag, is_terminated, actor_weight,
# and reward array. Channels absent from a trajectory are marked
# inactive with zero rewards — they contribute nothing to GAE or
# advantage combination but still occupy array slots for uniform
# indexing.
# ---------------------------------------------------------------------------

class PPOBufferV2:
    """PPO buffer built from a list of :class:`Trajectory` objects.

    The buffer performs a single batched ``actor.evaluate_actions`` call
    to fill ``log_prob`` for all frames, then slices the results back
    into per-trajectory segments.

    All per-channel data (rewards, termination, actor_weight) comes from
    ``Trajectory.channels``.  Channels absent from a trajectory are
    marked inactive for that segment.

    Layout (per channel, parallel lists indexed by trajectory):
    - ``reward_data[key]``       — reward arrays, one per trajectory
    - ``key_seg_active[key]``    — whether this channel is active per traj
    - ``key_seg_terminated[key]``— whether this channel is terminated per traj
    - ``key_seg_actor_weight[key]`` — scalar or ``(T,)`` array aw per traj

    Flat arrays (concatenated across trajectories):
    - ``obs``, ``actions``, ``log_probs``, ``sample_weights``, ``frame_modes``
    - ``final_obs`` — per-trajectory last observation (for bootstrap)
    - ``ep_lengths`` — per-trajectory frame count
    """

    def __init__(
        self,
        trajectories: List[Trajectory],
        actor: TrainablePolicy,
        device: torch.device,
        reward_keys: Tuple[str, ...],
    ):
        self.reward_keys: Tuple[str, ...] = reward_keys

        self.reward_data: Dict[str, List[np.ndarray]] = {
            k: [] for k in reward_keys
        }
        self.key_seg_active: Dict[str, List[bool]] = {
            k: [] for k in reward_keys
        }
        self.key_seg_terminated: Dict[str, List[bool]] = {
            k: [] for k in reward_keys
        }
        self.key_seg_actor_weight: Dict[str, List[Union[float, np.ndarray]]] = {
            k: [] for k in reward_keys
        }

        if not trajectories:
            self.obs = np.zeros((0,), np.float32)
            self.actions = np.zeros((0,), np.float32)
            self.log_probs = np.zeros(0, dtype=np.float32)
            self.sample_weights = np.zeros(0, dtype=np.float32)
            self.frame_modes: Optional[np.ndarray] = None
            self.final_obs: List[np.ndarray] = []
            self.ep_lengths: List[int] = []
            return

        # --- Batched evaluate_actions on all trajectory frames ---
        # Concatenate all observations and actions, run a single forward
        # pass through the actor to get log_probs, then slice back.
        # This is the most expensive call in buffer construction; batching
        # is essential for GPU efficiency.
        all_obs = np.concatenate(
            [t.obs for t in trajectories], axis=0,
        ).astype(np.float32)
        all_acts = np.concatenate(
            [t.actions for t in trajectories], axis=0,
        ).astype(np.float32)
        all_obs_t = torch.as_tensor(all_obs, dtype=torch.float32, device=device)
        all_acts_t = torch.as_tensor(all_acts, dtype=torch.float32, device=device)

        any_mode = any(t.mode is not None for t in trajectories)
        kwargs: Dict[str, Any] = {}
        if any_mode:
            all_modes = np.concatenate([
                np.full(len(t.obs), float(t.mode) if t.mode is not None else 1.0,
                        dtype=np.float32)
                for t in trajectories
            ])
            kwargs["frame_modes"] = torch.as_tensor(
                all_modes, dtype=torch.float32, device=device,
            )

        with torch.no_grad():
            all_lp, _ = actor.evaluate_actions(all_obs_t, all_acts_t, **kwargs)
        all_lp_np = all_lp.cpu().numpy().astype(np.float32)

        # --- Slice log_probs back into per-trajectory segments ---
        # Walk through trajectories in order, extracting the corresponding
        # log_prob slice for each. Also extract per-channel data from
        # Trajectory.channels, marking absent channels as inactive.
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        lp_list: List[np.ndarray] = []
        fin_list: List[np.ndarray] = []
        weight_list: List[np.ndarray] = []
        ep_lens: List[int] = []
        modes_list: List[np.ndarray] = []

        offset = 0
        for traj in trajectories:
            T_seg = len(traj.obs)
            if T_seg == 0:
                continue

            obs_seg = np.asarray(traj.obs, dtype=np.float32)
            acts_seg = np.asarray(traj.actions, dtype=np.float32)
            lp_seg = all_lp_np[offset:offset + T_seg]
            offset += T_seg
            mode_seg = np.full(
                T_seg, float(traj.mode) if traj.mode is not None else 1.0,
                dtype=np.float32,
            )

            # Per-key data from trajectory channels.
            # If a channel key is present in traj.channels, it is active
            # for this trajectory segment. Otherwise it is inactive
            # (zero rewards, terminated=True, actor_weight=0).
            for key in reward_keys:
                if key in traj.channels:
                    cd = traj.channels[key]
                    self.key_seg_active[key].append(True)
                    self.key_seg_terminated[key].append(cd.is_terminated)
                    self.key_seg_actor_weight[key].append(cd.actor_weight)
                    self.reward_data[key].append(
                        np.asarray(cd.reward, dtype=np.float32)
                    )
                else:
                    self.key_seg_active[key].append(False)
                    self.key_seg_terminated[key].append(True)
                    self.key_seg_actor_weight[key].append(0.0)
                    self.reward_data[key].append(
                        np.zeros(T_seg, dtype=np.float32)
                    )

            obs_list.append(obs_seg)
            act_list.append(acts_seg)
            lp_list.append(lp_seg)
            fin_list.append(np.asarray(traj.last_obs, dtype=np.float32))
            weight_list.append(
                np.full(T_seg, traj.importance, dtype=np.float32)
            )
            ep_lens.append(T_seg)
            modes_list.append(mode_seg)

        if not ep_lens:
            self.obs = np.zeros((0,), np.float32)
            self.actions = np.zeros((0,), np.float32)
            self.log_probs = np.zeros(0, dtype=np.float32)
            self.sample_weights = np.zeros(0, dtype=np.float32)
            self.frame_modes = None
            self.final_obs = []
            self.ep_lengths = []
            return

        self.obs = np.concatenate(obs_list, axis=0)
        self.actions = np.concatenate(act_list, axis=0)
        self.log_probs = np.concatenate(lp_list, axis=0)
        self.sample_weights = np.concatenate(weight_list, axis=0)
        self.frame_modes = np.concatenate(modes_list, axis=0) if any_mode else None
        self.final_obs = fin_list
        self.ep_lengths = ep_lens

    def __len__(self) -> int:
        return sum(self.ep_lengths)

    def is_empty(self) -> bool:
        return len(self.ep_lengths) == 0

    def buffer_stats(self) -> Dict[str, Any]:
        """Comprehensive buffer statistics for logging.

        Global stats:
        - n_trajectories, total_steps, traj_len_mean/min/max

        Per-channel stats (active segments only):
        - n_active_trajs, active_ratio
        - traj_len_min/max/mean (length of active trajectories)
        - reward_min/max/mean/std (over all active frames)
        - actor_weight_mean/min/max
        """
        if not self.ep_lengths:
            return {
                "n_trajectories": 0,
                "total_steps": 0,
                "traj_len_mean": 0.0,
                "traj_len_min": 0,
                "traj_len_max": 0,
                "per_channel": {},
            }

        ep_lens = np.array(self.ep_lengths)
        total_steps = int(ep_lens.sum())

        per_channel: Dict[str, Dict[str, float]] = {}
        for key in self.reward_keys:
            active_flags = self.key_seg_active[key]
            aws = self.key_seg_actor_weight[key]
            segs = self.reward_data[key]

            active_indices = [i for i, a in enumerate(active_flags) if a]
            n_active = len(active_indices)
            n_total = len(active_flags)

            if n_active == 0:
                per_channel[key] = {
                    "n_active_trajs": 0,
                    "active_ratio": 0.0,
                    "traj_len_min": 0,
                    "traj_len_max": 0,
                    "traj_len_mean": 0.0,
                    "reward_min": 0.0,
                    "reward_max": 0.0,
                    "reward_mean": 0.0,
                    "reward_std": 0.0,
                    "actor_weight_mean": 0.0,
                    "actor_weight_min": 0.0,
                    "actor_weight_max": 0.0,
                }
                continue

            active_lens = [self.ep_lengths[i] for i in active_indices]
            active_aw = [aws[i] for i in active_indices]
            active_segs = [segs[i] for i in active_indices]
            concat = np.concatenate(active_segs)
            concat_aw = np.concatenate([
                np.atleast_1d(np.asarray(aw, dtype=np.float32)) for aw in active_aw
            ])

            per_channel[key] = {
                "n_active_trajs": n_active,
                "active_ratio": n_active / n_total if n_total else 0.0,
                "traj_len_min": int(min(active_lens)),
                "traj_len_max": int(max(active_lens)),
                "traj_len_mean": float(np.mean(active_lens)),
                "reward_min": float(concat.min()) if concat.size else 0.0,
                "reward_max": float(concat.max()) if concat.size else 0.0,
                "reward_mean": float(concat.mean()) if concat.size else 0.0,
                "reward_std": float(concat.std()) if concat.size else 0.0,
                "actor_weight_mean": float(concat_aw.mean()) if concat_aw.size else 0.0,
                "actor_weight_min": float(concat_aw.min()) if concat_aw.size else 0.0,
                "actor_weight_max": float(concat_aw.max()) if concat_aw.size else 0.0,
            }

        return {
            "n_trajectories": len(self.ep_lengths),
            "total_steps": total_steps,
            "traj_len_mean": float(ep_lens.mean()),
            "traj_len_min": int(ep_lens.min()),
            "traj_len_max": int(ep_lens.max()),
            "per_channel": per_channel,
        }


# ---------------------------------------------------------------------------
# PPO update — fixed defaults, no experiment overrides
#
# The update function is a pure function of (actor, critics, buffer,
# hyperparameters). It does not call back into the experiment. This
# makes the PPO algorithm fully reproducible and testable in isolation.
#
# Pipeline:
#   1. Compute critic values V_c(s) for all frames, all channels
#   2. Compute bootstrap values V_c(s_terminal) for truncated segments
#   3. Per-channel GAE → advantages and returns
#   4. Explained variance → confidence weights
#   5. Combined advantage = Σ_c aw_c × conf_c × norm_adv_c
#   6. Multi-epoch minibatch PPO with clipped surrogate + value loss
#   7. Early stop on target_kl to prevent policy collapse
# ---------------------------------------------------------------------------

def _normalize_adv(adv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalization on active frames.  Inactive frames get zero.

    Normalization is per-channel and per-update: each channel's advantages
    are independently centered and scaled to unit std. This means the
    *absolute scale* of rewards across channels is irrelevant — only the
    *relative pattern* within each channel matters. The experiment
    controls cross-channel importance via actor_weight, not via reward
    magnitudes.

    Edge cases:
    - No active frames → all zeros (channel contributes nothing).
    - Zero variance (all advantages equal) → all zeros (no gradient signal).
    """
    active = adv[mask]
    if active.size == 0:
        return np.zeros_like(adv, dtype=np.float32)
    mean = float(active.mean())
    std = float(active.std())
    if std < 1e-8:
        return np.zeros_like(adv, dtype=np.float32)
    result = np.zeros_like(adv, dtype=np.float32)
    result[mask] = ((active - mean) / std).astype(np.float32)
    return result


def ppo_update_v2(
    actor: TrainablePolicy,
    critics: Dict[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    buf: PPOBufferV2,
    reward_channels: Tuple[RewardChannel, ...],
    pp: PPOParams,
    grad_clip_norm: float,
    device: torch.device,
    use_confidence: bool = True,
) -> Dict[str, float]:
    """Multi-critic PPO update with fixed defaults.

    - Advantage normalization: z-score on active frames (per-channel).
    - Advantage combination: ``Σ_c actor_weight_c * confidence_c * norm_adv_c``.
    - Sample weight normalization: divide by mean.
    - Early stop on target_kl.

    The actor sees a single combined advantage (not per-channel). Critics
    are updated independently on their own return targets. This decouples
    value learning (per-channel) from policy learning (combined).

    Args:
        actor: Trainable actor with ``evaluate_actions``.
        critics: One critic per reward channel.
        actor_optimizer, critic_optimizers: Optimizers.
        buf: PPO buffer with training data.
        reward_channels: Channel configs (name, gamma, gae_lambda).
        pp: PPO hyperparameters.
        grad_clip_norm: Max gradient norm.
        device: Torch device.
        use_confidence: If True, weight advantages by ``clip(EV, 0, 1)**0.5``.
            This down-weights channels whose critic has low explained variance,
            preventing noisy advantage estimates from destabilizing the actor.

    Returns:
        Stats dict for logging.
    """
    reward_keys = tuple(ch.name for ch in reward_channels)
    gammas = {ch.name: ch.gamma for ch in reward_channels}
    gae_lambdas = {ch.name: ch.gae_lambda for ch in reward_channels}

    obs_t = torch.as_tensor(buf.obs, dtype=torch.float32, device=device)

    # --- 1. Compute critic values V_c(s) for all frames ---
    # One forward pass per critic. Values are used for GAE (advantage)
    # and explained variance (confidence).
    values_all: Dict[str, np.ndarray] = {}
    for key in reward_keys:
        with torch.no_grad():
            values_all[key] = (
                critics[key](obs_t).reshape(-1).cpu().numpy().astype(np.float32)
            )

    # --- 2. Bootstrap values for truncated segments ---
    # A trajectory segment is "truncated" (not terminated) for a channel
    # if is_terminated=False. In that case, GAE needs V_c(s_next) as the
    # bootstrap value. We batch all truncated segments' final_obs into
    # a single forward pass per critic.
    bootstrap_indices: List[int] = []
    bootstrap_obs: List[np.ndarray] = []
    for i, T in enumerate(buf.ep_lengths):
        needs_boot = any(
            buf.key_seg_active[key][i] and not buf.key_seg_terminated[key][i]
            for key in reward_keys
        )
        if needs_boot and buf.final_obs[i] is not None:
            bootstrap_indices.append(i)
            bootstrap_obs.append(np.asarray(buf.final_obs[i], dtype=np.float32))

    bootstrap_values: Dict[str, np.ndarray] = {}
    bootstrap_pos: Dict[int, int] = {}
    if bootstrap_obs:
        boot_t = torch.as_tensor(
            np.stack(bootstrap_obs), dtype=torch.float32, device=device,
        )
        for key in reward_keys:
            with torch.no_grad():
                bootstrap_values[key] = (
                    critics[key](boot_t).reshape(-1).cpu().numpy().astype(np.float32)
                )
        bootstrap_pos = {ep_idx: pos for pos, ep_idx in enumerate(bootstrap_indices)}

    # --- Segment offsets ---
    # Precompute flat-array offsets for each trajectory segment so we
    # can slice per-trajectory data without repeated cumsum.
    seg_offsets: List[int] = []
    _off = 0
    for T in buf.ep_lengths:
        seg_offsets.append(_off)
        _off += T

    # --- Per-key frame-level active mask ---
    # Expand per-segment active flags into per-frame boolean masks.
    # These masks drive GAE (skip inactive segments), normalization
    # (only active frames), and critic loss (only active frames).
    n = sum(buf.ep_lengths)
    key_frame_mask: Dict[str, np.ndarray] = {}
    for key in reward_keys:
        mask = np.zeros(n, dtype=bool)
        for i, is_active in enumerate(buf.key_seg_active[key]):
            if is_active:
                s = seg_offsets[i]
                e = s + buf.ep_lengths[i]
                mask[s:e] = True
        key_frame_mask[key] = mask

    # --- 3. Per-channel GAE ---
    # For each channel, iterate over trajectory segments:
    # - Inactive segments → zero advantages and returns (no signal)
    # - Active terminated segments → last_value=0 (no bootstrap)
    # - Active truncated segments → last_value=V_c(s_next) (bootstrap)
    #
    # GAE uses per-channel gamma and lambda from RewardChannel config.
    advs_all: Dict[str, np.ndarray] = {}
    rets_all: Dict[str, np.ndarray] = {}

    for key in reward_keys:
        advs_list = []
        rets_list = []

        for i, T in enumerate(buf.ep_lengths):
            s = seg_offsets[i]
            values = values_all[key][s : s + T]

            if not buf.key_seg_active[key][i]:
                advs_list.append(np.zeros(T, dtype=np.float32))
                rets_list.append(np.zeros(T, dtype=np.float32))
                continue

            last_value = 0.0
            key_terminated = buf.key_seg_terminated[key][i]
            if not key_terminated and buf.final_obs[i] is not None and i in bootstrap_pos:
                last_value = float(bootstrap_values[key][bootstrap_pos[i]])

            rewards = buf.reward_data[key][i]
            adv, ret = compute_gae(
                rewards=rewards,
                values=values,
                last_value=last_value,
                gamma=gammas[key],
                lam=gae_lambdas[key],
            )
            advs_list.append(adv)
            rets_list.append(ret)

        advs_all[key] = np.concatenate(advs_list)
        rets_all[key] = np.concatenate(rets_list)
        mask = key_frame_mask[key]
        r_active = rets_all[key][mask]
        if r_active.size > 0:
            print(
                f"  {key}: return=[{r_active.min():+.3f}, {r_active.max():+.3f}] "
                f"mean={r_active.mean():+.3f} std={r_active.std():.3f} "
                f"(active={mask.sum()}/{len(mask)})",
                flush=True,
            )

    # --- 4. Explained variance per critic ---
    # EV = 1 - Var(y_true - y_pred) / Var(y_true)
    # Measures how well the critic predicts actual returns. Used to
    # compute confidence weights: low EV → low confidence → channel's
    # advantage is down-weighted in the combined signal.
    explained_variances: Dict[str, float] = {}
    for key in reward_keys:
        mask = key_frame_mask[key]
        y_true = rets_all[key][mask]
        y_pred = values_all[key][mask]
        var_y = np.var(y_true) if y_true.size > 0 else 0.0
        if var_y < 1e-8:
            ev = 0.0
        else:
            ev = float(1.0 - np.var(y_true - y_pred) / var_y)
        explained_variances[f"ev_{key}"] = ev

    # --- Prepare tensors ---
    act_t = torch.as_tensor(buf.actions, dtype=torch.float32, device=device)
    old_lp_t = torch.as_tensor(buf.log_probs, dtype=torch.float32, device=device)
    rets_t: Dict[str, torch.Tensor] = {}
    ret_masks_t: Dict[str, torch.Tensor] = {}
    for key in reward_keys:
        rets_t[key] = torch.as_tensor(rets_all[key], dtype=torch.float32, device=device)
        ret_masks_t[key] = torch.as_tensor(key_frame_mask[key], device=device)

    # --- 5a. Per-key per-frame actor_weight ---
    # Expand per-segment actor_weight into per-frame arrays.
    # actor_weight can be a scalar (broadcast to all frames) or a (T,)
    # array (per-step weight for time-varying channel importance).
    # This is where the experiment's curriculum scheduling (set during
    # build_trajectories) enters the PPO update. The framework never
    # modifies these values — it just applies them.
    key_actor_weight_frame: Dict[str, np.ndarray] = {}
    for key in reward_keys:
        aw_frame = np.zeros(n, dtype=np.float32)
        for i, is_active in enumerate(buf.key_seg_active[key]):
            if is_active:
                s = seg_offsets[i]
                T_seg = buf.ep_lengths[i]
                e = s + T_seg
                aw = buf.key_seg_actor_weight[key][i]
                if np.isscalar(aw) or (isinstance(aw, np.ndarray) and aw.ndim == 0):
                    aw_frame[s:e] = float(aw)
                else:
                    aw_frame[s:e] = np.asarray(aw, dtype=np.float32)[:T_seg]
        key_actor_weight_frame[key] = aw_frame

    # --- 5b. Confidence from explained variance ---
    # confidence_c = sqrt(clip(EV_c, 0, 1))
    # Square root dampens the effect: a critic with EV=0.5 gets conf=0.71,
    # not 0.5. This prevents overly aggressive down-weighting of partially
    # trained critics, which would slow their learning signal.
    confidences: Dict[str, float] = {}
    for key in reward_keys:
        ev = explained_variances.get(f"ev_{key}", 0.0)
        confidences[key] = float(np.clip(ev, 0.0, 1.0) ** 0.5) if use_confidence else 1.0

    # --- 5c. Combined advantage: Σ_c aw_c * conf_c * norm_adv_c ---
    # This is the single advantage signal the actor optimizes against.
    # Each channel contributes independently:
    #   - norm_adv_c: z-score normalized (scale-invariant across channels)
    #   - conf_c: down-weights unreliable critics
    #   - aw_c: experiment-controlled importance weight
    #
    # Channels with aw=0 or all-zero advantage are skipped entirely.
    # The combined advantage is NOT re-normalized — its scale reflects
    # the sum of weighted channel contributions.
    combined_adv = np.zeros(n, dtype=np.float32)
    for key in reward_keys:
        aw_frame = key_actor_weight_frame[key]
        if not np.any(aw_frame > 0.0):
            continue
        conf = confidences[key]
        combined_adv = combined_adv + aw_frame * conf * _normalize_adv(
            advs_all[key], key_frame_mask[key],
        )
    adv_t = torch.as_tensor(combined_adv, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(buf.sample_weights, dtype=torch.float32, device=device)

    # --- Frame modes (optional) ---
    # Some actors use frame_modes for mode-conditioned policies (e.g.
    # gating between safe/explore modes). Passed through to evaluate_actions.
    frame_modes_t: Optional[torch.Tensor] = None
    if buf.frame_modes is not None:
        frame_modes_t = torch.as_tensor(
            buf.frame_modes, dtype=torch.float32, device=device,
        )

    # --- Diagnostics: episode lengths and actor std ---
    # These are logged but do not influence the update.
    ep_lengths = buf.ep_lengths
    ep_len_mean = float(np.mean(ep_lengths)) if ep_lengths else 0.0
    ep_len_min = float(np.min(ep_lengths)) if ep_lengths else 0.0
    ep_len_max = float(np.max(ep_lengths)) if ep_lengths else 0.0

    # Actor std diagnostics — tracked to detect policy collapse
    # (std → 0 means deterministic, no exploration) or explosion.
    with torch.no_grad():
        clamped_log_std = torch.clamp(actor.log_std, pp.log_std_min, pp.log_std_max)
        clamped_std = clamped_log_std.exp()
        std_mean = float(clamped_std.mean().item())
        std_min = float(clamped_std.min().item())
        std_max = float(clamped_std.max().item())

    n_batches = max(1, n // pp.minibatch_size)
    n_episodes = len(buf.ep_lengths)

    # --- 6. Training loop: multi-epoch minibatch PPO ---
    # Each epoch shuffles all frames and iterates in minibatches.
    # Critic and actor are updated alternately per minibatch.
    # Early stop on target_kl prevents the policy from moving too far
    # from the rollout policy (which would break the on-policy assumption).
    pol_losses: List[float] = []
    val_losses: Dict[str, List[float]] = {key: [] for key in reward_keys}
    epoch_kl_stats: List[Dict[str, float]] = []
    early_stop_kl = 0.0
    all_entropies: List[float] = []
    all_clip_fracs: List[float] = []
    all_ratio_means: List[float] = []
    all_ratio_maxs: List[float] = []
    all_grad_norms_actor: List[float] = []
    all_grad_norms_critic: Dict[str, List[float]] = {key: [] for key in reward_keys}

    for epoch in range(pp.update_epochs):
        perm = torch.randperm(n, device=device)
        epoch_kls: List[float] = []
        epoch_pol_losses: List[float] = []

        for start in range(0, n, pp.minibatch_size):
            end = min(start + pp.minibatch_size, n)
            idx = perm[start:end]

            # Sample weight normalization: divide by mean so the
            # effective batch size is preserved regardless of individual
            # trajectory importance weights. Currently all trajectories
            # have importance=1.0, so this is a no-op.
            batch_weights = w_t[idx]
            batch_weights = batch_weights / (batch_weights.mean() + 1e-8)

            # --- Critic updates ---
            # Each critic is trained only on its active frames (mask).
            # Loss is MSE weighted by sample_weights, normalized by
            # active count (not total batch size) so inactive frames
            # don't dilute the gradient.
            for key in reward_keys:
                critic_optimizers[key].zero_grad()
                new_val = critics[key](obs_t[idx]).squeeze(-1)
                ret_val = rets_t[key][idx]
                mask = ret_masks_t[key][idx].to(new_val.dtype)
                n_active = mask.sum()
                if n_active == 0:
                    continue
                val_loss = (
                    ((new_val - ret_val) ** 2) * mask * batch_weights
                ).sum() / n_active
                val_loss.backward()
                grad_norm_c = torch.nn.utils.clip_grad_norm_(
                    critics[key].parameters(), grad_clip_norm,
                )
                all_grad_norms_critic[key].append(float(grad_norm_c))
                critic_optimizers[key].step()
                val_losses[key].append(float(val_loss))

            # --- Actor update ---
            # Standard PPO clipped surrogate:
            #   L = -E[min(ratio * adv, clip(ratio) * adv)] - entropy_coef * H
            # The advantage here is the combined_adv from step 5c.
            # ratio = exp(new_log_prob - old_log_prob), clipped to [1±eps].
            if frame_modes_t is not None:
                new_lp, entropy = actor.evaluate_actions(
                    obs_t[idx], act_t[idx], frame_modes=frame_modes_t[idx],
                )
            else:
                new_lp, entropy = actor.evaluate_actions(obs_t[idx], act_t[idx])

            with torch.no_grad():
                approx_kl = float((old_lp_t[idx] - new_lp).mean().item())
            epoch_kls.append(approx_kl)

            log_ratio = torch.clamp(new_lp - old_lp_t[idx], -20.0, 20.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv_t[idx]
            surr2 = (
                torch.clamp(ratio, 1.0 - pp.clip_eps, 1.0 + pp.clip_eps) * adv_t[idx]
            )
            policy_loss = -(torch.min(surr1, surr2) * batch_weights).mean()

            with torch.no_grad():
                clip_frac = float(
                    ((ratio - 1.0).abs() > pp.clip_eps).float().mean().item()
                )
                all_clip_fracs.append(clip_frac)
                all_ratio_means.append(float(ratio.mean().item()))
                all_ratio_maxs.append(float(ratio.max().item()))

            loss = policy_loss - pp.entropy_coef * entropy.mean()
            all_entropies.append(float(entropy.mean().item()))

            actor_optimizer.zero_grad()
            loss.backward()
            grad_norm_a = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), grad_clip_norm,
            )
            all_grad_norms_actor.append(float(grad_norm_a))
            actor_optimizer.step()
            epoch_pol_losses.append(float(policy_loss))

        # --- Epoch KL diagnostics ---
        # Track KL divergence per epoch for early stopping and anomaly
        # detection. Three warning conditions are checked:
        # 1. KL too small → policy may be stuck or LR too low
        # 2. KL monotonically increasing → risk of overshoot
        # 3. KL jump (2x in one step) → potential instability
        mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
        max_epoch_kl = float(np.max(epoch_kls)) if epoch_kls else 0.0
        std_epoch_kl = float(np.std(epoch_kls)) if epoch_kls else 0.0

        if epoch_kls:
            kl_cv = std_epoch_kl / (mean_epoch_kl + 1e-8)
            print(
                f"  [kl_stats] epoch={epoch} mean={mean_epoch_kl:.4f} std={std_epoch_kl:.4f} "
                f"cv={kl_cv:.2f} n_batches={len(epoch_kls)}",
                flush=True,
            )

        if mean_epoch_kl < 0.001:
            print(
                f"  [warn] epoch={epoch} mean_kl={mean_epoch_kl:.6f} too small, "
                f"policy may be stuck or LR too low",
                flush=True,
            )

        if len(epoch_kls) >= 3:
            kls = np.array(epoch_kls)
            if np.all(np.diff(kls) > 0):
                print(
                    f"  [warn] epoch={epoch} KL monotonically increasing "
                    f"({kls[0]:.4f} -> {kls[-1]:.4f}), risk of overshoot",
                    flush=True,
                )
            for i in range(1, len(kls)):
                if kls[i] > kls[i-1] * 2 and kls[i] > 0.01:
                    print(
                        f"  [warn] epoch={epoch} KL jump at step {i}: "
                        f"{kls[i-1]:.4f} -> {kls[i]:.4f}",
                        flush=True,
                    )
                    break

        epoch_kl_stats.append({
            "epoch": epoch,
            "mean_kl": mean_epoch_kl,
            "max_kl": max_epoch_kl,
            "std_kl": std_epoch_kl,
            "n_minibatches": len(epoch_kls),
        })

        pol_losses.extend(epoch_pol_losses)

        # Early stop: if mean KL exceeds target, stop training epochs
        # to prevent the policy from diverging too far from the rollout.
        if pp.target_kl > 0.0 and mean_epoch_kl > pp.target_kl:
            print(
                f"  [early_stop] epoch={epoch} mean_kl={mean_epoch_kl:.4f} > target",
                flush=True,
            )
            early_stop_kl = mean_epoch_kl
            break

    # --- 7. Aggregate stats for logging ---
    # Collect per-channel and global statistics into a flat dict.
    # The training loop formats these into human-readable lines and
    # a machine-readable __RAW_STATS__ JSON line.
    per_critic_losses: Dict[str, float] = {
        f"vloss_{key}": float(np.mean(val_losses[key])) if val_losses[key] else 0.0
        for key in reward_keys
    }
    per_adv_stats: Dict[str, float] = {}
    for key in reward_keys:
        a = advs_all[key]
        per_adv_stats[f"adv_mean_{key}"] = float(a.mean())
        per_adv_stats[f"adv_std_{key}"] = float(a.std())
    per_ret_stats: Dict[str, float] = {}
    for key in reward_keys:
        r = rets_all[key]
        per_ret_stats[f"ret_mean_{key}"] = float(r.mean())
        per_ret_stats[f"ret_std_{key}"] = float(r.std())

    total_steps = sum(buf.ep_lengths)
    final_kl = epoch_kl_stats[-1]["mean_kl"] if epoch_kl_stats else 0.0
    max_kl_overall = max((s["max_kl"] for s in epoch_kl_stats), default=0.0)

    clip_frac_mean = float(np.mean(all_clip_fracs)) if all_clip_fracs else 0.0
    ratio_mean = float(np.mean(all_ratio_means)) if all_ratio_means else 1.0
    ratio_max = float(max(all_ratio_maxs)) if all_ratio_maxs else 1.0
    grad_norm_actor = float(np.mean(all_grad_norms_actor)) if all_grad_norms_actor else 0.0
    per_critic_grad_norms: Dict[str, float] = {
        f"grad_norm_{key}": float(np.mean(all_grad_norms_critic[key]))
        if all_grad_norms_critic[key] else 0.0
        for key in reward_keys
    }

    return {
        "policy_loss": float(np.mean(pol_losses)) if pol_losses else 0.0,
        "value_loss": float(np.mean([
            per_critic_losses[f"vloss_{key}"] for key in reward_keys
        ])) if reward_keys else 0.0,
        "approx_kl": final_kl,
        "max_kl": max_kl_overall,
        "early_stop_kl": early_stop_kl,
        "epochs_done": len(epoch_kl_stats),
        "entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
        "std_mean": std_mean,
        "std_min": std_min,
        "std_max": std_max,
        "ep_len_mean": ep_len_mean,
        "ep_len_min": ep_len_min,
        "ep_len_max": ep_len_max,
        "epoch_kl_stats": epoch_kl_stats,
        "n_batches": n_batches,
        "n_episodes": n_episodes,
        "total_steps": total_steps,
        "clip_frac": clip_frac_mean,
        "ratio_mean": ratio_mean,
        "ratio_max": ratio_max,
        "grad_norm_actor": grad_norm_actor,
        **per_critic_grad_norms,
        **per_ret_stats,
        **per_critic_losses,
        **per_adv_stats,
        **explained_variances,
        **{f"confidence_{key}": confidences[key] for key in reward_keys},
    }
