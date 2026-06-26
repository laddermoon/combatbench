"""Generic PPO buffer, update, and rollout helpers.

Moves from V2's ``train_curriculum_v2.py``, parameterized by
``ExperimentConfig`` so that reward keys / extraction are not hardcoded.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from baseline.common.algos import compute_gae
from baseline.common.policies import CriticMLP, TanhGaussianMLPPolicy
from baseline.common.rollout import Episode

from .config import ExperimentConfig

# ---------------------------------------------------------------------------
# Data helpers – work directly on Episode numpy arrays
# ---------------------------------------------------------------------------

def _coerce_per_step(values: Any, expected_len: int) -> np.ndarray:
    """Coerce a raw observer leaf into a (T,) float32 array of length ``expected_len``."""
    if values is None:
        return np.zeros(expected_len, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_len:
        if expected_len == 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.linspace(0, len(arr) - 1, expected_len)
        arr = np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)
    return arr


def _extract_per_step_scalar(
    observer_outputs: Any,
    observer_name: str,
    expected_len: int,
) -> np.ndarray:
    """Pull a (T,) float32 reward signal from stacked observer outputs.

    If the observer emits a dict (e.g. ``{"reward": ..., "in_zone": ...}``),
    the first value is used. Use :func:`_extract_per_step_field` to read a
    specific named field.
    """
    node = observer_outputs.get(observer_name)
    if node is None:
        return np.zeros(expected_len, dtype=np.float32)
    values = next(iter(node.values())) if isinstance(node, dict) else node
    return _coerce_per_step(values, expected_len)


def _extract_per_step_field(
    observer_outputs: Any,
    observer_name: str,
    field: str,
    expected_len: int,
) -> Optional[np.ndarray]:
    """Pull a specific named field from a dict-valued observer output.

    Returns ``None`` if the observer is absent or not a dict.
    """
    node = observer_outputs.get(observer_name)
    if not isinstance(node, dict) or field not in node:
        return None
    return _coerce_per_step(node[field], expected_len)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set numpy + torch (CPU/CUDA) seeds in one call."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# PPO buffer – flat numpy arrays assembled from a list of Episodes
# ---------------------------------------------------------------------------

class PPOBuffer:
    """PPO buffer built from a list of :class:`Episode` objects.

    Generic over reward keys — delegates reward extraction and episode
    metrics entirely to ``experiment``.
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        stage_weights: Tuple[float, ...],
        actor: TanhGaussianMLPPolicy,
        device: torch.device,
        experiment: ExperimentConfig,
    ):
        self.reward_data: Dict[str, List[np.ndarray]] = {
            k: [] for k in experiment.reward_keys
        }
        self.episode_metrics: List[Dict[str, float]] = []
        self.episode_lengths: List[int] = []  # original episode lengths (for logging)

        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        lp_list: List[np.ndarray] = []
        fin_list: List[np.ndarray] = []
        weight_list: List[np.ndarray] = []
        terms: List[bool] = []
        ep_lens: List[int] = []

        # Phase 1: Collect valid episodes and pre-compute per-episode data
        # (segments, rewards) without any GPU calls.
        valid_eps: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, list, Dict[str, np.ndarray], Episode]] = []
        for ep in episodes:
            ep_target = str(ep.episode_options.get("agent_id", "robot_a"))
            obs = ep.observations.get(ep_target)
            acts = ep.actions.get(ep_target)
            fin = ep.final_observation.get(ep_target)
            if obs is None or acts is None or fin is None:
                print(
                    f"[DEBUG] Skipping episode {len(obs_list)+1}: "
                    f"obs={obs is not None} acts={acts is not None} fin={fin is not None}",
                    flush=True,
                )
                continue
            T_full = int(acts.shape[0])
            if T_full == 0:
                continue

            # Episode-level data: metrics and original length (per-episode).
            self.episode_metrics.append(
                experiment.compute_episode_metrics(ep)
            )
            self.episode_lengths.append(T_full)

            # Split into training segments (sub-episodes).
            segments = experiment.segment_episode(ep)
            if not segments:
                continue  # entire episode excluded from training

            # Extract rewards once for the full episode, then slice per segment.
            rewards_full = experiment.extract_rewards(ep)

            valid_eps.append((obs, acts, fin, T_full, segments, rewards_full, ep))

        # Phase 2: Batched evaluate_actions — one GPU call for all episodes
        # instead of 2048 per-episode calls.
        if valid_eps:
            all_obs = np.concatenate([e[0] for e in valid_eps], axis=0).astype(np.float32)
            all_acts = np.concatenate([e[1] for e in valid_eps], axis=0).astype(np.float32)
            all_obs_t = torch.as_tensor(all_obs, dtype=torch.float32, device=device)
            all_acts_t = torch.as_tensor(all_acts, dtype=torch.float32, device=device)
            with torch.no_grad():
                all_lp, _ = actor.evaluate_actions(all_obs_t, all_acts_t)
            all_lp_np = all_lp.cpu().numpy().astype(np.float32)
        else:
            all_lp_np = np.zeros(0, dtype=np.float32)

        # Phase 3: Slice into segments using pre-computed log probs.
        offset = 0
        for obs, acts, fin, T_full, segments, rewards_full, ep in valid_eps:
            lp_full_np = all_lp_np[offset:offset + T_full]
            offset += T_full

            for (start, end) in segments:
                T_seg = end - start
                if T_seg == 0:
                    continue

                obs_seg = np.asarray(obs[start:end], dtype=np.float32)
                acts_seg = np.asarray(acts[start:end], dtype=np.float32)
                lp_seg = lp_full_np[start:end]

                # Bootstrap observation: the obs right after the segment end.
                # If the segment ends mid-episode (e.g. fallback boundary),
                # bootstrap from the next step's observation and mark as
                # truncated (not terminated) so GAE bootstraps correctly.
                if end < T_full:
                    fin_seg = np.asarray(obs[end], dtype=np.float32)
                    term_seg = False
                else:
                    fin_seg = np.asarray(fin, dtype=np.float32)
                    term_seg = bool(ep.is_terminated)

                ep_weight = min(200.0 / T_seg, 10.0)
                weight_arr = np.full(T_seg, ep_weight, dtype=np.float32)

                for key in experiment.reward_keys:
                    r_full = rewards_full.get(key, np.zeros(T_full, dtype=np.float32))
                    self.reward_data[key].append(
                        np.asarray(r_full[start:end], dtype=np.float32)
                    )

                obs_list.append(obs_seg)
                act_list.append(acts_seg)
                lp_list.append(lp_seg)
                fin_list.append(fin_seg)
                weight_list.append(weight_arr)
                terms.append(term_seg)
                ep_lens.append(T_seg)

        if not ep_lens:
            print(
                f"[DEBUG] PPOBuffer: no valid episodes from {len(episodes)} input episodes",
                flush=True,
            )
            self.obs = np.zeros((0,), np.float32)
            self.actions = np.zeros((0,), np.float32)
            self.log_probs = np.zeros((0,), np.float32)
            self.sample_weights = np.zeros((0,), np.float32)
            self.final_obs: List[np.ndarray] = []
            self.is_terminated: List[bool] = []
            self.ep_lengths: List[int] = []
            return

        self.obs = np.concatenate(obs_list, axis=0)
        self.actions = np.concatenate(act_list, axis=0)
        self.log_probs = np.concatenate(lp_list, axis=0)
        self.sample_weights = np.concatenate(weight_list, axis=0)
        self.final_obs = fin_list
        self.is_terminated = terms
        self.ep_lengths = ep_lens
        self.experiment = experiment

    def __len__(self) -> int:
        return sum(self.ep_lengths)

    def is_empty(self) -> bool:
        return len(self.ep_lengths) == 0

    def batch_summary(self) -> Dict[str, float]:
        """Compute batch-level summary.

        Aggregates episode lengths and all keys from ``episode_metrics``
        (computed by the experiment's ``compute_episode_metrics``).
        No experiment-specific keys are hardcoded.
        """
        if not self.episode_lengths:
            return {"mean_length": 0.0}
        result: Dict[str, float] = {"mean_length": float(np.mean(self.episode_lengths))}
        if self.episode_metrics:
            for k in self.episode_metrics[0].keys():
                result[k] = float(np.mean([m.get(k, 0.0) for m in self.episode_metrics]))
        return result

    def reward_summary(self) -> Dict[str, Any]:
        """Return per-step reward statistics (mean/std) for diagnostics."""
        result: Dict[str, Any] = {}
        if not self.ep_lengths:
            for key in self.reward_data:
                result[f"{key}_mean"] = 0.0
                result[f"{key}_std"] = 0.0
            return result

        def _concat_mean_std(reward_list: List[np.ndarray]) -> Tuple[float, float]:
            if not reward_list:
                return 0.0, 0.0
            concat = np.concatenate(reward_list)
            if concat.size == 0:
                return 0.0, 0.0
            return float(concat.mean()), float(concat.std())

        for key, reward_list in self.reward_data.items():
            mean_val, std_val = _concat_mean_std(reward_list)
            result[f"{key}_mean"] = mean_val
            result[f"{key}_std"] = std_val

        return result


'''
1. Rollback 是保命逻辑，一定要有
2. 基于数据等分，每个Batch占总数据量的1/8
3. 如果有需要调学习率
'''

# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    actor: TanhGaussianMLPPolicy,
    critics: Dict[str, CriticMLP],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    buf: PPOBuffer,
    reward_keys: Tuple[str, ...],
    gammas: Dict[str, float],
    gae_lambda: float,
    clip_eps: float,
    entropy_coef: float,
    grad_clip_norm: float,
    target_kl: float,
    update_epochs: int,
    device: torch.device,
    stage_weights: Tuple[float, ...],
    minibatch_size: int = 4096,
) -> Dict[str, float]:
    """Multi-critic PPO update, parameterized by reward_keys and gammas.

    Data is split into minibatches of ``minibatch_size`` samples per epoch.
    The last minibatch may be smaller if ``n`` is not evenly divisible.
    """
    obs_all_t = torch.as_tensor(buf.obs, dtype=torch.float32, device=device)

    # Compute values for each critic
    values_all: Dict[str, np.ndarray] = {}
    for key, critic in critics.items():
        with torch.no_grad():
            values_all[key] = (
                critic(obs_all_t).squeeze(-1).cpu().numpy().astype(np.float32)
            )

    # Compute GAE for each reward component
    advs_all: Dict[str, np.ndarray] = {}
    rets_all: Dict[str, np.ndarray] = {}

    for key in reward_keys:
        advs_list = []
        rets_list = []
        offset = 0

        for i, T in enumerate(buf.ep_lengths):
            values = values_all[key][offset : offset + T]
            offset += T
            last_value = 0.0
            if not buf.is_terminated[i] and buf.final_obs[i] is not None:
                fin_t = torch.as_tensor(
                    buf.final_obs[i][None], dtype=torch.float32, device=device,
                )
                with torch.no_grad():
                    last_value = float(critics[key](fin_t).squeeze(-1).item())

            rewards = buf.reward_data[key][i]
            adv, ret = compute_gae(
                rewards=rewards,
                values=values,
                last_value=last_value,
                gamma=gammas[key],
                lam=gae_lambda,
            )
            advs_list.append(adv)
            rets_list.append(ret)

        advs_all[key] = np.concatenate(advs_list)
        rets_all[key] = np.concatenate(rets_list)
        r = rets_all[key]
        print(
            f"  {key}: return=[{r.min():+.3f}, {r.max():+.3f}] "
            f"mean={r.mean():+.3f} std={r.std():.3f}",
            flush=True,
        )

    # Compute explained variance for each critic before updates
    explained_variances: Dict[str, float] = {}
    for key in reward_keys:
        y_true = rets_all[key]
        y_pred = values_all[key]
        var_y = np.var(y_true)
        if var_y < 1e-8:
            ev = 0.0
        else:
            ev = float(1.0 - np.var(y_true - y_pred) / var_y)
        explained_variances[f"ev_{key}"] = ev

    # Compute episode length diagnostics from buffer
    ep_lengths = buf.ep_lengths
    ep_len_mean = float(np.mean(ep_lengths)) if ep_lengths else 0.0
    ep_len_min = float(np.min(ep_lengths)) if ep_lengths else 0.0
    ep_len_max = float(np.max(ep_lengths)) if ep_lengths else 0.0

    # Prepare tensors
    obs_t = torch.as_tensor(buf.obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(buf.actions, dtype=torch.float32, device=device)
    old_lp_t = torch.as_tensor(buf.log_probs, dtype=torch.float32, device=device)

    # Normalize advantages per component and combine with stage weights
    def _normalize_adv(adv: np.ndarray) -> np.ndarray:
        mean = float(adv.mean())
        std = float(adv.std())
        if std < 1e-8:
            return np.zeros_like(adv, dtype=np.float32)
        return ((adv - mean) / std).astype(np.float32)

    if len(stage_weights) != len(reward_keys):
        raise ValueError(
            f"stage_weights must have {len(reward_keys)} entries (one per "
            f"reward in {reward_keys}); got {stage_weights!r}"
        )
    combined_adv = np.zeros_like(advs_all[reward_keys[0]], dtype=np.float32)
    for w, key in zip(stage_weights, reward_keys):
        if w == 0.0:
            continue
        combined_adv = combined_adv + float(w) * _normalize_adv(advs_all[key])
    adv_t = torch.as_tensor(combined_adv, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(buf.sample_weights, dtype=torch.float32, device=device)

    n = obs_t.shape[0]
    pol_losses: List[float] = []
    val_losses: Dict[str, List[float]] = {key: [] for key in reward_keys}
    epoch_kl_stats: List[Dict[str, float]] = []  # Per-epoch KL statistics
    early_stop_kl = 0.0
    all_entropies: List[float] = []

    # Get baseline action standard deviation
    with torch.no_grad():
        clamped_log_std = torch.clamp(actor.log_std, actor.log_std_min, actor.log_std_max)
        clamped_std = clamped_log_std.exp()
        std_mean = float(clamped_std.mean().item())
        std_min = float(clamped_std.min().item())
        std_max = float(clamped_std.max().item())

    # Number of minibatches derived from minibatch_size (standard PPO).
    n_batches = max(1, n // minibatch_size)

    for epoch in range(update_epochs):
        perm = torch.randperm(n, device=device)
        epoch_kls: List[float] = []
        epoch_pol_losses: List[float] = []
        epoch_early_stop = False

        # Iterate over minibatches of size minibatch_size; last batch may be smaller.
        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            idx = perm[start:end]
            idx_cpu = idx.cpu().numpy()

            # Compute normalized difficulty weights for this minibatch
            batch_weights = w_t[idx]
            batch_weights = batch_weights / (batch_weights.mean() + 1e-8)

            # Step 1: Update each critic independently
            for key in reward_keys:
                critic_optimizers[key].zero_grad()
                new_val = critics[key](obs_t[idx]).squeeze(-1)
                ret_val = torch.as_tensor(
                    rets_all[key][idx_cpu], dtype=torch.float32, device=device,
                )
                val_loss = (((new_val - ret_val) ** 2) * batch_weights).mean()
                val_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critics[key].parameters(), grad_clip_norm,
                )
                critic_optimizers[key].step()
                val_losses[key].append(float(val_loss))

            # Step 2: Update actor (after all critics are updated)
            new_lp, entropy = actor.evaluate_actions(obs_t[idx], act_t[idx])

            with torch.no_grad():
                approx_kl = float((old_lp_t[idx] - new_lp).mean().item())
            epoch_kls.append(approx_kl)

            # Policy loss with combined normalized advantages
            log_ratio = torch.clamp(new_lp - old_lp_t[idx], -20.0, 20.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv_t[idx]
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t[idx]
            )
            policy_loss = -(torch.min(surr1, surr2) * batch_weights).mean()

            # Actor loss (no value loss here - critics are updated separately)
            loss = policy_loss - entropy_coef * entropy.mean()
            all_entropies.append(float(entropy.mean().item()))

            actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), grad_clip_norm,
            )
            actor_optimizer.step()
            epoch_pol_losses.append(float(policy_loss))

        # Epoch-level KL statistics
        mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
        max_epoch_kl = float(np.max(epoch_kls)) if epoch_kls else 0.0
        std_epoch_kl = float(np.std(epoch_kls)) if epoch_kls else 0.0

        # Log KL variance across minibatches (diagnostic)
        if epoch_kls:
            kl_cv = std_epoch_kl / (mean_epoch_kl + 1e-8)  # Coefficient of variation
            print(
                f"  [kl_stats] epoch={epoch} mean={mean_epoch_kl:.4f} std={std_epoch_kl:.4f} "
                f"cv={kl_cv:.2f} n_batches={len(epoch_kls)}",
                flush=True,
            )

        # Warning: suspiciously small KL (policy barely changing)
        if mean_epoch_kl < 0.001:
            print(
                f"  [warn] epoch={epoch} mean_kl={mean_epoch_kl:.6f} too small, "
                f"policy may be stuck or LR too low",
                flush=True,
            )

        # Warning: analyze intra-epoch KL trend
        if len(epoch_kls) >= 3:
            # Check for monotonic increase (potential runaway)
            kls = np.array(epoch_kls)
            if np.all(np.diff(kls) > 0):
                print(
                    f"  [warn] epoch={epoch} KL monotonically increasing "
                    f"({kls[0]:.4f} -> {kls[-1]:.4f}), risk of overshoot",
                    flush=True,
                )
            # Check for sudden jump (>2x from prev step)
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

        # Accumulate losses from this epoch (before early-stop break,
        # otherwise pol_losses is empty and policy_loss reports 0.0).
        pol_losses.extend(epoch_pol_losses)

        # Normal early stop: if mean KL exceeds target
        if target_kl > 0.0 and mean_epoch_kl > target_kl:
            print(
                f"  [early_stop] epoch={epoch} mean_kl={mean_epoch_kl:.4f} > target",
                flush=True,
            )
            early_stop_kl = mean_epoch_kl
            break



    # Aggregate value losses per critic
    total_val_losses = [
        np.mean(val_losses[key]) if val_losses[key] else 0.0
        for key in reward_keys
    ]

    per_critic_losses: Dict[str, float] = {
        f"vloss_{key}": float(np.mean(val_losses[key])) if val_losses[key] else 0.0
        for key in reward_keys
    }

    # Per-reward advantage stats
    per_adv_stats: Dict[str, float] = {}
    for key in reward_keys:
        a = advs_all[key]
        per_adv_stats[f"adv_mean_{key}"] = float(a.mean())
        per_adv_stats[f"adv_std_{key}"] = float(a.std())

    total_steps = sum(buf.ep_lengths)

    # Final KL summary
    final_kl = epoch_kl_stats[-1]["mean_kl"] if epoch_kl_stats else 0.0
    max_kl_overall = max((s["max_kl"] for s in epoch_kl_stats), default=0.0)

    return {
        "policy_loss": float(np.mean(pol_losses)) if pol_losses else 0.0,
        "value_loss": float(np.mean(total_val_losses)),
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
        "final_lr": float(actor_optimizer.param_groups[0]["lr"]),
        "n_batches": n_batches,  # derived from minibatch_size
        "total_steps": total_steps,
        **per_critic_losses,
        **per_adv_stats,
        **explained_variances,
    }



