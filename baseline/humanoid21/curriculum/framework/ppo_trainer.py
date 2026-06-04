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

        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        lp_list: List[np.ndarray] = []
        fin_list: List[np.ndarray] = []
        terms: List[bool] = []
        ep_lens: List[int] = []

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
            T = int(acts.shape[0])
            if T == 0:
                continue

            oo = ep.observer_outputs

            # Extract rewards from experiment
            rewards = experiment.extract_rewards(
                oo, T, ep.termination_proposals
            )

            # Store reward arrays
            for key in experiment.reward_keys:
                self.reward_data[key].append(
                    rewards.get(key, np.zeros(T, dtype=np.float32))
                )

            # Episode metrics
            self.episode_metrics.append(
                experiment.compute_episode_metrics(
                    oo, T, ep.termination_proposals
                )
            )

            # Compute log probs
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(acts, dtype=torch.float32, device=device)
            with torch.no_grad():
                lp, _ = actor.evaluate_actions(obs_t, act_t)
            lp_np = lp.cpu().numpy().astype(np.float32)

            obs_list.append(obs)
            act_list.append(acts)
            lp_list.append(lp_np)
            fin_list.append(np.asarray(fin, dtype=np.float32))
            terms.append(bool(ep.is_terminated))
            ep_lens.append(T)

        if not ep_lens:
            print(
                f"[DEBUG] PPOBuffer: no valid episodes from {len(episodes)} input episodes",
                flush=True,
            )
            self.obs = np.zeros((0,), np.float32)
            self.actions = np.zeros((0,), np.float32)
            self.log_probs = np.zeros((0,), np.float32)
            self.final_obs: List[np.ndarray] = []
            self.is_terminated: List[bool] = []
            self.ep_lengths: List[int] = []
            return

        self.obs = np.concatenate(obs_list, axis=0)
        self.actions = np.concatenate(act_list, axis=0)
        self.log_probs = np.concatenate(lp_list, axis=0)
        self.final_obs = fin_list
        self.is_terminated = terms
        self.ep_lengths = ep_lens

    def __len__(self) -> int:
        return sum(self.ep_lengths)

    def is_empty(self) -> bool:
        return len(self.ep_lengths) == 0


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
    minibatch_size: int,
    device: torch.device,
    stage_weights: Tuple[float, ...],
) -> Dict[str, float]:
    """Multi-critic PPO update, parameterized by reward_keys and gammas."""
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

    n = obs_t.shape[0]
    pol_losses: List[float] = []
    val_losses: Dict[str, List[float]] = {key: [] for key in reward_keys}
    kls: List[float] = []
    early_stop_kl = 0.0

    for _ in range(update_epochs):
        perm = torch.randperm(n, device=device)
        early_stop = False

        for s in range(0, n, minibatch_size):
            idx = perm[s : s + minibatch_size]
            idx_cpu = idx.cpu().numpy()

            # Step 1: Update each critic independently
            for key in reward_keys:
                critic_optimizers[key].zero_grad()
                new_val = critics[key](obs_t[idx]).squeeze(-1)
                ret_val = torch.as_tensor(
                    rets_all[key][idx_cpu], dtype=torch.float32, device=device,
                )
                val_loss = ((new_val - ret_val) ** 2).mean()
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
            kls.append(approx_kl)
            if target_kl > 0.0 and approx_kl > target_kl:
                early_stop_kl = approx_kl
                early_stop = True
                break

            # Policy loss with combined normalized advantages
            log_ratio = torch.clamp(new_lp - old_lp_t[idx], -20.0, 20.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv_t[idx]
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t[idx]
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Actor loss (no value loss here - critics are updated separately)
            loss = policy_loss - entropy_coef * entropy.mean()

            actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), grad_clip_norm,
            )
            actor_optimizer.step()
            pol_losses.append(float(policy_loss))

        if early_stop:
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

    return {
        "policy_loss": float(np.mean(pol_losses)) if pol_losses else 0.0,
        "value_loss": float(np.mean(total_val_losses)),
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "early_stop_kl": early_stop_kl,
        **per_critic_losses,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def batch_summary(buf: PPOBuffer, max_steps: int) -> Dict[str, float]:
    """Compute batch-level summary from a PPOBuffer.

    Returns generic training metrics (``mean_length``, ``len_ratio``) plus
    the mean of every key in ``episode_metrics`` (computed by the experiment's
    ``compute_episode_metrics``).  No experiment-specific keys are hardcoded.
    """
    n = len(buf.ep_lengths)
    if n == 0:
        return {"mean_length": 0.0, "len_ratio": 0.0}
    mean_len = float(np.mean(buf.ep_lengths))
    result: Dict[str, float] = {
        "mean_length": mean_len,
        "len_ratio": mean_len / float(max_steps),
    }
    # Aggregate all episode-level metrics (from experiment.compute_episode_metrics)
    if buf.episode_metrics:
        keys = buf.episode_metrics[0].keys()
        for k in keys:
            result[k] = float(np.mean([m.get(k, 0.0) for m in buf.episode_metrics]))
    return result


def reward_summary(buf: PPOBuffer) -> Dict[str, Any]:
    """Return per-step reward statistics (mean/std) for diagnostics."""
    result: Dict[str, Any] = {}
    if not buf.ep_lengths:
        for key in buf.reward_data:
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

    for key, reward_list in buf.reward_data.items():
        mean_val, std_val = _concat_mean_std(reward_list)
        result[f"{key}_mean"] = mean_val
        result[f"{key}_std"] = std_val

    return result
