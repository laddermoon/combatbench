"""Offline measurement of reward equivalence quantities.

Implements the two measurements from guidance item #2:

1. Canonical error: ||V^Delta(s) + φ(s) - V^Dense(s)||
   - Loads trained critics from ST-4 and ST-6 runs
   - Evaluates both on a common set of observations (from rollouts)
   - Reports L2 norm, mean abs error, and per-state breakdown

2. Offline advantage correlation: corr(A^Delta, A^Dense)
   - On the same batch of trajectories, computes GAE for both reward types
   - Uses each experiment's respective critic for value estimates
   - Reports Pearson correlation
   - Also computes corr(A^ST1, A^ST2) as a sanity check (should be ≈1.0)

Usage:
    PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/curriculum/experiments/measure_reward_equivalence.py

    # With custom checkpoint paths:
    PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/curriculum/experiments/measure_reward_equivalence.py \
        --st4-ckpt baseline/runs/train_st4_delta_ppo_noconf_20250715/checkpoints/checkpoint_u10000.pt \
        --st6-ckpt baseline/runs/train_st6_dense_ppo_noconf_20250715/checkpoints/checkpoint_u00600.pt \
        --st1-ckpt baseline/runs/train_st1_terminal_ppo_noconf_20250715/checkpoints/checkpoint_u00550.pt \
        --st2-ckpt baseline/runs/train_st2_survival_ppo_noconf_20250715/checkpoints/checkpoint_u00550.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from baseline.common.policies import CriticMLP
from baseline.common.rollout import Episode, ParallelRollouter
from baseline.common.algos import compute_gae
from baseline.framework.ppo_trainer import _extract_per_step_field
from baseline.humanoid21.curriculum.experiments.phi_critic import PhiSubtractedCritic

# Experiment configs
from baseline.humanoid21.curriculum.experiments.exp_st1_terminal import ST1TerminalConfig
from baseline.humanoid21.curriculum.experiments.exp_st2_survival import ST2SurvivalConfig
from baseline.humanoid21.curriculum.experiments.exp_st4_delta import ST4DeltaConfig
from baseline.humanoid21.curriculum.experiments.exp_st6_dense import ST6DenseConfig

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_critic_from_checkpoint(
    ckpt_path: str,
    obs_dim: int = 96,
    hidden_dim: int = 256,
    device: torch.device = torch.device("cpu"),
) -> CriticMLP:
    """Load a CriticMLP from a training checkpoint."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    critics_state = payload["critics_state_dict"]
    # All ST experiments use "r_fall" as the single reward key
    critic = CriticMLP(obs_dim=obs_dim, hidden_dim=hidden_dim).to(device)
    critic.load_state_dict(critics_state["r_fall"])
    critic.eval()
    return critic


def compute_phi_from_obs(obs: np.ndarray, standing_height: float = 1.28) -> np.ndarray:
    """Compute φ = uprightness * (height / standing_height) from observation.

    obs layout (humanoid21):
      [42:45] = c1 (1st col of world rot mat)
      [45:48] = c2 (2nd col of world rot mat)
      [48]    = height (Z)
    """
    c1 = obs[..., 42:45]
    c2 = obs[..., 45:48]
    height = obs[..., 48]
    uprightness = c1[..., 0] * c2[..., 1] - c1[..., 1] * c2[..., 0]
    return uprightness * height / standing_height


def collect_rollout_observations(
    env_blueprint_path: str,
    n_episodes: int = 64,
    seed: int = 999,
    rollout_workers: int = 4,
    policy_dir: Optional[str] = None,
) -> Tuple[np.ndarray, List[Episode]]:
    """Collect rollout episodes using a trained or initial policy.

    If policy_dir is given, loads that policy blueprint. Otherwise uses
    the initial (untrained) policy.

    Returns (all_observations, episodes) for offline analysis.
    """
    from envs.framework.policy import PolicyBlueprint

    if policy_dir is not None:
        policy_bp = PolicyBlueprint.load(Path(policy_dir) / "policy_blueprint.yaml")
        policy_bp.config["stochastic"] = True
    else:
        blueprint_dir = Path(env_blueprint_path).resolve().parent.parent / "blueprints"
        bp = PolicyBlueprint.load(blueprint_dir / "init_policy.yaml")
        actor = bp.build()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "policy_export"
            policy_bp = actor.to_blueprint(dest_path=str(export_dir))
            policy_bp.config["stochastic"] = True

    # Load env blueprint
    env_pb = ParameterizedEnvBlueprint.load(Path(env_blueprint_path))

    # Build rollout jobs (self-play, robot_a perspective)
    exp = ST4DeltaConfig()
    max_steps = exp.custom_config["max_steps"]
    rng = np.random.default_rng(seed)
    env_bps = {
        aid: env_pb.materialize(max_steps=max_steps, agent_id=aid)
        for aid in ("robot_a", "robot_b")
    }
    jobs = []
    for i in range(n_episodes):
        ep_seed = int(seed + i)
        agent_id = "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"
        initial_distance = float(rng.uniform(1.5, 3.5))
        jobs.append((
            policy_bp, policy_bp,
            env_bps[agent_id], ep_seed,
            {"agent_id": agent_id, "initial_distance": initial_distance},
        ))

    with ParallelRollouter(num_workers=rollout_workers) as rollouter:
        episodes = rollouter.collect(jobs)

    # Collect all observations from the target agent
    all_obs = []
    for ep in episodes:
        target = str(ep.episode_options.get("agent_id", "robot_a"))
        obs = ep.observations.get(target)
        if obs is not None:
            all_obs.append(obs.astype(np.float32))

    if all_obs:
        all_obs = np.concatenate(all_obs, axis=0)
    else:
        all_obs = np.zeros((0, 96), dtype=np.float32)

    return all_obs, episodes


def extract_phi_and_rewards(episode: Episode) -> Dict[str, np.ndarray]:
    """Extract φ, Delta rewards, Dense rewards, ST-1 rewards, ST-2 rewards
    from a single episode using the same observer outputs."""
    T = episode.num_frames
    phi = _extract_per_step_field(episode.observer_outputs, "height_phi", "phi", T)
    if phi is None:
        phi = np.zeros(T, dtype=np.float32)
    initial_phi_arr = _extract_per_step_field(episode.observer_outputs, "height_phi", "initial_phi", T)
    initial_phi = float(initial_phi_arr[0]) if initial_phi_arr is not None else 0.0

    fell = "imbalance" in episode.termination_proposals

    # Delta rewards (ST-4): r_t = φ(t) - φ(t-1), γ_s=1.0
    r_delta = np.zeros(T, dtype=np.float32)
    r_delta[0] = phi[0] - initial_phi
    if T > 1:
        r_delta[1:] = phi[1:] - phi[:-1]

    # Dense rewards (ST-6): r_t = 0.01 * φ(t)
    r_dense = 0.01 * phi.astype(np.float32)

    # ST-1 terminal penalty: 0 for alive, -1 on fall
    r_st1 = np.zeros(T, dtype=np.float32)
    if fell:
        r_st1[-1] = -1.0

    # ST-2 survival reward: +0.01 per alive step, 0 on fall
    r_st2 = np.full(T, 0.01, dtype=np.float32)
    if fell:
        r_st2[-1] = 0.0

    return {
        "phi": phi,
        "r_delta": r_delta,
        "r_dense": r_dense,
        "r_st1": r_st1,
        "r_st2": r_st2,
    }


def compute_offline_gae(
    episode: Episode,
    rewards: np.ndarray,
    critic: nn.Module,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns for a single episode using
    the given critic for value estimates."""
    import torch.nn as nn

    T = episode.num_frames
    target = str(episode.episode_options.get("agent_id", "robot_a"))
    obs = episode.observations.get(target).astype(np.float32)
    fin_obs = episode.final_observation.get(target)
    is_terminated = episode.is_terminated

    # Compute values
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = critic(obs_t).reshape(-1).cpu().numpy().astype(np.float32)

    # Bootstrap value
    last_value = 0.0
    if not is_terminated and fin_obs is not None:
        fin_t = torch.as_tensor(
            np.asarray(fin_obs, dtype=np.float32)[None, :],
            dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            last_value = float(critic(fin_t).reshape(-1).cpu().numpy()[0])

    adv, ret = compute_gae(
        rewards=rewards,
        values=values,
        last_value=last_value,
        gamma=gamma,
        lam=gae_lambda,
    )
    return adv, ret


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure reward equivalence: canonical error & advantage correlation"
    )
    parser.add_argument("--st4-ckpt", type=str,
                        default="baseline/runs/train_st4_delta_ppo_noconf_20250715/checkpoints/checkpoint_u10000.pt")
    parser.add_argument("--st6-ckpt", type=str,
                        default="baseline/runs/train_st6_dense_ppo_noconf_20250715/checkpoints/checkpoint_u00600.pt")
    parser.add_argument("--st1-ckpt", type=str,
                        default="baseline/runs/train_st1_terminal_ppo_noconf_20250715/checkpoints/checkpoint_u00550.pt")
    parser.add_argument("--st2-ckpt", type=str,
                        default="baseline/runs/train_st2_survival_ppo_noconf_20250715/checkpoints/checkpoint_u00550.pt")
    parser.add_argument("--n-episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--rollout-workers", type=int, default=4)
    parser.add_argument("--policy-dir", type=str, default=None,
                        help="Trained policy directory (with policy_blueprint.yaml). "
                             "Default: use ST-6 best policy if available.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    env_blueprint = str(
        Path(__file__).resolve().parent.parent.parent / "blueprints" / "basic_balance_phi_env.yaml"
    )

    # ------------------------------------------------------------------
    # Step 1: Load trained critics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1: Loading trained critics")
    print("=" * 70)

    critics = {}
    for name, path in [("st1", args.st1_ckpt), ("st2", args.st2_ckpt),
                        ("st4", args.st4_ckpt), ("st6", args.st6_ckpt)]:
        if not os.path.exists(path):
            print(f"  [WARN] {name} checkpoint not found: {path}")
            continue
        critics[name] = load_critic_from_checkpoint(path, device=device)
        print(f"  Loaded {name} critic from {path}")

    # ------------------------------------------------------------------
    # Step 2: Collect rollout episodes (common trajectory set)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    # Use trained ST-6 policy for realistic trajectories if not specified
    policy_dir = args.policy_dir
    if policy_dir is None:
        default_policy = "baseline/runs/train_st6_dense_ppo_noconf_20250715/policy"
        if os.path.exists(default_policy):
            policy_dir = default_policy

    print(f"Step 2: Collecting {args.n_episodes} rollout episodes")
    if policy_dir:
        print(f"  Using trained policy: {policy_dir}")
    else:
        print(f"  Using initial (untrained) policy")
    print("=" * 70)

    all_obs, episodes = collect_rollout_observations(
        env_blueprint,
        n_episodes=args.n_episodes,
        seed=args.seed,
        rollout_workers=args.rollout_workers,
        policy_dir=policy_dir,
    )
    print(f"  Collected {len(episodes)} episodes, {all_obs.shape[0]} total observations")

    # ------------------------------------------------------------------
    # Measurement 1: Canonical Error
    # ||V^Delta(s) + φ(s) - V^Dense(s)||
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Measurement 1: Canonical Error ||V^Delta(s) + φ(s) - V^Dense(s)||")
    print("=" * 70)

    if "st4" in critics and "st6" in critics:
        obs_t = torch.as_tensor(all_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            v_delta = critics["st4"](obs_t).cpu().numpy()
            v_dense = critics["st6"](obs_t).cpu().numpy()

        phi_all = compute_phi_from_obs(all_obs)

        # Canonical error: V^Delta + φ - V^Dense should be ≈ 0 if theory holds
        canonical_error = v_delta + phi_all - v_dense

        l2_norm = float(np.linalg.norm(canonical_error))
        mean_abs = float(np.mean(np.abs(canonical_error)))
        mean_sq = float(np.mean(canonical_error ** 2))
        rms = float(np.sqrt(mean_sq))

        # Also report individual value function stats
        print(f"\n  V^Delta  stats: mean={v_delta.mean():+.4f} std={v_delta.std():.4f} range=[{v_delta.min():+.4f}, {v_delta.max():+.4f}]")
        print(f"  V^Dense  stats: mean={v_dense.mean():+.4f} std={v_dense.std():.4f} range=[{v_dense.min():+.4f}, {v_dense.max():+.4f}]")
        print(f"  φ        stats: mean={phi_all.mean():+.4f} std={phi_all.std():.4f} range=[{phi_all.min():+.4f}, {phi_all.max():+.4f}]")
        print(f"  V^Delta+φ stats: mean={(v_delta+phi_all).mean():+.4f} std={(v_delta+phi_all).std():.4f}")

        print(f"\n  *** Canonical Error ***")
        print(f"  L2 norm:  {l2_norm:.4f}")
        print(f"  Mean abs: {mean_abs:.4f}")
        print(f"  RMS:      {rms:.4f}")
        print(f"  Mean:     {canonical_error.mean():+.6f}")

        # Interpretation
        v_dense_std = float(v_dense.std())
        if v_dense_std > 1e-8:
            relative_error = rms / v_dense_std
            print(f"  Relative error (RMS / std(V^Dense)): {relative_error:.4f}")
            print(f"\n  Interpretation:")
            if relative_error < 0.1:
                print(f"    → LOW error: V^Delta + φ ≈ V^Dense, accounting deadlock theory SUPPORTED")
            elif relative_error < 0.5:
                print(f"    → MODERATE error: partial correspondence, critic hasn't fully learned -φ")
            else:
                print(f"    → HIGH error: V^Delta + φ ≠ V^Dense, critic failed to learn the transformation")
    else:
        print("  [SKIP] Need both ST-4 and ST-6 checkpoints")

    # ------------------------------------------------------------------
    # Measurement 2: Offline Advantage Correlation
    # corr(A^Delta, A^Dense) and corr(A^ST1, A^ST2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Measurement 2: Offline Advantage Correlation")
    print("=" * 70)

    gamma = 0.99
    gae_lambda = 0.95

    # Collect advantages per reward type
    adv_data: Dict[str, List[np.ndarray]] = {
        "delta": [], "dense": [], "st1": [], "st2": []
    }
    ret_data: Dict[str, List[np.ndarray]] = {
        "delta": [], "dense": [], "st1": [], "st2": []
    }

    valid_episodes = 0
    for ep in episodes:
        target = str(ep.episode_options.get("agent_id", "robot_a"))
        if ep.observations.get(target) is None:
            continue
        if ep.num_frames == 0:
            continue

        rewards = extract_phi_and_rewards(ep)

        # Compute GAE for each reward type using its respective critic
        for rname, rkey, ckey in [
            ("delta", "r_delta", "st4"),
            ("dense", "r_dense", "st6"),
            ("st1", "r_st1", "st1"),
            ("st2", "r_st2", "st2"),
        ]:
            if ckey not in critics:
                continue
            adv, ret = compute_offline_gae(
                ep, rewards[rkey], critics[ckey],
                gamma=gamma, gae_lambda=gae_lambda, device=device,
            )
            adv_data[rname].append(adv)
            ret_data[rname].append(ret)

        valid_episodes += 1

    print(f"  Computed GAE for {valid_episodes} episodes")

    # Concatenate all advantages
    adv_concat: Dict[str, np.ndarray] = {}
    ret_concat: Dict[str, np.ndarray] = {}
    for name in adv_data:
        if adv_data[name]:
            adv_concat[name] = np.concatenate(adv_data[name])
            ret_concat[name] = np.concatenate(ret_data[name])

    # Report advantage statistics
    print(f"\n  Advantage statistics:")
    for name in sorted(adv_concat.keys()):
        a = adv_concat[name]
        r = ret_concat[name]
        print(f"    {name:6s}: adv mean={a.mean():+.6f} std={a.std():.6f} | "
              f"ret mean={r.mean():+.6f} std={r.std():.6f} | "
              f"n={a.shape[0]}")

    # Compute correlations
    def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) != len(y) or len(x) < 2:
            return float("nan")
        sx, sy = x.std(), y.std()
        if sx < 1e-12 or sy < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    print(f"\n  *** Advantage Correlations ***")

    # Key prediction: corr(A^ST1, A^ST2) ≈ 1.0
    if "st1" in adv_concat and "st2" in adv_concat:
        corr_12 = pearson_corr(adv_concat["st1"], adv_concat["st2"])
        print(f"  corr(A^ST1, A^ST2) = {corr_12:.4f}  (theory: ≈1.0)")

    # Key prediction: corr(A^Delta, A^Dense) ≈ 0
    if "delta" in adv_concat and "dense" in adv_concat:
        corr_46 = pearson_corr(adv_concat["delta"], adv_concat["dense"])
        print(f"  corr(A^Delta, A^Dense) = {corr_46:.4f}  (theory: ≈0.0)")

    # Additional correlations for context
    pairs = [
        ("delta", "st1", "Delta vs ST-1"),
        ("dense", "st1", "Dense vs ST-1"),
        ("delta", "st2", "Delta vs ST-2"),
        ("dense", "st2", "Dense vs ST-2"),
        ("st1", "delta", "ST-1 vs Delta"),
    ]
    for n1, n2, label in pairs:
        if n1 in adv_concat and n2 in adv_concat and n1 != n2:
            c = pearson_corr(adv_concat[n1], adv_concat[n2])
            print(f"  corr(A^{n1}, A^{n2}) = {c:.4f}  ({label})")

    # Return correlation
    print(f"\n  *** Return Correlations ***")
    if "st1" in ret_concat and "st2" in ret_concat:
        corr_ret_12 = pearson_corr(ret_concat["st1"], ret_concat["st2"])
        print(f"  corr(R^ST1, R^ST2) = {corr_ret_12:.4f}  (theory: ≈1.0, differ by constant)")
    if "delta" in ret_concat and "dense" in ret_concat:
        corr_ret_46 = pearson_corr(ret_concat["delta"], ret_concat["dense"])
        print(f"  corr(R^Delta, R^Dense) = {corr_ret_46:.4f}  (theory: ≈1.0 algebraically)")

    # ------------------------------------------------------------------
    # Control: Same-critic advantage correlation
    # Use a SINGLE critic (ST-6's) to compute GAE for all reward types.
    # This isolates the reward formula effect from the critic effect.
    # ------------------------------------------------------------------
    print(f"\n  *** Control: Same-critic (ST-6) Advantage Correlations ***")
    print(f"  (Using ST-6 critic for ALL reward types to isolate reward formula effect)")

    control_adv: Dict[str, List[np.ndarray]] = {"delta": [], "dense": [], "st1": [], "st2": []}
    if "st6" in critics:
        for ep in episodes:
            target = str(ep.episode_options.get("agent_id", "robot_a"))
            if ep.observations.get(target) is None or ep.num_frames == 0:
                continue
            rewards = extract_phi_and_rewards(ep)
            for rname, rkey in [("delta", "r_delta"), ("dense", "r_dense"),
                                 ("st1", "r_st1"), ("st2", "r_st2")]:
                adv, _ = compute_offline_gae(
                    ep, rewards[rkey], critics["st6"],
                    gamma=gamma, gae_lambda=gae_lambda, device=device,
                )
                control_adv[rname].append(adv)

        control_concat = {k: np.concatenate(v) for k, v in control_adv.items() if v}

        if "st1" in control_concat and "st2" in control_concat:
            c = pearson_corr(control_concat["st1"], control_concat["st2"])
            print(f"  corr(A^ST1, A^ST2)   = {c:.4f}  (theory: ≈1.0, same critic)")
        if "delta" in control_concat and "dense" in control_concat:
            c = pearson_corr(control_concat["delta"], control_concat["dense"])
            print(f"  corr(A^Delta, A^Dense) = {c:.4f}  (theory: ≈1.0 algebraically, same critic)")
        if "st1" in control_concat and "delta" in control_concat:
            c = pearson_corr(control_concat["st1"], control_concat["delta"])
            print(f"  corr(A^ST1, A^Delta)   = {c:.4f}  (same critic)")
        if "st1" in control_concat and "dense" in control_concat:
            c = pearson_corr(control_concat["st1"], control_concat["dense"])
            print(f"  corr(A^ST1, A^Dense)   = {c:.4f}  (same critic)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if "st4" in critics and "st6" in critics:
        print(f"\n  Measurement 1 (Canonical Error):")
        print(f"    ||V^Delta + φ - V^Dense||  = {l2_norm:.4f} (L2), {rms:.4f} (RMS)")
        print(f"    Relative to std(V^Dense)   = {rms / v_dense_std:.4f}")
        print(f"    → {'SUPPORTS' if rms / v_dense_std < 0.3 else 'REFUTES'} accounting deadlock theory")

    if "st1" in adv_concat and "st2" in adv_concat:
        print(f"\n  Measurement 2 (Advantage Correlation):")
        print(f"    corr(A^ST1, A^ST2)   = {corr_12:.4f}  (predicted: ≈1.0)")
    if "delta" in adv_concat and "dense" in adv_concat:
        print(f"    corr(A^Delta, A^Dense) = {corr_46:.4f}  (predicted: ≈0.0)")
        print(f"    → {'SUPPORTS' if abs(corr_46) < 0.3 else 'REFUTES'} signal collapse theory")


if __name__ == "__main__":
    main()
