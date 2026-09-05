"""Collect a state pool for gating classifier training.

Each run collects states from a single path (A or B), configured via a YAML
config file. This makes it easy to tune data quantity and parameters per path
independently.

  Path A: ConstantForcePlugin applies impulse perturbation (parameters
          sampled from boundary-weighted distribution via ImpulseSampler).
          The rollout policy controls the robot during force application.
          Episode ends → final state captured.

  Path B: InitialStatePerturbationPlugin applies geometric perturbation.
          The rollout policy drives K settling steps. Episode ends →
          final state captured.

Output: .npz file with states, observations, and metadata.

Usage::

    python3 baseline/humanoid21/balance_recover/gating/collect_state_pool.py \\
        --config baseline/humanoid21/balance_recover/gating/collect_path_a.yaml \\
        --output state_pool_a.npz \\
        --workers 32

    # Smoke test (10 episodes)
    python3 baseline/humanoid21/balance_recover/gating/collect_state_pool.py \\
        --config collect_path_a.yaml --output /tmp/smoke.npz --workers 4 --smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from baseline.framework.rollout import ParallelRollouter, Job
from baseline.humanoid21.balance_recover.sample_distribution import ImpulseSampler
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

CORE_STATE_FIELDS = [
    "root_pos",                # 3
    "root_rot",                # 4
    "root_vel_local",          # 3
    "root_angular_vel_local",  # 3
    "joint_pos_norm",          # 21
    "joint_vel_norm",          # 21
]
CORE_STATE_DIMS = [3, 4, 3, 3, 21, 21]
CORE_STATE_TOTAL = sum(CORE_STATE_DIMS)  # 55

BATCH_SIZE = 2000


def flatten_core_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([state[f] for f in CORE_STATE_FIELDS]).astype(np.float32)


def build_noisy_policy_blueprint(
    policy_dir: str,
    sigma: float,
) -> PolicyBlueprint:
    base_bp = PolicyBlueprint.load(Path(policy_dir) / "policy_blueprint.yaml")
    return PolicyBlueprint(
        cls="baseline.humanoid21.balance_recover.gating.noisy_policy:NoisyPolicyWrapper",
        config={
            "base_cls": base_bp.cls,
            "base_config": dict(base_bp.config),
            "sigma": float(sigma),
        },
    )


def build_random_policy_blueprint() -> PolicyBlueprint:
    return PolicyBlueprint.load(Path("policy/blueprints/random.yaml"))


def build_policy_variants(
    policy_dirs: Dict[str, str],
    policy_ids: Dict[str, int],
    noise_sigmas: List[float],
) -> List[Tuple[str, int, float, PolicyBlueprint]]:
    variants = []
    for ckpt_name, policy_dir in policy_dirs.items():
        policy_id = policy_ids[ckpt_name]
        for sigma in noise_sigmas:
            noisy_bp = build_noisy_policy_blueprint(policy_dir, sigma)
            variants.append((ckpt_name, policy_id, sigma, noisy_bp))
    return variants


def build_path_a_jobs(
    env_pb: ParameterizedEnvBlueprint,
    policy_variants: List[Tuple[str, int, float, PolicyBlueprint]],
    random_bp: PolicyBlueprint,
    impulse_samplers: Dict[str, ImpulseSampler],
    n_episodes: int,
    rng: np.random.RandomState,
    cfg: Dict[str, Any],
) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
    jobs = []
    n_variants = len(policy_variants)
    eps_per_variant = n_episodes // n_variants
    impulse_body = cfg.get("impulse_body", "torso")
    margin_steps = cfg.get("margin_steps", 2)
    initial_distance_range = tuple(cfg.get("initial_distance_range", [1.5, 3.0]))
    episode_seed = cfg.get("episode_seed", 42)

    for vi, (ckpt_name, policy_id, sigma, noisy_bp) in enumerate(policy_variants):
        sampler = impulse_samplers[ckpt_name]
        n_eps = eps_per_variant + (n_episodes - eps_per_variant * n_variants if vi == 0 else 0)
        for ei in range(n_eps):
            params_a = sampler.sample(rng)
            params_b = sampler.sample(rng)
            duration = int(params_a["duration_action_steps"])
            max_steps = duration + margin_steps

            env_bp = env_pb.materialize(
                max_steps=max_steps,
                agent_id="robot_a",
                impulse_body=impulse_body,
                initial_distance=float(rng.uniform(*initial_distance_range)),
            )

            options: Dict[str, Any] = {
                "impulse_params": {
                    "robot_a": {
                        "force": params_a["force"],
                        "direction_angle": params_a["direction_angle"],
                        "duration_action_steps": duration,
                        "body": impulse_body,
                    },
                    "robot_b": {
                        "force": params_b["force"],
                        "direction_angle": params_b["direction_angle"],
                        "duration_action_steps": int(params_b["duration_action_steps"]),
                        "body": impulse_body,
                    },
                },
                "_meta": {
                    "path_type": 0,
                    "policy_id": policy_id,
                    "noise_sigma": sigma,
                    "impulse_force": params_a["force"],
                    "impulse_duration": duration,
                    "impulse_angle": params_a["direction_angle"],
                    "initial_distance": env_bp.simulator.config.get("initial_distance", 2.0),
                },
            }

            jobs.append(Job(
    policy_a_bp=noisy_bp,
    policy_b_bp=random_bp,
    env_bp=env_bp,
    seed=episode_seed,
    episode_options=options,
))

    return jobs


def build_path_b_jobs(
    env_pb: ParameterizedEnvBlueprint,
    policy_variants: List[Tuple[str, int, float, PolicyBlueprint]],
    random_bp: PolicyBlueprint,
    n_episodes: int,
    rng: np.random.RandomState,
    cfg: Dict[str, Any],
) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
    jobs = []
    n_variants = len(policy_variants)
    eps_per_variant = n_episodes // n_variants
    perturb_ranges = cfg.get("perturb_ranges", {})
    settling_steps_options = cfg.get("settling_steps_options", [4, 8, 12, 16])
    margin_steps = cfg.get("margin_steps", 2)
    initial_distance_range = tuple(cfg.get("initial_distance_range", [1.5, 3.0]))
    episode_seed = cfg.get("episode_seed", 42)

    for vi, (ckpt_name, policy_id, sigma, noisy_bp) in enumerate(policy_variants):
        n_eps = eps_per_variant + (n_episodes - eps_per_variant * n_variants if vi == 0 else 0)
        for ei in range(n_eps):
            perturb_params = {}
            for pname, (lo, hi) in perturb_ranges.items():
                perturb_params[pname] = float(rng.uniform(lo, hi))

            K = int(rng.choice(settling_steps_options))
            max_steps = K + margin_steps

            initial_distance = float(rng.uniform(*initial_distance_range))

            env_bp = env_pb.materialize(
                max_steps=max_steps,
                agent_id="robot_a",
                initial_distance=initial_distance,
                **perturb_params,
            )

            perturb_values = [perturb_params[k] for k in perturb_ranges]

            options: Dict[str, Any] = {
                "_meta": {
                    "path_type": 1,
                    "policy_id": policy_id,
                    "noise_sigma": sigma,
                    "perturb_scales": perturb_values,
                    "settling_steps": K,
                    "initial_distance": initial_distance,
                },
            }

            jobs.append(Job(
    policy_a_bp=noisy_bp,
    policy_b_bp=random_bp,
    env_bp=env_bp,
    seed=episode_seed,
    episode_options=options,
))

    return jobs


def extract_from_episode(ep: Any) -> Optional[Dict[str, Any]]:
    sc = ep.observer_outputs.get("episode_end_capture")
    if sc is None:
        return None

    cs = sc.get("core_state")
    if cs is None:
        return None

    last_cs = {}
    for name in CORE_STATE_FIELDS:
        arr = np.asarray(cs[name])
        last_cs[name] = arr[-1] if arr.ndim > 1 else arr
    state_vec = flatten_core_state(last_cs)

    obs = sc.get("observation")
    if obs is not None:
        obs_arr = np.asarray(obs)
        obs_vec = obs_arr[-1] if obs_arr.ndim > 1 else obs_arr
        obs_vec = obs_vec.astype(np.float32)
    else:
        obs_vec = np.zeros(96, dtype=np.float32)

    meta = ep.episode_options.get("_meta", {})

    return {
        "state": state_vec,
        "observation": obs_vec,
        "path_type": int(meta.get("path_type", -1)),
        "policy_id": int(meta.get("policy_id", -1)),
        "noise_sigma": float(meta.get("noise_sigma", 0.0)),
        "impulse_force": float(meta.get("impulse_force", 0.0)),
        "impulse_duration": int(meta.get("impulse_duration", 0)),
        "impulse_angle": float(meta.get("impulse_angle", 0.0)),
        "perturb_scales": np.asarray(meta.get("perturb_scales", [0.0] * 5), dtype=np.float32),
        "settling_steps": int(meta.get("settling_steps", 0)),
        "initial_distance": float(meta.get("initial_distance", 0.0)),
        "ep_length": ep.num_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect state pool for gating classifier")
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--smoke", action="store_true", help="Small-scale test (10 episodes)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    path = cfg["path"]
    n_episodes = cfg["episodes"]

    if args.smoke:
        n_episodes = 10

    rng = np.random.RandomState(cfg.get("seed", 12345))

    policy_variants = build_policy_variants(
        cfg["policy_dirs"], cfg["policy_ids"], cfg["noise_sigmas"],
    )
    print(f"Built {len(policy_variants)} policy variants")

    random_bp = build_random_policy_blueprint()
    env_pb = ParameterizedEnvBlueprint.load(cfg["env_yaml"])

    if path == "A":
        weights_dir = cfg["impulse_weights_dir"]
        impulse_samplers: Dict[str, ImpulseSampler] = {}
        for ckpt_name in cfg["policy_dirs"]:
            gen_num = int(ckpt_name.replace("gen", ""))
            weights_path = str(Path(weights_dir) / f"sample_weights_gen{gen_num + 1}.npz")
            impulse_samplers[ckpt_name] = ImpulseSampler(weights_path)
        print(f"Loaded {len(impulse_samplers)} per-gen impulse samplers from {weights_dir}")
        print(f"Building Path A jobs ({n_episodes} episodes)...")
        jobs = build_path_a_jobs(
            env_pb, policy_variants, random_bp, impulse_samplers, n_episodes, rng, cfg,
        )
    elif path == "B":
        print(f"Building Path B jobs ({n_episodes} episodes)...")
        jobs = build_path_b_jobs(
            env_pb, policy_variants, random_bp, n_episodes, rng, cfg,
        )
    else:
        print(f"ERROR: unknown path '{path}' in config (expected 'A' or 'B')")
        return

    total = len(jobs)
    print(f"Total jobs: {total}")

    t0 = time.perf_counter()
    all_episodes = []
    rollouter = ParallelRollouter(num_workers=args.workers)
    for i in range(0, total, BATCH_SIZE):
        batch = jobs[i:i + BATCH_SIZE]
        eps = rollouter.collect(batch)
        all_episodes.extend(eps)
        done = min(i + BATCH_SIZE, total)
        print(f"  Batch {i//BATCH_SIZE + 1}: {done}/{total} episodes done")
    rollouter.close()
    episodes = all_episodes
    elapsed = time.perf_counter() - t0
    print(f"\nRollout time: {elapsed:.1f}s ({elapsed/total:.3f}s/episode)")

    results: List[Dict[str, Any]] = []
    failed = 0
    for ep in episodes:
        rec = extract_from_episode(ep)
        if rec is None:
            failed += 1
            continue
        results.append(rec)

    if failed > 0:
        print(f"WARNING: {failed} episodes had no episode_end_capture data")

    n = len(results)
    if n == 0:
        print("ERROR: no valid data extracted. Aborting.")
        return

    all_states = np.stack([r["state"] for r in results])
    all_obs = np.stack([r["observation"] for r in results])
    all_path_types = np.array([r["path_type"] for r in results], dtype=np.int32)
    all_policy_ids = np.array([r["policy_id"] for r in results], dtype=np.int32)
    all_noise_sigmas = np.array([r["noise_sigma"] for r in results], dtype=np.float32)
    all_impulse_forces = np.array([r["impulse_force"] for r in results], dtype=np.float32)
    all_impulse_durations = np.array([r["impulse_duration"] for r in results], dtype=np.int32)
    all_impulse_angles = np.array([r["impulse_angle"] for r in results], dtype=np.float32)
    all_perturb_scales = np.stack([r["perturb_scales"] for r in results])
    all_settling_steps = np.array([r["settling_steps"] for r in results], dtype=np.int32)
    all_initial_distances = np.array([r["initial_distance"] for r in results], dtype=np.float32)
    all_ep_lengths = np.array([r["ep_length"] for r in results], dtype=np.int32)

    print(f"\nTotal states: {n}")
    print(f"State dim: {all_states.shape[1]}  Obs dim: {all_obs.shape[1]}")
    print(f"Episode length: mean={all_ep_lengths.mean():.1f}  "
          f"min={all_ep_lengths.min()}  max={all_ep_lengths.max()}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=all_states,
        observations=all_obs,
        path_types=all_path_types,
        policy_ids=all_policy_ids,
        noise_sigmas=all_noise_sigmas,
        impulse_forces=all_impulse_forces,
        impulse_durations=all_impulse_durations,
        impulse_angles=all_impulse_angles,
        perturb_scales=all_perturb_scales,
        settling_steps=all_settling_steps,
        initial_distances=all_initial_distances,
        ep_lengths=all_ep_lengths,
        core_state_fields=np.array(CORE_STATE_FIELDS),
        core_state_dims=np.array(CORE_STATE_DIMS),
    )
    print(f"\nState pool saved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    loaded = np.load(out_path, allow_pickle=True)
    assert loaded["states"].shape == (n, CORE_STATE_TOTAL)
    assert loaded["observations"].shape[0] == n
    print(f"Verification: loaded shapes OK "
          f"(states={loaded['states'].shape}, obs={loaded['observations'].shape})")


if __name__ == "__main__":
    main()
