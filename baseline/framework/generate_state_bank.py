"""Generate a labeled state bank for balance recovery training.

Uses ParallelRollouter to run episodes with ImpulsePerturbationPlugin.
StateCapturePlugin + StateCaptureObserver capture the perturbed core_state
and observation at the first action step (before any physics runs).

Each record contains:
  - core_state (55-dim): root_pos(3), root_rot(4), root_vel_local(3),
    root_angular_vel_local(3), joint_pos_norm(21), joint_vel_norm(21)
  - observation (96-dim): flat observation vector at perturbed state
  - impulse_force, impulse_duration, impulse_direction (3)
  - label: 1.0 = survived, 0.0 = fell
  - episode_length: action steps before termination

Output: .npz file with arrays:
  states       (N, 55)  float32
  observations (N, 96)  float32
  forces       (N,)     float32
  durations    (N,)     int32
  directions   (N, 3)   float32
  labels       (N,)     float32
  ep_lengths   (N,)     int32

Usage::

    python3 baseline/framework/generate_state_bank.py \
        --policy-export baseline/runs/<run>/policy \
        --force-grid 10,20,30,50,70,100,150 \
        --duration-grid 1,2,3,4,6,8 \
        --episodes-per-cell 20 \
        --output baseline/runs/recovery_iter/gen0_state_bank.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.common.rollout import ParallelRollouter
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


def flatten_core_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([state[f] for f in CORE_STATE_FIELDS]).astype(np.float32)


def unflatten_core_state(vec: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    offset = 0
    for name, dim in zip(CORE_STATE_FIELDS, CORE_STATE_DIMS):
        out[name] = vec[offset:offset + dim].astype(np.float32)
        offset += dim
    return out


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]


def _extract_from_episode(ep, agent_id: str) -> Dict[str, Any] | None:
    """Extract captured state data from an Episode's observer_outputs.

    observer_outputs are stacked across all frames by EpisodeRecorder.
    We only need the first frame (the perturbed state).
    """
    sc = ep.observer_outputs.get("state_capture")
    if sc is None:
        return None

    cs = sc.get("core_state")
    if cs is None:
        return None

    # core_state fields are stacked (T, *shape) — take first frame
    first_cs = {}
    for name in CORE_STATE_FIELDS:
        arr = np.asarray(cs[name])
        first_cs[name] = arr[0] if arr.ndim > 1 else arr
    state_vec = flatten_core_state(first_cs)

    obs = sc.get("observation")
    if obs is not None:
        obs_arr = np.asarray(obs)
        obs_vec = obs_arr[0] if obs_arr.ndim > 1 else obs_arr
        obs_vec = obs_vec.astype(np.float32)
    else:
        obs_vec = np.zeros(96, dtype=np.float32)

    force_arr = sc.get("impulse_force")
    force = float(np.asarray(force_arr).flat[0]) if force_arr is not None else 0.0

    dur_arr = sc.get("impulse_duration")
    duration = int(np.asarray(dur_arr).flat[0]) if dur_arr is not None else 0

    dir_arr = sc.get("impulse_direction")
    if dir_arr is not None:
        dir_vec = np.asarray(dir_arr, dtype=np.float32)
        dir_vec = dir_vec[0] if dir_vec.ndim > 1 else dir_vec
    else:
        dir_vec = np.zeros(3, dtype=np.float32)

    fell = all(r.startswith("imbalance") for r in ep.agent_termination_reason.values())
    label = 0.0 if fell else 1.0

    return {
        "state": state_vec,
        "observation": obs_vec,
        "force": force,
        "duration": duration,
        "direction": dir_vec,
        "label": label,
        "ep_length": ep.num_frames,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate labeled state bank")
    p.add_argument("--policy-export", required=True)
    p.add_argument("--blueprint", type=str,
                   default="baseline/humanoid21/blueprints/impulse_boundary_env.yaml")
    p.add_argument("--force-grid", type=str, default="10,20,30,50,70,100,150")
    p.add_argument("--duration-grid", type=str, default="1,2,3,4,6,8")
    p.add_argument("--episodes-per-cell", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--tolerance", type=int, default=6)
    p.add_argument("--output", required=True)
    p.add_argument("--agent-id", type=str, default="robot_a")
    args = p.parse_args()

    forces = _parse_float_list(args.force_grid)
    durations = _parse_int_list(args.duration_grid)

    policy_bp = PolicyBlueprint.load(Path(args.policy_export) / "policy_blueprint.yaml")
    policy_path_abs = str((Path(args.policy_export) / "policy_blueprint.yaml").resolve())

    env_pb = ParameterizedEnvBlueprint.load(args.blueprint)

    print(f"=== State Bank Generation ===")
    print(f"policy: {policy_path_abs}")
    print(f"forces: {forces}")
    print(f"durations: {durations}")
    print(f"episodes/cell: {args.episodes_per_cell}")
    print(f"workers: {args.workers}")
    print(f"max_steps: {args.max_steps}")
    print(f"tolerance: {args.tolerance}")
    print()

    all_jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
    cell_map: List[Tuple[float, int, int, int]] = []

    base_seed = args.seed
    for force in forces:
        for duration in durations:
            env_bp = env_pb.materialize(
                max_steps=args.max_steps,
                agent_id=args.agent_id,
                tolerance=args.tolerance,
                policy_blueprint_path=policy_path_abs,
                force_magnitude=force,
                duration_action_steps=duration,
                direction_mode="random_horizontal",
            )
            start = len(all_jobs)
            for i in range(args.episodes_per_cell):
                seed = base_seed + i
                all_jobs.append((
                    policy_bp, policy_bp,
                    env_bp, seed,
                    {"agent_id": args.agent_id, "initial_distance": 2.0},
                ))
            cell_map.append((force, duration, start, args.episodes_per_cell))
            base_seed += args.episodes_per_cell

    total = len(all_jobs)
    print(f"Total episodes: {total}")
    t0 = time.perf_counter()

    rollouter = ParallelRollouter(num_workers=args.workers)
    episodes = rollouter.collect(all_jobs)
    rollouter.close()

    elapsed = time.perf_counter() - t0
    print(f"Rollout time: {elapsed:.1f}s ({elapsed/total:.2f}s/episode)")
    print()

    results: List[Dict[str, Any]] = []
    failed = 0
    for ep in episodes:
        rec = _extract_from_episode(ep, args.agent_id)
        if rec is None:
            failed += 1
            continue
        results.append(rec)

    if failed > 0:
        print(f"WARNING: {failed} episodes had no state_capture data")

    n = len(results)
    if n == 0:
        print("ERROR: no valid data extracted. Aborting.")
        return

    all_states = np.stack([r["state"] for r in results])
    all_obs = np.stack([r["observation"] for r in results])
    all_forces = np.array([r["force"] for r in results], dtype=np.float32)
    all_durations = np.array([r["duration"] for r in results], dtype=np.int32)
    all_directions = np.stack([r["direction"] for r in results])
    all_labels = np.array([r["label"] for r in results], dtype=np.float32)
    all_ep_lengths = np.array([r["ep_length"] for r in results], dtype=np.int32)

    survived = int(all_labels.sum())
    fell = n - survived
    print(f"Total states: {n}")
    print(f"Survived: {survived}  Fell: {fell}  Rate: {survived/n:.3f}")
    print(f"State dim: {all_states.shape[1]}  Obs dim: {all_obs.shape[1]}")
    print(f"Episode length: mean={all_ep_lengths.mean():.1f}  "
          f"min={all_ep_lengths.min()}  max={all_ep_lengths.max()}")
    print()

    print(f"{'force':>7} {'dur':>4} {'surv':>5} {'fell':>5} {'total':>6} {'rate':>6} {'mean_len':>9}")
    print("-" * 50)
    for force, duration, start, count in cell_map:
        cell_results = results[start:start + count]
        if len(cell_results) == 0:
            continue
        cell_labels = np.array([r["label"] for r in cell_results])
        cell_lens = np.array([r["ep_length"] for r in cell_results])
        s = int(cell_labels.sum())
        print(f"{force:>7.0f} {duration:>4d} {s:>5d} {len(cell_results) - s:>5d} "
              f"{len(cell_results):>6d} {s/len(cell_results):>6.3f} {cell_lens.mean():>9.1f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=all_states,
        observations=all_obs,
        forces=all_forces,
        durations=all_durations,
        directions=all_directions,
        labels=all_labels,
        ep_lengths=all_ep_lengths,
        core_state_fields=np.array(CORE_STATE_FIELDS),
        core_state_dims=np.array(CORE_STATE_DIMS),
    )
    print(f"\nState bank saved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    loaded = np.load(out_path, allow_pickle=True)
    assert loaded["states"].shape == (n, CORE_STATE_TOTAL)
    assert loaded["observations"].shape[0] == n
    print(f"Verification: loaded shapes OK "
          f"(states={loaded['states'].shape}, obs={loaded['observations'].shape})")


if __name__ == "__main__":
    main()
