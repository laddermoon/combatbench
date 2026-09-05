"""Label states in a state pool by running frozen π_recover and checking recovery.

For each state in the pool, inject it as the initial state of robot_a,
run π_recover for up to 100 action steps, and label:
  - label=0 (unsafe): robot fell (imbalance termination)
  - label=1 (safe): robot stayed standing (timeout or no termination)

Output: labeled_state_pool.npz = original fields + labels (N,)

Usage::

    python3 baseline/humanoid21/balance_recover/gating/label_state_pool.py \\
        --input state_pool.npz \\
        --output labeled_state_pool.npz \\
        --policy baseline/runs/recovery_v5_gen9/policy_exports/u00635/policy_blueprint.yaml \\
        --workers 32
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.rollout import ParallelRollouter, Job
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

from baseline.humanoid21.balance_recover.gating.noisy_policy import NoisyPolicyWrapper

EPISODE_SEED = 42
BATCH_SIZE = 2000


def build_labeling_jobs(
    env_pb: ParameterizedEnvBlueprint,
    recover_bp: PolicyBlueprint,
    n_states: int,
    state_bank_path: str,
    valid_indices: np.ndarray,
) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
    """Build one job per state: inject state by index, run π_recover.

    ``valid_indices`` maps job position → original state pool index.
    """
    env_bp = env_pb.materialize(state_bank_path=state_bank_path)
    jobs = []
    for i in range(n_states):
        options: Dict[str, Any] = {
            "state_bank_index": int(valid_indices[i]),
        }
        jobs.append(Job(
    policy_a_bp=recover_bp,
    policy_b_bp=recover_bp,
    env_bp=env_bp,
    seed=EPISODE_SEED,
    episode_options=options,
))
    return jobs


def extract_label_from_episode(ep: Any) -> int:
    """Extract label from episode termination reason.

    Returns 0 if robot_a fell (imbalance), 1 if it survived.
    """
    reason = ep.agent_termination_reason.get("robot_a", "")
    if reason.startswith("imbalance"):
        return 0
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Label state pool with π_recover recovery outcomes")
    parser.add_argument("--input", type=str, required=True, help="Input state_pool.npz path")
    parser.add_argument("--output", type=str, required=True, help="Output labeled_state_pool.npz path")
    parser.add_argument("--policy", type=str, required=True, help="Path to π_recover policy_blueprint.yaml")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--env-yaml", type=str,
                        default="baseline/humanoid21/balance_recover/gating/label_state_pool_env.yaml")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise std on actions (0=clean)")
    parser.add_argument("--min-torso-height", type=float, default=0.0,
                        help="Skip states with root_pos[2] below this threshold (0=no filter)")
    parser.add_argument("--smoke", action="store_true", help="Small-scale test (20 states)")
    args = parser.parse_args()

    # Load state pool
    data = np.load(args.input, allow_pickle=True)
    all_states = data["states"]
    n_total = len(all_states)
    print(f"Loaded state pool: {n_total} states")

    # Height filter: root_pos[2] is index 2 in the flattened state vector
    if args.min_torso_height > 0:
        heights = all_states[:, 2]
        mask = heights >= args.min_torso_height
        valid_indices = np.where(mask)[0]
        n_filtered = n_total - len(valid_indices)
        print(f"Height filter (>= {args.min_torso_height}): kept {len(valid_indices)}, filtered {n_filtered} ({100*n_filtered/n_total:.1f}%)")
    else:
        valid_indices = np.arange(n_total)

    n_states = len(valid_indices)

    if args.smoke:
        n_states = min(20, n_states)
        valid_indices = valid_indices[:n_states]

    # Load π_recover policy (optionally wrapped with noise)
    base_bp = PolicyBlueprint.load(Path(args.policy))
    if args.noise_std > 0:
        recover_bp = PolicyBlueprint(
            cls="baseline.humanoid21.balance_recover.gating.noisy_policy:NoisyPolicyWrapper",
            config={
                "base_cls": base_bp.cls,
                "base_config": dict(base_bp.config),
                "sigma": args.noise_std,
            },
        )
        print(f"Loaded π_recover from {args.policy} (noise_std={args.noise_std})")
    else:
        recover_bp = base_bp
        print(f"Loaded π_recover from {args.policy} (clean)")

    # Load env blueprint
    env_pb = ParameterizedEnvBlueprint.load(args.env_yaml)

    # Build jobs: only for valid (filtered) indices
    print(f"Building {n_states} labeling jobs...")
    jobs = build_labeling_jobs(env_pb, recover_bp, n_states, str(Path(args.input).resolve()),
                               valid_indices)

    # Collect in batches
    t0 = time.perf_counter()
    all_labels = np.ones(n_states, dtype=np.float32)
    rollouter = ParallelRollouter(num_workers=args.workers)
    for i in range(0, n_states, BATCH_SIZE):
        batch = jobs[i:i + BATCH_SIZE]
        eps = rollouter.collect(batch)
        for j, ep in enumerate(eps):
            all_labels[i + j] = extract_label_from_episode(ep)
        done = min(i + BATCH_SIZE, n_states)
        n_safe = int(all_labels[:done].sum())
        print(f"  Batch {i//BATCH_SIZE + 1}: {done}/{n_states} done "
              f"(safe={n_safe}, unsafe={done - n_safe})")
    rollouter.close()
    elapsed = time.perf_counter() - t0
    print(f"\nLabeling time: {elapsed:.1f}s ({elapsed/n_states:.3f}s/state)")

    # Stats
    n_safe = int(all_labels[:n_states].sum())
    n_unsafe = n_states - n_safe
    print(f"\nResults:")
    print(f"  Safe (label=1):   {n_safe} ({100*n_safe/n_states:.1f}%)")
    print(f"  Unsafe (label=0): {n_unsafe} ({100*n_unsafe/n_states:.1f}%)")

    # Save: copy original data (filtered) + add labels
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    for key in data.files:
        arr = data[key]
        if len(arr) == n_total:
            save_dict[key] = arr[valid_indices]
        else:
            save_dict[key] = arr
    save_dict["labels"] = all_labels[:n_states]

    np.savez_compressed(out_path, **save_dict)
    print(f"\nLabeled state pool saved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify
    loaded = np.load(out_path, allow_pickle=True)
    assert "labels" in loaded.files
    assert loaded["labels"].shape == (n_states,)
    assert loaded["states"].shape[0] == n_states
    print(f"Verification: labels shape OK ({loaded['labels'].shape}), states shape OK ({loaded['states'].shape})")


if __name__ == "__main__":
    main()
