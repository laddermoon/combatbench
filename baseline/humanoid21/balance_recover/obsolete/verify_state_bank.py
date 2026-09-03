"""Verify StateBankInitPlugin: inject states from a state bank and check
that survival/fall outcomes match the labels recorded in the bank.

Workflow:
1. Load the state bank (.npz)
2. For each state in the bank, run an episode with StateBankInitPlugin
   injecting that exact state (via state_bank_index option)
3. Compare the resulting fall/survive with the bank's label
4. Report match rate

Usage::

    python3 baseline/framework/verify_state_bank.py \
        --policy-export baseline/runs/<run>/policy \
        --state-bank /tmp/state_bank_verify.npz \
        --workers 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.rollout import ParallelRollouter
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


def _build_verify_blueprint(
    env_pb: ParameterizedEnvBlueprint,
    policy_path_abs: str,
    state_bank_path_abs: str,
    agent_id: str,
    max_steps: int,
    tolerance: int,
) -> EnvBlueprint:
    """Materialize a blueprint with StateBankInitPlugin instead of ImpulsePerturbationPlugin."""
    env_bp = env_pb.materialize(
        max_steps=max_steps,
        agent_id=agent_id,
        tolerance=tolerance,
        policy_blueprint_path=policy_path_abs,
        force_magnitude=0,
        duration_action_steps=0,
        direction_mode="random_horizontal",
    )

    d = env_bp.to_dict()
    new_plugins = []
    for spec in d.get("plugins", []):
        cls = spec.get("cls", "")
        if "ImpulsePerturbationPlugin" in cls:
            new_plugins.append({
                "cls": "envs.humanoid21.disturbance_plugins:StateBankInitPlugin",
                "config": {
                    "state_bank_path": state_bank_path_abs,
                    "target_robot": agent_id,
                    "seed": 42,
                },
            })
        elif "StateCapturePlugin" in cls:
            continue
        else:
            new_plugins.append(spec)
    d["plugins"] = new_plugins

    observer_plugins = d.get("observer_plugins", {})
    observer_plugins.pop("state_capture", None)
    d["observer_plugins"] = observer_plugins

    return EnvBlueprint.from_dict(d)


def main() -> None:
    p = argparse.ArgumentParser(description="Verify StateBankInitPlugin")
    p.add_argument("--policy-export", required=True)
    p.add_argument("--state-bank", required=True, help="Path to .npz state bank file")
    p.add_argument("--blueprint", type=str,
                   default="baseline/humanoid21/blueprints/impulse_boundary_env.yaml")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--tolerance", type=int, default=6)
    p.add_argument("--agent-id", type=str, default="robot_a")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    policy_bp = PolicyBlueprint.load(Path(args.policy_export) / "policy_blueprint.yaml")
    policy_path_abs = str((Path(args.policy_export) / "policy_blueprint.yaml").resolve())
    state_bank_path_abs = str(Path(args.state_bank).resolve())

    env_pb = ParameterizedEnvBlueprint.load(args.blueprint)
    env_bp = _build_verify_blueprint(
        env_pb, policy_path_abs, state_bank_path_abs,
        args.agent_id, args.max_steps, args.tolerance,
    )

    bank = np.load(state_bank_path_abs, allow_pickle=True)
    n = len(bank["states"])
    bank_labels = bank["labels"]
    bank_ep_lengths = bank["ep_lengths"]

    print(f"=== StateBankInitPlugin Verification ===")
    print(f"state bank: {state_bank_path_abs}")
    print(f"states: {n}")
    print(f"bank labels: survived={int(bank_labels.sum())}  fell={int(n - bank_labels.sum())}")
    print(f"workers: {args.workers}")
    print()

    all_jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
    for i in range(n):
        seed = args.seed + i
        all_jobs.append((
            policy_bp, policy_bp,
            env_bp, seed,
            {
                "agent_id": args.agent_id,
                "initial_distance": 2.0,
                "state_bank_index": i,
            },
        ))

    total = len(all_jobs)
    print(f"Total episodes: {total}")
    t0 = time.perf_counter()

    rollouter = ParallelRollouter(num_workers=args.workers)
    episodes = rollouter.collect(all_jobs)
    rollouter.close()

    elapsed = time.perf_counter() - t0
    print(f"Rollout time: {elapsed:.1f}s ({elapsed/total:.2f}s/episode)")
    print()

    matches = 0
    mismatches: List[Tuple[int, float, float, int, int]] = []
    verify_labels = []
    verify_ep_lengths = []

    for i, ep in enumerate(episodes):
        fell = all(r.startswith("imbalance") for r in ep.agent_termination_reason.values())
        verify_label = 0.0 if fell else 1.0
        verify_labels.append(verify_label)
        verify_ep_lengths.append(ep.num_frames)

        bank_label = float(bank_labels[i])
        if verify_label == bank_label:
            matches += 1
        else:
            mismatches.append((
                i, bank_label, verify_label,
                int(bank_ep_lengths[i]), ep.num_frames,
            ))

    verify_labels = np.array(verify_labels, dtype=np.float32)
    verify_ep_lengths = np.array(verify_ep_lengths, dtype=np.int32)

    match_rate = matches / n
    print(f"=== Results ===")
    print(f"Matches:   {matches}/{n}  ({match_rate:.1%})")
    print(f"Mismatches: {n - matches}/{n}")
    print()

    if mismatches:
        print(f"{'idx':>4} {'bank_label':>11} {'verify_label':>13} {'bank_len':>9} {'verify_len':>11}")
        print("-" * 55)
        for idx, bl, vl, blen, vlen in mismatches:
            print(f"{idx:>4d} {bl:>11.0f} {vl:>13.0f} {blen:>9d} {vlen:>11d}")
    else:
        print("All labels match! StateBankInitPlugin is verified.")

    label_corr = np.corrcoef(bank_labels, verify_labels)[0, 1] if n > 1 else 1.0
    len_corr = np.corrcoef(bank_ep_lengths.astype(float), verify_ep_lengths.astype(float))[0, 1] if n > 1 else 1.0
    print(f"\nLabel correlation: {label_corr:.4f}")
    print(f"Ep length correlation: {len_corr:.4f}")
    print(f"Bank ep_len:  mean={bank_ep_lengths.mean():.1f}  min={bank_ep_lengths.min()}  max={bank_ep_lengths.max()}")
    print(f"Verify ep_len: mean={verify_ep_lengths.mean():.1f}  min={verify_ep_lengths.min()}  max={verify_ep_lengths.max()}")


if __name__ == "__main__":
    main()
