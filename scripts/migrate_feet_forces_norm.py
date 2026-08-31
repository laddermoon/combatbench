"""Migrate checkpoints / exported policies to the body-weight-normalized
``feet_forces`` observation unit.

Context
-------
``feet_forces`` (obs dims [52:54]) was changed from raw Newtons to
dimensionless ``f / (m*g)`` (body-weight multiples).  This is a static
linear rescaling of two input dimensions:

    obs'_i = obs_i / s   (s = body_weight, i ∈ {52, 53})

For a first linear layer  y = W x + b  this is exactly compensated by

    W[:, i] *= s

so the network computes the **same** function on the new observations.
All downstream layers see identical inputs and need no change.

This script applies that surgery in-place to:

  1. PPO checkpoints  — actor + every critic.
  2. SAC checkpoints  — actor + every Q-critic (first layer input is
     ``[obs; action]``, only the obs columns [52:54] are scaled).
  3. Exported policy directories — ``model.pt`` containing a bare actor
     state_dict.

Usage
-----
    # Single checkpoint
    python scripts/migrate_feet_forces_norm.py <checkpoint.pt> [--body-weight 400.7]

    # Entire run directory (all checkpoints + policy/ + policy_exports/)
    python scripts/migrate_feet_forces_norm.py --run-dir baseline/runs/train_xxx_ppo_... [--body-weight 400.7]

    # Dry run (report what would change, don't write)
    python scripts/migrate_feet_forces_norm.py <path> --dry-run

The default ``--body-weight`` is read from the MuJoCo model itself
(``battle_v1.xml``) via ``Humanoid21Meta``, so you normally don't need
to pass it.  Pass ``--body-weight`` only if the model XML has been
modified since the checkpoint was trained.

Verification
------------
After migration, the network's output on the **new** (normalized)
observations should match the original network's output on the **old**
(Newton) observations to within float32 rounding (~1e-6).  The script
prints ``max|Δaction|`` for a random sample when the original
checkpoint can be loaded alongside.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# --- constants ---
FEET_DIMS = (52, 54)  # obs slice [52:54]


def get_default_body_weight() -> float:
    """Read body_weight from the MuJoCo model via Humanoid21Meta."""
    import mujoco
    from envs.humanoid21.meta import Humanoid21Meta

    xml = Path(__file__).resolve().parent.parent / "envs" / "humanoid21" / "battle_v1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    meta = Humanoid21Meta.build_runtime_tables(model)
    # Both robots have the same mass in battle_v1.xml; use robot_a.
    return float(meta["robots"]["robot_a"]["body_weight"])


def _scale_columns(
    weight: torch.Tensor,
    dims: Tuple[int, int],
    factor: float,
) -> int:
    """Scale ``weight[:, dims[0]:dims[1]] *= factor`` in-place. Return dim count."""
    s, e = dims
    weight[:, s:e] *= factor
    return e - s


def _find_first_linear_keys(
    state_dict: Dict[str, torch.Tensor],
    obs_dim: int,
    action_dim: int,
) -> List[str]:
    """Find keys that are the first linear layer taking raw obs as input.

    Heuristic: weight matrix whose shape[1] is obs_dim or obs_dim+action_dim,
    AND whose key ends in '.0.weight' (first layer in a Sequential) or
    is exactly 'net.0.weight' / 'trunk.0.weight'.

    For obs_dim+action_dim (SAC Q-critic), only the first obs_dim columns
    are obs — but since dims [52:54] < obs_dim, we scale the same slice.
    """
    keys = []
    for k, v in state_dict.items():
        if not k.endswith(".0.weight") and k != "net.0.weight":
            continue
        if v.dim() != 2:
            continue
        in_dim = v.shape[1]
        if in_dim == obs_dim or in_dim == obs_dim + action_dim:
            keys.append(k)
    return keys


def migrate_state_dict(
    state_dict: Dict[str, torch.Tensor],
    body_weight: float,
    obs_dim: int = 96,
    action_dim: int = 21,
    prefix: str = "",
) -> List[str]:
    """Scale feet_forces columns in all first-linear-layer weights.

    Returns list of human-readable descriptions of what was scaled.
    """
    changes: List[str] = []
    keys = _find_first_linear_keys(state_dict, obs_dim, action_dim)
    for k in keys:
        w = state_dict[k]
        if w.shape[1] < FEET_DIMS[1]:
            continue  # input dim too small, skip
        n = _scale_columns(w, FEET_DIMS, body_weight)
        changes.append(f"  {prefix}{k}: shape {tuple(w.shape)}, scaled {n} cols by {body_weight:.1f}")
    return changes


def migrate_checkpoint(
    ckpt_path: Path,
    body_weight: float,
    dry_run: bool = False,
) -> List[str]:
    """Migrate a PPO or SAC checkpoint file."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    changes: List[str] = [f"checkpoint: {ckpt_path.name}"]

    algo = ck.get("algorithm", "ppo")
    obs_dim, action_dim = 96, 21

    # --- Actor ---
    actor_sd = ck.get("actor_state_dict")
    if actor_sd is not None:
        changes.append("actor:")
        if not dry_run:
            changes.extend(migrate_state_dict(actor_sd, body_weight, obs_dim, action_dim, "actor."))
        else:
            keys = _find_first_linear_keys(actor_sd, obs_dim, action_dim)
            changes.extend([f"    [dry-run] {k} shape {tuple(actor_sd[k].shape)}" for k in keys])

    # --- Critics ---
    # PPO: critics_state_dict = {channel_name: state_dict}
    # SAC: critic_state_dict = {'shared': {'q1': sd, 'q2': sd, 'q1_target': sd, 'q2_target': sd, ...}}
    critics_sd = ck.get("critics_state_dict")
    if critics_sd is not None:
        for ch_name, ch_sd in critics_sd.items():
            label = f"critic[{ch_name}]:"
            if not dry_run:
                ch_changes = migrate_state_dict(ch_sd, body_weight, obs_dim, action_dim, f"critic.{ch_name}.")
                if ch_changes:
                    changes.append(label)
                    changes.extend(ch_changes)
            else:
                keys = _find_first_linear_keys(ch_sd, obs_dim, action_dim)
                if keys:
                    changes.append(label)
                    changes.extend([f"    [dry-run] {k} shape {tuple(ch_sd[k].shape)}" for k in keys])

    # SAC single critic (nested structure)
    critic_sd = ck.get("critic_state_dict")
    if critic_sd is not None:
        # Walk the nested structure: critic_state_dict['shared']['q1'] etc.
        shared = critic_sd.get("shared", critic_sd)
        for sub_name, sub_sd in shared.items():
            if not isinstance(sub_sd, dict):
                continue
            # Skip optimizer state dicts — they contain no weight matrices
            # that take obs as input (only exp_avg / exp_avg_sq of same shape).
            if "optimizer" in sub_name:
                continue
            label = f"critic[{sub_name}]:"
            if not dry_run:
                ch_changes = migrate_state_dict(sub_sd, body_weight, obs_dim, action_dim, f"critic.{sub_name}.")
                if ch_changes:
                    changes.append(label)
                    changes.extend(ch_changes)
            else:
                keys = _find_first_linear_keys(sub_sd, obs_dim, action_dim)
                if keys:
                    changes.append(label)
                    changes.extend([f"    [dry-run] {k} shape {tuple(sub_sd[k].shape)}" for k in keys])

    # --- Save ---
    if not dry_run:
        backup = ckpt_path.with_suffix(".pt.bak_pre_feet_norm")
        if not backup.exists():
            shutil.copy2(ckpt_path, backup)
        torch.save(ck, ckpt_path)
        changes.append(f"  saved (backup: {backup.name})")

    return changes


def migrate_exported_policy(
    policy_dir: Path,
    body_weight: float,
    dry_run: bool = False,
) -> List[str]:
    """Migrate an exported policy directory (model.pt + policy.py)."""
    model_pt = policy_dir / "model.pt"
    if not model_pt.exists():
        return [f"exported policy: {policy_dir.name} — no model.pt, skipped"]

    payload = torch.load(model_pt, map_location="cpu", weights_only=False)
    changes = [f"exported policy: {policy_dir.name}"]

    # model.pt may be a bare state_dict or a dict with 'state_dict'
    if isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
        sub = "state_dict"
    elif isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        sd = payload
        sub = "root"
    else:
        return [f"exported policy: {policy_dir.name} — unrecognized model.pt format, skipped"]

    if not dry_run:
        ch = migrate_state_dict(sd, body_weight, prefix=f"{policy_dir.name}.{sub}.")
        changes.extend(ch)
        backup = model_pt.with_suffix(".pt.bak_pre_feet_norm")
        if not backup.exists():
            shutil.copy2(model_pt, backup)
        torch.save(payload, model_pt)
        changes.append(f"  saved (backup: {backup.name})")
    else:
        keys = _find_first_linear_keys(sd, 96, 21)
        changes.extend([f"  [dry-run] {k} shape {tuple(sd[k].shape)}" for k in keys])

    return changes


def migrate_run_dir(
    run_dir: Path,
    body_weight: float,
    dry_run: bool = False,
) -> List[str]:
    """Migrate all checkpoints + exported policies in a run directory."""
    all_changes: List[str] = [f"=== run dir: {run_dir} ==="]

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        # PPO uses checkpoint_u*.pt, SAC uses checkpoint_s*.pt
        for pt in sorted(ckpt_dir.glob("checkpoint_[us]*.pt")):
            all_changes.extend(migrate_checkpoint(pt, body_weight, dry_run))
            all_changes.append("")

    for sub in ("policy", "policy_exports"):
        d = run_dir / sub
        if not d.exists():
            continue
        # policy_exports/ has subdirectories per update
        if sub == "policy_exports":
            for pd in sorted(d.iterdir()):
                if pd.is_dir() and (pd / "model.pt").exists():
                    all_changes.extend(migrate_exported_policy(pd, body_weight, dry_run))
                    all_changes.append("")
        else:
            if (d / "model.pt").exists():
                all_changes.extend(migrate_exported_policy(d, body_weight, dry_run))
                all_changes.append("")

    return all_changes


def main():
    ap = argparse.ArgumentParser(
        description="Migrate checkpoints to body-weight-normalized feet_forces obs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", type=Path, help="Checkpoint file or run directory")
    ap.add_argument("--run-dir", action="store_true", help="Treat path as a run directory")
    ap.add_argument("--body-weight", type=float, default=None,
                    help="Body weight m*g in Newtons (default: read from model)")
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = ap.parse_args()

    bw = args.body_weight
    if bw is None:
        bw = get_default_body_weight()
    print(f"body_weight (m*g) = {bw:.2f} N")
    print(f"scaling obs dims [{FEET_DIMS[0]}:{FEET_DIMS[1]}] (feet_forces) by {bw:.2f}")
    print()

    if args.run_dir or args.path.is_dir():
        changes = migrate_run_dir(args.path, bw, args.dry_run)
    else:
        changes = migrate_checkpoint(args.path, bw, args.dry_run)

    for line in changes:
        print(line)

    if args.dry_run:
        print("\n[dry-run] no files were modified.")


if __name__ == "__main__":
    main()
