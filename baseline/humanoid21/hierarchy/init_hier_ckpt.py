"""Migrate a 96-dim checkpoint to the 65-dim hierarchy contract.

Slices the actor/critic first-layer weight matrices from (H, 96) to
(H, 65): keeps columns [0:57] (proprioception) and appends 8 zero
columns for the new command dims.  The resulting checkpoint is
compatible with --resume-from: all other layers pass through
unchanged, missing critics (r_fall/r_cross/r_guard) get fresh init.

Usage::

    python3 baseline/humanoid21/hierarchy/init_hier_ckpt.py \
        --src baseline/runs/standup_mbs_ef05_s42/checkpoints/checkpoint_u00350.pt \
        --dst baseline/humanoid21/hierarchy/ckpts/hier_s0_init.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

BODY_DIM = 57
CMD_DIM = 8
NEW_OBS_DIM = BODY_DIM + CMD_DIM  # 65


def _migrate_2d(w: torch.Tensor, tag: str) -> torch.Tensor:
    """(H, 96) → (H, 65): keep [0:57], zero-pad [57:65]."""
    if w.ndim != 2 or w.shape[1] != 96:
        print(f"  skip {tag}: shape {tuple(w.shape)} (not (?, 96))")
        return w
    body = w[:, :BODY_DIM]          # (H, 57)
    pad = torch.zeros(
        w.shape[0], CMD_DIM, dtype=w.dtype, device=w.device)
    out = torch.cat([body, pad], dim=1)  # (H, 65)
    print(f"  {tag}: {tuple(w.shape)} → {tuple(out.shape)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Source 96-dim checkpoint path")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Destination 65-dim checkpoint path")
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    print(f"[init_hier_ckpt] loading {src.name} …")
    payload = torch.load(src, map_location="cpu", weights_only=False)

    # --- Actor ---
    actor_sd = payload["actor_state_dict"]
    print("[init_hier_ckpt] actor:")
    for k in list(actor_sd.keys()):
        actor_sd[k] = _migrate_2d(actor_sd[k], f"actor.{k}")

    # --- Critics ---
    critics_sd = payload.get("critics_state_dict", {})
    for ch_name, ch_sd in critics_sd.items():
        print(f"[init_hier_ckpt] critic '{ch_name}':")
        for k in list(ch_sd.keys()):
            ch_sd[k] = _migrate_2d(ch_sd[k], f"critic.{ch_name}.{k}")

    # --- Clear optimizer moments (param shapes changed → stale) ---
    # Keep the keys present so load_checkpoint's dict access doesn't
    # raise KeyError; empty ``state`` = fresh optimizer moments.
    if "actor_optimizer_state_dict" in payload:
        payload["actor_optimizer_state_dict"]["state"] = {}
        print("[init_hier_ckpt] cleared actor optimizer state")
    if "critic_optimizers_state_dict" in payload:
        for k in payload["critic_optimizers_state_dict"]:
            payload["critic_optimizers_state_dict"][k]["state"] = {}
        print("[init_hier_ckpt] cleared critic optimizer states")

    # --- Update metadata ---
    payload["update"] = 0
    payload.pop("loop_state", None)
    payload.pop("rollout_state", None)
    payload.pop("rng_state", None)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)
    print(f"[init_hier_ckpt] saved → {dst}")


if __name__ == "__main__":
    main()
