"""Reduce log_std in a checkpoint to give the policy more precise control.

Usage:
    python3 baseline/humanoid21/scripts/reduce_std.py <checkpoint_path> <target_log_std>
"""
import sys
import torch
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 reduce_std.py <checkpoint_path> <target_log_std>")
        sys.exit(1)

    ckpt_path = Path(sys.argv[1])
    target_log_std = float(sys.argv[2])

    payload = torch.load(ckpt_path, map_location="cpu")

    # Try actor_state_dict first, then state_dict
    state_dict = payload.get("actor_state_dict", payload.get("state_dict", payload))

    # Find log_std parameter
    log_std_key = None
    for k in state_dict:
        if "log_std" in k:
            log_std_key = k
            break

    if log_std_key is None:
        print("ERROR: No log_std parameter found in checkpoint")
        print("Available keys:", list(state_dict.keys()))
        sys.exit(1)

    old_log_std = state_dict[log_std_key]
    old_std = old_log_std.exp()
    print(f"Old log_std: mean={float(old_log_std.mean()):.4f} std={float(old_std.mean()):.4f}")
    print(f"Setting to: {target_log_std:.4f} (std={torch.exp(torch.tensor(target_log_std)):.4f})")

    state_dict[log_std_key] = torch.full_like(old_log_std, target_log_std)

    # Save modified checkpoint
    out_path = ckpt_path.parent / (ckpt_path.stem + f"_std{target_log_std:.1f}.pt")
    if "actor_state_dict" in payload:
        payload["actor_state_dict"] = state_dict
    else:
        payload["state_dict"] = state_dict
    torch.save(payload, out_path)
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
