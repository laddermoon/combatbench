"""Render broadcast-view images from a saved state pool .npz file.

Loads each state, injects it into a Humanoid21Simulator, and captures
a broadcast-view image. Useful for visual inspection of state diversity.

Usage::

    python3 baseline/humanoid21/balance_recover/gating/render_state_images.py \\
        --input state_pool.npz \\
        --output-dir state_pool_images \\
        --max-images 100
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

CORE_STATE_FIELDS = [
    "root_pos",
    "root_rot",
    "root_vel_local",
    "root_angular_vel_local",
    "joint_pos_norm",
    "joint_vel_norm",
]
CORE_STATE_DIMS = [3, 4, 3, 3, 21, 21]


def unflatten_core_state(vec: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    offset = 0
    for name, dim in zip(CORE_STATE_FIELDS, CORE_STATE_DIMS):
        out[name] = vec[offset:offset + dim].astype(np.float32)
        offset += dim
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render state pool images")
    parser.add_argument("--input", type=str, required=True, help="Input .npz path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output image directory")
    parser.add_argument("--max-images", type=int, default=100, help="Max images to render")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in state pool")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    states = data["states"]
    n_total = len(states)

    rng = np.random.RandomState(args.seed)
    n = min(args.max_images, n_total)
    indices = rng.choice(n_total, size=n, replace=False)
    indices.sort()
    if n <= 0:
        print("No states to render.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from envs.humanoid21.simulator import Humanoid21Simulator

    sim = Humanoid21Simulator()
    sim.reset()

    import imageio.v2 as imageio

    for i, idx in enumerate(indices):
        state_vec = states[idx]
        core_state = unflatten_core_state(state_vec)

        sim.set_core_state({"robot_a": core_state})
        image = sim.get_broadcastview_image()

        img_path = out_dir / f"state_{idx:06d}.png"
        imageio.imwrite(str(img_path), np.asarray(image))

        if (i + 1) % 10 == 0:
            print(f"Rendered {i + 1}/{n} images")

    print(f"\nDone. {n} images saved to {out_dir}")


if __name__ == "__main__":
    main()
