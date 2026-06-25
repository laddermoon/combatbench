"""Visualize retargeted MoCap motion in MuJoCo.

Loads the humanoid21 model, applies retargeted joint angles per frame,
and renders the motion for visual verification.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mujoco
import mujoco.viewer

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from baseline.humanoid21.mocap.amc_parser import parse_amc
from baseline.humanoid21.mocap.asf_parser import parse_asf
from baseline.humanoid21.mocap.retarget_v2 import retarget_motion, JOINT_ORDER, JOINT_LIMITS, angles_to_normalized_action
from envs.humanoid21.meta import Humanoid21Meta


def load_model():
    """Load the humanoid21 MuJoCo model (single robot)."""
    arena_xml = Humanoid21Meta.ARENA_XML
    spec = mujoco.MjSpec.from_file(arena_xml)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def apply_joint_angles(model, data, joint_angles_deg, robot_suffix="_a"):
    """Apply joint angles (degrees) to the MuJoCo model.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_angles_deg: (21,) array of joint angles in degrees
        robot_suffix: "_a" or "_b"
    """
    for i, joint_name in enumerate(JOINT_ORDER):
        full_name = f"{joint_name}{robot_suffix}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
        if jid < 0:
            continue
        qpos_adr = model.jnt_qposadr[jid]
        data.qpos[qpos_adr] = np.radians(joint_angles_deg[i])

    mujoco.mj_forward(model, data)


def visualize_motion(motion_deg, fps=30, robot_suffix="_a"):
    """Visualize retargeted motion in MuJoCo viewer.

    Args:
        motion_deg: (T, 21) array of joint angles in degrees
        fps: playback framerate
        robot_suffix: which robot to animate
    """
    model, data = load_model()

    # Set initial standing pose
    standing = Humanoid21Meta.INITIAL_POSES["standing"]
    root_pos_adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"root{robot_suffix}")
    if root_pos_adr >= 0:
        qpos_adr = model.jnt_qposadr[root_pos_adr]
        data.qpos[qpos_adr:qpos_adr + 3] = standing["root_pos"]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = standing["root_quat"]

    # Apply initial joint angles
    apply_joint_angles(model, data, motion_deg[0], robot_suffix)

    dt = 1.0 / fps
    model.opt.timestep = dt

    frame_idx = 0
    total_frames = len(motion_deg)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and frame_idx < total_frames:
            apply_joint_angles(model, data, motion_deg[frame_idx], robot_suffix)
            mujoco.mj_forward(model, data)
            viewer.sync()
            frame_idx += 1

            # Wait for next frame
            import time
            time.sleep(dt)

    print(f"Played {frame_idx}/{total_frames} frames")


def save_motion_video(motion_deg, output_path, fps=30, robot_suffix="_a"):
    """Save retargeted motion as a video using mujoco rendering.

    Args:
        motion_deg: (T, 21) array of joint angles in degrees
        output_path: path to save video
        fps: framerate
        robot_suffix: which robot to animate
    """
    import mediapy as media

    model, data = load_model()
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Set initial pose
    standing = Humanoid21Meta.INITIAL_POSES["standing"]
    root_pos_adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"root{robot_suffix}")
    if root_pos_adr >= 0:
        qpos_adr = model.jnt_qposadr[root_pos_adr]
        data.qpos[qpos_adr:qpos_adr + 3] = standing["root_pos"]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = standing["root_quat"]

    frames = []
    for t in range(len(motion_deg)):
        apply_joint_angles(model, data, motion_deg[t], robot_suffix)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="side" if "side" in [model.camera(i).name for i in range(model.ncam)] else 0)
        pixels = renderer.render()
        frames.append(pixels)

    media.write_video(output_path, frames, fps=fps)
    print(f"Saved {len(frames)} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize retargeted MoCap motion")
    parser.add_argument("--asf", default="baseline/humanoid21/mocap/data/raw/13.asf")
    parser.add_argument("--amc", default="baseline/humanoid21/mocap/data/raw/13_01.amc")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video", default=None, help="Save video to this path instead of interactive viewer")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames")
    args = parser.parse_args()

    frames = parse_amc(args.amc)
    if args.max_frames:
        frames = frames[:args.max_frames]

    print(f"Loaded {len(frames)} frames from {args.amc}")
    motion = retarget_motion(frames)
    print(f"Retargeted motion shape: {motion.shape}")

    if args.video:
        save_motion_video(motion, args.video, fps=args.fps)
    else:
        visualize_motion(motion, fps=args.fps)


if __name__ == "__main__":
    main()
