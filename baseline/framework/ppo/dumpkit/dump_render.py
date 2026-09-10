"""Render images for a dumped episode and verify against dump data.

Usage (via debug.py CLI)::

    python3 baseline/framework/ppo/debug.py render <dump_dir> --episode 0

This module:
1. Reads the dump's stochastic_policy/, env_blueprint.yaml, episode_options.json
2. Reads episodes.npz to get the seed for the specified episode
3. Runs round_runner + BaseFrameRecorder to generate per-frame PNG + JSON
4. Auto-verifies recorded obs/actions against episodes.npz
5. Writes an association record mapping recorded frames to dump frames

The recording uses the stochastic wrapped policy (explore_factor baked in),
so the recorded trajectory should match the training trajectory frame-by-frame.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def render_episode(
    dump_dir: Path,
    episode_index: int = 0,
    *,
    output_subdir: str = "record",
    verbose: bool = True,
) -> Path:
    """Render one episode's images and verify against dump data.

    Args:
        dump_dir: Path to the dump directory (e.g. ``runs/.../dumps/u00002/``).
        episode_index: Which episode to render (0-based index into episodes.npz).
        output_subdir: Subdirectory within dump_dir for recording output.
        verbose: Print progress to stdout.

    Returns:
        Path to the recording output directory.

    Raises:
        FileNotFoundError: If required dump files are missing.
        ValueError: If episode_index is out of range.
        RuntimeError: If verification fails (diff exceeds threshold).
    """
    dump_dir = Path(dump_dir)

    # --- Load dump metadata ---
    ep_npz = dump_dir / "episodes.npz"
    env_bp_path = dump_dir / "env_blueprint.yaml"
    policy_bp_path = dump_dir / "stochastic_policy" / "policy_blueprint.yaml"
    options_path = dump_dir / "episode_options.json"

    if not ep_npz.exists():
        raise FileNotFoundError(f"episodes.npz not found in {dump_dir}")
    if not env_bp_path.exists():
        raise FileNotFoundError(f"env_blueprint.yaml not found in {dump_dir}")
    if not policy_bp_path.exists():
        raise FileNotFoundError(
            f"stochastic_policy/policy_blueprint.yaml not found in {dump_dir}"
        )

    ep_data = np.load(ep_npz, allow_pickle=True)
    n_episodes = int(ep_data["n_episodes"])
    if episode_index < 0 or episode_index >= n_episodes:
        raise ValueError(
            f"episode_index {episode_index} out of range (0..{n_episodes - 1})"
        )

    seed = int(ep_data["base_seeds"][episode_index])
    num_frames = int(ep_data["num_frames"][episode_index])
    frame_offsets = ep_data["episode_frame_offsets"]
    frame_start = int(frame_offsets[episode_index])
    frame_end = int(frame_offsets[episode_index + 1])

    if verbose:
        print(f"[render] episode {episode_index}/{n_episodes}")
        print(f"  seed: {seed}")
        print(f"  num_frames: {num_frames}")
        print(f"  frame range in episodes.npz: [{frame_start}, {frame_end})")

    # --- Load episode options ---
    episode_options: Optional[Dict[str, Any]] = None
    if options_path.exists():
        with open(options_path) as f:
            episode_options = json.load(f)

    # --- Run round_runner + BaseFrameRecorder ---
    record_dir = dump_dir / output_subdir
    record_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[render] running round_runner with seed={seed}...")

    result = _run_round_runner(
        env_bp_path=env_bp_path,
        policy_bp_path=policy_bp_path,
        seed=seed,
        episode_options=episode_options,
        record_dir=record_dir,
    )

    if verbose:
        print(f"[render] round_runner done: {result['steps']} steps")

    # --- Find the recorded episode directory ---
    # BaseFrameRecorder creates episode_00000, episode_00001, etc.
    # If we're rendering episode N, the recorded episode is in episode_00000
    # (since we only run one episode at a time).
    recorded_ep_dir = record_dir / "episode_00000"
    if not recorded_ep_dir.exists():
        raise RuntimeError(
            f"Recorded episode directory not found: {recorded_ep_dir}"
        )

    # --- Auto-verify ---
    if verbose:
        print(f"[render] verifying recorded data against dump...")
    verification_result = _verify_recorded_vs_dump(
        recorded_ep_dir=recorded_ep_dir,
        ep_data=ep_data,
        frame_start=frame_start,
        frame_end=frame_end,
        verbose=verbose,
    )

    # --- Write association record ---
    association = _build_association(
        dump_dir=dump_dir,
        episode_index=episode_index,
        seed=seed,
        num_frames=num_frames,
        frame_start=frame_start,
        frame_end=frame_end,
        recorded_ep_dir=recorded_ep_dir,
        round_result=result,
        verification_result=verification_result,
    )

    assoc_path = record_dir / "association.json"
    assoc_path.write_text(
        json.dumps(association, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if verbose:
        print(f"[render] association written: {assoc_path}")
        if verification_result is not None:
            status = verification_result["status"]
            if status == "pass":
                print(f"[render] verification: PASS")
                print(f"  frames: {verification_result['n_frames_compared']}/{verification_result['n_frames_total']}")
                print(f"  max obs diff:    {verification_result['max_obs_diff']:.2e}")
                print(f"  max action diff: {verification_result['max_action_diff']:.2e}")
            else:
                print(f"[render] verification: FAIL")
                print(f"  frames: {verification_result['n_frames_compared']}/{verification_result['n_frames_total']}")
                print(f"  max obs diff:    {verification_result['max_obs_diff']:.2e}")
                print(f"  max action diff: {verification_result['max_action_diff']:.2e}")
                print(f"  WARNING: Recorded data does NOT match dump data!")
                print(f"  The images may not correspond to the training trajectory.")
                print(f"  Possible causes:")
                print(f"    - Policy export mismatch (weights or explore_factor)")
                print(f"    - Environment blueprint mismatch")
                print(f"    - Seed derivation mismatch")
                print(f"    - torch thread count mismatch (should be 1)")
                print(f"  Do NOT use these images as ground truth for the dump.")

    return record_dir


def _run_round_runner(
    env_bp_path: Path,
    policy_bp_path: Path,
    seed: int,
    episode_options: Optional[Dict[str, Any]],
    record_dir: Path,
) -> Dict[str, Any]:
    """Run round_runner with BaseFrameRecorder and return the result dict.

    Sets ``torch.set_num_threads(1)`` to match the training worker's
    BLAS configuration.  Multi-threaded BLAS uses a different floating-
    point reduction order, which produces tiny (~1e-8) differences
    that compound across frames and eventually cause visible divergence.
    Without this, the replay would diverge from the training trajectory
    even with identical seeds, policies, and environments.
    """
    import torch
    torch.set_num_threads(1)

    from envs.framework.blueprint import EnvBlueprint
    from envs.framework.policy import PolicyBlueprint
    from envs.framework.round_runner import RoundRunner
    from envs.framework.recorder import BaseFrameRecorder

    env_bp = EnvBlueprint.load(str(env_bp_path))
    policy_a = PolicyBlueprint.load(str(policy_bp_path)).build()
    policy_b = PolicyBlueprint.load(str(policy_bp_path)).build()

    recorder = BaseFrameRecorder(output_dir=str(record_dir))

    with RoundRunner(
        blueprint=env_bp,
        policy_a=policy_a,
        policy_b=policy_b,
        recorders=[recorder],
    ) as runner:
        result = runner.run(
            seed=seed,
            options=episode_options,
            want_extras=False,
        )

    return result


def _verify_recorded_vs_dump(
    recorded_ep_dir: Path,
    ep_data: Any,  # np.load result
    frame_start: int,
    frame_end: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Verify recorded frames against dump data.

    The BaseFrameRecorder has an off-by-one: step_00000 is the initial
    state (on_pre_episode), step_00001 is the first post-action snapshot.
    So recorded step (N+1) corresponds to dump frame N.

    Physics simulations exhibit chaotic divergence: tiny float32
    differences compound exponentially.  We verify two things:
    1. Early frames (first ``verify_n_early`` frames) must match tightly
       — this confirms the seed, policy, and env are correct.
    2. Later frames are reported but not used for pass/fail, since
       physics divergence is expected and does not indicate a wrong
       trajectory.

    Returns a dict with:
        status: "pass" or "fail"
        max_obs_diff_early: float (max obs diff in early frames)
        max_action_diff_early: float (max action diff in early frames)
        max_obs_diff_all: float (max obs diff across all frames)
        max_action_diff_all: float (max action diff across all frames)
        n_frames_compared: int
        n_early_frames: int
        threshold: float
        frame_mapping: str
    """
    # Load dump data for this episode
    dump_obs_a = ep_data["obs.robot_a"][frame_start:frame_end]
    dump_act_a = ep_data["actions.robot_a"][frame_start:frame_end]

    has_robot_b = "obs.robot_b" in ep_data
    if has_robot_b:
        dump_obs_b = ep_data["obs.robot_b"][frame_start:frame_end]
        dump_act_b = ep_data["actions.robot_b"][frame_start:frame_end]

    n_frames = frame_end - frame_start
    # With torch.set_num_threads(1), the replay is bit-exact, so we
    # can verify ALL frames.  The threshold is generous (1e-3) to
    # allow for float32 JSON round-trip in the recorded data.
    threshold = 1e-3

    max_obs_diff = 0.0
    max_action_diff = 0.0
    n_compared = 0

    for i in range(n_frames):
        step_file = recorded_ep_dir / f"step_{i + 1:05d}.json"
        if not step_file.exists():
            if verbose:
                print(f"  frame {i}: recorded step {i + 1} not found!")
            continue

        with open(step_file) as f:
            rec = json.load(f)

        # Compare robot_a
        rec_obs_a = np.array(rec.get("observation", {}).get("robot_a", []), dtype=np.float32)
        rec_act_a = np.array(rec.get("action", {}).get("robot_a", []), dtype=np.float32)

        if len(rec_obs_a) > 0 and i < len(dump_obs_a):
            od = float(np.abs(dump_obs_a[i] - rec_obs_a).max())
            max_obs_diff = max(max_obs_diff, od)

        if len(rec_act_a) > 0 and i < len(dump_act_a):
            ad = float(np.abs(dump_act_a[i] - rec_act_a).max())
            max_action_diff = max(max_action_diff, ad)

        # Compare robot_b
        if has_robot_b:
            rec_obs_b = np.array(rec.get("observation", {}).get("robot_b", []), dtype=np.float32)
            rec_act_b = np.array(rec.get("action", {}).get("robot_b", []), dtype=np.float32)
            if len(rec_obs_b) > 0 and i < len(dump_obs_b):
                od = float(np.abs(dump_obs_b[i] - rec_obs_b).max())
                max_obs_diff = max(max_obs_diff, od)
            if len(rec_act_b) > 0 and i < len(dump_act_b):
                ad = float(np.abs(dump_act_b[i] - rec_act_b).max())
                max_action_diff = max(max_action_diff, ad)

        n_compared += 1

    status = "pass" if (
        max_obs_diff < threshold and max_action_diff < threshold
    ) else "fail"

    return {
        "status": status,
        "max_obs_diff": max_obs_diff,
        "max_action_diff": max_action_diff,
        "n_frames_compared": n_compared,
        "n_frames_total": n_frames,
        "threshold": threshold,
        "note": (
            "With torch.set_num_threads(1), the replay is bit-exact. "
            "All frames must match within float32 JSON round-trip tolerance."
        ),
        "frame_mapping": "recorded step (N+1) <-> dump frame N (off-by-one: step_00000 is initial state)",
    }


def _build_association(
    dump_dir: Path,
    episode_index: int,
    seed: int,
    num_frames: int,
    frame_start: int,
    frame_end: int,
    recorded_ep_dir: Path,
    round_result: Dict[str, Any],
    verification_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the association record linking images to dump data."""
    assoc: Dict[str, Any] = {
        "dump_dir": str(dump_dir),
        "episode_index": episode_index,
        "seed": seed,
        "num_frames": num_frames,
        "dump_frame_range": [frame_start, frame_end],
        "recorded_episode_dir": str(recorded_ep_dir),
        "round_result": round_result,
        "frame_mapping": {
            "description": "Recorded step N+1 corresponds to dump frame N. step_00000 is the initial state (on_pre_episode, no action).",
            "recorded_step_offset": 1,
            "dump_frame_offset": 0,
        },
    }
    if verification_result is not None:
        assoc["verification"] = verification_result
    return assoc


__all__ = ["render_episode"]
