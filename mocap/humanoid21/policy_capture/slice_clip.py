"""Slice a recorded policy-capture episode into a clip.

Takes a ``ep_<seed>_<robot>.npz`` produced by ``TrajectoryRecorder`` plus
its ``ep_<seed>_meta.json``, cuts ``[start_step, end_step)`` on every
time-indexed array, and writes ``<out>.npz`` + ``<out>.meta.json`` that
carry the source provenance and window.  Optionally cuts the matching
episode mp4 by the same time range (video fps vs action fps).

Usage::

    PYTHONPATH=. python3 -m mocap.humanoid21.policy_capture.slice_clip \
        --npz ep_1010_robot_a.npz --meta ep_1010_meta.json \
        --start 10 --end 60 --robot robot_a \
        --out clips/guard_jab_r_u08760_robot_a.npz \
        --video-in ep_1010.mp4 --video-out clips/guard_jab_r_u08760.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def slice_npz(npz_path: str, start: int, end: int, out_path: str) -> dict:
    d = np.load(npz_path)
    out = {}
    for k in d.files:
        arr = d[k]
        if arr.ndim >= 1 and arr.shape[0] == d["timestamps"].shape[0]:
            out[k] = arr[start:end]
        else:
            out[k] = arr
    out["timestamps"] = np.arange(
        end - start, dtype=np.float32) / (1.0 / (d["timestamps"][1] - d["timestamps"][0]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return {"n_frames": int(end - start), "arrays": sorted(out)}


def slice_video(src: str, dst: str, start_s: float, end_s: float,
                fps: float) -> None:
    import imageio.v2 as imageio
    rd = imageio.get_reader(src)
    meta = rd.get_meta_data()
    vfps = float(meta.get("fps", fps))
    wr = imageio.get_writer(dst, fps=vfps)
    try:
        for i, frame in enumerate(rd):
            t = i / vfps
            if t < start_s - 1e-6:
                continue
            if t >= end_s - 1e-6:
                break
            wr.append_data(frame)
    finally:
        wr.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--start", type=int, required=True,
                    help="start action step (inclusive)")
    ap.add_argument("--end", type=int, required=True,
                    help="end action step (exclusive)")
    ap.add_argument("--out", required=True, help="clip npz path")
    ap.add_argument("--video-in", default=None)
    ap.add_argument("--video-out", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    src_meta = json.load(open(args.meta))
    fps = float(src_meta["fps_action"])

    info = slice_npz(args.npz, args.start, args.end, args.out)
    clip_meta = {
        "label": args.label,
        "robot": args.robot,
        "source": {
            "npz": str(Path(args.npz).resolve()),
            "meta": str(Path(args.meta).resolve()),
        },
        "window_steps": {"start": args.start, "end": args.end},
        "window_seconds": {
            "start": args.start / fps, "end": args.end / fps},
        "n_frames": info["n_frames"],
        "fps_action": fps,
        "provenance": src_meta.get("provenance"),
        "termination_reasons": src_meta.get("termination_reasons"),
        "body_names": src_meta.get("body_names", {}).get(args.robot),
    }
    meta_out = Path(args.out).with_suffix(".meta.json")
    with open(meta_out, "w") as f:
        json.dump(clip_meta, f, indent=1)

    if args.video_in and args.video_out:
        Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
        slice_video(args.video_in, args.video_out,
                    args.start / fps, args.end / fps, fps)
    print(json.dumps({"clip": args.out, "meta": str(meta_out),
                      "n_frames": info["n_frames"]}, indent=1))


if __name__ == "__main__":
    main()
