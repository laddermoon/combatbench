"""Download CMU boxing MoCap data, retarget to humanoid21, and save.

Pipeline:
  1. Download ASF skeleton + AMC motion files for CMU boxing subjects (13, 14, 15, 17)
  2. Retarget each motion to humanoid21 21-DOF joint angles
  3. Save as .npy files (T, 21) in degrees
  4. Also save normalized actions [-1, 1] for direct policy training

Usage:
  python3 -m baseline.humanoid21.mocap.prepare_data --output-dir baseline/humanoid21/mocap/data/retargeted
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import List

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from baseline.humanoid21.mocap.amc_parser import parse_amc
from baseline.humanoid21.mocap.retarget_v2 import retarget_motion, angles_to_normalized_action, JOINT_ORDER

CMU_BASE_URL = "http://mocap.cs.cmu.edu/subjects"

# Boxing subjects and their motion numbers
BOXING_SUBJECTS = {
    13: list(range(1, 25)),   # 13_01 to 13_24
    14: [1, 2, 3],
    15: [13],
    17: [10],
}


def download_file(url: str, dest: str, use_proxy: bool = True) -> bool:
    """Download a file with optional proxy."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if use_proxy:
        proxy = os.environ.get("http_proxy", "http://192.168.16.76:18000")
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy,
            "https": proxy,
        })
        urllib.request.install_opener(urllib.request.build_opener(proxy_handler))

    try:
        print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def download_boxing_data(raw_dir: str) -> List[tuple]:
    """Download all boxing ASF/AMC files.

    Returns:
        List of (subject_num, motion_num, asf_path, amc_path) tuples for successful downloads
    """
    raw_dir = os.path.abspath(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)

    results = []

    for subject_num, motion_nums in BOXING_SUBJECTS.items():
        # Download ASF (one per subject)
        asf_url = f"{CMU_BASE_URL}/{subject_num}/{subject_num}.asf"
        asf_path = os.path.join(raw_dir, f"{subject_num}.asf")
        if not download_file(asf_url, asf_path):
            print(f"  Skipping subject {subject_num}: no ASF")
            continue

        # Download AMC files
        for motion_num in motion_nums:
            amc_name = f"{subject_num}_{motion_num:02d}.amc"
            amc_url = f"{CMU_BASE_URL}/{subject_num}/{amc_name}"
            amc_path = os.path.join(raw_dir, amc_name)
            if download_file(amc_url, amc_path):
                results.append((subject_num, motion_num, asf_path, amc_path))

    return results


def retarget_and_save(
    asf_path: str,
    amc_path: str,
    output_dir: str,
    subject_num: int,
    motion_num: int,
) -> str:
    """Retarget a single motion and save.

    Returns:
        Path to saved .npy file
    """
    frames = parse_amc(amc_path)
    if len(frames) == 0:
        print(f"  No frames in {amc_path}")
        return None

    motion = retarget_motion(frames)

    # Save in degrees
    name = f"{subject_num}_{motion_num:02d}"
    out_path = os.path.join(output_dir, f"{name}.npy")
    os.makedirs(output_dir, exist_ok=True)
    np.save(out_path, motion)

    # Also save normalized actions
    norm_motion = angles_to_normalized_action(motion)
    norm_path = os.path.join(output_dir, f"{name}_norm.npy")
    np.save(norm_path, norm_motion)

    print(f"  Saved {name}: {motion.shape} → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare CMU boxing MoCap data for humanoid21")
    parser.add_argument("--raw-dir", default="baseline/humanoid21/mocap/data/raw",
                        help="Directory for raw ASF/AMC files")
    parser.add_argument("--output-dir", default="baseline/humanoid21/mocap/data/retargeted",
                        help="Directory for retargeted .npy files")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing raw files")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Don't use proxy for downloads")
    args = parser.parse_args()

    args.raw_dir = os.path.abspath(args.raw_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    # Step 1: Download
    if not args.skip_download:
        print("=== Step 1: Downloading CMU boxing data ===")
        files = download_boxing_data(args.raw_dir)
        print(f"\nDownloaded {len(files)} motion files")
    else:
        print("=== Step 1: Using existing raw files ===")
        files = []
        for subject_num, motion_nums in BOXING_SUBJECTS.items():
            asf_path = os.path.join(args.raw_dir, f"{subject_num}.asf")
            if not os.path.exists(asf_path):
                continue
            for motion_num in motion_nums:
                amc_path = os.path.join(args.raw_dir, f"{subject_num}_{motion_num:02d}.amc")
                if os.path.exists(amc_path):
                    files.append((subject_num, motion_num, asf_path, amc_path))
        print(f"Found {len(files)} existing motion files")

    # Step 2: Retarget
    print(f"\n=== Step 2: Retargeting to humanoid21 ===")
    all_motions = []
    for subject_num, motion_num, asf_path, amc_path in files:
        print(f"\nProcessing subject {subject_num}, motion {motion_num:02d}...")
        out_path = retarget_and_save(asf_path, amc_path, args.output_dir, subject_num, motion_num)
        if out_path:
            all_motions.append(out_path)

    # Step 3: Summary
    print(f"\n=== Summary ===")
    print(f"Retargeted {len(all_motions)} motions")
    print(f"Output directory: {args.output_dir}")
    print(f"Joint order: {JOINT_ORDER}")

    # Print total frames
    total_frames = 0
    for mp in all_motions:
        motion = np.load(mp)
        total_frames += len(motion)
        print(f"  {os.path.basename(mp)}: {motion.shape}")
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main()
