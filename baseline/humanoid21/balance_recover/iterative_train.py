#!/usr/bin/env python3
"""Iterative balance recovery training automation.

Runs the full loop: probe -> sample -> train -> monitor -> log,
repeating across generations.

Usage:
    # From scratch (no checkpoint):
    python3 iterative_train.py \
        --policy baseline/runs/standing/policy/policy_blueprint.yaml \
        --run-root /data1/dev/recovery_run1 \
        --exp-name recovery_v2 \
        --start-gen 0 \
        --max-gens 10

    # Resume from a previous generation:
    python3 iterative_train.py \
        --policy baseline/runs/weighted_impulse_gen2/policy/policy_blueprint.yaml \
        --checkpoint baseline/runs/weighted_impulse_gen2/checkpoints/checkpoint_u01620.pt \
        --run-root /data1/dev/recovery_run2 \
        --exp-name recovery_v3 \
        --start-gen 0 \
        --max-gens 10
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # /data1/mono/things/combatbench

# 全局变量，由 main() 设置
_RUN_ROOT: Path = SCRIPT_DIR
_LOG_FILE: Path = SCRIPT_DIR / "iterative_train.log"


def log(gen: int, phase: str, details: str = ""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[Gen {gen}] {ts} | {phase}"
    if details:
        line += f" | {details}"
    print(line, flush=True)
    with open(_LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_probe(gen: int, policy_path: str, workers: int) -> dict:
    csv_out = str(_RUN_ROOT / f"boundary_gen{gen}.csv")
    json_out = str(_RUN_ROOT / f"boundary_gen{gen}.json")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "probe_boundary.py"),
        "--policy-blueprint-path", policy_path,
        "--output", csv_out,
        "--json-output", json_out,
        "--workers", str(workers),
    ]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    log(gen, "PROBE_START", f"policy={policy_path}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        log(gen, "PROBE_ERROR", result.stderr[-500:])
        raise RuntimeError(f"Probe failed: {result.returncode}")

    with open(json_out) as f:
        data = json.load(f)
    forces = [40.0, 100.0, 200.0]
    stats = {}
    for f_val in forces:
        cds = [r["critical_duration"] for r in data["results"] if r["force"] == f_val]
        stats[f"F={int(f_val)}N"] = f"mean={sum(cds)/len(cds):.1f}" if cds else "n/a"
    log(gen, "PROBE_DONE", " ".join(f"{k}={v}" for k, v in stats.items()))
    return data


def run_sample(gen: int) -> dict:
    json_in = str(_RUN_ROOT / f"boundary_gen{gen}.json")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "sample_distribution.py"),
        "--input", json_in,
        "--output-dir", str(_RUN_ROOT),
    ]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    log(gen, "SAMPLE_START")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        log(gen, "SAMPLE_ERROR", result.stderr[-500:])
        raise RuntimeError(f"Sample failed: {result.returncode}")

    # Copy to gen-suffixed names
    npz_src = _RUN_ROOT / "sample_weights.npz"
    json_src = _RUN_ROOT / "sample_distribution.json"
    csv_src = _RUN_ROOT / "samples.csv"
    npz_dst = _RUN_ROOT / f"sample_weights_gen{gen}.npz"
    json_dst = _RUN_ROOT / f"sample_distribution_gen{gen}.json"
    csv_dst = _RUN_ROOT / f"samples_gen{gen}.csv"
    shutil.copy2(npz_src, npz_dst)
    shutil.copy2(json_src, json_dst)
    shutil.copy2(csv_src, csv_dst)

    # Copy gen-suffixed versions of base heatmap/plot files (skip existing _gen files)
    for pattern in ["heatmap_*.png", "boundary_and_distribution.png"]:
        for src in _RUN_ROOT.glob(pattern):
            if f"_gen" in src.stem:
                continue
            dst = _RUN_ROOT / f"{src.stem}_gen{gen}{src.suffix}"
            shutil.copy2(src, dst)

    with open(json_dst) as f:
        sample_info = json.load(f)
    stats = sample_info.get("sample_statistics", {})
    dur_mean = stats.get("duration", {}).get("mean", "?")
    force_dist = stats.get("force_distribution", {})
    log(gen, "SAMPLE_DONE",
        f"duration_mean={dur_mean} "
        f"F40={force_dist.get('40.0', '?')}% "
        f"F100={force_dist.get('100.0', '?')}% "
        f"F200={force_dist.get('200.0', '?')}%")
    return str(npz_dst)


def launch_train(gen: int, checkpoint: str, policy_path: str, sample_npz: str,
                 exp_name: str) -> tuple:
    run_name = f"{exp_name}_gen{gen}"
    run_dir = REPO_ROOT / "baseline" / "runs" / run_name
    cmd = [
        sys.executable, str(REPO_ROOT / "baseline" / "framework" / "train.py"),
        "--experiment", "balance_recover",
        "--algo", "ppo",
        "--background",
        "--set", f"policy_blueprint_path={policy_path}",
        "--set", f"weight_npz_path={sample_npz}",
        "--set", "reset_best=True",
        "--run-name", run_name,
    ]
    if checkpoint:
        cmd += ["--resume-from", checkpoint, "--reset-update"]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    log(gen, "TRAIN_START", f"checkpoint={checkpoint}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        log(gen, "TRAIN_ERROR", result.stderr[-500:])
        raise RuntimeError(f"Train launch failed: {result.returncode}")

    # Parse PID and run_dir from output
    output = result.stdout
    pid_match = re.search(r"\[run\] pid: (\d+)", output)
    dir_match = re.search(r"\[run\] dir: (.+)", output)
    pid = int(pid_match.group(1)) if pid_match else None
    train_dir = Path(dir_match.group(1).strip()) if dir_match else run_dir
    log_path = train_dir / "train.log"
    log(gen, "TRAIN_LAUNCHED", f"pid={pid} run_dir={train_dir}")
    return pid, train_dir, log_path


def monitor_train(gen: int, pid: int, log_path: Path) -> dict:
    best_survived = -1
    best_update = -1
    start_update = -1
    last_update = -1
    eval_re = re.compile(r"^\[eval\s+(\d+)\] survived=([\d.]+)")
    new_best_re = re.compile(r"\[new_best\]")
    early_stop_re = re.compile(r"^\[early_stop\]")

    last_pos = 0
    while True:
        # Check if process is still alive
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                # Process exited
                log(gen, "TRAIN_PROCESS_EXIT", f"pid={pid} exited")
                break

        # Read new log lines
        try:
            with open(log_path, "r") as f:
                f.seek(last_pos)
                new_lines = f.readlines()
                last_pos = f.tell()
        except FileNotFoundError:
            time.sleep(5)
            continue

        for line in new_lines:
            m = eval_re.match(line)
            if m:
                u = int(m.group(1))
                survived = int(float(m.group(2)))
                if start_update < 0:
                    start_update = u
                last_update = u
                is_new_best = bool(new_best_re.search(line))
                if is_new_best and survived > best_survived:
                    best_survived = survived
                    best_update = u
                    log(gen, "NEW_BEST", f"update={u} survived={survived}")
                elif u % 50 == 0:
                    log(gen, "EVAL", f"update={u} survived={survived} best={best_survived}")

            if early_stop_re.match(line):
                log(gen, "EARLY_STOP", line.strip())
                break
        else:
            time.sleep(30)
            continue
        break  # early_stop detected

    # Wait for process to fully exit
    if pid is not None:
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass

    updates_trained = (last_update - start_update) if start_update >= 0 else 0
    info = {
        "best_survived": best_survived,
        "best_update": best_update,
        "start_update": start_update,
        "last_update": last_update,
        "updates_trained": updates_trained,
    }
    log(gen, "TRAIN_DONE",
        f"updates_trained={updates_trained} best_survived={best_survived} best_update={best_update}")
    return info


def find_latest_exports(run_dir: Path) -> tuple:
    """Find latest policy_exports dir and checkpoint."""
    exports_dir = run_dir / "policy_exports"
    if not exports_dir.exists():
        return None, None

    # Find latest non-eval export
    export_dirs = sorted(
        [d for d in exports_dir.iterdir() if d.is_dir() and not d.name.endswith("_eval")],
        key=lambda d: int(d.name.replace("u", "")),
    )
    if not export_dirs:
        return None, None
    latest_export = export_dirs[-1]
    policy_bp = latest_export / "policy_blueprint.yaml"
    if not policy_bp.exists():
        return None, None

    # Find latest checkpoint
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return policy_bp, None
    ckpts = sorted(ckpt_dir.glob("checkpoint_u*.pt"),
                   key=lambda p: int(p.stem.replace("checkpoint_u", "")))
    latest_ckpt = ckpts[-1] if ckpts else None

    return policy_bp, latest_ckpt


def main():
    global _RUN_ROOT, _LOG_FILE

    parser = argparse.ArgumentParser(description="Iterative balance recovery training")
    parser.add_argument("--policy", type=str, required=True,
                        help="Initial policy blueprint path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Initial checkpoint path (omit for from-scratch training)")
    parser.add_argument("--run-root", type=str, required=True,
                        help="Root directory for all intermediate files (boundary, samples, heatmaps, logs)")
    parser.add_argument("--exp-name", type=str, default="weighted_impulse",
                        help="Base name for training run directories (e.g. recovery_v2 -> recovery_v2_gen0)")
    parser.add_argument("--start-gen", type=int, default=0,
                        help="Starting generation number")
    parser.add_argument("--max-gens", type=int, default=10,
                        help="Maximum generations to run")
    parser.add_argument("--workers", type=int, default=96,
                        help="Probe workers")
    args = parser.parse_args()

    _RUN_ROOT = Path(args.run_root).resolve()
    _RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = _RUN_ROOT / "iterative_train.log"

    policy_path = args.policy
    checkpoint = args.checkpoint

    log(args.start_gen, "ITERATIVE_TRAIN_START",
        f"run_root={_RUN_ROOT} exp_name={args.exp_name} "
        f"policy={policy_path} checkpoint={checkpoint} max_gens={args.max_gens}")

    for gen in range(args.start_gen, args.start_gen + args.max_gens):
        try:
            # 1. Probe
            run_probe(gen, policy_path, args.workers)

            # 2. Sample
            sample_npz = run_sample(gen)

            # 3. Train
            pid, train_dir, log_path = launch_train(
                gen, checkpoint, policy_path, sample_npz, args.exp_name)

            # 4. Monitor
            info = monitor_train(gen, pid, log_path)

            # 5. Prepare next gen
            next_policy, next_ckpt = find_latest_exports(train_dir)
            if next_policy is None or next_ckpt is None:
                log(gen, "ERROR", "No policy_exports or checkpoint found, stopping")
                break

            log(gen, "NEXT_GEN",
                f"policy={next_policy} checkpoint={next_ckpt}")
            policy_path = str(next_policy)
            checkpoint = str(next_ckpt)

        except RuntimeError as e:
            log(gen, "ERROR", str(e))
            break
        except KeyboardInterrupt:
            log(gen, "INTERRUPTED", "user interrupted")
            break

    log(args.start_gen + args.max_gens - 1, "ITERATIVE_TRAIN_END", "all generations done")


if __name__ == "__main__":
    main()
