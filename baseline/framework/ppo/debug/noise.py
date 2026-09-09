"""``noise`` — noise baseline from multi-seed training (S6).

S6: Launches multiple training runs with different seeds, then
computes per-metric mean ± std across runs.  The resulting "noise band"
is used by ``compare --with-noise-band`` to determine whether a
difference between two runs is significant or just noise.

"没有噪声带的对照结论一律不采信" (DEBUG_GUIDE.md §3.9) — this tool
establishes the noise band that makes that discipline enforceable.

Launches training via ``train.py --background --seed S --max-updates U``
and waits for all runs to complete before computing the band.

See ``DEBUG_GUIDE.md`` §3.9 ``noise``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compare import parse_run_metrics, NoiseBand


# ---------------------------------------------------------------------------
# Training launch
# ---------------------------------------------------------------------------

def launch_training_run(
    experiment: str,
    seed: int,
    max_updates: int,
    algo: str = "ppo",
    run_name: Optional[str] = None,
    run_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> Path:
    """Launch a single background training run.

    Uses ``train.py --background --seed S --max-updates U``.

    Args:
        experiment: Experiment name.
        seed: Random seed.
        max_updates: Maximum number of updates.
        algo: Algorithm ("ppo" or "sac").
        run_name: Optional custom run name.
        run_dir: Optional explicit run directory.
        project_root: Project root (default: auto-detect).
        extra_args: Additional CLI args.

    Returns:
        Path to the run directory.
    """
    if project_root is None:
        # Auto-detect: this file is at
        # baseline/framework/ppo/debug/noise.py
        project_root = Path(__file__).resolve().parents[4]

    train_py = project_root / "baseline" / "framework" / "train.py"

    cmd = [
        sys.executable, str(train_py),
        "--experiment", experiment,
        "--algo", algo,
        "--background",
        "--seed", str(seed),
        "--max-updates", str(max_updates),
    ]
    if run_name is not None:
        cmd += ["--run-name", run_name]
    if run_dir is not None:
        cmd += ["--run-dir", str(run_dir)]
    if extra_args:
        cmd += extra_args

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    # Launch — train.py --background forks + setsid and prints run info
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(project_root),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to launch training run: {result.stderr}"
        )

    # Parse the run directory from stdout
    # train.py prints: "[run] dir: /path/to/run_dir"
    run_dir_path = None
    for line in result.stdout.splitlines():
        if "[run] dir:" in line:
            run_dir_path = line.split("dir:", 1)[1].strip()
            break
    if run_dir_path is None:
        raise RuntimeError(
            f"Could not parse run directory from train.py output:\n{result.stdout}"
        )
    return Path(run_dir_path)


def wait_for_runs(
    run_dirs: List[Path],
    poll_interval: float = 5.0,
    timeout: float = 3600.0,
) -> None:
    """Wait for all training runs to complete.

    Polls each run's PID file; when the PID no longer exists, the run
    is considered complete.

    Args:
        run_dirs: List of run directories.
        poll_interval: Seconds between polls.
        timeout: Maximum seconds to wait.
    """
    deadline = time.time() + timeout
    pending = list(run_dirs)
    while pending and time.time() < deadline:
        still_running: List[Path] = []
        for rd in pending:
            pid_file = rd / "pid"
            if not pid_file.exists():
                # No PID file — might have finished or failed early
                continue
            try:
                pid = int(pid_file.read_text().strip())
            except (ValueError, OSError):
                continue
            # Check if process is alive
            if _pid_alive(pid):
                still_running.append(rd)
        pending = still_running
        if pending:
            time.sleep(poll_interval)
    if pending:
        raise TimeoutError(
            f"Timed out waiting for {len(pending)} run(s): "
            f"{[str(rd) for rd in pending]}"
        )


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ---------------------------------------------------------------------------
# Noise band computation
# ---------------------------------------------------------------------------

def compute_noise_band(
    run_dirs: List[Path],
    window: int = 5,
) -> NoiseBand:
    """Compute a noise band from multiple training runs.

    For each metric, computes mean and std across all runs.  Each run's
    metric value is the average of its last ``window`` updates.

    Args:
        run_dirs: List of completed run directories.
        window: Number of recent updates to average per run.

    Returns:
        NoiseBand with per-metric mean ± std.
    """
    all_metrics: List[Dict[str, float]] = []
    for rd in run_dirs:
        metrics = parse_run_metrics(rd, window=window)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        return NoiseBand()

    # Collect all metric names
    all_keys: set = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    band_metrics: Dict[str, Dict[str, float]] = {}
    for key in sorted(all_keys):
        vals: List[float] = []
        for m in all_metrics:
            if key in m:
                vals.append(m[key])
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            band_metrics[key] = {"mean": mean, "std": std}

    return NoiseBand(
        metrics=band_metrics,
        n_seeds=len(all_metrics),
        n_updates=window,
    )


def save_noise_band(band: NoiseBand, path: Path) -> None:
    """Save a noise band to a JSON file."""
    with open(path, "w") as f:
        json.dump(band.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_noise_summary(band: NoiseBand, run_dirs: List[Path]) -> str:
    """Render a human-readable noise band summary."""
    lines: List[str] = []
    lines.append(f"噪声基线   {band.n_seeds} seeds × {band.n_updates} updates")
    lines.append(f"  runs: {[rd.name for rd in run_dirs]}")
    lines.append("")
    lines.append(f"{'指标':<30s} {'mean':<12s} {'std':<12s}")
    for key in sorted(band.metrics.keys()):
        m = band.metrics[key]
        lines.append(f"{key:<30s} {m['mean']:<12.4f} {m['std']:<12.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def noise(
    experiment: str,
    seeds: List[int],
    updates: int,
    algo: str = "ppo",
    output: Optional[Path] = None,
    project_root: Optional[Path] = None,
    window: int = 5,
    wait: bool = True,
) -> str:
    """Launch multi-seed training runs and compute a noise band.

    Args:
        experiment: Experiment name.
        seeds: List of seeds to run.
        updates: Maximum updates per run.
        algo: Algorithm ("ppo" or "sac").
        output: Path to save noise band JSON (optional).
        project_root: Project root (default: auto-detect).
        window: Number of recent updates to average per run.
        wait: If True, wait for runs to complete before computing.

    Returns:
        Human-readable summary string.
    """
    if not seeds:
        raise ValueError("At least one seed is required.")

    # Launch all runs
    run_dirs: List[Path] = []
    for seed in seeds:
        run_name = f"noise_{experiment}_{algo}_seed{seed}"
        rd = launch_training_run(
            experiment=experiment,
            seed=seed,
            max_updates=updates,
            algo=algo,
            run_name=run_name,
            project_root=project_root,
        )
        run_dirs.append(rd)

    # Wait for completion
    if wait:
        wait_for_runs(run_dirs)

    # Compute noise band
    band = compute_noise_band(run_dirs, window=window)

    # Save if requested
    if output is not None:
        save_noise_band(band, Path(output))

    return render_noise_summary(band, run_dirs)
