"""Git-based code snapshot for experiment reproducibility.

Creates a lightweight git branch that captures the exact code state
at the start of a training run, without disturbing the working tree.

Usage::

    from baseline.framework.code_snapshot import create_code_snapshot

    info = create_code_snapshot(run_name="train_xxx_ppo_20260707", run_dir=run_dir)
    # info is written to run_dir / "code_snapshot.json"
    # a branch exp/train_xxx_ppo_20260707 is created

Reproduce::

    git worktree add /tmp/repro_train_xxx exp/train_xxx_ppo_20260707
    cd /tmp/repro_train_xxx
    python3 baseline/framework/train.py --experiment xxx --algo ppo
"""
from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional


def _git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _git_ok(repo_dir: Path, *args: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def create_code_snapshot(
    run_name: str,
    run_dir: Path,
    *,
    repo_search_from: Optional[Path] = None,
) -> Optional[dict]:
    """Create a git branch snapshot of the current code state.

    Uses ``git commit-tree`` to create a commit without moving HEAD,
    then creates a branch ``exp/<run_name>`` pointing to it.
    The staging area is restored to HEAD afterwards.

    Args:
        run_name: Name of the training run (used for branch name).
        run_dir: Directory where ``code_snapshot.json`` will be written.
        repo_search_from: Directory to search for the git repo root.
            Defaults to the parent of this file.

    Returns:
        Snapshot info dict, or None if git is unavailable.
    """
    if not shutil.which("git"):
        print("[snapshot] git not found, skipping code snapshot", flush=True)
        return None

    search_from = repo_search_from or Path(__file__).resolve().parent
    if not _git_ok(search_from, "rev-parse", "--is-inside-work-tree"):
        print("[snapshot] not a git repo, skipping code snapshot", flush=True)
        return None

    repo_root = Path(_git(search_from, "rev-parse", "--show-toplevel"))
    head_commit = _git(repo_root, "rev-parse", "HEAD")
    head_branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    branch_name = f"exp/{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Acquire a file lock on .git/snapshot.lock to serialize concurrent snapshots
    lock_path = repo_root / ".git" / "snapshot.lock"
    lock_fd = None
    try:
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
    except (OSError, IOError):
        print("[snapshot] could not acquire lock, skipping code snapshot", flush=True)
        if lock_fd:
            lock_fd.close()
        return None

    # Save original index state so we can restore it exactly
    orig_index_tree = _git(repo_root, "write-tree")

    try:
        # Stage everything (including untracked files)
        _git(repo_root, "add", "-A")
        tree_hash = _git(repo_root, "write-tree")
        commit_msg = f"snapshot: {run_name}"
        commit_hash = _git(
            repo_root, "commit-tree", tree_hash,
            "-m", commit_msg, "-p", head_commit,
        )
        _git(repo_root, "branch", branch_name, commit_hash)
    finally:
        # Restore staging area to its original state (preserves user's staged changes)
        _git(repo_root, "read-tree", orig_index_tree)
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()

    dirty = head_branch == "HEAD"  # detached HEAD
    info = {
        "branch": branch_name,
        "commit": commit_hash,
        "base_branch": head_branch if not dirty else "(detached)",
        "base_commit": head_commit,
        "repo_root": str(repo_root),
        "run_name": run_name,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / "code_snapshot.json"
    with open(snapshot_path, "w") as f:
        json.dump(info, f, indent=2)

    return info


def format_repro_command(
    info: dict, *, args,
    original_run_dir: Path,
    original_repo_root: Path,
) -> str:
    """Format a human-friendly reproduction command string.

    Args:
        info: Snapshot info dict from create_code_snapshot.
        args: argparse.Namespace with all original CLI arguments.
        original_run_dir: Absolute path of the original run directory.
        original_repo_root: Absolute path of the original git repo root.
    """
    branch = info["branch"]
    repro_dir = f"/tmp/repro_{info['run_name']}"

    lines = [
        f"# --- Reproduce this run ---",
        f"# Code snapshot: branch {branch} (commit {info['commit'][:8]})",
        f"git worktree add {repro_dir} {branch}",
        f"cd {repro_dir}",
    ]

    parts = ["python3 baseline/framework/train.py"]
    parts.append(f"--experiment {args.experiment}")
    parts.append(f"--algo {args.algo}")
    if args.smoke:
        parts.append("--smoke")
    if args.no_confidence:
        parts.append("--no-confidence")

    repro_run_name = f"repro_{info['run_name']}"

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = (original_repo_root / resume_path).resolve()
        if resume_path.exists():
            parts.append(f"--resume-from {resume_path}")

    parts.append(f"--run-name {repro_run_name}")
    repro_run_dir = original_repo_root / "baseline" / "runs" / repro_run_name
    parts.append(f"--run-dir {repro_run_dir}")

    # Include --set params for reproducibility
    for s in getattr(args, "set", []) or []:
        parts.append(f"--set {s}")

    # Always skip snapshot on repro to avoid creating another branch
    parts.append("--no-snapshot")

    lines.append(" ".join(parts))
    lines.append(f"# cleanup: git worktree remove {repro_dir}")

    return "\n".join(lines)
