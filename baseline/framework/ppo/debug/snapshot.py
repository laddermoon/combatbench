"""Snapshot capture — sentinel-file trigger + on-disk snapshot writer.

S2: A running training loop checks for a sentinel file
``<run_dir>/debug_request.json`` at the top of each update.  When found,
it captures a full snapshot of the current update's state into
``<run_dir>/debug/u{u:05d}/``:

- ``request.json`` — the original request (moved atomically from sentinel)
- ``manifest.json`` — update, timestamp, git commit, episodes mode
- ``episodes/`` — ``EpisodeCollection.save()`` (subset or all)
- ``actor/`` — θ_old policy blueprint (copied from ``policy_exports/``)
- ``critics.pt`` — critic state dicts (NOT in policy_exports)
- ``rng_state.pt`` — torch + numpy + cuda RNG state
- ``replay/`` — empty dir, filled by ``replay.py``

See ``DESIGN_debug_system.md`` §4.1 / §4.2.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from baseline.framework.rollout import Episode, EpisodeCollection
from envs.framework.blueprint import EnvBlueprint


SENTINEL_FILENAME = "debug_request.json"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class DebugRequest:
    """A parsed snapshot request from the sentinel file.

    ``hypothesis`` is mandatory (§6 discipline 6) — writing a snapshot
    without a hypothesis is rejected.  This forces the user to articulate
    what they're looking for before grabbing data.
    """
    hypothesis: str
    episodes_mode: str = "subset"  # "subset" | "all"
    episodes_n: int = 8
    include_full_grad: bool = False

    def __post_init__(self):
        if not self.hypothesis or not self.hypothesis.strip():
            raise ValueError(
                "debug_request.json: 'hypothesis' is required and must be "
                "non-empty (DEBUG_GUIDE.md §6 discipline 6). "
                "If you can't state a hypothesis, run `health` or `chain` first."
            )
        if self.episodes_mode not in ("subset", "all"):
            raise ValueError(
                f"episodes_mode must be 'subset' or 'all', "
                f"got {self.episodes_mode!r}"
            )
        if self.episodes_mode == "subset" and self.episodes_n <= 0:
            raise ValueError(
                f"episodes_n must be > 0 for subset mode, got {self.episodes_n}"
            )


def poll_request(
    run_dir: Path,
    update: int,
) -> Optional[DebugRequest]:
    """Check for a snapshot request at the top of the training loop.

    Returns ``None`` when no sentinel file exists (the common case —
    single ``os.path.exists`` check, negligible overhead per P3).

    When the sentinel exists:
    1. Read + parse it (rejecting if ``hypothesis`` is missing).
    2. Atomically move it to ``<run_dir>/debug/u{u:05d}/request.json``
       (same filesystem → ``os.rename`` is atomic on POSIX).
    3. Return the parsed :class:`DebugRequest`.

    The atomic rename makes the request one-shot: a second poll in the
    same or a later update won't see it again.
    """
    sentinel = run_dir / SENTINEL_FILENAME
    if not sentinel.exists():
        return None

    try:
        with open(sentinel) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Corrupt sentinel — don't silently delete it; leave it for the
        # user to inspect, but don't crash training.
        print(
            f"[debug] sentinel {sentinel} is corrupt ({e}); "
            f"ignoring. Remove it to stop this warning.",
            flush=True,
        )
        return None

    try:
        req = DebugRequest(
            hypothesis=raw.get("hypothesis", ""),
            episodes_mode=raw.get("episodes_mode", "subset"),
            episodes_n=raw.get("episodes_n", 8),
            include_full_grad=raw.get("include_full_grad", False),
        )
    except ValueError as e:
        print(f"[debug] rejecting snapshot request: {e}", flush=True)
        # Move the bad request aside so it doesn't block future requests.
        reject_dir = run_dir / "debug" / "rejected"
        reject_dir.mkdir(parents=True, exist_ok=True)
        reject_path = reject_dir / f"request_u{update:05d}_{int(datetime.now().timestamp())}.json"
        try:
            os.rename(str(sentinel), str(reject_path))
        except OSError:
            pass
        return None

    # Atomic move to the per-update debug directory.
    snapshot_dir = run_dir / "debug" / f"u{update:05d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    dest = snapshot_dir / "request.json"
    try:
        os.rename(str(sentinel), str(dest))
    except OSError as e:
        # If rename fails (cross-filesystem?), fall back to copy + delete.
        shutil.copy2(str(sentinel), str(dest))
        os.unlink(str(sentinel))

    return req


def capture_snapshot(
    run_dir: Path,
    update: int,
    request: DebugRequest,
    *,
    episodes: List[Episode],
    actor: torch.nn.Module,
    actor_export_dir: Path,
    critics: Dict[str, torch.nn.Module],
    experiment_name: str,
    env_blueprint: Optional[EnvBlueprint] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a full snapshot to ``<run_dir>/debug/u{update:05d}/``.

    Must be called AFTER ``build_trajectories`` and BEFORE ``ppo_update``
    so the snapshot captures θ_old (pre-update actor) + pre-update critics
    + the exact episodes that will feed this update.

    Args:
        run_dir: Training run directory.
        update: Current update number.
        request: Parsed debug request.
        episodes: All rollout episodes for this update.
        actor: The actor model (θ_old, pre-update state).
        actor_export_dir: Path to ``policy_exports/u{u:05d}`` (θ_old blueprint).
        critics: Critic models (pre-update state).
        experiment_name: Experiment name (for replay reconstruction).
        env_blueprint: The EnvBlueprint used for this update's episodes.
            Required if ``episodes`` is non-empty (EpisodeCollection needs it).
        config: Optional config dict (from ``config.json``) for manifest.

    Returns:
        Path to the snapshot directory.
    """
    snapshot_dir = run_dir / "debug" / f"u{update:05d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # --- request.json already moved here by poll_request ---

    # --- manifest.json ---
    manifest: Dict[str, Any] = {
        "update": update,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "episodes_mode": request.episodes_mode,
        "episodes_n": request.episodes_n if request.episodes_mode == "subset" else len(episodes),
        "include_full_grad": request.include_full_grad,
        "hypothesis": request.hypothesis,
    }
    # Git commit from code_snapshot.json if present.
    snapshot_info = run_dir / "code_snapshot.json"
    if snapshot_info.exists():
        try:
            with open(snapshot_info) as f:
                si = json.load(f)
            manifest["git_commit"] = si.get("commit")
            manifest["git_branch"] = si.get("branch")
        except (json.JSONDecodeError, OSError):
            pass
    # Config hash for quick identity check.
    if config is not None:
        manifest["config_keys"] = sorted(config.keys())

    (snapshot_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- episodes/ ---
    # Subset: first N episodes (deterministic, P1 — no random sampling).
    # All: every episode.
    if request.episodes_mode == "subset":
        subset = episodes[:request.episodes_n]
    else:
        subset = episodes

    if subset:
        if env_blueprint is None:
            raise ValueError(
                "env_blueprint is required when episodes is non-empty "
                "(EpisodeCollection needs an EnvBlueprint)"
            )
        coll = EpisodeCollection(
            blueprint=env_blueprint,
            episodes=subset,
        )
        coll.save(snapshot_dir / "episodes")
    else:
        # Empty episodes — create the dir so replay knows it's intentional.
        (snapshot_dir / "episodes").mkdir(parents=True, exist_ok=True)

    # --- actor/ (θ_old) ---
    # Copy the already-exported policy blueprint for this update.
    actor_dest = snapshot_dir / "actor"
    if actor_export_dir.exists():
        if actor_dest.exists():
            shutil.rmtree(actor_dest)
        shutil.copytree(actor_export_dir, actor_dest)
    else:
        # Actor export missing — record the fact in manifest.
        manifest["actor_export_missing"] = str(actor_export_dir)
        (snapshot_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- critics.pt ---
    # policy_exports only contains the actor; critics must be saved
    # separately (DESIGN §2 fact: "导出只含 actor，不含 critic").
    critics_state = {k: v.state_dict() for k, v in critics.items()}
    torch.save(critics_state, snapshot_dir / "critics.pt")

    # --- actor_state.pt ---
    # The blueprint export is for deployment; replay needs the full
    # TrainablePolicy state_dict to reconstruct θ_old for ppo_update.
    torch.save(actor.state_dict(), snapshot_dir / "actor_state.pt")

    # --- rng_state.pt ---
    rng_state: Dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng_state, snapshot_dir / "rng_state.pt")

    # --- replay/ (empty, filled by replay.py) ---
    (snapshot_dir / "replay").mkdir(parents=True, exist_ok=True)

    print(
        f"[debug] snapshot captured at update {update} "
        f"→ {snapshot_dir} "
        f"(episodes={request.episodes_mode}:{len(subset)}, "
        f"full_grad={request.include_full_grad})",
        flush=True,
    )

    return snapshot_dir


__all__ = [
    "DebugRequest",
    "poll_request",
    "capture_snapshot",
    "SENTINEL_FILENAME",
    "MANIFEST_FILENAME",
]
