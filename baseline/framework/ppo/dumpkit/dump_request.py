"""Dump request — sentinel-file trigger for one-shot update data capture.

A running training loop checks for a sentinel file
``<run_dir>/dump_request.json`` at the top of each update.  When found,
it captures the complete update data (episodes, trajectories, GAE,
combine, gradients) into ``<run_dir>/dumps/u{u:05d}/``.

``hypothesis`` is mandatory — writing a dump without a hypothesis is
rejected.  This forces the user to articulate what they're looking for
before grabbing data.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


SENTINEL_FILENAME = "dump_request.json"


@dataclass(frozen=True)
class DumpRequest:
    """A parsed dump request from the sentinel file.

    Attributes:
        hypothesis: Mandatory — forces the user to articulate intent.
            Empty/whitespace → rejected at parse time.
        include_full_grad: When True, capture the full flat actor
            gradient (epoch 0, minibatch 0).  Off by default (large).
    """

    hypothesis: str
    include_full_grad: bool = False

    def __post_init__(self):
        if not self.hypothesis or not self.hypothesis.strip():
            raise ValueError(
                "dump_request.json: 'hypothesis' is required and must be "
                "non-empty. If you can't state a hypothesis, you're not "
                "ready to dump."
            )


def poll_dump_request(
    run_dir: Path,
    update: int,
) -> Optional[DumpRequest]:
    """Check for a dump request at the top of the training loop.

    Returns ``None`` when no sentinel file exists (the common case —
    single ``os.path.exists`` check, negligible overhead).

    When the sentinel exists:
    1. Read + parse it (rejecting if ``hypothesis`` is missing).
    2. Atomically move it to ``<run_dir>/dumps/u{u:05d}/request.json``
       (same filesystem → ``os.rename`` is atomic on POSIX).
    3. Return the parsed :class:`DumpRequest`.

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
            f"[dump] sentinel {sentinel} is corrupt ({e}); "
            f"ignoring. Remove it to stop this warning.",
            flush=True,
        )
        return None

    try:
        req = DumpRequest(
            hypothesis=raw.get("hypothesis", ""),
            include_full_grad=raw.get("include_full_grad", False),
        )
    except ValueError as e:
        print(f"[dump] rejecting dump request: {e}", flush=True)
        # Move the bad request aside so it doesn't block future requests.
        reject_dir = run_dir / "dumps" / "rejected"
        reject_dir.mkdir(parents=True, exist_ok=True)
        reject_path = reject_dir / f"request_u{update:05d}_{int(datetime.now().timestamp())}.json"
        try:
            os.rename(str(sentinel), str(reject_path))
        except OSError:
            pass
        return None

    # Atomic move to the per-update dump directory.
    dump_dir = run_dir / "dumps" / f"u{update:05d}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dest = dump_dir / "request.json"
    try:
        os.rename(str(sentinel), str(dest))
    except OSError:
        # If rename fails (cross-filesystem?), fall back to copy + delete.
        shutil.copy2(str(sentinel), str(dest))
        os.unlink(str(sentinel))

    return req


__all__ = [
    "DumpRequest",
    "poll_dump_request",
    "SENTINEL_FILENAME",
]
