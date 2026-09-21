"""Dump request — sentinel-file trigger for one-shot update data capture.

A running training loop checks for a sentinel file
``<run_dir>/dump_request.json`` at the top of each update.  When found,
it captures the complete update data (episodes, trajectories, GAE,
combine, gradients, gradsig diagnostic) into ``<run_dir>/dumps/u{u:05d}/``.

``hypothesis`` is optional but encouraged — writing down what you're
looking for makes the dump self-describing later.  Dumps may also be
scheduled at launch via ``train.py --dump-at`` (``source="cli"``) —
see ``loop.py``.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SENTINEL_FILENAME = "dump_request.json"


@dataclass(frozen=True)
class DumpRequest:
    """A parsed dump request.

    Attributes:
        hypothesis: Free-text intent recorded into the dump (optional —
            the dump is a general-purpose tool, but a hypothesis makes
            it self-describing).
        include_full_grad: When True, capture the full flat actor
            gradient (epoch 0, minibatch 0).  Off by default (large).
        source: Provenance — ``"sentinel"`` for a sentinel-file request,
            ``"cli"`` for a launch-time ``--dump-at`` schedule.  Written
            into request.json / manifest so the dump records how it was
            triggered.
    """

    hypothesis: str = ""
    include_full_grad: bool = False
    source: str = "sentinel"


def poll_dump_request(
    run_dir: Path,
    update: int,
) -> Optional[DumpRequest]:
    """Check for a dump request at the top of the training loop.

    Returns ``None`` when no sentinel file exists (the common case —
    single ``os.path.exists`` check, negligible overhead).

    When the sentinel exists:
    1. Read + parse it.
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

    req = DumpRequest(
        hypothesis=raw.get("hypothesis", "") or "",
        include_full_grad=raw.get("include_full_grad", False),
        source="sentinel",
    )

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
