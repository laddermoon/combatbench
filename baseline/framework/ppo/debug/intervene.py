"""``intervene-check`` — knob verification tool (S6).

S6: Verifies that each configured knob actually entered the training
data pathway.  "配置里写了" ≠ "生效了" — this tool makes the check
automatic rather than relying on someone remembering to verify
(DEBUG_GUIDE.md §3.8).

Combines:
- Framework built-in knobs (from ``knobs.builtin_knob_checks()``)
- Experiment-specific knobs (from ``ExperimentPPO.knob_checks()``)

Log-only by default; snapshot-dependent knobs (observer) are skipped
with a "需快照" note when no snapshot is available.

See ``DEBUG_GUIDE.md`` §3.8 ``intervene-check`` and
``DESIGN_debug_system.md`` §5.4.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .knobs import (
    KnobCheck,
    KnobContext,
    KnobResult,
    builtin_knob_checks,
    load_config,
    parse_log_entry,
    run_knob_checks,
    render_knob_table,
)


# ---------------------------------------------------------------------------
# Snapshot auto-discovery
# ---------------------------------------------------------------------------

def find_latest_snapshot(run_dir: Path) -> Optional[Path]:
    """Find the latest snapshot directory under ``<run_dir>/debug/``.

    Returns the path to ``<run_dir>/debug/u<NNNNN>/`` with the highest
    update number, or None if no snapshots exist.
    """
    debug_dir = run_dir / "debug"
    if not debug_dir.exists():
        return None
    snapshots = []
    for d in debug_dir.iterdir():
        if d.is_dir() and re.match(r"u\d+", d.name):
            snapshots.append(d)
    if not snapshots:
        return None
    # Sort by update number (u00001 → 1)
    def _update_num(p: Path) -> int:
        m = re.match(r"u(\d+)", p.name)
        return int(m.group(1)) if m else -1
    snapshots.sort(key=_update_num)
    return snapshots[-1]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(
    run_dir: Path,
    update: Optional[int] = None,
    snapshot_dir: Optional[Path] = None,
    load_experiment: bool = True,
) -> KnobContext:
    """Build a KnobContext from a run directory.

    Loads config.json, parses the latest (or target) ``__RAW_STATS__``
    entry, auto-finds a snapshot if not given, and optionally loads the
    experiment instance.

    Args:
        run_dir: Training run directory.
        update: Target update number (default: latest from log).
        snapshot_dir: Explicit snapshot directory (default: auto-find).
        load_experiment: If True, load the experiment via the registry.

    Returns:
        KnobContext with all fields populated.
    """
    run_dir = Path(run_dir).resolve()
    config = load_config(run_dir)
    log_entry = parse_log_entry(run_dir, update)
    log_stats = log_entry.get("stats") if log_entry else None

    # Auto-find snapshot
    if snapshot_dir is None:
        snapshot_dir = find_latest_snapshot(run_dir)
    else:
        snapshot_dir = Path(snapshot_dir).resolve()

    # Load experiment
    experiment = None
    if load_experiment:
        try:
            from .probes import load_experiment_from_run
            experiment = load_experiment_from_run(run_dir)
        except Exception:
            pass  # Experiment loading is best-effort

    return KnobContext(
        run_dir=run_dir,
        config=config,
        log_entry=log_entry,
        log_stats=log_stats,
        snapshot_dir=snapshot_dir,
        experiment=experiment,
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def intervene_check(
    run_dir: Path,
    update: Optional[int] = None,
    knob_name: Optional[str] = None,
    snapshot_dir: Optional[Path] = None,
) -> str:
    """Run intervene-check and return the rendered report.

    Args:
        run_dir: Training run directory.
        update: Target update number (default: latest from log).
        knob_name: If given, only run this knob (filter by name).
        snapshot_dir: Explicit snapshot directory (default: auto-find).

    Returns:
        Human-readable report string.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    ctx = build_context(run_dir, update=update, snapshot_dir=snapshot_dir)

    # Determine update for display
    display_update = update
    if display_update is None and ctx.log_entry is not None:
        display_update = ctx.log_entry.get("update")

    # Collect all knob checks
    all_checks: List[KnobCheck] = list(builtin_knob_checks())

    # Add experiment-specific knobs
    if ctx.experiment is not None:
        try:
            exp_checks = ctx.experiment.knob_checks()
            all_checks.extend(exp_checks)
        except Exception:
            pass  # Best-effort

    # Filter by knob_name if given
    if knob_name is not None:
        all_checks = [c for c in all_checks if c.name == knob_name]
        if not all_checks:
            available = [c.name for c in builtin_knob_checks()]
            if ctx.experiment is not None:
                try:
                    available.extend(c.name for c in ctx.experiment.knob_checks())
                except Exception:
                    pass
            return (f"Knob {knob_name!r} not found. "
                    f"Available: {available}")

    # Run checks
    results = run_knob_checks(ctx, all_checks)

    # Render
    return render_knob_table(results, update=display_update)
