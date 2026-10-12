"""SAC resume bit-identicality across a real process boundary.

Property under test:

    continuous run A  ==  run B1 (to a checkpoint) -> NEW PROCESS -> resume B2

on the ``sac_balance`` experiment (real MuJoCo), shrunk to a few short
rounds.  ``train_sac`` runs in separate subprocesses because resume
equivalence is a process-boundary property: an in-process rerun would
inherit module state a real resume cannot see.

Checks:
  1. Terminal checkpoints at the same env_step — the ENTIRE payload is
     compared recursively: actor/critic params, all optimizer states,
     alpha state, replay arrays + rng + identity maps, runtime RNG and
     clocks, and experiment state.  Bitwise equality implies everything
     upstream (rollouts, slice construction, batch draws, gradients,
     optimizer math, replay evictions) was bitwise identical.
  2. ``run_terminal.json`` records agree on status/env_step/critic_tick.

Requires MuJoCo; skipped when unavailable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch


def _can_import_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _can_import_mujoco(),
    reason="MuJoCo not available — resume equivalence needs the real env",
)

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parents[3]  # things/combatbench
_DRIVER = _TESTS_DIR / "_sac_resume_driver.py"

SPLIT_STEPS = 200   # B1 stops at/after this env_step
TOTAL_STEPS = 500   # both branches reach this env_step

# Keys whose values are legitimately different between two runs even when
# every computed quantity is identical.
_VOLATILE_CONFIG_KEYS = ("saved_at",)

# Per-sample ``collection`` provenance that records WHICH process produced
# the data rather than WHAT was computed: worker_id is a PID, job_key and
# policy_fingerprint are hashes over run-scoped artifacts (export paths
# embed the run dir).  The underlying computed quantities they label —
# episode contents and policy weights — are compared bitwise elsewhere
# (replay arrays, trainer.pt).
_VOLATILE_COLLECTION_KEYS = ("job_key", "policy_fingerprint", "worker_id")

# clocks.checkpoint_tick counts checkpoint *writes* across the run
# lineage — a resumed run legitimately ticks more (it inherits the
# parent's counter and then writes its own).  It is bookkeeping about
# I/O, not learning state; every learning-relevant clock (env_step,
# critic_tick, collection_round, ...) is still compared exactly.
_VOLATILE_CLOCK_KEYS = ("checkpoint_tick",)


def _run_driver(
    run_dir: Path,
    max_env_steps: int,
    resume_from: Path = None,
) -> None:
    cmd = [
        sys.executable, str(_DRIVER),
        "--run-dir", str(run_dir),
        "--max-env-steps", str(max_env_steps),
    ]
    if resume_from is not None:
        cmd += ["--resume-from", str(resume_from)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
    proc = subprocess.run(
        cmd, env=env, cwd=_PROJECT_ROOT,
        capture_output=True, text=True, timeout=900,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "_driver_stdout.log").write_text(
        proc.stdout + "\n===== STDERR =====\n" + proc.stderr
    )
    assert proc.returncode == 0, (
        f"driver failed ({run_dir}):\n{proc.stderr[-3000:]}"
    )


def _terminal(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "run_terminal.json").read_text())


def _deep_equal(a: Any, b: Any, path: str = "") -> List[str]:
    diffs: List[str] = []
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if not torch.equal(a.cpu(), b.cpu()):
            diffs.append(f"{path}: tensor differs {tuple(a.shape)}")
        return diffs
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.dtype != b.dtype:
            diffs.append(f"{path}: dtype {a.dtype} != {b.dtype}")
            return diffs
        if not np.array_equal(a, b):
            diffs.append(f"{path}: ndarray differs {a.shape}")
        return diffs
    if isinstance(a, (bytes, bytearray)) or isinstance(b, (bytes, bytearray)):
        if a != b:
            diffs.append(f"{path}: bytes differ")
        return diffs
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(
                f"{path}: keys differ {sorted(set(a) ^ set(b))[:10]}"
            )
            return diffs
        for k in a:
            diffs += _deep_equal(a[k], b[k], f"{path}.{k}")
        return diffs
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
            return diffs
        for i, (x, y) in enumerate(zip(a, b)):
            diffs += _deep_equal(x, y, f"{path}[{i}]")
        return diffs
    if isinstance(a, float) and isinstance(b, float):
        if a != b and not (np.isnan(a) and np.isnan(b)):
            diffs.append(f"{path}: {a} != {b}")
        return diffs
    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def _load_payload(path: Path) -> Any:
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        for key in _VOLATILE_CONFIG_KEYS:
            data.pop(key, None)
        return data
    raise AssertionError(f"unexpected file in checkpoint: {path.name}")


def _normalize_run_ids(obj: Any, names: tuple) -> Any:
    """source_key/collection provenance legitimately embeds the run_id
    (``run_a/r000006/...`` vs ``run_b2/r000006/...``) — that is the
    correct answer to "which run produced this sample".  Normalize it
    away before comparing so only *computed* differences remain."""
    if isinstance(obj, str):
        for name in names:
            obj = obj.replace(name, "<RUN>")
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "collection" and isinstance(v, dict):
                v = {
                    kk: vv for kk, vv in v.items()
                    if kk not in _VOLATILE_COLLECTION_KEYS
                }
            if k == "clocks" and isinstance(v, dict):
                v = {
                    kk: vv for kk, vv in v.items()
                    if kk not in _VOLATILE_CLOCK_KEYS
                }
            out[k] = _normalize_run_ids(v, names)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(_normalize_run_ids(v, names) for v in obj)
    if isinstance(obj, set):
        return {_normalize_run_ids(v, names) for v in obj}
    return obj


def _compare_checkpoint_dirs(
    dir_a: Path, dir_b: Path, run_names: tuple = ()
) -> List[str]:
    diffs: List[str] = []
    files_a = {
        p.name for p in dir_a.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    files_b = {
        p.name for p in dir_b.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    if files_a != files_b:
        diffs.append(f"file sets differ: {sorted(files_a ^ files_b)}")
        return diffs
    for name in sorted(files_a):
        pa, pb = dir_a / name, dir_b / name
        try:
            va = _normalize_run_ids(_load_payload(pa), run_names)
            vb = _normalize_run_ids(_load_payload(pb), run_names)
            diffs += _deep_equal(va, vb, name)
        except Exception as exc:  # unreadable format -> raw bytes compare
            if pa.read_bytes() != pb.read_bytes():
                diffs.append(f"{name}: bytes differ ({exc})")
    return diffs


def test_sac_resume_continuation_is_bit_identical(tmp_path):
    run_a = tmp_path / "run_a"
    run_b1 = tmp_path / "run_b1"
    run_b2 = tmp_path / "run_b2"

    # Continuous branch: straight through to TOTAL_STEPS.
    _run_driver(run_a, TOTAL_STEPS)

    # Split branch: stop at/after SPLIT_STEPS, resume in a new process.
    _run_driver(run_b1, SPLIT_STEPS)
    term_b1 = _terminal(run_b1)
    assert term_b1["status"] == "completed"
    split_ckpt = Path(term_b1["final_checkpoint"])
    assert (split_ckpt / "manifest.json").exists()

    _run_driver(run_b2, TOTAL_STEPS, resume_from=split_ckpt)

    term_a = _terminal(run_a)
    term_b2 = _terminal(run_b2)
    for key in ("status", "env_step", "critic_tick", "collection_round"):
        assert term_a[key] == term_b2[key], (
            f"terminal {key}: continuous={term_a[key]} "
            f"resumed={term_b2[key]}"
        )

    ckpt_a = Path(term_a["final_checkpoint"])
    ckpt_b2 = Path(term_b2["final_checkpoint"])
    assert ckpt_a.name == ckpt_b2.name, (
        "terminal checkpoints at different env_steps: "
        f"{ckpt_a.name} vs {ckpt_b2.name}"
    )

    diffs = _compare_checkpoint_dirs(
        ckpt_a, ckpt_b2, run_names=(run_a.name, run_b2.name)
    )
    assert not diffs, (
        "resumed run diverged from continuous run:\n"
        + "\n".join(diffs[:40])
    )
