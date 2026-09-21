"""Resume bit-identicality — the flagship equivalence test.

Property under test:

    continuous run A  ==  run B1 (to checkpoint) -> NEW PROCESS -> resume B2

compared update-by-update on the overlap.  train_ppo runs in *separate*
subprocesses because resume equivalence is a process-boundary property:
an in-process rerun would inherit module state a real resume cannot see.

Checks, strongest first:
  1. The checkpoint at the final update — the ENTIRE payload is compared
     recursively: actor/critic params, Adam exp_avg/exp_avg_sq/step,
     experiment state, and the saved RNG states themselves.  Params
     bitwise-equal implies everything upstream (rollouts, ADV, minibatch
     order, gradients, optimizer math) was bitwise identical.
  2. ``__RAW_STATS__`` lines for the resumed updates — all fields equal
     except the ``timing`` block.
  3. ``dir_cos`` present at the first resumed *dump* update — regression
     check that prev_gvec was restored (a v1 resume would emit none).
     The gradsig diagnostic is dump-only, so the driver schedules dump
     requests at u1 (seeds prev_gvec into ckpt_u2) and u3 (first resumed
     update reads it back).
  4. gradsig npz arrays inside the resumed update's dump —
     ``dumps/u00003/gradsig.npz``, np.array_equal on update-local keys.

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
_DRIVER = _TESTS_DIR / "_resume_driver.py"

CKPT_UPDATE = 2      # checkpoint boundary: eval_interval=2 -> u2 exists
END_UPDATE = 4       # both branches reach u4


def _run_driver(
    run_dir: Path,
    max_updates: int,
    resume_from: Path = None,
    dump_updates: tuple = (),
) -> str:
    """Run the driver as its own process; return captured stdout."""
    cmd = [
        sys.executable, str(_DRIVER),
        "--run-dir", str(run_dir),
        "--max-updates", str(max_updates),
    ]
    if resume_from is not None:
        cmd += ["--resume-from", str(resume_from)]
    if dump_updates:
        cmd += ["--dump-updates", ",".join(str(u) for u in dump_updates)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
    proc = subprocess.run(
        cmd, env=env, cwd=_PROJECT_ROOT,
        capture_output=True, text=True, timeout=600,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "_driver_stdout.log").write_text(
        proc.stdout + "\n===== STDERR =====\n" + proc.stderr
    )
    assert proc.returncode == 0, (
        f"driver failed ({run_dir}):\n{proc.stderr[-2000:]}"
    )
    return proc.stdout


def _parse_raw_stats(stdout: str) -> Dict[int, Dict[str, Any]]:
    """Extract {update: raw_log_dict} from __RAW_STATS__ lines."""
    out = {}
    for line in stdout.splitlines():
        if line.startswith("__RAW_STATS__ "):
            rec = json.loads(line[len("__RAW_STATS__ "):])
            out[int(rec["update"])] = rec
    return out


def _deep_equal(a: Any, b: Any, path: str = "") -> List[str]:
    """Recursively compare two structures; return list of diff paths."""
    diffs: List[str] = []
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if not torch.equal(a.cpu(), b.cpu()):
            diffs.append(f"{path}: tensor differs {tuple(a.shape)}")
        return diffs
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not np.array_equal(a, b):
            diffs.append(f"{path}: ndarray differs {a.shape}")
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


def _stats_without_timing(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Drop wall-clock fields — the only legitimately non-deterministic
    part of a stats record (identical computation, different durations)."""
    rec = dict(rec)
    rec.pop("timing", None)
    stats = dict(rec.get("stats", {}))
    rec["stats"] = {
        k: v for k, v in stats.items() if not k.endswith("_time_s")
    }
    return rec


def test_resume_continuation_is_bit_identical(tmp_path):
    dir_a = tmp_path / "run_A"    # continuous u1..u4
    dir_b1 = tmp_path / "run_B1"  # u1..u2, produces checkpoint_u2
    dir_b2 = tmp_path / "run_B2"  # resume -> u3..u4

    # Dump requests schedule the (dump-only) gradsig diagnostic: u1
    # seeds prev_gvec so ckpt_u2 carries it; u3 — the first resumed
    # update — must then emit dir_cos against the restored vector.
    out_a = _run_driver(dir_a, END_UPDATE, dump_updates=(1, CKPT_UPDATE + 1))
    out_b1 = _run_driver(dir_b1, CKPT_UPDATE, dump_updates=(1,))
    ckpt = dir_b1 / "checkpoints" / f"checkpoint_u{CKPT_UPDATE:05d}.pt"
    assert ckpt.exists(), f"missing checkpoint {ckpt}"
    out_b2 = _run_driver(
        dir_b2, END_UPDATE, resume_from=ckpt,
        dump_updates=(CKPT_UPDATE + 1,),
    )

    stats_a = _parse_raw_stats(out_a)
    stats_b1 = _parse_raw_stats(out_b1)
    stats_b2 = _parse_raw_stats(out_b2)

    # Sanity: independent runs still agree on the shared prefix.
    for u in (1, CKPT_UPDATE):
        diffs = _deep_equal(
            _stats_without_timing(stats_a[u]),
            _stats_without_timing(stats_b1[u]),
            f"prefix.u{u}",
        )
        assert not diffs, "prefix runs diverged:\n" + "\n".join(diffs[:20])

    # The actual property: resumed u3..u4 == continuous u3..u4.
    for u in range(CKPT_UPDATE + 1, END_UPDATE + 1):
        assert u in stats_b2, f"resumed run missing update {u}"
        diffs = _deep_equal(
            _stats_without_timing(stats_a[u]),
            _stats_without_timing(stats_b2[u]),
            f"resumed.u{u}",
        )
        assert not diffs, (
            f"resumed update {u} not bit-identical:\n"
            + "\n".join(diffs[:30])
        )

    # prev_gvec regression: dir_cos must exist at the first resumed
    # dump update (would be absent on a v1/no-RNG resume).  u3 is the
    # only diagnostic update after the boundary, and both sides computed
    # it against the u1 aggregate gradient.
    s3 = stats_b2[CKPT_UPDATE + 1]["stats"]
    assert "grad_sig_dir_cos" in s3, (
        "resumed run missing dir_cos at first dump update — "
        "the dump-only diagnostic did not run"
    )
    assert s3["grad_sig_dir_cos"] == (
        stats_a[CKPT_UPDATE + 1]["stats"]["grad_sig_dir_cos"]
    ), "dir_cos differs — prev_gvec was not restored from checkpoint"

    # Strongest check: the final checkpoint payloads — params, Adam
    # moments, experiment state, and the saved RNG streams — all equal.
    ckpt_a = torch.load(
        dir_a / "checkpoints" / f"checkpoint_u{END_UPDATE:05d}.pt",
        map_location="cpu", weights_only=False,
    )
    ckpt_b = torch.load(
        dir_b2 / "checkpoints" / f"checkpoint_u{END_UPDATE:05d}.pt",
        map_location="cpu", weights_only=False,
    )
    diffs = _deep_equal(ckpt_a, ckpt_b, "ckpt_u4")
    assert not diffs, (
        "final checkpoints differ:\n" + "\n".join(diffs[:30])
    )

    # gradsig artifacts live inside the dump now
    # (dumps/uNNNNN/gradsig.npz).  Only update-local data is compared:
    # the raw per-frame arrays and the diagnostic scalars.  Axis-derived
    # artifacts (hist, *_edges, hist_over/under, n_*in_hist) are
    # intentionally excluded — they depend on the run-local frozen
    # meta.json, which a fresh run dir rederives from ITS first
    # diagnostic update.  Resume into the SAME dir keeps meta.json and
    # reproduces even those identically; resume into a NEW dir re-bins
    # identical raw data.
    _GRADSIG_LOCAL_KEYS = (
        "grad_norm", "cos", "proj", "valid", "norm_quantiles",
        "gnorm", "coherence", "dir_cos", "proj_mean", "proj_std",
        "frac_neg", "n_sampled", "n_valid", "n_excluded",
        "n_nonfinite", "n_params", "sampled_idx", "w_adv", "floor_pen",
    )
    for u in range(CKPT_UPDATE + 1, END_UPDATE + 1):
        npz_a = dir_a / "dumps" / f"u{u:05d}" / "gradsig.npz"
        npz_b = dir_b2 / "dumps" / f"u{u:05d}" / "gradsig.npz"
        if u == CKPT_UPDATE + 1:
            # u3 carries a dump request on both sides — the merged
            # gradsig.npz must exist (silent skip would mask a missing
            # artifact).
            assert npz_a.exists(), f"missing {npz_a}"
            assert npz_b.exists(), f"missing {npz_b}"
        if not (npz_a.exists() and npz_b.exists()):
            continue
        da, db = np.load(npz_a), np.load(npz_b)
        for k in _GRADSIG_LOCAL_KEYS:
            if k not in da.files or k not in db.files:
                continue
            assert np.array_equal(da[k], db[k]), (
                f"gradsig u{u} array '{k}' differs"
            )

    print("test_resume_continuation_is_bit_identical: PASS")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_resume_continuation_is_bit_identical(Path(d))
