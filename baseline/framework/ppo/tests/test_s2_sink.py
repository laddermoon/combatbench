"""S2 DebugSink / NpzSink tests.

Tests cover:
- NpzSink records + writes per-stage .npz files.
- record_minibatch groups by (epoch, mb).
- --full-grad captures flat gradient tensor + param names.
- Stage/name contract: fixed list of stages.
- Empty sink writes empty files.
- None values are silently dropped.
- Unknown stage raises ValueError.

Conventions follow test_trainer.py / test_s1_provenance.py:
- Simple print("test_xxx: PASS") at the end of each test.
- `if __name__ == "__main__":` block runs all tests sequentially.
- No pytest fixtures required (but pytest-compatible).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.sink import DebugSink, NpzSink, STAGES


# ---------------------------------------------------------------------------
# NpzSink basics
# ---------------------------------------------------------------------------

def test_npz_sink_records_and_writes(tmp_path):
    """NpzSink records arrays and writes per-stage .npz on close."""
    sink = NpzSink(tmp_path / "out")
    sink.record("buffer", "old_log_prob", np.zeros(5, dtype=np.float32))
    sink.record("buffer", "explore_factor", np.ones(5, dtype=np.float32))
    sink.record("combine", "combined_adv", np.full(5, 0.5, dtype=np.float32))
    sink.close()

    assert (tmp_path / "out" / "buffer.npz").exists()
    assert (tmp_path / "out" / "combine.npz").exists()
    assert (tmp_path / "out" / "gae.npz").exists()  # empty but written
    assert (tmp_path / "out" / "update.npz").exists()  # empty but written

    data = np.load(tmp_path / "out" / "buffer.npz")
    assert np.allclose(data["old_log_prob"], 0.0)
    assert np.allclose(data["explore_factor"], 1.0)

    data = np.load(tmp_path / "out" / "combine.npz")
    assert np.allclose(data["combined_adv"], 0.5)

    print("test_npz_sink_records_and_writes: PASS")


def test_npz_sink_minibatch_grouping(tmp_path):
    """record_minibatch groups by (epoch, mb) and stacks uniform shapes."""
    sink = NpzSink(tmp_path / "out")
    sink.record_minibatch(0, 0, "ratio", np.ones(3, dtype=np.float32))
    sink.record_minibatch(0, 1, "ratio", np.full(3, 2.0, dtype=np.float32))
    sink.record_minibatch(1, 0, "ratio", np.full(3, 3.0, dtype=np.float32))
    sink.close()

    data = np.load(tmp_path / "out" / "update.npz")
    # Per-minibatch flattened keys.
    assert "000:000.ratio" in data
    assert "000:001.ratio" in data
    assert "001:000.ratio" in data
    # Stacked (all same shape → 3,).
    assert "ratio" in data
    stacked = data["ratio"]  # (3 minibatches, 3)
    assert stacked.shape == (3, 3)
    assert np.allclose(stacked[0, 0], 1.0)
    assert np.allclose(stacked[1, 0], 2.0)
    assert np.allclose(stacked[2, 0], 3.0)

    print("test_npz_sink_minibatch_grouping: PASS")


def test_npz_sink_full_grad(tmp_path):
    """--full-grad captures flat gradient + param names for epoch 0, mb 0."""
    sink = NpzSink(tmp_path / "out")

    # Simulate a small model's gradients.
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    # Fake gradients.
    for name, p in model.named_parameters():
        p.grad = torch.randn_like(p)

    flat_grads = []
    param_names = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            flat_grads.append(p.grad.detach().cpu().numpy().ravel())
            param_names.append(name)
    full_grad = np.concatenate(flat_grads)
    sink.record_minibatch(0, 0, "full_grad", full_grad)
    sink.record_minibatch(0, 0, "full_grad_param_names",
                          np.array(param_names, dtype=object))
    sink.close()

    data = np.load(tmp_path / "out" / "update.npz", allow_pickle=True)
    assert "full_grad" in data
    assert "full_grad_param_names" in data
    # Total params: 3*4+4 + 4*2+2 = 26
    # Stacked across 1 minibatch → shape (1, 26)
    assert data["full_grad"].shape == (1, 26)
    # Param names: stacked across 1 minibatch → (1, N)
    names_arr = data["full_grad_param_names"]
    # Flatten in case of stacking.
    names = names_arr.ravel() if names_arr.ndim > 1 else names_arr
    name_list = list(names)
    assert "0.weight" in name_list
    assert "1.weight" in name_list

    print("test_npz_sink_full_grad: PASS")


def test_npz_sink_empty_writes_files(tmp_path):
    """Empty sink (no records) still writes empty .npz files."""
    sink = NpzSink(tmp_path / "out")
    sink.close()

    for stage in STAGES:
        assert (tmp_path / "out" / f"{stage}.npz").exists()

    print("test_npz_sink_empty_writes_files: PASS")


def test_npz_sink_none_dropped(tmp_path):
    """None values are silently dropped (no array stored)."""
    sink = NpzSink(tmp_path / "out")
    sink.record("buffer", "frame_ids", None)  # should be dropped
    sink.record("buffer", "old_log_prob", np.zeros(3, dtype=np.float32))
    sink.close()

    data = np.load(tmp_path / "out" / "buffer.npz")
    assert "old_log_prob" in data
    assert "frame_ids" not in data

    print("test_npz_sink_none_dropped: PASS")


def test_npz_sink_unknown_stage_raises(tmp_path):
    """Unknown stage name raises ValueError."""
    sink = NpzSink(tmp_path / "out")
    with pytest.raises(ValueError):
        sink.record("unknown_stage", "foo", np.zeros(3))
    sink.close()

    print("test_npz_sink_unknown_stage_raises: PASS")


def test_npz_sink_close_idempotent(tmp_path):
    """Double close is safe (no error)."""
    sink = NpzSink(tmp_path / "out")
    sink.record("buffer", "x", np.zeros(2, dtype=np.float32))
    sink.close()
    sink.close()  # should not raise
    print("test_npz_sink_close_idempotent: PASS")


def test_npz_sink_record_after_close_raises(tmp_path):
    """Recording after close raises RuntimeError."""
    sink = NpzSink(tmp_path / "out")
    sink.close()
    with pytest.raises(RuntimeError):
        sink.record("buffer", "x", np.zeros(2, dtype=np.float32))
    print("test_npz_sink_record_after_close_raises: PASS")


def test_stages_contract():
    """STAGES is the fixed contract: buffer, gae, combine, update."""
    assert STAGES == ("buffer", "gae", "combine", "update")
    print("test_stages_contract: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_npz_sink_records_and_writes(Path("/tmp/test_s2_sink"))
    test_npz_sink_minibatch_grouping(Path("/tmp/test_s2_sink_mb"))
    test_npz_sink_full_grad(Path("/tmp/test_s2_sink_fg"))
    test_npz_sink_empty_writes_files(Path("/tmp/test_s2_sink_empty"))
    test_npz_sink_none_dropped(Path("/tmp/test_s2_sink_none"))
    test_npz_sink_unknown_stage_raises(Path("/tmp/test_s2_sink_unknown"))
    test_npz_sink_close_idempotent(Path("/tmp/test_s2_sink_idem"))
    test_npz_sink_record_after_close_raises(Path("/tmp/test_s2_sink_close"))
    test_stages_contract()
    print("\nAll S2 sink tests passed.")
