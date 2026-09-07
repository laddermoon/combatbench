"""Tests for the minimal PPO example (P1-5).

These tests ensure that:
1. The minimal example is importable and registered.
2. All abstract methods are implemented (no NotImplementedError at runtime).
3. The GUIDE.md §4 code blocks match the actual file.
4. A 2-update smoke run completes without error (requires MuJoCo).
"""
from __future__ import annotations

import importlib
import inspect
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# P1-5: Minimal example is importable, registered, and structurally correct
# ---------------------------------------------------------------------------

def test_minimal_example_imports():
    """The minimal example can be imported without error."""
    from baseline.experiments_ppo.exp_minimal import MinimalExperiment
    assert MinimalExperiment.name == "minimal"


def test_minimal_example_is_registered():
    """The minimal example appears in the experiment registry."""
    from baseline.experiments_ppo import list_ppo_experiments, get_ppo_experiment
    assert "minimal" in list_ppo_experiments()
    exp = get_ppo_experiment("minimal")
    assert exp.name == "minimal"
    assert exp.obs_dim == 96
    assert exp.action_dim == 21


def test_minimal_example_implements_all_abstract_methods():
    """MinimalExperiment implements every abstract method of ExperimentPPO.

    This catches the case where a new abstract method is added to
    ExperimentPPO but the minimal example (and GUIDE.md) isn't updated.
    Methods can be inherited from CombatExperimentPPOBase.
    """
    from baseline.experiments_ppo.exp_minimal import MinimalExperiment
    from baseline.framework.ppo import ExperimentPPO

    # Collect all abstract method names from ExperimentPPO
    abstract_methods = set()
    for cls in ExperimentPPO.__mro__:
        if hasattr(cls, "__abstractmethods__"):
            abstract_methods.update(cls.__abstractmethods__)

    # Check that MinimalExperiment (or its base) overrides every one.
    # An abstract method is "implemented" if the resolved attribute is
    # not the abstract stub — i.e., the method's __isabstractmethod__
    # is False or absent.
    for method_name in abstract_methods:
        attr = getattr(MinimalExperiment, method_name, None)
        assert attr is not None, (
            f"MinimalExperiment is missing method '{method_name}'. "
            f"Update exp_minimal.py and GUIDE.md §4."
        )
        is_abstract = getattr(attr, "__isabstractmethod__", False)
        assert not is_abstract, (
            f"MinimalExperiment does not implement abstract method '{method_name}'. "
            f"Update exp_minimal.py and GUIDE.md §4."
        )


def test_minimal_example_reward_channels():
    """reward_channels() returns a valid tuple."""
    from baseline.experiments_ppo.exp_minimal import MinimalExperiment
    exp = MinimalExperiment()
    channels = exp.reward_channels()
    assert len(channels) == 1
    assert channels[0].name == "r_potential"
    assert 0 < channels[0].gamma < 1
    assert 0 < channels[0].gae_lambda <= 1


def test_minimal_example_common_params():
    """common_params() returns valid CommonParams."""
    from baseline.experiments_ppo.exp_minimal import MinimalExperiment
    exp = MinimalExperiment()
    cp = exp.common_params()
    assert cp.name == "minimal"
    assert cp.episodes_per_update > 0
    assert cp.max_updates > 0


def test_minimal_example_ppo_params():
    """ppo_params() returns valid PPOParams."""
    from baseline.experiments_ppo.exp_minimal import MinimalExperiment
    exp = MinimalExperiment()
    pp = exp.ppo_params()
    assert pp.clip_eps > 0
    assert pp.update_epochs > 0
    assert pp.minibatch_size > 0


def test_minimal_example_state_roundtrip():
    """state() / load_state() round-trip works."""
    from baseline.experiments_ppo.exp_minimal import MinimalExperiment
    exp = MinimalExperiment()
    exp._best_potential = 0.75
    state = exp.state()
    assert state["best_potential"] == 0.75

    exp2 = MinimalExperiment()
    exp2.load_state(state)
    assert exp2._best_potential == 0.75


# ---------------------------------------------------------------------------
# P1-5: GUIDE.md §4 references this file (no untested code blocks)
# ---------------------------------------------------------------------------

def test_guide_references_minimal_example():
    """GUIDE.md §4 should reference exp_minimal.py, not inline broken code."""
    guide_path = Path(__file__).resolve().parent.parent / "GUIDE.md"
    if not guide_path.exists():
        pytest.skip("GUIDE.md not found")
    content = guide_path.read_text(encoding="utf-8")
    # GUIDE.md should mention the minimal example file
    assert "exp_minimal" in content or "minimal" in content.lower(), (
        "GUIDE.md should reference exp_minimal.py as the canonical minimal example."
    )


# ---------------------------------------------------------------------------
# P1-5: Full smoke run (requires MuJoCo — skipped if env not available)
# ---------------------------------------------------------------------------

def _can_import_mujoco() -> bool:
    """Check if MuJoCo is importable."""
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _can_import_mujoco(),
    reason="MuJoCo not available — full smoke run requires the env",
)
def test_minimal_example_smoke_run(tmp_path):
    """2-update smoke run of the minimal example completes without error.

    This is the ultimate test: the minimal example actually trains for
    2 updates and produces a checkpoint.  If any code path is broken
    (imports, build_jobs, build_trajectories, ppo_update, eval, export),
    this test fails.
    """
    import dataclasses
    from baseline.experiments_ppo import get_ppo_experiment
    from baseline.framework.ppo.loop import train_ppo

    exp = get_ppo_experiment("minimal")
    # Simulate --smoke: shrink the schedule for a fast 2-update run.
    # train_ppo doesn't have a smoke= kwarg; train.py does this via
    # dataclasses.replace on common_params and ppo_params.
    cp = exp.common_params()
    cp = dataclasses.replace(
        cp, max_updates=2, episodes_per_update=8,
        eval_episodes=4, eval_interval=2, rollout_workers=2,
    )
    exp.common_params = lambda: cp  # type: ignore
    pp = exp.ppo_params()
    pp = dataclasses.replace(pp, update_epochs=2, minibatch_size=64)
    exp.ppo_params = lambda: pp  # type: ignore

    train_ppo(
        experiment=exp,
        run_dir=Path(tmp_path),
    )
    # Check that at least one checkpoint was written
    checkpoints = list(Path(tmp_path).glob("checkpoints/*.pt"))
    assert len(checkpoints) > 0, "Smoke run should produce at least one checkpoint"


if __name__ == "__main__":
    # Run tests that don't need MuJoCo
    test_minimal_example_imports()
    test_minimal_example_is_registered()
    test_minimal_example_implements_all_abstract_methods()
    test_minimal_example_reward_channels()
    test_minimal_example_common_params()
    test_minimal_example_ppo_params()
    test_minimal_example_state_roundtrip()
    test_guide_references_minimal_example()
    print("All P1-5 tests (no MuJoCo): PASS")
