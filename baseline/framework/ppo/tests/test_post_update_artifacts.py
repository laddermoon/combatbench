"""Post-update artifacts & experiment-side reference-chain tests.

Covers the lifecycle refactor:
- ``ExperimentPPO.post_update(stats, update, *, artifacts)`` contract
- ``UpdateArtifacts`` semantics (this update's produced objects)
- ``CombatExperimentPPOBase`` reference-history accumulation, horizon
  bound, ``_sampling_spec`` assembly, and ``state()``/``load_state()``
  round-trip of the reference chain.
- ``build_jobs`` required ``update`` kwarg and ``run_name()`` default.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.experiment import UpdateArtifacts
from baseline.framework.rollout.job import ReferenceSpec
from envs.framework.policy import PolicyBlueprint


def _bp(tmp_path: Path, name: str) -> PolicyBlueprint:
    """Minimal file: blueprint (PolicyBlueprint has cls/config only)."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "policy.py").write_text("# stub\n")
    return PolicyBlueprint(cls=f"file:{d}/policy.py:Stub")


def _bp_dirname(bp: PolicyBlueprint) -> str:
    """Directory name embedded in a file: blueprint's cls path."""
    # cls = "file:<dir>/policy.py:Stub"
    return Path(bp.cls.split("file:", 1)[1]).parent.name


def _exp(tmp_path, **kwargs):
    """A bare CombatExperimentPPOBase subclass instance (no env needed)."""
    from baseline.experiments_ppo.base import CombatExperimentPPOBase

    class _E(CombatExperimentPPOBase):
        name = "t"
        def reward_channels(self):
            return ()
        def build_trajectories(self, episodes):
            return []
        def on_eval(self, episodes, update):
            return {}

    return _E(**kwargs)


class TestPostUpdateContract:
    def test_artifacts_fields(self, tmp_path):
        bp = _bp(tmp_path, "u00001")
        a = UpdateArtifacts(policy_bp=bp, checkpoint_path=tmp_path / "ck.pt")
        assert a.policy_bp is bp
        assert a.checkpoint_path.name == "ck.pt"
        assert UpdateArtifacts().policy_bp is None
        assert UpdateArtifacts().checkpoint_path is None

    def test_base_post_update_accepts_no_artifacts(self, tmp_path):
        e = _exp(tmp_path)
        assert e.post_update(SimpleNamespace(), 1) is None
        assert e.post_update(SimpleNamespace(), 1,
                             artifacts=UpdateArtifacts()) is None
        assert e._ref_history == []

    def test_ref_history_accumulates_and_bounds(self, tmp_path):
        e = _exp(tmp_path, reference_horizon="3")
        for u in range(1, 6):
            e.post_update(
                SimpleNamespace(), u,
                artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, f"u{u}")),
            )
        # Buffer keeps H+1 entries: the newest is the current policy
        # itself; the ensemble draws the H strictly-past versions.
        assert len(e._ref_history) == 4
        assert [_bp_dirname(b) for b in e._ref_history] == [
            "u2", "u3", "u4", "u5",
        ]

    def test_ref_history_unbounded_when_horizon_zero(self, tmp_path):
        e = _exp(tmp_path)
        for u in range(1, 4):
            e.post_update(
                SimpleNamespace(), u,
                artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, f"u{u}")),
            )
        assert len(e._ref_history) == 3


class TestDeltaValidation:
    def test_delta_factor_requires_horizon(self, tmp_path):
        with pytest.raises(ValueError, match="reference_horizon"):
            _exp(tmp_path, delta_factor="1.0")

    def test_delta_factor_with_horizon_ok(self, tmp_path):
        e = _exp(tmp_path, delta_factor="1.0", reference_horizon="10")
        assert e.delta_factor == 1.0
        assert e.reference_horizon == 10


class TestSamplingSpecAssembly:
    def test_warmup_no_history_plain_spec(self, tmp_path):
        e = _exp(tmp_path, delta_factor="5.0",
                 reference_horizon="10")
        spec = e._sampling_spec()
        assert spec.reference is None
        assert spec.delta_factor == 0.0

    def test_delta_spec_uniform_weights(self, tmp_path):
        e = _exp(tmp_path, delta_factor="5.0",
                 reference_horizon="10")
        # Partial window (9 strictly-past < H=10): still plain spec —
        # the warmup gate keeps Δ semantics uniform at n=H.
        for u in range(1, 11):
            e.post_update(
                SimpleNamespace(), u,
                artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, f"u{u}")),
            )
        spec = e._sampling_spec()
        assert spec.reference is None
        assert spec.delta_factor == 0.0
        # One more export completes the window: u11 is the current
        # rollout policy (Gen0, excluded); the ensemble is u1..u10.
        e.post_update(
            SimpleNamespace(), 11,
            artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, "u11")),
        )
        spec = e._sampling_spec()
        assert isinstance(spec.reference, ReferenceSpec)
        assert [_bp_dirname(b) for b in spec.reference.policies] == [
            f"u{u}" for u in range(1, 11)
        ]
        assert spec.reference.weights == pytest.approx(
            tuple([0.1] * 10))
        assert spec.delta_factor == 5.0

    def test_self_only_history_plain_spec(self, tmp_path):
        """A single history member is the current policy itself —
        nothing strictly-past exists, so delta stays off (plain spec)."""
        e = _exp(tmp_path, delta_factor="5.0",
                 reference_horizon="10")
        e.post_update(
            SimpleNamespace(), 1,
            artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, "u1")),
        )
        spec = e._sampling_spec()
        assert spec.reference is None
        assert spec.delta_factor == 0.0

    def test_frozen_ensemble_excludes_self(self, tmp_path):
        """Frozen mode: ensemble is also strictly-past — Gen0 already
        supplies the current policy via its own deterministic action;
        a self member would only bias a_ref toward μ₀ (Δ diluted by
        a mechanically-zero member).  Same full-window warmup gate as
        dynamic: activates at update H+2."""
        e = _exp(tmp_path, delta_factor="5.0",
                 reference_horizon="10", delta_mode="frozen")
        for u in range(1, 12):
            e.post_update(
                SimpleNamespace(), u,
                artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, f"u{u}")),
            )
        spec = e._sampling_spec()
        assert spec.delta_mode == "frozen"
        assert isinstance(spec.reference, ReferenceSpec)
        # u11 is the current rollout policy — excluded, same as dynamic.
        assert [_bp_dirname(b) for b in spec.reference.policies] == [
            f"u{u}" for u in range(1, 11)
        ]
        assert spec.reference.weights == pytest.approx((0.1,) * 10)

    def test_frozen_self_only_history_plain_spec(self, tmp_path):
        """Frozen mode with a single history member (= the current
        policy) finds no strictly-past version — the mechanism stays
        off (plain spec) exactly like dynamic warmup."""
        e = _exp(tmp_path, delta_factor="5.0",
                 reference_horizon="10", delta_mode="frozen")
        e.post_update(
            SimpleNamespace(), 1,
            artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, "u1")),
        )
        spec = e._sampling_spec()
        assert spec.reference is None
        assert spec.delta_factor == 0.0

    def test_frozen_requires_delta_factor(self, tmp_path):
        with pytest.raises(ValueError, match="delta_factor"):
            _exp(tmp_path, delta_mode="frozen")

    def test_delta_mode_invalid_value(self, tmp_path):
        with pytest.raises(ValueError, match="delta_mode"):
            _exp(tmp_path, delta_factor="1.0", reference_horizon="10",
                 delta_mode="banana")

    def test_delta_off_plain_spec_despite_history(self, tmp_path):
        e = _exp(tmp_path, reference_horizon="10")
        e.post_update(
            SimpleNamespace(), 1,
            artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, "u1")),
        )
        spec = e._sampling_spec()
        assert spec.reference is None and spec.delta_factor == 0.0


class TestRefHistoryPersistence:
    def test_state_roundtrip(self, tmp_path):
        e = _exp(tmp_path, reference_horizon="10")
        for u in range(1, 4):
            e.post_update(
                SimpleNamespace(), u,
                artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, f"u{u}")),
            )
        e2 = _exp(tmp_path, reference_horizon="10")
        e2.load_state(e.state())
        assert len(e2._ref_history) == 3
        assert [b.cls for b in e2._ref_history] == [
            b.cls for b in e._ref_history
        ]
        # Restored chain still assembles a valid spec.
        spec = e2._sampling_spec()
        assert spec.reference is not None or e2.delta_factor == 0.0

    def test_state_merges_with_subclass(self, tmp_path):
        """Subclass state() dicts merging super() keep ref_history."""
        from baseline.experiments_ppo.base import CombatExperimentPPOBase

        class _E(CombatExperimentPPOBase):
            name = "t"
            def reward_channels(self):
                return ()
            def build_trajectories(self, episodes):
                return []
            def on_eval(self, episodes, update):
                return {}
            def state(self):
                return {**super().state(), "custom": 7}

        e = _E()
        e.post_update(
            SimpleNamespace(), 1,
            artifacts=UpdateArtifacts(policy_bp=_bp(tmp_path, "u1")),
        )
        s = e.state()
        assert s["custom"] == 7
        assert len(s["ref_history"]) == 1


class TestInterface:
    def test_build_jobs_requires_update_kwarg(self, tmp_path):
        e = _exp(tmp_path)
        with pytest.raises(TypeError):
            # Missing required kw-only ``update``.
            e.build_jobs(None, 0, 4)  # type: ignore[call-arg]

    def test_run_name_default(self, tmp_path):
        e = _exp(tmp_path)
        name = e.run_name()
        assert re.fullmatch(
            r"train_t_ppo_\d{8}_\d{6}", name
        ), name
