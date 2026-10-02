"""E5 W0/W4 — 迁移审计与 manifest 失效追踪（无 GPU 依赖）。"""
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent


def _bp(name="basic_balance_v2_phi_dual_env.yaml", **knobs):
    from envs.framework.parameterized_blueprint import (
        ParameterizedEnvBlueprint)
    pb = ParameterizedEnvBlueprint.load(
        project_root / "baseline/humanoid21/blueprints" / name)
    return pb.materialize(**knobs)


def test_audit_no_unknown_keys():
    """standup + basic_balance 审计无 unknown 处置。"""
    from envs.batchframework.migration_audit import audit_blueprint
    for name in ("standup_4stage_dense_v2_env.yaml",
                 "basic_balance_v2_phi_dual_env.yaml"):
        rep = audit_blueprint(_bp(name))
        assert rep.unknown_keys() == [], rep.unknown_keys()
        assert rep.source_bp_hash


def test_audit_reports_cpu_surface():
    """审计附 CPU 单元的 accessor/hook 静态面（供迁移排期）。"""
    from envs.batchframework.migration_audit import audit_blueprint
    rep = audit_blueprint(_bp())
    imb = rep.unit(
        "baseline.humanoid21.plugins.imbalance_termination"
        ":DualImbalanceTerminationPlugin")
    assert imb is not None and imb.capability == "native"
    assert "get_derived_state" in imb.cpu_reads
    assert "on_post_action_step" in imb.cpu_hooks


def test_unregistered_unit_rejected():
    """未注册单元 → unsupported + 无设备路径猜测映射。"""
    from envs.batchframework.capability_registry import lookup
    entry = lookup("nonexistent.module:Nope")
    assert entry.capability.value == "unsupported"


def test_manifest_freshness_and_stale():
    """cls+config 漂移 → 该单元 stale；其余不受影响。"""
    import dataclasses
    from envs.batchframework.migration_audit import audit_blueprint
    from envs.batchframework.migration_manifest import MigrationManifest

    env_bp = _bp()
    rep = audit_blueprint(env_bp)
    m = MigrationManifest.from_audit(rep, experiment_name="basic_balance",
                                     blueprint_name="bb")
    m.unit("DualImbalanceTerminationPlugin").add_evidence(
        "unit_replay", True, input_hash="abc", detail="13 tests")
    assert m.validate_freshness(env_bp) == []

    # 改一个单元的 config → 该单元 stale（EnvBlueprint frozen，整体 replace）
    env_bp = dataclasses.replace(env_bp, plugins=tuple(
        dataclasses.replace(spec, config={
            **spec.config,
            "tolerance": spec.config.get("tolerance", 1) + 1})
        if "imbalance_termination" in spec.cls else spec
        for spec in env_bp.plugins))
    stale = m.validate_freshness(env_bp)
    assert stale == ["DualImbalanceTerminationPlugin"]
    assert m.unit("DualImbalanceTerminationPlugin").stale
    assert not m.unit("cross_support_a").stale


def test_manifest_save_load_roundtrip(tmp_path):
    from envs.batchframework.migration_audit import audit_blueprint
    from envs.batchframework.migration_manifest import MigrationManifest
    m = MigrationManifest.from_audit(
        audit_blueprint(_bp()), experiment_name="bb",
        blueprint_name="bb_env")
    p = m.save(tmp_path / "m.json")
    m2 = MigrationManifest.load(p)
    assert m2.source_bp_hash == m.source_bp_hash
    assert len(m2.units) == len(m.units)
    assert m2.units[0].unit_hash == m.units[0].unit_hash


def test_evidence_levels_guarded():
    from envs.batchframework.migration_manifest import UnitManifest
    u = UnitManifest(name="x", cls="a:B", capability="native")
    with pytest.raises(AssertionError):
        u.add_evidence("bogus_level", True)
    u.add_evidence("e2e_collect", True, detail="4 eps contract ok")
    assert u.evidence[0].level == "e2e_collect"
