"""ROADMAP §11 工作包 C：负例与维护试验——迁移工具链对坏输入的
拒绝/定位能力（对应 R3_RESULTS.md §6 未覆盖项）。

- `unit_hash` v2 = {cls, config, CPU 模块文件, device_cls 模块文件}——
  源码/模块常量漂移判 stale（v1 只哈希 cls 名+config，源码漂移静默放过）；
- blueprint 含未注册插件 → audit 判 unsupported + collect 启动即拒；
- manifest config 漂移 → collect 启动即 RuntimeError（stale 证据不沿用）；
- observer schema 失配 → 报错点名单元+叶；
- 畸形 manifest 文件 → warn 且不冒充新鲜（不再静默跳过）。
"""
from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).resolve().parent.parent


def _gpu_ok() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _gpu_ok(), reason="CUDA unavailable")


@pytest.fixture(scope="module")
def env_bp():
    from envs.framework.parameterized_blueprint import (
        ParameterizedEnvBlueprint)
    pb = ParameterizedEnvBlueprint.load(
        project_root / "baseline/humanoid21/blueprints"
        / "standup_4stage_dense_v2_env.yaml")
    return pb.materialize(max_steps=32)


@pytest.fixture(scope="module")
def policy_bp(tmp_path_factory):
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)
    pol = TruncatedNormalPolicy(obs_dim=96, action_dim=21, hidden_dim=256,
                                device="cpu")
    dest = tmp_path_factory.mktemp("pol_neg") / "export"
    return pol.to_blueprint(str(dest))


def _import_tmp(tmp_path, monkeypatch, name, body):
    """在 tmp_path 写一个模块并导入（源码漂移测试的注入手段）。"""
    (tmp_path / f"{name}.py").write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# 指纹：源码漂移（v1 只哈希 cls+config——本组测试是机制的 RED 先行）
# ---------------------------------------------------------------------------
def test_unit_hash_tracks_module_constant(tmp_path, monkeypatch):
    """模块级常量（类外）漂移 → 指纹变。

    覆盖 foot_state 的 STANDING_FOOT_Z 类场景：公式依赖的常量定义在
    模块级，`inspect.getsource(cls)` 抓不到，模块文件指纹能抓到。
    """
    _import_tmp(tmp_path, monkeypatch, "negmod",
                "THRESH = 1\n\nclass Foo:\n    pass\n")
    from envs.batchframework.migration_audit import _unit_hash
    h1 = _unit_hash("negmod:Foo", {})
    (tmp_path / "negmod.py").write_text(
        "THRESH = 2\n\nclass Foo:\n    pass\n")
    importlib.reload(sys.modules["negmod"])
    h2 = _unit_hash("negmod:Foo", {})
    assert h1 != h2


def test_unit_hash_tracks_device_impl_source(tmp_path, monkeypatch):
    """entry.device_cls 声明的设备实现模块漂移 → 指纹变。

    discuss D14：目标实现变化同样使既有验证失效——CPU 侧没动时
    设备实现改动也要 stale。
    """
    _import_tmp(tmp_path, monkeypatch, "negcpu", "class Foo:\n    pass\n")
    _import_tmp(tmp_path, monkeypatch, "negdev", "class Dev:\n    pass\n")
    from envs.batchframework.migration_audit import _unit_hash
    h1 = _unit_hash("negcpu:Foo", {}, device_cls="negdev:Dev")
    (tmp_path / "negdev.py").write_text("class Dev:\n    X = 1\n")
    importlib.reload(sys.modules["negdev"])
    h2 = _unit_hash("negcpu:Foo", {}, device_cls="negdev:Dev")
    assert h1 != h2


def test_unit_hash_stable_and_config_sensitive(tmp_path, monkeypatch):
    """同输入恒等 + config 漂移照样变（v1 行为不回退）。"""
    _import_tmp(tmp_path, monkeypatch, "negmod2",
                "class Bar:\n    pass\n")
    from envs.batchframework.migration_audit import _unit_hash
    assert _unit_hash("negmod2:Bar", {"a": 1}) == \
        _unit_hash("negmod2:Bar", {"a": 1})
    assert _unit_hash("negmod2:Bar", {"a": 1}) != \
        _unit_hash("negmod2:Bar", {"a": 2})


def test_freshness_marks_source_edited_unit_stale(tmp_path, monkeypatch,
                                                env_bp):
    """真实 blueprint + 临时插件：改该插件模块源 → 仅该单元 stale。"""
    _import_tmp(tmp_path, monkeypatch, "negplug",
                "class FooPlugin:\n"
                "    def __init__(self, x=0):\n        self.x = x\n")
    from envs.framework.blueprint import ClassSpec
    bp = dataclasses.replace(
        env_bp, plugins=env_bp.plugins + (
            ClassSpec(cls="negplug:FooPlugin", config={}),))
    from envs.batchframework.migration_audit import audit_blueprint
    from envs.batchframework.migration_manifest import MigrationManifest
    m = MigrationManifest.from_audit(audit_blueprint(bp))
    assert m.validate_freshness(bp) == []
    (tmp_path / "negplug.py").write_text(
        "class FooPlugin:\n"
        "    def __init__(self, x=0):\n        self.x = x + 1\n")
    importlib.reload(sys.modules["negplug"])
    stale = m.validate_freshness(bp)
    assert stale == ["FooPlugin"]
    assert m.unit("FooPlugin").stale
    assert not m.unit("RandomFallenStatePlugin").stale


def test_native_units_source_resolvable():
    """所有 NATIVE 注册单元的源码指纹非空——防指纹静默空转。

    指纹机制若对某单元解析不出源码文件，等于该单元永久无源码
    保护——本测试把"保护覆盖了谁"变成显式断言。
    """
    from envs.batchframework.migration_audit import _src_fingerprint
    from envs.batchframework.capability_registry import (
        Capability, REGISTRY)
    missing = []
    for cls, entry in REGISTRY.items():
        if entry.capability is not Capability.NATIVE:
            continue
        if _src_fingerprint(cls) is None:
            missing.append(f"cpu:{cls}")
        dev = getattr(entry, "device_cls", None)
        if dev and _src_fingerprint(dev) is None:
            missing.append(f"dev:{dev}")
    assert not missing


# ---------------------------------------------------------------------------
# 拒迁：未注册插件 → audit 判 unsupported + collect 启动即拒
# ---------------------------------------------------------------------------
def test_audit_flags_unregistered_plugin_unsupported(env_bp):
    """未注册 cls：audit 不崩、判 unsupported、config 键标 unknown。"""
    from dataclasses import replace
    from envs.framework.blueprint import ClassSpec
    from envs.batchframework.migration_audit import audit_blueprint
    bogus = ClassSpec(cls="nonexistent.mod:BogusPlugin", config={"k": 1})
    rep = audit_blueprint(dataclasses.replace(
        env_bp, plugins=env_bp.plugins + (bogus,)))
    u = [u for u in rep.units if u.cls == "nonexistent.mod:BogusPlugin"]
    assert len(u) == 1
    assert u[0].capability == "unsupported"
    assert any(d.key == "k" and d.disposition == "unknown"
               for d in u[0].config_disposition)


@requires_cuda
def test_collect_rejects_unregistered_plugin(env_bp, policy_bp):
    """运行时同样启动即拒——不静默忽略插件（不猜测映射）。"""
    from envs.framework.blueprint import ClassSpec
    from envs.batchframework.device_rollouter import DeviceRollouter
    from baseline.framework.rollout.job import Job, SamplingSpec
    bogus = ClassSpec(cls="nonexistent.mod:BogusPlugin", config={})
    bp = dataclasses.replace(env_bp,
                             plugins=env_bp.plugins + (bogus,))
    job = Job(policy_a_bp=policy_bp, policy_b_bp=policy_bp, env_bp=bp,
              seed=0, episode_options={}, sampling_a=SamplingSpec(0.0),
              sampling_b=SamplingSpec(0.0), stochastic=True)
    with DeviceRollouter(batch_size=4, device="cuda") as dr:
        with pytest.raises(ValueError, match="unsupported"):
            dr.collect([job])


# ---------------------------------------------------------------------------
# stale：config 漂移 → collect 以过期证据拒绝执行
# ---------------------------------------------------------------------------
@requires_cuda
def test_collect_rejects_config_drifted_blueprint(env_bp, policy_bp):
    """simulator.config 改动 → manifest stale → collect RuntimeError。

    manifest 在 _set_env 的建 bundle 之前校验——拒绝发生在任何
    物理执行之前。
    """
    from envs.batchframework.device_rollouter import DeviceRollouter
    from baseline.framework.rollout.job import Job, SamplingSpec
    drifted_sim = dataclasses.replace(
        env_bp.simulator,
        config={**env_bp.simulator.config, "initial_distance": 9.99})
    bp = dataclasses.replace(env_bp, simulator=drifted_sim)
    job = Job(policy_a_bp=policy_bp, policy_b_bp=policy_bp, env_bp=bp,
              seed=0, episode_options={}, sampling_a=SamplingSpec(0.0),
              sampling_b=SamplingSpec(0.0), stochastic=True)
    with DeviceRollouter(batch_size=4, device="cuda") as dr:
        with pytest.raises(RuntimeError, match="stale"):
            dr.collect([job])


# ---------------------------------------------------------------------------
# 定位：observer schema 失配 → 报错点名单元+叶
# ---------------------------------------------------------------------------
def test_observer_missing_leaf_names_unit_and_leaf():
    """schema 声明的叶缺失 → ValueError 点名 observer 与 leaf。"""
    from envs.batchframework.binding_registry import IoSchema
    from envs.batchframework.record_store import ObserverLeaf, RecordStore
    rs = RecordStore(
        IoSchema(agent_ids=("robot_a", "robot_b"), obs_dim=4,
                 action_dim=2),
        T=2, B=2, device=torch.device("cpu"),
        observer_schemas={
            "foot_state_a": {"left_foot_contact":
                             ObserverLeaf(torch.bool, ())}})
    with pytest.raises(ValueError, match="foot_state_a.*left_foot_contact"):
        rs.write_observer_step(0, {"foot_state_a": {}})


def test_observer_wrong_shape_names_leaf():
    from envs.batchframework.binding_registry import IoSchema
    from envs.batchframework.record_store import ObserverLeaf, RecordStore
    rs = RecordStore(
        IoSchema(agent_ids=("robot_a", "robot_b"), obs_dim=4,
                 action_dim=2),
        T=2, B=2, device=torch.device("cpu"),
        observer_schemas={
            "obs_x": {"h": ObserverLeaf(torch.float32, (3,))}})
    with pytest.raises(ValueError, match="obs_x.*h"):
        rs.write_observer_step(0, {"obs_x": {"h": torch.zeros(2, 5)}})


# ---------------------------------------------------------------------------
# 畸形 manifest：warn + 按无关跳过，不冒充新鲜也不静默
# ---------------------------------------------------------------------------
def test_malformed_manifest_warns_and_skips(tmp_path, monkeypatch, env_bp):
    """MANIFEST_DIR 里的坏 json → UserWarning 且 find 返回 None。

    此前的 `except Exception: continue` 是静默跳过——坏文件会让
    stale 保护静默失效（E8 实际踩过缺字段坑）。负例要求"定位"。
    """
    import envs.batchframework.migration_manifest as mm
    (tmp_path / "broken.json").write_text("{ not valid json")
    monkeypatch.setattr(mm, "MANIFEST_DIR", tmp_path)
    with pytest.warns(UserWarning, match="manifest"):
        assert mm.find_manifest_for(env_bp) is None
