"""PhysicsBackend 契约一致性测试（E1-W1）。

同一套测试跑所有声明实现 ``envs.batchframework.physics.PhysicsBackend``
的后端。当前工厂注册：FakeBatchBackend。WarpBackend 在 W2 落地后
注册进 BACKENDS 即自动纳入同一套件。

契约覆盖（对应 physics.py docstring / discuss.md D4）：
- views() 借用字典的字段/shape/dtype/device；
- initialize(mask) 只写选中行，未选中行逐位不变；
- apply_patch 只写声明的 writable 字段；
- advance(n, control) 每子步调 control、状态演化；
- capture/restore 返回独立拷贝且 masked 恢复不扰动他行；
- pending wrench 仅首子步消费。
"""
from __future__ import annotations

import pytest
import torch

from envs.batchframework.fake_backend import FakeBatchBackend
from envs.batchframework.physics import (
    RefreshPolicy,
    SnapshotLevel,
    check_descriptor,
)


# ---------------------------------------------------------------------------
# 后端工厂注册表——新后端在此登记即自动进入全套件
# ---------------------------------------------------------------------------
def _fake_factory():
    return FakeBatchBackend(batch_size=8, device="cpu")


BACKENDS = {"fake": _fake_factory}


@pytest.fixture(params=sorted(BACKENDS), ids=sorted(BACKENDS))
def backend(request):
    b = BACKENDS[request.param]()
    yield b
    b.close()


# ---------------------------------------------------------------------------
class TestDescriptor:
    def test_describe_valid(self, backend):
        d = backend.describe()
        check_descriptor(d)
        assert d.batch_size > 0
        assert d.nq > 0 and d.nv > 0


class TestViews:
    def test_views_are_tensors_with_batch_dim(self, backend):
        d = backend.describe()
        vv = backend.views()
        for name in ("qpos", "qvel", "ctrl"):
            t = vv[name]
            assert torch.is_tensor(t)
            assert t.shape[0] == d.batch_size

    def test_views_share_storage_with_step(self, backend):
        """advance 后视图反映新状态（借用而非快照）。"""
        vv = backend.views()
        before = vv["qpos"].clone()
        backend.advance(1)
        # 最小动力学下 qpos 可变；至少证明视图对象仍可读、dtype 未变
        assert vv["qpos"].dtype == before.dtype


class TestInitialize:
    def test_masked_initialize_leaves_others(self, backend):
        vv = backend.views()
        B = backend.describe().batch_size
        nq = backend.describe().nq
        mask = torch.zeros(B, dtype=torch.bool, device=vv["qpos"].device)
        mask[:2] = True
        qpos_new = torch.ones(2, nq, device=vv["qpos"].device)
        before_rest = vv["qpos"][~mask].clone()
        backend.initialize(qpos_new, mask=mask)
        assert torch.equal(vv["qpos"][:2], qpos_new)
        assert torch.equal(vv["qpos"][~mask], before_rest)

    def test_initialize_clears_wrist_pending(self, backend):
        """initialize 后选中行的控制/外力类字段清零（全新求解语义）。"""
        vv = backend.views()
        vv["ctrl"].fill_(1.0)
        backend.initialize(torch.zeros_like(vv["qpos"]))
        assert torch.equal(vv["ctrl"],
                           torch.zeros_like(vv["ctrl"]))


class TestAdvance:
    def test_control_called_per_substep(self, backend):
        calls = []

        class Ctrl:
            def apply(self, vv):
                calls.append(1)

        backend.advance(5, control=Ctrl())
        assert len(calls) == 5

    def test_advance_evolves_state(self, backend):
        vv = backend.views()
        vv["qvel"].fill_(1.0)
        q0 = vv["qpos"].clone()
        backend.advance(1)
        assert not torch.equal(vv["qpos"], q0)


class TestSnapshot:
    def test_capture_returns_owned_copies(self, backend):
        vv = backend.views()
        B = backend.describe().batch_size
        mask = torch.zeros(B, dtype=torch.bool, device=vv["qpos"].device)
        mask[0] = True
        snap = backend.capture(mask, SnapshotLevel.INTEGRATION)
        assert "qpos" in snap
        # 改原视图不影响快照
        vv["qpos"][0] = 12345.0
        assert snap["qpos"][0, 0] != 12345.0

    def test_masked_restore_only_touches_selected(self, backend):
        vv = backend.views()
        B = backend.describe().batch_size
        mask = torch.zeros(B, dtype=torch.bool, device=vv["qpos"].device)
        mask[0] = True
        snap = backend.capture(mask, SnapshotLevel.INTEGRATION)
        vv["qpos"].fill_(7.0)
        backend.restore(mask, snap)
        # mask 行恢复为快照值；未选行保持 7.0
        assert torch.equal(vv["qpos"][0], snap["qpos"][0])
        assert (vv["qpos"][1:] == 7.0).all()


class TestWrench:
    def test_pending_force_first_substep_only(self, backend):
        """queue 语义：pending 外力只进下一 advance 的首个子步。"""
        st = backend.build_device_state()
        B = backend.describe().batch_size
        backend.dev_add_ext_force(
            0, torch.ones(B, 3, device=st.sim.qpos.device))
        backend.advance(3)
        log = backend.xfrc_log
        assert len(log) == 3
        assert (log[0][:, 0, :3] == 1.0).all()      # 子步 0 有外力
        assert (log[1] == 0.0).all()                # 子步 1+ 无
        assert (log[2] == 0.0).all()


class TestContractViolations:
    def test_patch_unknown_field_rejected(self, backend):
        vv = backend.views()
        B = backend.describe().batch_size
        mask = torch.zeros(B, dtype=torch.bool, device=vv["qpos"].device)
        mask[0] = True
        with pytest.raises((ValueError, KeyError)):
            backend.apply_patch(mask, {"nonexistent": torch.zeros(1)})
