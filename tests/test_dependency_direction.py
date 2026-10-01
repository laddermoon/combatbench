"""E1-W5 依赖方向静态检查：核心契约层不得依赖具体后端/任务。

允许方向：
    契约层 (physics/device_state/backend/device_plugin/device_runtime/
            fake_backend)  ←  组装层 (rollouter/registry)  ←  任务层
    backends/warp_backend  ←  warp_simulator（facade）← 任务绑定
    batch_binding（任务语义）  ←  各后端 facade

禁止方向：
    - 核心契约层 import warp/mujoco_warp/jax/mjx/humanoid21/baseline
    - warp_simulator import mjx_simulator（W2 解耦断言）
    - warp_backend import humanoid21 / mjx / device_*（后端不得反向
      依赖任务绑定或 runtime 层）
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BF = ROOT / "envs" / "batchframework"

_CORE_FILES = [
    "physics.py",
    "device_state.py",
    "backend.py",
    "device_plugin.py",
    "device_runtime.py",
    "fake_backend.py",
    "host_compat.py",
]

_FORBIDDEN_CORE = re.compile(
    r"^\s*(?:from|import)\s+[\w.]*"
    r"(?:warp|mujoco_warp|mjx|humanoid21|baseline)", re.M)


@pytest.mark.parametrize("fname", _CORE_FILES)
def test_core_layer_has_no_backend_or_task_deps(fname):
    src = (BF / fname).read_text()
    hits = _FORBIDDEN_CORE.findall(src)
    assert not hits, f"{fname} forbidden imports: {hits}"


def test_warp_simulator_does_not_import_mjx_simulator():
    src = (BF / "warp_simulator.py").read_text()
    assert "mjx_simulator" not in src
    assert not re.search(
        r"from\s+.*mjx|import\s+.*\bmjx\b|\bimport\s+jax\b", src), \
        "warp_simulator still references jax/mjx"


def test_warp_backend_is_pure_physics():
    src = (BF / "backends" / "warp_backend.py").read_text()
    forbidden = re.compile(
        r"^\s*(?:from|import)\s+[\w.]*"
        r"(?:humanoid21|mjx_simulator|device_runtime|device_plugin|"
        r"device_rollouter|baseline)", re.M)
    hits = forbidden.findall(src)
    assert not hits, f"warp_backend forbidden imports: {hits}"


def test_no_hardcoded_default_device_in_generic_path():
    """通用路径不得写死 cuda:0——device 必须可注入（构造默认值除外）。"""
    pattern = re.compile(r'["\']cuda:0["\']')
    viol = []
    for f in BF.rglob("*.py"):
        if f.name.startswith("probe_"):  # 探针脚本非通用路径
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            m = pattern.search(line)
            if m and "default" not in line.lower() \
                    and "device:" not in line.lower() \
                    and "docstring" not in line.lower():
                viol.append(f"{f.name}:{i}: {line.strip()[:70]}")
    # 构造签名默认 "cuda:0" 是注入点而非假设；过滤签名行
    viol = [v for v in viol if "device: str" not in v
            and "device = " not in v and 'device="cuda:0"' not in v]
    assert not viol, "hardcoded cuda:0:\n" + "\n".join(viol)
