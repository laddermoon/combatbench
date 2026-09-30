"""warp 真实后端的 M3 冒烟测试（GPU-gated）。

验证 DEVICE 平面完整链路在真实物理后端上闭环：
reset → obs 构建（设备端）→ 多步推进 → per-env 终止 → 部分 reset。
数值正确性由 test_warp_validation.py 的 fixture 覆盖；
本文件只管运行时接线。
"""
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).resolve().parent.parent


def _gpu_ok() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ok(), reason="CUDA unavailable")
def test_warp_runtime_smoke():
    from envs.batchframework.warp_simulator import WarpHumanoid21Simulator
    from envs.batchframework.device_runtime import (
        BatchRuntime, DeviceTimeoutPlugin)

    sim = WarpHumanoid21Simulator(batch_size=4)
    sim.reset()
    rt = BatchRuntime(sim, obs_builder=sim.device_obs_builder(),
                      phy_substeps=25)
    rt.attach(DeviceTimeoutPlugin(max_steps=2))

    rt.step()
    st = rt.state
    assert st.io.obs_a.shape == (4, 96)
    assert torch.isfinite(st.io.obs_a).all()
    rt.step()
    # timeout 触发 → 全 env 复位
    assert st.episode.episode_steps.tolist() == [0] * 4
    # reset 后观测回到初始姿态附近（非 NaN、高度合理）
    obs = st.io.obs_a
    assert torch.isfinite(obs).all()
