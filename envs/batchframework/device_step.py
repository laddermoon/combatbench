"""step 实验的设备端单元（端到端步态迁移）。

- ``WarpGaitClockSimulator``    GaitClockSimulator 的 warp 对应——
  ``WarpHumanoid21Simulator`` 子类，观测构造器换成
  ``GaitClockObsBuilder``；
- ``GaitClockObsBuilder``       96 维基础观测 + 3 维步态时钟
  （cmd_L / cmd_R / wprog）——时钟源是 ``state.episode.episode_steps``
  （runtime 簿记：进入 step() 的无条件调用计数，obs 构造前已 +1，
  对齐 CPU 侧 "frame t 的观测读到 _action_step=t" 的约定）；
- ``DeviceFootStateObserver``   FootStateObserver 的设备原生实现：
  足心高（xipos.z）、sole 净空（4 端点最小世界 z − 半径）、
  足-地接触（flat 接触视图，无主力门限）。

语义单一来源在 CPU 侧：
``baseline/humanoid21/end2end/gait_clock_simulator.py``（常量与调度
公式）与 ``foot_state_observer.py``（STANDING_FOOT_Z / 端点几何 /
接触判定）。本文件只复刻为张量化实现，不重定义常量。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from baseline.humanoid21.end2end.foot_state_observer import (
    FOOT_ENDPOINTS_LOCAL,
    FOOT_GEOM_RADIUS,
    STANDING_FOOT_Z,
)
from baseline.humanoid21.end2end.gait_clock_simulator import (
    GaitClockSimulator,
)

from .device_obs import WarpObsBuilder, contact_forces_flat
from .device_plugin import BaseDeviceObserver, DeviceCtx
from .warp_simulator import WarpHumanoid21Simulator

_AGENT_RID = ("robot_a", "robot_b")
#: 与 CPU FootStateObserver.get_output 逐键对齐的输出叶顺序。
_FOOT_KEYS = ("h_left_foot", "h_right_foot", "sole_clear_left",
              "sole_clear_right", "left_foot_contact", "right_foot_contact")


# ---------------------------------------------------------------------------
# 步态时钟观测构造器
# ---------------------------------------------------------------------------
def _gait_clock_cols(steps: torch.Tensor, period: int) -> torch.Tensor:
    """``episode_steps`` (B,) i64 → (B,3) f32 [cmd_L, cmd_R, wprog]。

    公式与 ``GaitClockSimulator.gait_clock`` 逐项相同（纯张量，无 host
    同步——episode_steps 原位自增，CUDA graph 重放读到新值）。
    """
    half = period // 2
    pos = torch.remainder(steps, period)
    left = pos < half
    wprog = torch.where(left, pos, pos - half).to(torch.float32) \
        / float(half)
    return torch.stack([left.to(torch.float32),
                        (~left).to(torch.float32), wprog], dim=-1)


class GaitClockObsBuilder(WarpObsBuilder):
    """WarpObsBuilder + 尾部 3 维步态时钟（GaitClockSimulator 的设备形态）。

    双 agent 共享同一环境时钟——与 CPU ``get_observation`` 对两个
    robot 附加相同 ext 一致。
    """

    def __init__(self, tables, batch_size: int, gait_period: int = 40):
        super().__init__(tables, batch_size)
        self.gait_period = int(gait_period)

    def obs_dim(self) -> int:
        return super().obs_dim() + GaitClockSimulator.GAIT_DIM

    def _build_inner(self, state, *, dense_contacts: bool
                     ) -> Dict[str, torch.Tensor]:
        clock = _gait_clock_cols(state.episode.episode_steps,
                                 self.gait_period)
        obs = {
            "robot_a": torch.cat([
                self._robot_obs(state, "robot_a", "robot_b",
                                dense_contacts=dense_contacts),
                clock], dim=-1),
            "robot_b": torch.cat([
                self._robot_obs(state, "robot_b", "robot_a",
                                dense_contacts=dense_contacts),
                clock], dim=-1),
        }
        if state.io.obs_a is not None:
            state.io.obs_a.copy_(obs["robot_a"])
            state.io.obs_b.copy_(obs["robot_b"])
        return obs


class WarpGaitClockSimulator(WarpHumanoid21Simulator):
    """GaitClockSimulator 的设备端对应（warp 后端，obs 96→99）。

    时钟簿记由 ``BatchRuntime`` 的 ``episode_steps`` 承担——reset/部分
    reset 的清零语义经 ``reset_episode_rows`` 对齐 CPU 的
    ``_action_step = 0``；本类只做观测增广与维度上报。
    ``gait_period`` 默认 40（镜像 CPU ``GaitClockSimulator`` 默认值）。
    """

    OBS_BASE_DIM = GaitClockSimulator.OBS_BASE_DIM
    GAIT_DIM = GaitClockSimulator.GAIT_DIM

    def __init__(self, batch_size: int, gait_period: int = 40,
                 **kwargs: Any) -> None:
        super().__init__(batch_size=batch_size, **kwargs)
        self.gait_period = int(gait_period)

    def device_obs_builder(self) -> GaitClockObsBuilder:
        return GaitClockObsBuilder(
            self.task_tables(), self._batch_size,
            gait_period=self.gait_period)

    def obs_dim(self) -> int:
        return self.OBS_BASE_DIM + self.GAIT_DIM

    def get_observation(self) -> Dict[str, np.ndarray]:
        """host 校验/回放路径：基础 96 维 + 当前帧时钟。

        时钟源 = attach 的 runtime ``episode_steps``；未 attach
        （刚 reset、尚无簿记）时等价 CPU ``_action_step=0``——全零
        帧索引时钟。
        """
        base = super().get_observation()
        st = self._dev_state
        steps = (np.zeros(self._batch_size, dtype=np.int64)
                 if st is None else
                 st.episode.episode_steps.detach().cpu().numpy())
        pos = steps % self.gait_period
        half = self.gait_period // 2
        left = pos < half
        wprog = np.where(left, pos, pos - half).astype(np.float32) / half
        ext = np.stack([left.astype(np.float32),
                        (~left).astype(np.float32), wprog], axis=-1)
        return {
            rid: np.concatenate(
                [np.asarray(o, dtype=np.float32), ext], axis=-1)
            for rid, o in base.items()
        }


# ---------------------------------------------------------------------------
# FootStateObserver → DeviceFootStateObserver
# ---------------------------------------------------------------------------
class DeviceFootStateObserver(BaseDeviceObserver):
    """FootStateObserver 的设备端批量实现（per-agent 6 叶）。

    CPU 语义逐项对应（``foot_state_observer.py`` 为唯一语义源）：

    - ``h_*``     ``xipos[foot].z - STANDING_FOOT_Z``——足体惯性系原点
      世界高度相对站立基准；
    - ``sole_clear_*``   4 个足囊端点（body 系 fromto 局部坐标）经
      ``xquat`` 旋转后的最小世界 z − ``FOOT_GEOM_RADIUS``——只有整足
      离地才 >0（端点绕轴/侧缘摇摆抬高中点但抬不动净空）；
    - ``*_foot_contact``   任一活跃接触配对 (ground_geom ↔ 足 body)，
      无主力门限；aff 条件与 CPU ``_detect_contact`` 同式
      （env 侧 aff==0，robot 侧 aff==本 agent）。
    """

    def __init__(self, agent_idx: int, *, tables: Dict[str, Any],
                 standing_foot_z: float = STANDING_FOOT_Z):
        assert agent_idx in (0, 1)
        self.agent_idx = int(agent_idx)
        self.standing_foot_z = float(standing_foot_z)
        self._t = tables
        dev = tables["geom_bodyid"].device
        self._pts = torch.as_tensor(
            np.asarray(FOOT_ENDPOINTS_LOCAL), dtype=torch.float32,
            device=dev)                                # (4,3) body 系端点
        self._out: Optional[Dict[str, torch.Tensor]] = None

    @classmethod
    def from_sim(cls, sim, agent_idx: int,
                 standing_foot_z: float = STANDING_FOOT_Z, **_kw):
        tables = sim.task_tables()
        r = tables.robots[_AGENT_RID[agent_idx]]
        kp = r["keypoint_body_ids"]
        return cls(agent_idx, standing_foot_z=standing_foot_z, tables=dict(
            foot_l=int(kp["foot_left"]), foot_r=int(kp["foot_right"]),
            ground_gid=int(tables.ground_geom_id),
            agent_aff=agent_idx + 1,
            geom_bodyid=tables.geom_bodyid, geom_aff=tables.geom_aff))

    @property
    def output_schema(self):
        return {
            "h_left_foot": (torch.float32, ()),
            "h_right_foot": (torch.float32, ()),
            "sole_clear_left": (torch.float32, ()),
            "sole_clear_right": (torch.float32, ()),
            "left_foot_contact": (torch.bool, ()),
            "right_foot_contact": (torch.bool, ()),
        }

    def get_output(self) -> Any:
        return self._out

    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        self._out = None

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        self._out = None

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        pass

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        s, t = ctx.sim, self._t
        fl, fr = t["foot_l"], t["foot_r"]
        self._out = {
            "h_left_foot": s.xipos[:, fl, 2] - self.standing_foot_z,
            "h_right_foot": s.xipos[:, fr, 2] - self.standing_foot_z,
            "sole_clear_left": self._sole_clear(s, fl),
            "sole_clear_right": self._sole_clear(s, fr),
            "left_foot_contact": self._foot_ground(ctx.state, fl),
            "right_foot_contact": self._foot_ground(ctx.state, fr),
        }

    # ------------------------------------------------------------------
    def _sole_clear(self, s, body_id: int) -> torch.Tensor:
        """CPU ``sole_clearance`` 的批量版（公式逐项相同）。"""
        q = s.xquat[:, body_id]                        # (B,4) [w,x,y,z]
        w, x, y, z = q.unbind(-1)
        r31 = 2.0 * (x * z + w * y)
        r32 = 2.0 * (y * z - w * x)
        r33 = 1.0 - 2.0 * (x * x + y * y)
        p = self._pts                                  # (4,3)
        z4 = (s.xpos[:, body_id, 2:3]
              + p[:, 0] * r31[:, None]
              + p[:, 1] * r32[:, None]
              + p[:, 2] * r33[:, None])                # (B,4)
        return z4.min(dim=-1).values - FOOT_GEOM_RADIUS

    def _foot_ground(self, state, foot_body: int) -> torch.Tensor:
        """``FootStateObserver._detect_contact`` 单足版（flat 视图）。

        命中 = env 侧 (aff==0 ∧ geom==ground) 配对 robot 侧
        (aff==agent_aff ∧ body==foot)；无主力门限——与 CPU
        逐条扫描语义一致（dist<=0 活跃过滤由 contact_forces_flat
        承担，对应 CPU ncon 只含已生成接触）。
        """
        t = self._t
        cf = contact_forces_flat(state, t["geom_bodyid"], t["geom_aff"])
        env1 = (cf.aff1 == 0) & (cf.geom1 == t["ground_gid"])
        env2 = (cf.aff2 == 0) & (cf.geom2 == t["ground_gid"])
        hit = ((env1 & (cf.aff2 == t["agent_aff"])
                & (cf.body2 == foot_body))
               | (env2 & (cf.aff1 == t["agent_aff"])
                  & (cf.body1 == foot_body)))
        out = torch.zeros(state.batch_size, dtype=torch.bool,
                          device=cf.worldid.device)
        out.index_fill_(0, cf.worldid[hit], True)
        return out
