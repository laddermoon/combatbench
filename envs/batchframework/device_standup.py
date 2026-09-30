"""standup 任务的设备端原生插件（M4）。

转换自：
- ``baseline.humanoid21.rewards.standing_balance_4stage.StandingBalance4StageRewarder``
  → ``DeviceStandup4StageRewarder``（BaseDeviceObserver，per-agent 参数化）

所有阈值/常量**单一来源**：直接 import CPU 模块的常量，不复制字面量。
语义对照点（与 CPU 逐字段对齐）：
- extra_contact_count 是 distinct body 去重计数（非接触点计数）；
- 接触计入条件：env 侧 geom==ground 且 robot 侧 aff==本 agent，
  force_mag ≥ 1.0N；
- w_foot 门限 F_LOAD_MIN=10N 以下置 0；
- d_score 是 XY 平面距离（站立兼容，见 CPU 侧 Stage3 注释）；
- stage 自上而下判定，无迟滞。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from baseline.humanoid21.rewards.standing_balance_4stage import (
    D_GATE,
    D_MAX,
    D_MIN,
    F_ENTER,
    F_LOAD_MIN,
    H_CROUCH,
    H_FOOT_MAX,
    H_HAND_MAX,
    H_STAND,
    OTHER_PENALTY_K,
)

from .device_obs import contact_forces_flat
from .device_plugin import BaseDeviceObserver, DeviceCtx

# 输出字段顺序（get_output dict 的稳定键序）
OUT_KEYS = ("stage", "potential", "f_score", "contact_score", "d_score",
            "d_hf", "w_foot", "h_score", "h_torso")


class DeviceStandup4StageRewarder(BaseDeviceObserver):
    """StandingBalance4StageRewarder 的设备端批量版。

    Args:
        agent_idx: 0=robot_a / 1=robot_b。
        body_tables: 由 ``from_sim`` 构造时填——torso/hand/foot 的 body id
            与 contact 查找表。
    """

    def __init__(self, agent_idx: int, body_tables: Dict[str, Any]):
        assert agent_idx in (0, 1)
        self.agent_idx = int(agent_idx)
        self._t = body_tables
        self._out: Optional[Dict[str, torch.Tensor]] = None

    @classmethod
    def from_sim(cls, sim, agent_idx: int, obs_builder=None) -> "DeviceStandup4StageRewarder":
        """从 warp sim 的 meta 缓存提取 body id / 查找表。"""
        rid = "robot_a" if agent_idx == 0 else "robot_b"
        cache = sim._robots[rid]
        kp = cache["keypoint_body_ids"]
        torso_id = kp.get("torso", cache["root_body_id"])
        if obs_builder is None:
            obs_builder = sim.device_obs_builder()
        geom_bodyid, geom_aff = obs_builder.contact_tables
        return cls(agent_idx, dict(
            torso=int(torso_id),
            hand_l=int(kp["hand_left"]), hand_r=int(kp["hand_right"]),
            foot_l=int(kp["foot_left"]), foot_r=int(kp["foot_right"]),
            ground_gid=int(sim._ground_geom_id),
            robot_aff=agent_idx + 1,
            geom_bodyid=geom_bodyid, geom_aff=geom_aff,
            nbody=int(sim._model.nbody),
        ))

    # ------------------------------------------------------------------
    def _contacts_for_agent(self, state):
        """→ (per-env) hand/foot/extra 分类聚合。"""
        cf = contact_forces_flat(state, self._t["geom_bodyid"],
                                 self._t["geom_aff"])
        B = state.batch_size
        dev = cf.worldid.device
        t = self._t
        M = cf.sel.numel()
        zeros = torch.zeros(B, dtype=torch.float32, device=dev)
        extra_bool = torch.zeros(B, t["nbody"], dtype=torch.bool, device=dev)
        cat_force = torch.zeros(B, 4, dtype=torch.float32, device=dev)
        cat_touch = torch.zeros(B, 4, dtype=torch.bool, device=dev)
        # 类别码：0=foot_l 1=foot_r 2=hand_l 3=hand_r；其它 body 单独去重
        if M == 0:
            return dict(cat_force=cat_force, cat_touch=cat_touch,
                        extra_count=zeros)

        strong = cf.force_mag >= 1.0
        # CPU 语义：env 侧 aff==0 且 geom==ground，robot 侧 aff==本 agent
        ga = (cf.aff1 == 0) & (cf.geom1 == t["ground_gid"]) \
            & (cf.aff2 == t["robot_aff"])
        gb = (cf.aff2 == 0) & (cf.geom2 == t["ground_gid"]) \
            & (cf.aff1 == t["robot_aff"])
        rb = torch.where(ga, cf.body2,
                         torch.where(gb, cf.body1,
                                     torch.full_like(cf.body1, -1)))
        valid = strong & (ga | gb)
        m = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if m.numel() == 0:
            return dict(cat_force=cat_force, cat_touch=cat_touch,
                        extra_count=zeros)
        w, body = cf.worldid[m], rb[m]
        f = cf.force_mag[m]

        cat = torch.full((m.numel(),), -1, dtype=torch.long, device=dev)
        cat = torch.where(body == t["foot_l"], torch.zeros_like(cat), cat)
        cat = torch.where(body == t["foot_r"], torch.ones_like(cat), cat)
        cat = torch.where(body == t["hand_l"],
                          torch.full_like(cat, 2), cat)
        cat = torch.where(body == t["hand_r"],
                          torch.full_like(cat, 3), cat)
        hf = cat >= 0
        idx = w[hf] * 4 + cat[hf]
        cat_force.view(-1).index_add_(0, idx, f[hf])
        cat_touch.view(-1)[idx] = True
        # extra = robot 侧 body 不属于手/脚的接触，按 (env, body) 去重
        ex = ~hf
        extra_bool[w[ex], body[ex].clamp(min=0)] = True
        return dict(cat_force=cat_force, cat_touch=cat_touch,
                    extra_count=extra_bool.sum(dim=-1).float())

    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        self._out = None

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        # CPU on_pre_episode 全清零语义；out 缓存在下一步重建，
        # 本步已终止 env 的输出由采集端按 terminated 掩码丢弃。
        self._out = None

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        s = ctx.sim
        t = self._t

        # --- Signal 1: f_down（朝向） ---
        q = s.xquat[:, t["torso"]]                      # (B,4) wxyz
        x_world_z = 2.0 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        f_score = ((-x_world_z + 1.0) / 2.0).clamp(0.0, 1.0)

        # --- Signal 2: 支撑 proximity × 额外接触惩罚 ---
        h_hand = (s.xpos[:, t["hand_l"], 2] + s.xpos[:, t["hand_r"], 2]) / 2.0
        h_foot = (s.xpos[:, t["foot_l"], 2] + s.xpos[:, t["foot_r"], 2]) / 2.0
        hand_prox = (1.0 - h_hand / H_HAND_MAX).clamp(0.0, 1.0)
        foot_prox = (1.0 - h_foot / H_FOOT_MAX).clamp(0.0, 1.0)
        support = (hand_prox + foot_prox) / 2.0

        con = self._contacts_for_agent(ctx.state)
        other_pen = 1.0 / (1.0 + OTHER_PENALTY_K * con["extra_count"])
        contact_score = support * other_pen

        # --- Signal 3: hand-mid↔foot-mid XY 距离 ---
        hand_mid = (s.xpos[:, t["hand_l"], :2] + s.xpos[:, t["hand_r"], :2]) / 2
        foot_mid = (s.xpos[:, t["foot_l"], :2] + s.xpos[:, t["foot_r"], :2]) / 2
        d_hf = torch.linalg.norm(hand_mid - foot_mid, dim=-1)
        d_score = ((D_MAX - d_hf) / (D_MAX - D_MIN)).clamp(0.0, 1.0)

        # --- Signal 4: 承重转移 + torso 高度 ---
        f_foot = con["cat_force"][:, 0] + con["cat_force"][:, 1]
        f_hand = con["cat_force"][:, 2] + con["cat_force"][:, 3]
        total = f_foot + f_hand
        w_foot = torch.where(total >= F_LOAD_MIN, f_foot / total.clamp(min=1e-9),
                             torch.zeros_like(total))
        h_torso = s.xpos[:, t["torso"], 2]
        h_score = ((h_torso - H_CROUCH) / (H_STAND - H_CROUCH)).clamp(0.0, 1.0)
        p4 = w_foot * h_score

        # --- stage 自上而下 ---
        hf_touch = con["cat_touch"].any(dim=-1)
        only_hf = (con["extra_count"] == 0) & hf_touch
        stage = torch.ones(ctx.batch_size, dtype=torch.long, device=h_torso.device)
        stage = torch.where(f_score >= F_ENTER, torch.full_like(stage, 2), stage)
        stage = torch.where(only_hf, torch.full_like(stage, 3), stage)
        stage = torch.where(only_hf & (d_score >= D_GATE),
                            torch.full_like(stage, 4), stage)

        potential = torch.where(
            stage == 1, 0.10 * f_score,
            torch.where(
                stage == 2, 0.10 + 0.10 * contact_score,
                torch.where(
                    stage == 3, 0.20 + 0.10 * d_score,
                    0.30 + 0.70 * p4)))

        self._out = {
            "stage": stage.to(torch.float32),
            "potential": potential,
            "f_score": f_score,
            "contact_score": contact_score,
            "d_score": d_score,
            "d_hf": d_hf,
            "w_foot": w_foot,
            "h_score": h_score,
            "h_torso": h_torso,
        }

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        pass  # 输出缓存由下一步重建；episode 汇总由采集端负责

    def get_output(self) -> Dict[str, torch.Tensor]:
        """dict of (B,) 张量；首次 build 前为 None。"""
        return self._out
