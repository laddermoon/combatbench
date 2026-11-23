"""CMD_PROBE：obs96 + cmd22 → 实际 sim rollout → 完成度指标（Lib）。

用法：
    from baseline.humanoid21.v1.cmd_probe import CmdProbe
    probe = CmdProbe("steparm")               # 别名或 policy_exports 目录
    res = probe.evaluate(obs96, cmd22)        # obs(96,) 原始 / cmd(22,) 指令段

语义：
  obs96 经 RobotState.from_obs 解码为 ego 完整动力学状态（P5 系），
  写回仿真器；cmd22 = obs[57:79] 指令段字面值——
      [0]   flag_r     [1:4] tgt_r     [4] flag_l   [5:8] tgt_l
      [8:10] waist（预留，忽略）
      [10:16] step_cur 残差 [foot,Δx,Δy,Δψ,0,0]
      [16:22] step_nxt 残差（对 P̂ 反解，到期晋升）
  指令在注入姿态上锚定为世界目标；rollout 中残差按真值演化——
  测的是"此状态下 policy 对该指令的真实执行"，而非 obs 复读。

返回 dict：cmd_echo / warnings / step{res0,res_end,t_arrive,d_mv,
  d_st,d_yaw,res_traj,arr} / arm{err_r/l 轨迹, t_band, end} /
  balance{min_phi,end_phi,fall,feet_disp}。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_V1 = Path(__file__).resolve().parent
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))

from scipy.spatial.transform import Rotation as Rot

from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)
from point5.humanoid21.robot_state import (
    RobotState, joint_ranges,
)
from step_accept import _load_policy, _resolve_policy, _wrap_pi

_ENV_YAML = _V1 / "v2_probe_env.yaml"
_ROBOTS = ("robot_a", "robot_b")
_BAND = 0.15          # 臂令"进带"阈值（|err|<band 首达帧）


def decode_cmd22(v: np.ndarray) -> Dict[str, Any]:
    """22 维指令段 → 结构化（字面值，未反解）。"""
    v = np.asarray(v, np.float64)
    if v.shape != (22,):
        raise ValueError(f"cmd22: expected (22,), got {v.shape}")
    return {
        "flag_r": float(v[0]), "tgt_r": v[1:4].tolist(),
        "flag_l": float(v[4]), "tgt_l": v[5:8].tolist(),
        "waist": v[8:10].tolist(),
        "step_vec": v[10:16], "step_nxt_vec": v[16:22],
    }


class CmdProbe:
    """obs+cmd 完成度探针（确定性 act，单次 rollout）。"""

    def __init__(self, policy: str,
                 env_yaml: Optional[Path] = None) -> None:
        pdir = _resolve_policy(policy)
        self.policy_dir = pdir
        self.pol = _load_policy(pdir)
        self.env = ParameterizedEnvBlueprint.load(
            env_yaml or _ENV_YAML).materialize().build()
        self.sim = self.env.simulator
        self._rng = joint_ranges()          # (21,2) 关节弧度范围
        self._ref = (self._rng[:, 0] + self._rng[:, 1]) / 2
        self._sc = (self._rng[:, 1] - self._rng[:, 0]) / 2

    # ------------------------------------------------------------ 主 API --
    def evaluate(self, obs: np.ndarray, cmd: np.ndarray,
                 robot: str = "robot_a",
                 dur: int = 20, nxt_dur: int = 15,
                 horizon: Optional[int] = None,
                 tail_f: int = 10,
                 arm_seq: Optional[list] = None,
                 seed: int = 0) -> Dict[str, Any]:
        """obs96 + cmd22 → rollout → 完成度指标。

        dur    注入当前令的时长（到期→晋升 nxt→hold）
        horizon  rollout 帧数；None = dur + tail_f
        arm_seq 可选臂令分段 [(flags(2), tgt(6), dur_f), ...]——
                覆盖 cmd22 的臂段；首段立即生效、逐段轮转。
                建议调用方把 cmd22 臂段写成首段值以保持一致。
        """
        obs = np.asarray(obs, np.float64)
        if obs.shape != (96,):
            raise ValueError(f"obs: expected (96,), got {obs.shape}")
        spec = decode_cmd22(np.asarray(cmd))
        warns = self._warnings(spec)
        rs = RobotState.from_obs(obs)

        self.env.reset(seed=seed)
        sim = self.sim
        sim.inject_state(robot, rs)
        arm = (np.array([spec["flag_r"] > 0.5, spec["flag_l"] > 0.5]),
               np.concatenate([spec["tgt_r"], spec["tgt_l"]]))
        h = horizon if horizon is not None else dur + tail_f
        sim.probe_apply(robot, {
            "step_vec": spec["step_vec"],
            "dur": dur,
            "step_nxt_vec": spec["step_nxt_vec"],
            "nxt_dur": nxt_dur,
            "arm": arm,
            "arm_seq": arm_seq,
        })
        # 对侧机器人：自由臂 + 保持令（站着别动）
        other = "robot_b" if robot == "robot_a" else "robot_a"
        sim.probe_apply(other, {
            "step_vec": None, "dur": 15, "step_nxt_vec": None,
            "arm": (np.zeros(2, bool), np.zeros(6)),
        })
        self.pol.reset(seed + 1)

        c = sim._robot(robot)
        torso_id = int(c["root_body_id"])
        kp = c["keypoint_body_ids"]
        hid = {"r": int(kp["hand_right"]), "l": int(kp["hand_left"])}

        tr = {"res": [], "arm_r": [], "arm_l": [], "phi": [],
              "mid": [], "armq_r": [], "armq_l": [],
              "hpos_r": [], "hpos_l": [], "hspd_r": [], "hspd_l": []}
        evs = []
        oa, ob = self.env.get_observation()
        for t in range(h):
            aa, _ = self.pol.act(oa)
            ab, _ = self.pol.act(ob)
            self.env.step(aa, ab)
            sfx = "a" if robot == "robot_a" else "b"
            rep = self.env.get_observer_output(
                f"stepace_report_{sfx}")
            ad = self.env.get_observer_output(f"arm_dual_{sfx}")
            phi = self.env.get_observer_output(
                f"standing_balance_{sfx}")["potential"]
            oa, ob = self.env.get_observation()
            o = oa if robot == "robot_a" else ob
            tr["res"].append(np.asarray(o[67:73], float).tolist())
            tr["arm_r"].append(float(ad.get("err_r", 0.0)))
            tr["arm_l"].append(float(ad.get("err_l", 0.0)))
            tr["armq_r"].append(
                self._arm_qerr(robot, spec["tgt_r"], "r"))
            tr["armq_l"].append(
                self._arm_qerr(robot, spec["tgt_l"], "l"))
            tr["phi"].append(float(phi))
            fl, fr, _ = sim._feet_yaw(robot)
            tr["mid"].append([float(0.5 * (fl[0] + fr[0])),
                              float(0.5 * (fl[1] + fr[1]))])
            # 手部关键点：ego 系位置（躯干原点+朝向）+ 世界系速度
            tp = sim.data.xpos[torso_id]
            tq = sim.data.xquat[torso_id]            # wxyz
            rinv = Rot.from_quat(
                [tq[1], tq[2], tq[3], tq[0]]).inv()
            for sd in ("r", "l"):
                h = int(hid[sd])
                tr[f"hpos_{sd}"].append(
                    rinv.apply(sim.data.xpos[h] - tp).tolist())
                tr[f"hspd_{sd}"].append(
                    float(np.linalg.norm(sim.data.cvel[h, 3:6])))
            if rep.get("event", 0.0) > 0.5:
                evs.append({k: float(v) for k, v in rep.items()})
            if self.env.is_episode_over():
                break
        return {
            "cmd_echo": {k: (v.tolist() if isinstance(v, np.ndarray)
                             else v) for k, v in spec.items()},
            "warnings": warns,
            "frames": len(tr["phi"]),
            "step": self._step_metrics(tr, evs, dur),
            "arm": self._arm_metrics(spec, tr),
            "hands": self._hand_metrics(tr),
            "balance": self._balance_metrics(tr),
        }

    def _hand_metrics(self, tr):
        """手部关键点指标：挥击通道完成度（ego 系位置/世界速度）。"""
        out = {}
        for sd in ("r", "l"):
            pos = np.asarray(tr[f"hpos_{sd}"], float)   # (T,3) ego
            spd = np.asarray(tr[f"hspd_{sd}"], float)
            if pos.size == 0:
                continue
            i_vp = int(np.argmax(spd))
            i_rx = int(np.argmax(pos[:, 0]))            # 最大前伸
            out[sd] = {
                "v_peak": float(spd[i_vp]),
                "t_vpeak": i_vp,
                "reach_at_vpeak": float(pos[i_vp, 0]),
                "v_at_reach": float(spd[i_rx]),
                "t_reach": i_rx,
                "reach_fwd": float(pos[i_rx, 0]),
                "reach_pos": [round(float(v), 3)
                              for v in pos[i_rx]],
                "reach_h": float(
                    np.linalg.norm(pos[:, :2], axis=1).max()),
                "pos_end": [round(float(v), 3) for v in pos[-1]],
                "v_end": float(spd[-1]),
                "pos_traj": [[round(float(v), 3) for v in p]
                             for p in pos],
            }
        return out

    # ------------------------------------------------------------ 指标 ----
    def _arm_qerr(self, rid: str, tgt_nrm, side: str) -> float:
        """臂关节实际角度 vs 指令（norm→rad）的欧氏误差。"""
        c = self.sim._robot(rid)
        q = np.asarray(
            self.sim.data.qpos[np.asarray(c["qpos_indices"])], float)
        i = 15 if side == "r" else 18
        rad = np.asarray(tgt_nrm, float) * self._sc[i:i + 3] \
            + self._ref[i:i + 3]
        return float(np.linalg.norm(q[i:i + 3] - rad))

    def _warnings(self, spec: Dict[str, Any]) -> list:
        w = []
        sv = spec["step_vec"]
        if int(round(sv[0])) == 0 and abs(sv[3]) > 1e-3:
            w.append("step_vec: foot=0 且 Δψ≠0——保持令不可表达"
                     "转身（cmd_target 吞 dyaw），该维被忽略")
        if any(abs(x) > 1e-6 for x in spec["waist"]):
            w.append("waist 维非零——预留维度，忽略")
        return w

    def _step_metrics(self, tr, evs, dur):
        res = np.asarray(tr["res"], float)
        if res.size == 0:
            return {}
        # 晋升发生在 age=dur 那一步——res[dur-1] 已是新令残差，
        # 有效令内残差只到 dur-2
        n = min(dur - 1, len(res))
        res_xy = np.linalg.norm(res[:n, 1:3], axis=1)
        mv_evs = [e for e in evs if e.get("hold", 0.0) < 0.5]
        m = {
            "res0": float(res_xy[0]) if len(res_xy) else 0.0,
            "res_end": float(res_xy[n - 1]) if len(res_xy) else 0.0,
            "res_min": float(res_xy.min()) if len(res_xy) else 0.0,
            "res_traj": [round(float(x), 3) for x in res_xy],
            "dyaw_end": float(res[n - 1, 3]),
            "n_events": len(evs),
        }
        if mv_evs:
            e = mv_evs[0]
            m.update({
                "d_mv": e["d_mv"], "d_st": e["d_st"],
                "d_yaw": e["d_yaw"],
                "t_arrive": e["t_arrive"],
                "arr": bool(e["t_arrive"] >= 0),
                "settle_age": e["age"],
            })
        return m

    def _arm_metrics(self, spec, tr):
        def band(arr):
            a = np.asarray(arr, float)
            hit = np.where(a < _BAND)[0]
            return int(hit[0]) if len(hit) else -1
        fr, fl = spec["flag_r"] > 0.5, spec["flag_l"] > 0.5
        return {
            "flag_r": fr, "flag_l": fl,
            "err_r_end": float(tr["arm_r"][-1]) if tr["arm_r"] else 0.,
            "err_l_end": float(tr["arm_l"][-1]) if tr["arm_l"] else 0.,
            "err_r_mean": float(np.mean(tr["arm_r"]))
            if tr["arm_r"] else 0.,
            "err_l_mean": float(np.mean(tr["arm_l"]))
            if tr["arm_l"] else 0.,
            "t_band_r": band(tr["arm_r"]) if fr else -1,
            "t_band_l": band(tr["arm_l"]) if fl else -1,
            "err_q_r_end": (float(tr["armq_r"][-1])
                            if fr and tr["armq_r"] else None),
            "err_q_l_end": (float(tr["armq_l"][-1])
                            if fl and tr["armq_l"] else None),
            "err_r_traj": [round(float(x), 3)
                           for x in tr["arm_r"]],
            "err_l_traj": [round(float(x), 3)
                           for x in tr["arm_l"]],
        }

    def _balance_metrics(self, tr):
        phi = np.asarray(tr["phi"], float)
        mid = np.asarray(tr["mid"], float)
        return {
            "min_phi": float(phi.min()) if len(phi) else 0.0,
            "end_phi": float(phi[-1]) if len(phi) else 0.0,
            "fall": bool((phi < 0.3).any()),
            "feet_disp": float(np.linalg.norm(mid[-1] - mid[0]))
            if len(mid) else 0.0,
        }


__all__ = ["CmdProbe", "decode_cmd22"]
