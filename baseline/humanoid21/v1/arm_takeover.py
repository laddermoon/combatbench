"""手臂接管演示：脚本臂目标 + 原平衡策略控制躯干/双腿（D5）。

目的：验证"手臂欠驱动弱、可脚本化"路线——把双臂 6 个关节的控制权
从策略中剥离，用一次性的目标姿态切换驱动（抱架 / 前伸拳），观察
s42i_b 平衡策略在手臂被接管后能否维持站立。

语义约定（与用户对齐）：
- 初始全身姿态 = READY；**抱架不是 READY 手位**——READY 双手偏低。
  抱架/出拳直接用**归一化关节角**指定（即 action 空间值，
  ``norm_to_rad = norm*half+mid``），不经 IK——IK 分支选择对高位
  手目标不可控（能量最小支选出鸡翅构型），关节角直接可控。
  每臂 (shoulder1, shoulder2, elbow)，左右同值即镜像
  （左右关节轴已按镜像设计，同 norm 值=镜像运动）：
  - 抱架 ``(0.4, -0.3, 0.8)``：shoulder1 pos=夹肘收臂贴肋、
    shoulder2 neg=上臂略后收锁肘于体侧、elbow pos=屈肘前臂竖护脸
  - 出拳 ``(0.4, 0.8, -0.8)``：上臂前抬 + 肘近全伸（留 0.2 余量
    不顶限位）的直拳
- 接管方式：一次性把臂关节 action 切到目标值并保持，无插值；
  PD 控制器自行尽力追踪。另一只手保持抱架。
- 关节划分：臂 6 关节（action[15:21]）由脚本覆盖，其余 15 关节
  继续由原策略输出（策略 obs 里能看到手臂被移动）。
- 初始快照在 READY 上直接把臂关节烘焙为抱架——第 0 帧即抱架，
  无过渡段。

时序（50ms 控制周期，72 步 = 3.6s）：
    0-19 抱架 1.0s → 20-25 右拳 0.3s → 26-45 抱架 1.0s
    → 46-51 左拳 0.3s → 52-71 抱架 1.0s

用法：
    PYTHONPATH=. python3 -m baseline.humanoid21.v1.arm_takeover \
        --policy-yaml /tmp/v1_takeover/policy_a.yaml --out /tmp/v1_takeover
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from baseline.humanoid21.v1.simulator import V1SkillSimulator
from baseline.humanoid21.v1.state import ready_core_state
from envs.framework.env_runtime import EnvRuntime
from envs.framework.policy import Policy, PolicyBlueprint

# ------------------------------------------------------------ 臂目标设计 --

# 归一化关节角（=action 空间值）：每臂 (shoulder1, shoulder2, elbow)，
# 左右同值——关节轴按镜像设计，同 norm 值即镜像动作。
# 抱架：肘贴肋 + 前臂竖护脸（用户指定并渲染确认）。
# 出拳：上臂前抬 + 肘近全伸的直拳（留余量不顶限位）。
_GUARD_JOINT = (0.4, -0.3, 0.8)
_PUNCH_JOINT = (0.4, 0.8, -0.8)

# action 索引：右 [15,16,17]=shoulder1/2/elbow_r，左 [18,19,20]。
ARM_IDX = {"right": np.array([15, 16, 17]),
           "left": np.array([18, 19, 20])}


def _phase_action(phase: str) -> np.ndarray:
    """该相位下 6 关节的 action 覆盖值。"""
    out = np.zeros(6, np.float32)
    for side, idx in (("right", slice(0, 3)), ("left", slice(3, 6))):
        out[idx] = (_PUNCH_JOINT if phase.endswith("_" + side[0])
                    else _GUARD_JOINT)
    return out


# ------------------------------------------------------------- 时序编排 --

_PHASES: Tuple[Tuple[int, int, str], ...] = (
    (0, 20, "guard"),                  # 0.0-1.0s
    (20, 26, "punch_r"),               # 1.0-1.3s
    (26, 46, "guard"),                 # 1.3-2.3s
    (46, 52, "punch_l"),               # 2.3-2.6s
    (52, 72, "guard"),                 # 2.6-3.6s
)
N_STEPS = 72


class ArmTakeoverPolicy(Policy):
    """包装原策略：每步用相位表覆盖 action[15:21]，其余关节透传。

    mode="sequence" 按 _PHASES 执行完整序列；
    mode="guard"    全程抱架（对照组：接管但不发拳）。
    """

    def __init__(self, base: Policy, mode: str = "sequence") -> None:
        if mode not in ("sequence", "guard"):
            raise ValueError(f"mode must be sequence|guard, got {mode}")
        self._base = base
        self._mode = mode
        self._t = 0
        self._cache = {ph: _phase_action(ph) for _, _, ph in _PHASES}

    def _arm(self) -> np.ndarray:
        if self._mode == "guard":
            return self._cache["guard"]
        for lo, hi, ph in _PHASES:
            if lo <= self._t < hi:
                return self._cache[ph]
        return self._cache["guard"]

    def act(self, observation, *, want_extra: bool = False):
        action, extra = self._base.act(observation, want_extra=want_extra)
        action = np.asarray(action, np.float32).copy()
        action[15:21] = self._arm()
        self._t += 1
        return action, extra

    def reset(self, seed=None) -> None:
        self._t = 0
        self._base.reset(seed)


# ----------------------------------------------------------- 快照与环境 --

def _guard_snapshot(robot_id: str, distance: float = 4.0) -> Dict:
    """READY 全身 + 抱架臂，平移到出生槽位；robot_b 旋转 yaw=π 面向 a。"""
    st = ready_core_state()
    for idx in ARM_IDX.values():
        st["joint_pos_norm"][idx] = _GUARD_JOINT
    sign = -1.0 if robot_id == "robot_a" else 1.0
    st["root_pos"] = st["root_pos"].copy()
    st["root_pos"][0] += sign * distance / 2.0
    if robot_id == "robot_b":
        q = Rotation.from_quat(
            [st["root_rot"][1], st["root_rot"][2],
             st["root_rot"][3], st["root_rot"][0]])
        q = Rotation.from_euler("z", np.pi) * q
        wxyz = q.as_quat()[[3, 0, 1, 2]]
        st["root_rot"] = wxyz
    return st


def _build_runtime(video_path: Optional[str], observers: Dict) -> EnvRuntime:
    from baseline.humanoid21.plugins.imbalance_termination import (
        DualImbalanceTerminationPlugin)
    from envs.framework.common_plugins import VideoRecorderPlugin

    sim = V1SkillSimulator(
        cmd_version="v2", initial_distance=4.0,
        initial_pose_a="standing", initial_pose_b="standing")
    plugins: List = [
        DualImbalanceTerminationPlugin(force_threshold=1.0, min_height=0.0)]
    if video_path:
        plugins.append(VideoRecorderPlugin(fps=30, output_path=video_path))
    return EnvRuntime(
        sim, observer_plugins=observers, plugins=plugins,
        phy_steps_per_action=25, max_steps=N_STEPS)


def run_case(name: str, pol_a: Policy, pol_b: Policy,
             out_dir: str) -> Dict:
    """跑一遍 3.6s rollout，返回逐机指标。video 存 out_dir/<name>.mp4。"""
    import os
    from baseline.humanoid21.v1.observers import V1ClipObserver

    obs_a, obs_b = V1ClipObserver("robot_a"), V1ClipObserver("robot_b")
    rt = _build_runtime(
        os.path.join(out_dir, f"{name}.mp4"),
        {"clip_a": obs_a, "clip_b": obs_b})
    rt.reset(seed=0, options={
        "v1_snapshot": {rid: _guard_snapshot(rid)
                        for rid in ("robot_a", "robot_b")},
        "v1_cmd": {rid: np.zeros(9) for rid in ("robot_a", "robot_b")},
    })
    for rid, pol in (("robot_a", pol_a), ("robot_b", pol_b)):
        pol.reset(0)

    rec: Dict[str, List[Dict]] = {"robot_a": [], "robot_b": []}
    term = {"robot_a": None, "robot_b": None}
    steps = 0
    for t in range(N_STEPS):
        if not rt.is_episode_active:
            break
        o_a, o_b = rt.get_observation()
        a_a, _ = pol_a.act(o_a)
        a_b, _ = pol_b.act(o_b)
        rt.step(a_a, a_b)
        steps = t + 1
        for rid, ob in (("robot_a", obs_a), ("robot_b", obs_b)):
            out = ob.get_output()
            if out:
                rec[rid].append({"t": t, "root_pry": out["root_pry"],
                                 "hand_l": out["hand_l"],
                                 "hand_r": out["hand_r"],
                                 "com": out["com"]})
        term = rt.get_agent_termination()
    rt.close()

    summary = {"name": name, "steps": steps,
               "termination": {k: v for k, v in term.items()}}
    for rid in ("robot_a", "robot_b"):
        frames = rec[rid]
        if not frames:
            continue
        pry = np.array([f["root_pry"] for f in frames])
        hz = np.array([[f["hand_l"][2], f["hand_r"][2]] for f in frames])
        com = np.array([f["com"] for f in frames])
        summary[rid] = {
            "survived_steps": len(frames),
            "max_abs_pitch": float(np.abs(pry[:, 0]).max()),
            "max_abs_roll": float(np.abs(pry[:, 1]).max()),
            "hand_z_range": [float(hz.min()), float(hz.max())],
            "com_drift_xy": float(
                np.linalg.norm(com[-1, :2] - com[0, :2])),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-yaml", required=True,
                    help="导出策略 blueprint yaml（u01680 等）")
    ap.add_argument("--out", default="/tmp/v1_takeover")
    ap.add_argument("--cases", default="untouched,guard_hold,punch_seq")
    args = ap.parse_args()
    import os
    os.makedirs(args.out, exist_ok=True)

    def base_policy() -> Policy:
        return PolicyBlueprint.load(args.policy_yaml).build()

    cases = {
        # 对照：原策略不动，READY 起点——sanity（应全程站稳）
        "untouched": (base_policy(), base_policy()),
        # 双臂接管为抱架、全程不发拳——接管本身能否被平衡策略容忍
        "guard_hold": (ArmTakeoverPolicy(base_policy(), "guard"),
                       ArmTakeoverPolicy(base_policy(), "guard")),
        # 主实验：a 执行完整出拳序列，b 接管抱架作镜内对照
        "punch_seq": (ArmTakeoverPolicy(base_policy(), "sequence"),
                      ArmTakeoverPolicy(base_policy(), "guard")),
    }
    results = {}
    for name in args.cases.split(","):
        pa, pb = cases[name.strip()]
        results[name] = run_case(name.strip(), pa, pb, args.out)
        print(json.dumps(results[name], indent=2))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
