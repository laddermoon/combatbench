"""STEPBUDGET × 空击 Harness：向前走 → 站定交替出拳 → 视频。

两段式部署形态验证：
    Phase WALK  (t < walk_frames)：planner 发前进步令
                （foot 交替、dir=出生朝向、ψ*=出生朝向、bud=8）。
    Phase PUNCH：保持令（foot=0）+ 交替空击脚本——
                R 刺拳 6f → 收回警戒 14f → L 刺拳 6f → 收回 14f…

软硬劫持两种部署语义：
    --mode soft（v2_stepbudarm 策略）：臂令只写 obs[57:65]
        （flag+tgt），动作原样——策略自己仲裁跟不跟。
    --mode hard（v2_stepbudhard 策略）：obs 写承诺，且外置覆写
        执行器把 act[15:18]/[18:21] 强制替换为 tgt——复刻
        "策略 + 覆写执行器 + 指令源"三件套部署；站立门控复刻
        训练语义（h>1.10 持续 20f 才劫持）。

拳原语来自 punch_sweep 标定（attack_params.json）：
    punch = [1, 0, -1]（右臂最强单发；norm 空间两臂镜像同值）
    guard = [0.172, 0.172, 0.333]（收回警戒位）

用法：
    PYTHONPATH=.:baseline/humanoid21/v1 python3 -B \
        baseline/humanoid21/v1/stepbudget_punch_accept.py \
        --policy baseline/runs/<run>/policy_exports/uNNNNN \
        --mode soft --video
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_V1 = Path(__file__).resolve().parent
_REPO = _V1.parents[2]
for _p in (str(_REPO), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stepbudget_planner import _mk, _vec, BUDGET_MAX  # noqa: E402
from envs.framework.parameterized_blueprint import (  # noqa: E402
    ParameterizedEnvBlueprint,
)

_ENV_YAML = _V1 / "v2_stepbudget_clip_env.yaml"
_OUT = _V1 / "accept_reports" / "punch_clip"

PUNCH_TGT = np.array([1.0, 0.0, -1.0], np.float32)
GUARD_TGT = np.array([0.1724, 0.1724, 0.3333], np.float32)

# 空击脚本：[(dur_f, tgt_r, tgt_l), …] 循环播放
PUNCH_SCRIPT: List[Tuple[int, Optional[np.ndarray],
                         Optional[np.ndarray]]] = [
    (6,  PUNCH_TGT, None),      # R 刺拳
    (14, GUARD_TGT, None),      # 收回警戒
    (6,  None,      PUNCH_TGT), # L 刺拳
    (14, None,      GUARD_TGT),
    (6,  PUNCH_TGT, None),      # R 二次
    (8,  GUARD_TGT, None),
    (6,  None,      PUNCH_TGT), # L 二次
    (8,  None,      GUARD_TGT),
    (5,  PUNCH_TGT, PUNCH_TGT), # 双拳齐出（峰值扰动展示）
    (16, GUARD_TGT, GUARD_TGT), # 双收警戒
]


def _load_policy(export_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "exp_pol", export_dir / "policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("Exported"):
            return getattr(mod, name)()
    raise RuntimeError(f"no Exported* class in {export_dir}")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------------ 规划 --
class WalkPunchPlanner:
    """前走→保持 + 空击脚本的 cmd22 生成器。"""

    def __init__(self, walk_frames: int, bud_step: int = 8,
                 bud_hold: int = 40) -> None:
        self.walk_frames = int(walk_frames)
        self.bud_step = int(bud_step)
        self.bud_hold = int(bud_hold)
        self.cur: Optional[Dict[str, Any]] = None
        self.nxt: Optional[Dict[str, Any]] = None
        self._age = 0
        self._t = 0
        self._yaw0 = 0.0
        self._last_foot = 0

    def reset(self, yaw: float) -> None:
        self._yaw0 = yaw
        self._t = 0
        self._age = 0
        self._last_foot = 0
        self.cur = _mk(0, None, yaw, 8)
        self.nxt = None

    def _alt(self) -> int:
        self._last_foot = -self._last_foot if self._last_foot else -1
        return self._last_foot

    def _plan_nxt(self) -> Dict[str, Any]:
        if self._t < self.walk_frames:
            return _mk(self._alt(), self._yaw0, self._yaw0,
                     self.bud_step)
        return _mk(0, None, self._yaw0, self.bud_hold)

    def _arm_seg(self) -> np.ndarray:
        seg = np.zeros(8, np.float32)
        if self._t < self.walk_frames:
            return seg
        pt = self._t - self.walk_frames
        total = sum(d for d, _, _ in PUNCH_SCRIPT)
        pt %= total
        for dur, tr, tl in PUNCH_SCRIPT:
            if pt < dur:
                if tr is not None:
                    seg[0] = 1.0
                    seg[1:4] = tr
                if tl is not None:
                    seg[4] = 1.0
                    seg[5:8] = tl
                break
            pt -= dur
        return seg

    def update(self, yaw: float) -> np.ndarray:
        self._age += 1
        self._t += 1
        if self._age >= self.cur["budget"]:
            self.cur = (self.nxt if self.nxt is not None
                        else self._plan_nxt())
            self._age = 0
            self.nxt = None
        if self.nxt is None:
            self.nxt = self._plan_nxt()
        c = np.zeros(22, np.float32)
        c[0:8] = self._arm_seg()
        c[10:16] = _vec(self.cur, yaw,
                        self.cur["budget"] - self._age)
        c[16:22] = _vec(self.nxt, yaw, self.nxt["budget"])
        return c


# ------------------------------------------------------------------ 包装 --
class WalkPunchPolicy:
    """obs96 → planner → cmd22 → obs79 → 基座 → （hard 则覆写）。"""

    def __init__(self, base: Any, planner: WalkPunchPlanner,
                 hard: bool, flag_h: float = 1.10) -> None:
        self.base = base
        self.planner = planner
        self.hard = bool(hard)
        self.flag_h = float(flag_h)
        self._world = None          # () -> (yaw, height)
        self._gt = 0                # hard 站立门控 timer
        self._below = 0

    def attach_world(self, fn) -> None:
        """fn() → (ego_yaw, root_height)。"""
        self._world = fn

    def reset(self, seed: int = 0) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset(seed)
        self._gt = 0
        self._below = 0
        if self._world is not None:
            yaw, _ = self._world()
            self.planner.reset(yaw)

    def _standing_ok(self, h: float) -> bool:
        """复刻训练侧站立门控（h>1.10 持续 20f→置位，5f 宽限→清）。"""
        if h > self.flag_h:
            self._gt += 1
            self._below = 0
        else:
            self._below += 1
            if self._below > 5:
                self._gt = 0
        return self._gt >= 20

    def act(self, observation: np.ndarray, *,
            want_extra: bool = False):
        yaw, h = self._world()
        cmd = self.planner.update(yaw)
        if self.hard and not self._standing_ok(h):
            cmd[0] = cmd[4] = 0.0          # 门控拦截：obs flag→0
            cmd[1:4] = cmd[5:8] = 0.0
        obs79 = np.concatenate([observation[:57], cmd])
        act, extra = self.base.act(obs79, want_extra=want_extra)
        act = np.asarray(act, np.float32).copy()
        if self.hard:
            if cmd[0] > 0.5:
                act[15:18] = cmd[1:4]      # 覆写执行器
            if cmd[4] > 0.5:
                act[18:21] = cmd[5:8]
        if want_extra:
            extra = dict(extra or {})
            extra.update({"cmd22": cmd})
            return act, extra
        return act, None


# ------------------------------------------------------------------ 主程 --
def _yaw_h(env) -> Tuple[float, float]:
    core = env.simulator.get_core_state()
    q = np.asarray(core["robot_a"]["root_rot"], np.float64)
    w, x, y, z = q
    yaw = math.atan2(2 * (w * z + x * y),
                     1 - 2 * (y * y + z * z))
    return yaw, float(core["robot_a"]["root_pos"][2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--mode", choices=["soft", "hard"], required=True)
    ap.add_argument("--walk-frames", type=int, default=60)
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--distance", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    pdir = Path(args.policy)
    base = _load_policy(pdir)

    env = (ParameterizedEnvBlueprint.load(_ENV_YAML)
           .materialize(initial_distance=args.distance,
                        max_steps=args.frames)
           .build())
    from envs.humanoid21.plugins import FrozenRobotPlugin
    env.attach_plugin(FrozenRobotPlugin("robot_b"))
    if args.video:
        from envs.framework.common_plugins import VideoRecorderPlugin
        env.attach_plugin(VideoRecorderPlugin(fps=30))

    planner = WalkPunchPlanner(args.walk_frames)
    pol = WalkPunchPolicy(base, planner, hard=(args.mode == "hard"))
    pol.attach_world(lambda: _yaw_h(env))

    outdir = Path(args.out) / args.mode / pdir.name
    (outdir / "videos").mkdir(parents=True, exist_ok=True)

    results = []
    for ep in range(args.episodes):
        vpath = (str(outdir / "videos" / f"ep{ep}.mp4")
                 if args.video else None)
        env.reset(seed=args.seed + ep * 17,
                  options=({"video_output_path": vpath}
                           if vpath else {}))
        pol.reset(args.seed + ep * 17 + 1)
        core = env.simulator.get_core_state()
        d0 = float(np.linalg.norm(
            np.asarray(core["robot_a"]["root_pos"][:2])
            - np.asarray(core["robot_b"]["root_pos"][:2])))
        min_h, n_fall, walked = 9.0, 0, 0.0
        for t in range(args.frames):
            oa, _ = env.get_observation()
            aa, _ = pol.act(np.asarray(oa))
            env.step(aa, np.zeros(21, np.float32))
            core = env.simulator.get_core_state()
            h = float(core["robot_a"]["root_pos"][2])
            min_h = min(min_h, h)
            if h < 0.5:
                n_fall += 1
        core = env.simulator.get_core_state()
        d1 = float(np.linalg.norm(
            np.asarray(core["robot_a"]["root_pos"][:2])
            - np.asarray(core["robot_b"]["root_pos"][:2])))
        walked = d0 - d1
        rec = {"ep": ep, "walked_m": round(walked, 2),
               "min_h": round(min_h, 3), "fall_frames": n_fall,
               "video": vpath}
        results.append(rec)
        print(f"[ep{ep}] {rec}")

    (outdir / "summary.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"out → {outdir}")


if __name__ == "__main__":
    main()
