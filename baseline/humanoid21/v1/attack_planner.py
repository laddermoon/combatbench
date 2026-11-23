"""ATTACK_PLANNER：ego 系脚本状态机 → cmd22（开发版）。

    obs96 ─→ ego57 + opp 段解析 → AttackPlanner.update() → cmd22
    obs79 = cat(ego57, cmd22) → 基座 policy → action21

关键语义（与部署/训练契约对齐）：

- cmd22 指令段字面格式（obs[57:79]）：
    [0]   flag_r  [1:4] tgt_r   [4] flag_l  [5:8] tgt_l
    [8:10] waist（预留，恒 0）
    [10:16] step_cur = [foot, Δx, Δy, Δψ, 0, 0]
    [16:22] step_nxt = 同构（planner 队列诚实预览）
- 步令段装的是**残差**，部署侧逐字写入——实测写法语义敏感
  （战斗 env 单步对照，150f 位移）：
      恒定字面        (0.02,0.44)  原地踏步——违反"残差随执行衰减"
      纯对手锚定      (0.59,0.83)  前进但漂移大于前进
      落地清零        (0.50,1.05)  同
      **锚定幅度+令龄衰减+实时方向  (0.83,0.34)  ← 采用**
      衰减12帧+live方向 (0.83,0.34)  decay_k=12
  即：res = min(d−strike_range,cap)·decay(age/12)·dir(θ_live)，
  Δψ = θ_live·decay。
- 足底触地索引（实测）：**foot<0 ↔ obs[52]，foot>0 ↔ obs[53]**
  （文档"右/左"标注与实际flag→抬脚对应关系相反——以实测为准）。
- foot=0 时 Δψ 被 cmd_target 吞掉（已知边界）——转身对准用
  "位移=0 的移动令"（foot=±1, Δ=0, Δψ=θ），验收已验证此令型。
- 臂 tgt 是绝对量（非残差）每帧可更新；拳 = arm_seq 两段
  （punch_k 帧伸展 → 收回到 guard tgt）。

状态机：APPROACH（步令逼近）→ STRIKE（脚hold+拳序列）
       →（序列播完回 APPROACH）；upright 崩 → GUARD（全零令，
       让基座自己站稳）恢复后回 APPROACH。
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:                                   # 打包版内联 stub 兜底
    from envs.framework.policy import Policy
except Exception:                      # pragma: no cover
    class Policy:                      # type: ignore[no-redef]
        def act(self, observation, *, want_extra=False):
            raise NotImplementedError


_DEFAULT_PARAMS: Dict[str, Any] = {
    "strike_range": 0.62,    # 根间距 ≤ 此值且对准 → 出拳
    "approach_range": 0.20,  # 逼近目标深度（拳程~0.5，根距需
                           #   ≤~0.4 拳才能压进躯干半径~0.2）
                           #   必须 < strike_range，否则出现
                           #   (strike_range, strike_range+standoff_step)
                           #   既不迈步也不出拳的死锁区）
    "strike_yaw": 0.15,      # rad，|bearing|≤此值才算对准——实测
                             #   θ~0.3 出拳从体侧擦过（躯干半径~0.15）
    "abort_yaw": 0.40,       # STRIKE 中偏航超此 → 中止重对准
    "step_cap": 0.40,        # 单步残差位移上限 m
    "dyaw_cap": 0.45,        # 单步转角残差上限 rad
    "standoff_step": 0.12,   # 残余步距 < 此值不迈步（防原地碎步）
    "max_age": 15,           # 步令重规划间隔兜底帧
    "punch_ext": (0.5, 0.5, -1.0),    # 低位体干拳（z-.21 钻护臂下方
                                    #   打腹，reach .55）——冻结假人
                                    #   前臂护胸，正面拳全打在臂上
    "punch_high": (-1.0, 1.0, -1.0),  # 高位拳 tgt（reach .51 z+.14）
    "punch_k": 8,            # 伸展段帧数
    "recover_k": 8,          # 收回段帧数（之后回 APPROACH）
    "high_z": 0.20,          # 对手头 ego-z > 此值 → 选高位拳
                             #   （冻结假人前臂护胸，体干拳打在
                             #   臂上不记分——先压高位拳试越顶）
                             #   （站立对手头 ego-z~0.3，默认打躯干
                             #   ——目标大、reach 更长的体干拳优先）
    "guard_z": 0.45,         # upright < 此 → GUARD
    "guard_recover": 0.60,   # upright > 此持续 guard_frames → 解除
    "guard_frames": 10,
    "contact_hi": 0.30,      # 落地阈值（体重倍数）
    "contact_lo": 0.15,      # 离地阈值
    "contact_m1_idx": 52,    # 实测 foot<0 ↔ obs[52]
    "contact_p1_idx": 53,    # 实测 foot>0 ↔ obs[53]
    "decay_k": 12.0,         # 残差线性衰减常数（帧）——扫出最优
    "lunge_step": 0.25,      # STRIKE 期突进步幅 m
}


class AttackPlanner:
    """obs96 → cmd22。每帧调用，内部自持状态机。"""

    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 guard_tgt=(0.172, 0.172, 0.333)) -> None:
        self.p = dict(_DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.guard_tgt = np.asarray(guard_tgt, float)
        self.reset()

    # ------------------------------------------------------------ 状态 --
    def reset(self) -> None:
        self.state = "APPROACH"
        self._foot = -1              # 下一条步令的动脚（-1 左 / +1 右）
        self._age = 10 ** 9          # 当前步令令龄（强制首帧发令）
        self._armed = False          # 落地门控武装（离地→落地语义）
        self._strike_t = 0
        self._guard_ok = 0
        self._punch = np.asarray(self.p["punch_ext"], float)

    # ------------------------------------------------------------ 主入口 --
    def update(self, obs: np.ndarray) -> np.ndarray:
        p = self.p
        obs = np.asarray(obs, float)
        opp = obs[57:60]                       # 对手根：ego 系
        d = float(np.hypot(opp[0], opp[1]))
        th = float(math.atan2(opp[1], opp[0]))
        head_z = float(obs[68])                # 对手头 ego-z
        up = -float(obs[44])                   # uprightness（-g_z）
        c_m1 = float(obs[p["contact_m1_idx"]])
        c_p1 = float(obs[p["contact_p1_idx"]])

        if self.state != "GUARD" and up < p["guard_z"]:
            self.state, self._guard_ok = "GUARD", 0
        if self.state == "GUARD":
            if up > p["guard_recover"]:
                self._guard_ok += 1
                if self._guard_ok >= p["guard_frames"]:
                    self.state = "APPROACH"
                    self._age = 10 ** 9        # 立即重发步令
            else:
                self._guard_ok = 0
            return self._cmd(step=None, arm=None)   # 全零：基座自救

        self._age += 1
        if self.state == "STRIKE":
            self._strike_t += 1
            if (self._strike_t >= p["punch_k"] + p["recover_k"]
                    or abs(th) > p["abort_yaw"]):
                self.state, self._age = "APPROACH", 10 ** 9
                return self._cmd(step=None, arm=None)
            tgt = (self._punch if self._strike_t <= p["punch_k"]
                   else self.guard_tgt)
            # 突进拳：伸展段同时给小步令压上最后 0.2m
            # （实测站立拳 ~0.6m 根距仍差 0.25m 接触）
            lunge = np.array([self._foot,
                              p["lunge_step"] * math.cos(th),
                              p["lunge_step"] * math.sin(th),
                              0.3 * th, 0, 0])
            return self._cmd(step=(lunge, np.zeros(6)),
                             arm=(tgt, None))

        # APPROACH
        if d <= p["strike_range"] and abs(th) <= p["strike_yaw"]:
            self.state = "STRIKE"
            self._strike_t = 0
            self._punch = (np.asarray(p["punch_high"], float)
                           if head_z > p["high_z"]
                           else np.asarray(p["punch_ext"], float))
            return self._cmd(step=None,
                             arm=(self._punch, None))

        # 重规划门控：首令 / 动脚"离地→落地" / 超时
        c_mv = c_m1 if self._foot < 0 else c_p1
        if c_mv < p["contact_lo"]:
            self._armed = True
        landed = self._armed and c_mv >= p["contact_hi"]
        if self._age > p["max_age"] or (self._age > 2 and landed):
            self._foot = -self._foot
            self._age = 0
            self._armed = c_mv < p["contact_lo"]

        step = self._step_res(d, th, self._foot, self._age)
        nxt = self._step_res(d, th, -self._foot, 0)
        return self._cmd(step=(step, nxt), arm=None)

    # ------------------------------------------------------------ 构造 ----
    def _step_res(self, d: float, th: float, foot: int,
                  age: int) -> np.ndarray:
        """逼近步残差：min(d−range,cap)·decay(age) 沿 bearing 方向。

        幅度对对手距离锚定（身体推进→d收缩→幅度缩），
        剖面按令龄线性衰减（模拟"残差随脚踝接近目标"），
        方向对实时 bearing 跟踪——三者缺一实测就废（见模块doc）。
        """
        p = self.p
        k = max(0.0, 1.0 - age / p["decay_k"])
        L = max(0.0, min(d - p["approach_range"],
                         p["step_cap"])) * k
        dyaw = float(np.clip(th, -p["dyaw_cap"], p["dyaw_cap"])) * k
        if L < p["standoff_step"] * k:
            return np.array([foot, 0.0, 0.0, dyaw, 0, 0])
        return np.array([foot,
                         L * math.cos(th), L * math.sin(th),
                         dyaw, 0, 0])

    def _cmd(self, step: Optional[Tuple[np.ndarray, np.ndarray]],
             arm: Optional[Tuple[np.ndarray, Optional[np.ndarray]]]
             ) -> np.ndarray:
        c = np.zeros(22)
        if arm is not None:
            tr, tl = arm
            if tr is not None:
                c[0] = 1.0
                c[1:4] = tr
            if tl is not None:
                c[4] = 1.0
                c[5:8] = tl
        if step is not None:
            c[10:16] = step[0]
            c[16:22] = step[1]
        return c


# ============================================================ wrappers ==

class CmdPolicy(Policy):
    """obs96 + cmd_fn → obs79 → 基座 policy → action（对立方可用）。"""

    def __init__(self, base: Policy,
                 cmd_fn: Callable[[np.ndarray, int], np.ndarray]) -> None:
        self.base = base
        self.cmd_fn = cmd_fn
        self._t = 0

    def reset(self, seed: Optional[int] = None) -> None:
        self._t = 0
        if hasattr(self.base, "reset"):
            self.base.reset(seed)

    def act(self, observation: np.ndarray, *,
            want_extra: bool = False) -> Tuple[np.ndarray, Any]:
        obs = np.asarray(observation, float)
        cmd = self.cmd_fn(obs, self._t)
        self._t += 1
        act, extra = self.base.act(
            np.concatenate([obs[:57], cmd]), want_extra=want_extra)
        return act, extra


class AttackPolicy(Policy):
    """obs96 → AttackPlanner → cmd22 → obs79 → 基座 → action21。"""

    def __init__(self, base: Policy, planner: AttackPlanner) -> None:
        self.base = base
        self.planner = planner
        self._last_cmd = np.zeros(22)

    def reset(self, seed: Optional[int] = None) -> None:
        self.planner.reset()
        if hasattr(self.base, "reset"):
            self.base.reset(seed)

    def act(self, observation: np.ndarray, *,
            want_extra: bool = False) -> Tuple[np.ndarray, Any]:
        obs = np.asarray(observation, float)
        cmd = self.planner.update(obs)
        self._last_cmd = cmd
        act, extra = self.base.act(
            np.concatenate([obs[:57], cmd]), want_extra=want_extra)
        if want_extra:
            return act, {"cmd22": cmd, "planner": self.planner.state,
                         "base_extra": extra}
        return act, None


# ------------------------------------------------------------ 对立方 ----

def opp_hold_cmd(obs: np.ndarray, t: int) -> np.ndarray:
    """站立对手：恒零指令（hold）。"""
    return np.zeros(22)


def make_opp_walk_cmd(step: float = 0.15,
                      period: int = 14) -> Callable:
    """慢走对手：周期交替小步前移残差（朝自身 +x，即朝向对手出生位）。

    残差每帧按固定节奏写（不演化的"目标距离"近似——对手是
    陪练不是主角，够制造移动靶即可）。
    """
    def _fn(obs: np.ndarray, t: int) -> np.ndarray:
        c = np.zeros(22)
        ph = (t // period) % 2
        foot = -1.0 if ph == 0 else 1.0
        c[10:16] = [foot, step, 0.0, 0.0, 0.0, 0.0]
        c[16:22] = [-foot, step, 0.0, 0.0, 0.0, 0.0]
        return c
    return _fn


__all__ = ["AttackPlanner", "AttackPolicy", "CmdPolicy",
           "opp_hold_cmd", "make_opp_walk_cmd"]
