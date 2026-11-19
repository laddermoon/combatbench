"""步边界检测相位机 —— 纯 obs 流函数，训练/部署同源同码（D15/S4_PLAN §3）。

判据（每令一周期，两相位）：

    相位A 候抬: 脚 X 连续 lift_deb 帧 force < lift_thr → X 当选 swing
    相位B 摆动: force[swing] > land_thr → 落地边界（executed=swing）
    任一相位 age ≥ timeout_age → 超时边界（executed=None）

要点：脚必须先被观测到卸力才有落地权——已承重指令脚不能秒结、
错脚先抬则错脚当选照样结算。age 由本机内部维护并暴露 `.age`
（sim 写进 step_cur[5]，部署侧指令源自读）。

用法：
    bd = StepBoundary()
    bd.begin_order(cmd_foot=-1)      # 新令一周期（age 归零）
    ev = bd.update(feet_forces)      # feet_forces=[R, L]（obs 同源）
    if ev: ...                        # BoundaryEvent or None

部署侧：指令源拿策略收到的 obs 驱动同一实例，边界时机与训练
完全一致（输入相同 ⇒ 状态相同）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

WAIT_LIFT, IN_SWING = 0, 1


@dataclass(frozen=True)
class BoundaryEvent:
    """一步终了：executed=落地脚(−1左/+1右) 或 None(超时无落地)。"""

    executed: Optional[int]
    timed_out: bool
    age: float


class StepBoundary:
    """单指令周期相位机。feet_forces 顺序 = [R, L]（obs 原生序）。"""

    def __init__(
        self,
        lift_thr: float = 0.1,
        land_thr: float = 0.4,
        lift_deb: int = 2,
        timeout_age: float = 6.0,
        age_inc: float = 0.1,
    ) -> None:
        self.lift_thr = float(lift_thr)
        self.land_thr = float(land_thr)
        self.lift_deb = int(lift_deb)
        self.timeout_age = float(timeout_age)
        self.age_inc = float(age_inc)
        self.phase = WAIT_LIFT
        self.age = 0.0
        self._swing: Optional[int] = None       # −1左/+1右
        self._cmd: int = 0
        self._deb = [0, 0]                       # [R, L] 连续卸力帧数

    def begin_order(self, cmd_foot: int) -> None:
        """新指令周期：age 归零、相位复位。"""
        self.phase = WAIT_LIFT
        self.age = 0.0
        self._swing = None
        self._cmd = int(cmd_foot)
        self._deb = [0, 0]

    def update(self, forces) -> Optional[BoundaryEvent]:
        """喂一帧 feet_forces=[R, L]；返回 BoundaryEvent 或 None。"""
        self.age += self.age_inc
        fr, fl = float(forces[0]), float(forces[1])
        f = {-1: fl, +1: fr}

        if self.phase == WAIT_LIFT:
            for side, v in ((-1, fl), (+1, fr)):
                i = 1 if side < 0 else 0
                self._deb[i] = self._deb[i] + 1 if v < self.lift_thr else 0
            # 选举：先满去抖者；同帧并列 → 指令脚优先
            hit_l = self._deb[1] >= self.lift_deb
            hit_r = self._deb[0] >= self.lift_deb
            if hit_l or hit_r:
                if hit_l and hit_r:
                    self._swing = self._cmd if self._cmd in (-1, +1) else -1
                else:
                    self._swing = -1 if hit_l else +1
                self.phase = IN_SWING
                self._deb = [0, 0]
        elif f[self._swing] > self.land_thr:
            return BoundaryEvent(self._swing, False, self.age)

        if self.age >= self.timeout_age:
            return BoundaryEvent(None, True, self.age)
        return None


if __name__ == "__main__":
    # 自测：正常步 / 错脚 / 无抬超时 / 摆动超时 / 已承重指令脚不秒结
    bd = StepBoundary()

    # 1) 正常：指令左脚，左脚抬→落
    bd.begin_order(-1)
    ev = None
    seq = [(0.5, 0.5)] * 3 + [(0.5, 0.05)] * 2 + [(0.5, 0.02)] * 5 \
        + [(0.5, 0.9)]
    for t, f_ in enumerate(seq):
        ev = bd.update(f_)
        if ev:
            break
    assert ev and ev.executed == -1 and not ev.timed_out, ev
    print(f"正常步: t={t} age={ev.age:.1f} executed=L ✓")

    # 2) 错脚：指令左脚，右脚先抬先落 → 结算 executed=R
    bd.begin_order(-1)
    seq = [(0.5, 0.5)] + [(0.05, 0.5)] * 2 + [(0.02, 0.5)] * 4 + [(0.9, 0.5)]
    for t, f_ in enumerate(seq):
        ev = bd.update(f_)
        if ev:
            break
    assert ev and ev.executed == +1 and not ev.timed_out, ev
    print(f"错脚步: t={t} age={ev.age:.1f} executed=R ✓")

    # 3) 无抬超时：双脚恒承重
    bd.begin_order(+1)
    for t in range(80):
        ev = bd.update((0.5, 0.5))
        if ev:
            break
    assert ev and ev.executed is None and ev.timed_out and ev.age >= 6.0
    print(f"无抬超时: t={t} age={ev.age:.1f} ✓")

    # 4) 摆动超时：右脚抬起后不落
    bd.begin_order(+1)
    seq = [(0.5, 0.5)] + [(0.05, 0.5)] * 2 + [(0.02, 0.5)] * 100
    for t, f_ in enumerate(seq):
        ev = bd.update(f_)
        if ev:
            break
    assert ev and ev.timed_out and ev.executed is None
    print(f"摆动超时: t={t} age={ev.age:.1f} ✓")

    # 5) 指令脚已承重不秒结：左脚一直 >0.4，必须真抬真落
    bd.begin_order(-1)
    seq = [(0.5, 0.6)] * 8 + [(0.5, 0.05)] * 2 + [(0.5, 0.7)]
    evs = []
    for t, f_ in enumerate(seq):
        e = bd.update(f_)
        if e:
            evs.append((t, e))
    assert len(evs) == 1 and evs[0][0] == 10 and evs[0][1].executed == -1, evs
    print(f"已承重指令脚: t={evs[0][0]} 真抬落后才结 ✓")

    # 6) 双脚同帧齐抬 → 指令脚优先
    bd.begin_order(+1)
    ev = None
    seq = [(0.5, 0.5)] + [(0.05, 0.05)] * 2 + [(0.5, 0.9)]
    for t, f_ in enumerate(seq):
        ev = bd.update(f_)
        if ev:
            break
    assert ev and ev.executed == +1, ev
    print(f"齐抬指令脚优先: executed=R ✓")

    print("\nstep_boundary 自测全过")
