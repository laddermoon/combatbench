"""前序链式起点池（DECISION_LOG D3 简化一，内存版 v0）。

每轮迭代：
  begin_iteration  → 候选起点 = READY + ∪ 前序技能终止状态集
  build_jobs       → sample() 取候选起点
  build_trajectories→ 合格执行的末帧快照 record_terminal() 入终止集
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class StartEntry:
    """一个候选起点：完整 core_state 快照（jsonable lists）。"""
    key: str
    state: Dict[str, Any]
    source: str        # 产生它的技能名，或 "ready"
    update: int        # 产生时的 update 号


class StartPool:
    """起点池。候选 = READY + 各前序技能的终止状态集（D3）。"""

    def __init__(self, ready_state: Dict[str, Any]) -> None:
        self._ready = StartEntry("ready", ready_state, "ready", 0)
        self.candidates: Dict[str, List[StartEntry]] = {}
        self.terminals: Dict[str, List[StartEntry]] = {}
        self._next_idx = 0

    def begin_iteration(self, skills, update: int) -> None:
        """刷新各技能候选起点：READY + ∪ predecessors 的终止集。"""
        for name, sk in skills.items():
            cands = [self._ready]
            for pred in sk.predecessors:
                cands.extend(self.terminals.get(pred, ()))
            self.candidates[name] = cands

    def sample(self, skill: str, rng: np.random.Generator) -> StartEntry:
        cands = self.candidates.get(skill) or [self._ready]
        return cands[int(rng.integers(len(cands)))]

    def record_terminal(self, skill: str, state: Dict[str, Any],
                        update: int) -> StartEntry:
        e = StartEntry(f"{skill}@u{update}#{self._next_idx}",
                       state, skill, update)
        self._next_idx += 1
        self.terminals.setdefault(skill, []).append(e)
        return e

    def stats(self) -> Dict[str, Any]:
        return {
            "candidates": {k: len(v) for k, v in self.candidates.items()},
            "terminals": {k: len(v) for k, v in self.terminals.items()},
        }

    # ── checkpoint 持久化（experiment.state()/load_state 调）──
    def save(self) -> Dict[str, Any]:
        return {
            "next_idx": self._next_idx,
            "terminals": {
                k: [{"key": e.key, "state": e.state,
                     "source": e.source, "update": e.update}
                    for e in v]
                for k, v in self.terminals.items()},
        }

    def load(self, data: Dict[str, Any]) -> None:
        self._next_idx = int(data.get("next_idx", 0))
        self.terminals = {
            k: [StartEntry(**e) for e in v]
            for k, v in (data.get("terminals") or {}).items()}
