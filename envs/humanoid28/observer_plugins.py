"""
Humanoid28 observer 插件

当前包含：
  CombatScoringObserver — 战斗状态 observer（与 humanoid21 逻辑一致，
      消费 CombatScoringPlugin 写入 ctx.metrics / ctx.events 的数据）。

注：humanoid21 的 BalanceAnalysisObserver 尚未移植（依赖踝关节语义适配），
    见 README.md「未移植/后续工作」。
"""
from typing import Any, Dict

from envs.framework import BaseObserverPlugin
from envs.framework.context import ReadOnlySimContext

class CombatScoringObserver(BaseObserverPlugin):
    """
    Combat status observer that tracks both robots simultaneously.

    Consumes data produced by :class:`CombatScoringPlugin` (stored in
    ``ctx.metrics``) and surfaces health, damage, and hit events for
    ``robot_a`` and ``robot_b`` in a single output dict.

    Output keys (dict):
        * ``robot_a`` — dict with ``health``, ``cumulative_damage_taken``,
          ``step_damage_taken``, ``step_hit_events``, ``is_ko``.
        * ``robot_b`` — same structure for the opponent.
        * ``events`` — raw hit events list from the current step.

    Per-step semantics are derived by cursor-diff on ``ctx.events``: the
    event journal is **append-only within an episode** (see
    ``EventJournal``) and written by :class:`CombatScoringPlugin` (one
    entry per action step at ``on_post_action_step``), so "this step's
    events" = the slice appended since the previous refresh. The cursor
    is ``(events_epoch, count)``: an epoch change means the journal was
    reset by the framework — the whole current journal counts as new
    rather than continuing from the stale position. ``ctx.events``
    itself is never mutated here — this observer is the reference
    example for consuming the shared event journal.
    """

    def __init__(self):
        self._output: Any = None
        self._cursor_epoch = 0
        self._cursor_count = 0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._cursor_epoch = ctx.events_epoch
        self._cursor_count = len(ctx.events)
        self._output = self._build_output(ctx)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_output(ctx)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        return None

    def get_output(self) -> Any:
        return self._output

    def to_blueprint(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "CombatScoringObserver":
        return cls(**config)

    def _build_output(self, ctx: ReadOnlySimContext) -> Dict[str, Any]:
        metrics = ctx.metrics
        all_events = list(ctx.events)
        if ctx.events_epoch != self._cursor_epoch:
            # Journal was reset since the last read (new episode) — every
            # current entry is new; do not continue from the stale offset.
            self._cursor_epoch = ctx.events_epoch
            self._cursor_count = 0
        elif len(all_events) < self._cursor_count:
            # Defensive resync for a shrink that bypassed _reset — cannot
            # happen through the public EventJournal API.
            self._cursor_count = 0
        events = all_events[self._cursor_count:]
        self._cursor_count = len(all_events)

        def _robot_status(robot_id: str) -> Dict[str, Any]:
            health_key = f"health_{robot_id[-1]}"  # health_a / health_b
            damage_key = f"damage_taken_{robot_id[-1]}"
            robot_events = [e for e in events if e.get("defender") == robot_id]
            step_damage = sum(float(e.get("damage", 0.0)) for e in robot_events)
            health = float(metrics.get(health_key, 0.0))
            return {
                "health": health,
                "cumulative_damage_taken": float(metrics.get(damage_key, 0.0)),
                "step_damage_taken": float(step_damage),
                "step_hit_events": robot_events,
                "is_ko": health <= 0.0,
            }

        return {
            "robot_a": _robot_status("robot_a"),
            "robot_b": _robot_status("robot_b"),
            "events": events,
        }
