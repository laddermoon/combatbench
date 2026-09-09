"""``knobs`` — KnobCheck registry for ``intervene-check`` (S6).

S6: Defines the knob-checking infrastructure.  A "knob" is a configurable
parameter that should be observable in the data pathway.  ``intervene-check``
verifies that each configured knob actually entered the training data
path — "配置里写了" ≠ "生效了" (DEBUG_GUIDE.md §3.8).

Two layers:
1. **Built-in knobs** (framework-registered): explore_factor,
   uncertainty_floor, floor_weight, resume, observer.  These cover
   common PPO knobs that any experiment using this framework has.
2. **Experiment knobs** (via ``ExperimentPPO.knob_checks()``): experiments
   add their own knobs for experiment-specific parameters.

See ``DEBUG_GUIDE.md`` §3.8 ``intervene-check`` and
``DESIGN_debug_system.md`` §5.4.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Log parsing helpers (shared pattern with attribute.py)
# ---------------------------------------------------------------------------

_RAW_STATS_RE = re.compile(r"__RAW_STATS__\s*(\{.*\})")
_GRADDIAG_RE = re.compile(
    r"\[GradDiag\].*floor_active=([0-9.]+)"
)


def parse_log_entry(run_dir: Path, target_update: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Find the latest (or specific) ``__RAW_STATS__`` full entry.

    Unlike ``attribute.parse_log_stats`` which returns only the inner
    ``stats`` dict, this returns the full entry including ``update``,
    ``buffer_stats``, ``eval_info``, ``timing``, etc.

    Args:
        run_dir: Training run directory.
        target_update: If given, find the entry for this update number.

    Returns:
        Full parsed dict, or None if not found.
    """
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return None
    latest: Optional[Dict[str, Any]] = None
    with open(log_path) as f:
        for line in f:
            m = _RAW_STATS_RE.search(line)
            if not m:
                continue
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if target_update is not None:
                if data.get("update") == target_update:
                    return data
            else:
                latest = data
    return latest


def parse_floor_active(run_dir: Path, target_update: Optional[int] = None) -> Optional[float]:
    """Extract the latest ``floor_active`` value from ``[GradDiag]`` log lines.

    If ``target_update`` is given, scans only lines between that update's
    ``[update N]`` marker and the next.  Otherwise returns the latest.

    Returns:
        Latest floor_active value, or None if no GradDiag lines found.
    """
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return None
    latest: Optional[float] = None
    in_target = target_update is None
    with open(log_path) as f:
        for line in f:
            # Track update boundaries
            um = re.search(r"\[update\s+(\d+)\]", line)
            if um:
                u = int(um.group(1))
                if target_update is not None:
                    in_target = (u == target_update)
            m = _GRADDIAG_RE.search(line)
            if m and (in_target or target_update is None):
                latest = float(m.group(1))
    return latest


def load_config(run_dir: Path) -> Dict[str, Any]:
    """Load ``config.json`` from a run directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KnobContext:
    """Context passed to KnobCheck callbacks.

    Provides everything a knob check might need: the run directory,
    parsed config, the latest (or target) ``__RAW_STATS__`` entry,
    the inner ``stats`` dict, an optional snapshot directory, and the
    reconstructed experiment instance.
    """
    run_dir: Path
    config: Dict[str, Any]
    log_entry: Optional[Dict[str, Any]]      # full __RAW_STATS__ entry
    log_stats: Optional[Dict[str, Any]]     # inner "stats" key
    snapshot_dir: Optional[Path] = None
    experiment: Any = None                   # ExperimentPPO or None


@dataclass(frozen=True)
class KnobCheck:
    """A single knob verification.

    A "knob" is a configurable parameter.  ``configured`` reads what was
    set (from config/experiment).  ``observed`` reads what actually
    happened in the data pathway (from log/snapshot).  ``agree`` decides
    whether they match.
    """
    name: str
    configured: Callable[["KnobContext"], Any]
    observed: Callable[["KnobContext"], Any]
    agree: Callable[[Any, Any], bool]
    requires_snapshot: bool = False
    """If True, this knob needs snapshot data.  When no snapshot is
    available, the knob is skipped with a "需快照" note instead of
    being run."""


@dataclass
class KnobResult:
    """Result of running one KnobCheck."""
    name: str
    configured_value: Any
    observed_value: Any
    passed: bool
    note: str = ""
    skipped: bool = False
    """True if the knob was skipped (e.g. snapshot-dependent knob with
    no snapshot available)."""


# ---------------------------------------------------------------------------
# Built-in knob checks
# ---------------------------------------------------------------------------

def _cfg_explore_factor(ctx: KnobContext) -> Any:
    """Configured explore_factor: '相位相关' if experiment uses phase-dependent ef, else the value."""
    exp = ctx.experiment
    if exp is not None and hasattr(exp, "explore_factor"):
        # Check if experiment overrides build_jobs with phase_explore_factor
        # by looking for the callable in the experiment module or class.
        # The standup_step_v3 experiment uses phase_explore_factor.
        # We detect this by checking if build_jobs was overridden.
        import inspect
        build_jobs_method = getattr(type(exp), "build_jobs", None)
        if build_jobs_method is not None:
            src = inspect.getsource(build_jobs_method)
            if "phase_explore_factor" in src or "explore_factor_a=" in src:
                return "相位相关"
        return getattr(exp, "explore_factor", None)
    return None


def _obs_explore_factor(ctx: KnobContext) -> Any:
    """Observed: eff_std_mean / std_mean ratio from log stats."""
    stats = ctx.log_stats
    if stats is None:
        return None
    eff = stats.get("eff_std_mean")
    std = stats.get("std_mean")
    if eff is None or std is None or std == 0:
        return None
    return eff / std


def _agree_explore_factor(cfg: Any, obs: Any) -> bool:
    """Agree: if phase-dependent, ratio should be ~1.5 (1.3–1.8); if fixed, ~1.0."""
    if obs is None:
        return False
    if cfg == "相位相关":
        return 1.3 <= obs <= 1.8
    # Fixed ef: ratio should be close to 1.0
    return 0.85 <= obs <= 1.15


def _cfg_uncertainty_floor(ctx: KnobContext) -> Any:
    """Configured uncertainty_floor from experiment or config."""
    exp = ctx.experiment
    if exp is not None and hasattr(exp, "uncertainty_floor"):
        # Check if floor was disabled
        if getattr(exp, "_floor_disabled", False):
            return f"{exp.uncertainty_floor} (已关闭@u{getattr(exp, '_floor_disabled_at', '?')})"
        return exp.uncertainty_floor
    # Fallback: check config.json
    cfg = ctx.config.get("experiment", {}).get("initial_exploration", {})
    return cfg.get("uncertainty_floor", None)


def _obs_uncertainty_floor(ctx: KnobContext) -> Any:
    """Observed: floor_active from GradDiag log lines."""
    return parse_floor_active(ctx.run_dir)


def _agree_uncertainty_floor(cfg: Any, obs: Any) -> bool:
    """Agree: floor_active > 0 when floor is active; ≈0 when disabled."""
    if obs is None:
        return False
    if isinstance(cfg, str) and "已关闭" in cfg:
        # Floor was disabled — expect floor_active ≈ 0
        return obs < 0.01
    if cfg is None or cfg == 0:
        return obs < 0.01
    # Floor is configured and active — expect floor_active > 0
    return obs > 0.01


def _cfg_floor_weight(ctx: KnobContext) -> Any:
    """Configured floor_weight: 'balance_mask' if experiment uses mask-based."""
    exp = ctx.experiment
    if exp is not None:
        # Check if experiment has floor_weight pattern (balance_mask)
        import inspect
        src = inspect.getsource(type(exp))
        if "balance_mask" in src and "floor_weight" in src:
            return "balance_mask"
    return None


def _obs_floor_weight(ctx: KnobContext) -> Any:
    """Observed: unique actor_weight values across channels from buffer_stats."""
    entry = ctx.log_entry
    if entry is None:
        return None
    buffer_stats = entry.get("buffer_stats", {})
    per_channel = buffer_stats.get("per_channel", {})
    if not per_channel:
        return None
    unique_vals = set()
    for ch_name, ch_stats in per_channel.items():
        aw_min = ch_stats.get("actor_weight_min")
        aw_max = ch_stats.get("actor_weight_max")
        if aw_min is not None:
            unique_vals.add(round(float(aw_min), 4))
        if aw_max is not None:
            unique_vals.add(round(float(aw_max), 4))
    return sorted(unique_vals) if unique_vals else None


def _agree_floor_weight(cfg: Any, obs: Any) -> bool:
    """Agree: mask-based weights should have values ⊆ {0, 1} (or close)."""
    if obs is None:
        return False
    # All values should be close to 0 or 1
    return all(abs(v) < 0.01 or abs(v - 1.0) < 0.01 for v in obs)


def _cfg_resume(ctx: KnobContext) -> Any:
    """Configured resume: resume_from path from config, or '无'."""
    # train.py stores resume_from in config? Check config.json structure.
    # Actually, resume_from is a CLI arg, not stored in config.json.
    # We check the log for "[checkpoint] resuming from update X".
    log_path = ctx.run_dir / "train.log"
    if not log_path.exists():
        return "无"
    with open(log_path) as f:
        for line in f:
            if "[checkpoint] resuming from update" in line:
                m = re.search(r"resuming from update (\d+)", line)
                if m:
                    return f"resume@u{m.group(1)}"
            if "[checkpoint] update counter reset to 0" in line:
                return "reset_update"
    return "无"


def _obs_resume(ctx: KnobContext) -> Any:
    """Observed: parameter fingerprint match from latest checkpoint."""
    ckpt_dir = ctx.run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return "无checkpoint"
    ckpts = sorted(ckpt_dir.glob("checkpoint_u*.pt"))
    if not ckpts:
        return "无checkpoint"
    # We can't fully verify parameter fingerprint without loading the
    # checkpoint and comparing to the current policy export.  For now,
    # just report that a checkpoint exists and its update number.
    latest_ckpt = ckpts[-1]
    m = re.search(r"checkpoint_u(\d+)\.pt", latest_ckpt.name)
    if m:
        return f"ckpt@u{m.group(1)}"
    return str(latest_ckpt.name)


def _agree_resume(cfg: Any, obs: Any) -> bool:
    """Agree: if resume configured, checkpoint should exist; if 无, any ckpt is fine."""
    if obs == "无checkpoint":
        return cfg == "无"
    return True


def _cfg_observer(ctx: KnobContext) -> Any:
    """Configured: '—' (observer consistency is always checked against obs)."""
    return "—"


def _obs_observer(ctx: KnobContext) -> Any:
    """Observed: obs[45] vs h_torso max deviation from snapshot debug_arrays."""
    if ctx.snapshot_dir is None:
        return None
    replay_dir = ctx.snapshot_dir / "replay"
    debug_npz = replay_dir / "debug_arrays.npz"
    if not debug_npz.exists():
        return None
    # Check for h_torso debug array (exp_standup_step_v3 exports h_torso)
    data = np.load(debug_npz, allow_pickle=True)
    h_torso = data.get("h_torso", None)
    if h_torso is None:
        # Try other common names
        for key in data.files:
            if "torso" in key.lower() and "h_" in key.lower():
                h_torso = data[key]
                break
    if h_torso is None:
        return None
    # We'd need obs[45] to compare, but obs is in episodes/, not replay/.
    # For now, just report h_torso stats as a proxy.
    return f"max_dev=未测"  # Placeholder — full impl needs episode obs


def _agree_observer(cfg: Any, obs: Any) -> bool:
    """Agree: max deviation < 1e-4."""
    if obs is None:
        return False
    if "未测" in str(obs):
        return False
    return True  # Simplified


def builtin_knob_checks() -> List[KnobCheck]:
    """Return the framework's built-in knob checks.

    These cover common PPO knobs.  Experiments add their own via
    ``knob_checks()``.
    """
    return [
        KnobCheck(
            name="explore_factor",
            configured=_cfg_explore_factor,
            observed=_obs_explore_factor,
            agree=_agree_explore_factor,
        ),
        KnobCheck(
            name="uncertainty_floor",
            configured=_cfg_uncertainty_floor,
            observed=_obs_uncertainty_floor,
            agree=_agree_uncertainty_floor,
        ),
        KnobCheck(
            name="floor_weight",
            configured=_cfg_floor_weight,
            observed=_obs_floor_weight,
            agree=_agree_floor_weight,
        ),
        KnobCheck(
            name="resume",
            configured=_cfg_resume,
            observed=_obs_resume,
            agree=_agree_resume,
        ),
        KnobCheck(
            name="observer",
            configured=_cfg_observer,
            observed=_obs_observer,
            agree=_agree_observer,
            requires_snapshot=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_knob_checks(
    ctx: KnobContext,
    checks: List[KnobCheck],
) -> List[KnobResult]:
    """Run all knob checks against the context.

    Skips snapshot-dependent knobs when no snapshot is available.
    """
    results: List[KnobResult] = []
    for check in checks:
        if check.requires_snapshot and ctx.snapshot_dir is None:
            results.append(KnobResult(
                name=check.name,
                configured_value=check.configured(ctx),
                observed_value=None,
                passed=False,
                note="需快照",
                skipped=True,
            ))
            continue
        try:
            cfg_val = check.configured(ctx)
        except Exception as e:
            results.append(KnobResult(
                name=check.name,
                configured_value=f"错误: {e}",
                observed_value=None,
                passed=False,
                note="configured() 异常",
            ))
            continue
        try:
            obs_val = check.observed(ctx)
        except Exception as e:
            results.append(KnobResult(
                name=check.name,
                configured_value=cfg_val,
                observed_value=f"错误: {e}",
                passed=False,
                note="observed() 异常",
            ))
            continue
        try:
            passed = check.agree(cfg_val, obs_val)
        except Exception:
            passed = False
        results.append(KnobResult(
            name=check.name,
            configured_value=cfg_val,
            observed_value=obs_val,
            passed=passed,
        ))
    return results


def render_knob_table(results: List[KnobResult], update: Optional[int] = None) -> str:
    """Render knob results as a human-readable table.

    Format per DEBUG_GUIDE.md §3.8::
        旋钮                配置值    数据通路实测              判定
        explore_factor      相位相关   σ 比值 1.56（期望 1.5–1.7）   ✓ 已生效
    """
    lines: List[str] = []
    if update is not None:
        lines.append(f"干预验证 @ update {update}")
    else:
        lines.append("干预验证")
    lines.append("")
    # Column widths
    name_w = 20
    cfg_w = 18
    obs_w = 30
    header = (f"{'旋钮':<{name_w}} {'配置值':<{cfg_w}} {'数据通路实测':<{obs_w}} 判定")
    lines.append(header)
    for r in results:
        if r.skipped:
            mark = "—"
            note = r.note
        elif r.passed:
            mark = "✓"
            note = "已生效"
        else:
            mark = "✗"
            note = r.note or "未生效"
        cfg_str = _fmt_value(r.configured_value)
        obs_str = _fmt_value(r.observed_value)
        line = f"{r.name:<{name_w}} {cfg_str:<{cfg_w}} {obs_str:<{obs_w}} {mark} {note}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_value(v: Any) -> str:
    """Format a knob value for display."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 0.01 and v != 0:
            return f"{v:.6f}"
        return f"{v:.4f}"
    if isinstance(v, list):
        return str([round(x, 4) if isinstance(x, float) else x for x in v])
    return str(v)
