"""``compare`` — cross-run comparison tool (S6).

S6: Compares two training runs metric-by-metric, optionally with a
noise band for significance testing.  "没有噪声带的对照结论一律不采信"
(DEBUG_GUIDE.md §3.9) — this tool enforces that discipline by marking
each difference as "✓ 显著" or "噪声内" based on a noise band.

The noise band comes from ``debug.py noise`` output (a JSON file with
per-metric mean ± std across multiple seeds).

See ``DEBUG_GUIDE.md`` §3.9 ``compare``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Log parsing (shared pattern)
# ---------------------------------------------------------------------------

_RAW_STATS_RE = re.compile(r"__RAW_STATS__\s*(\{.*\})")


def parse_all_entries(run_dir: Path) -> List[Dict[str, Any]]:
    """Parse all ``__RAW_STATS__`` entries from a run directory's log.

    Returns a list of full entry dicts (including ``update``,
    ``stats``, ``eval_info``, ``buffer_stats``, etc.), ordered by
    update number.
    """
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with open(log_path) as f:
        for line in f:
            m = _RAW_STATS_RE.search(line)
            if not m:
                continue
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            entries.append(data)
    return entries


def extract_metrics(
    entries: List[Dict[str, Any]],
    window: int = 1,
) -> Dict[str, float]:
    """Extract a flat metric dict from log entries.

    Pulls from both ``eval_info`` (eval metrics like max_pot, success)
    and ``stats`` (training metrics like uncertainty, approx_kl, EV,
    S0 aggregates like influence_share, dead_frame_ratio, grad_dim_NN).

    Args:
        entries: List of parsed ``__RAW_STATS__`` entries.
        window: Number of recent entries to average (default: 1 = latest).

    Returns:
        Dict mapping metric name to averaged float value.
    """
    if not entries:
        return {}
    recent = entries[-window:]
    metrics: Dict[str, float] = {}

    # Collect all numeric keys from stats and eval_info
    all_keys: set = set()
    for entry in recent:
        stats = entry.get("stats", {})
        all_keys.update(k for k, v in stats.items()
                        if isinstance(v, (int, float)))
        eval_info = entry.get("eval_info", {})
        all_keys.update(k for k, v in eval_info.items()
                        if isinstance(v, (int, float)))

    for key in sorted(all_keys):
        vals: List[float] = []
        for entry in recent:
            stats = entry.get("stats", {})
            if key in stats and isinstance(stats[key], (int, float)):
                vals.append(float(stats[key]))
                continue
            eval_info = entry.get("eval_info", {})
            if key in eval_info and isinstance(eval_info[key], (int, float)):
                vals.append(float(eval_info[key]))
        if vals:
            metrics[key] = sum(vals) / len(vals)

    return metrics


def parse_run_metrics(
    run_dir: Path,
    window: int = 1,
) -> Dict[str, float]:
    """Parse a run directory's log and extract averaged metrics.

    Args:
        run_dir: Training run directory.
        window: Number of recent updates to average (default: 1 = latest).

    Returns:
        Dict mapping metric name to averaged float value.
    """
    entries = parse_all_entries(run_dir)
    return extract_metrics(entries, window=window)


# ---------------------------------------------------------------------------
# Noise band
# ---------------------------------------------------------------------------

@dataclass
class NoiseBand:
    """Per-metric noise band from multi-seed runs."""
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """metric_name → {"mean": float, "std": float}"""
    n_seeds: int = 0
    n_updates: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NoiseBand":
        return cls(
            metrics=d.get("metrics", {}),
            n_seeds=d.get("n_seeds", 0),
            n_updates=d.get("n_updates", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "n_seeds": self.n_seeds,
            "n_updates": self.n_updates,
        }

    def std(self, metric: str) -> Optional[float]:
        """Return the std for a metric, or None if not in band."""
        m = self.metrics.get(metric)
        if m is None:
            return None
        return m.get("std")

    def mean(self, metric: str) -> Optional[float]:
        """Return the mean for a metric, or None if not in band."""
        m = self.metrics.get(metric)
        if m is None:
            return None
        return m.get("mean")


def load_noise_band(path: Path) -> NoiseBand:
    """Load a noise band from a JSON file produced by ``debug.py noise``."""
    with open(path) as f:
        data = json.load(f)
    return NoiseBand.from_dict(data)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class MetricComparison:
    """Comparison of one metric between two runs."""
    name: str
    value_a: Optional[float]
    value_b: Optional[float]
    diff: Optional[float]
    noise_std: Optional[float]
    significant: Optional[bool]
    """True if |diff| > 2*std (significant), False if within noise,
    None if no noise band available."""


@dataclass
class CompareReport:
    """Result of comparing two runs."""
    run_a: Path
    run_b: Path
    window: int
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    comparisons: List[MetricComparison] = field(default_factory=list)
    noise_band: Optional[NoiseBand] = None


def compare_runs(
    run_a: Path,
    run_b: Path,
    window: int = 1,
    noise_band: Optional[NoiseBand] = None,
) -> CompareReport:
    """Compare two training runs metric-by-metric.

    Args:
        run_a: First run directory.
        run_b: Second run directory.
        window: Number of recent updates to average (default: 1).
        noise_band: Optional NoiseBand for significance testing.

    Returns:
        CompareReport with per-metric comparisons.
    """
    metrics_a = parse_run_metrics(run_a, window=window)
    metrics_b = parse_run_metrics(run_b, window=window)

    all_keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
    comparisons: List[MetricComparison] = []
    for key in all_keys:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        diff = (vb - va) if (va is not None and vb is not None) else None
        noise_std = noise_band.std(key) if noise_band else None
        if diff is not None and noise_std is not None:
            significant = abs(diff) > 2 * noise_std
        else:
            significant = None
        comparisons.append(MetricComparison(
            name=key,
            value_a=va,
            value_b=vb,
            diff=diff,
            noise_std=noise_std,
            significant=significant,
        ))

    return CompareReport(
        run_a=Path(run_a),
        run_b=Path(run_b),
        window=window,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        comparisons=comparisons,
        noise_band=noise_band,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_compare_table(report: CompareReport) -> str:
    """Render a comparison report as a human-readable table.

    Format per DEBUG_GUIDE.md §3.9::
        指标            A          B          差异     噪声带    判定
        max_pot        0.999      0.999      +0.000   ±0.003   无差异
    """
    lines: List[str] = []
    lines.append(f"跨 run 对照   A: {report.run_a.name}")
    lines.append(f"              B: {report.run_b.name}")
    if report.noise_band is not None:
        lines.append(f"              （噪声基线: {report.noise_band.n_seeds} seeds "
                      f"× {report.noise_band.n_updates} updates）")
    lines.append("")

    # Column widths
    name_w = 25
    val_w = 12
    diff_w = 10
    noise_w = 10
    judge_w = 12

    header = (f"{'指标':<{name_w}} {'A':<{val_w}} {'B':<{val_w}} "
              f"{'差异':<{diff_w}} {'噪声带':<{noise_w}} 判定")
    lines.append(header)

    for c in report.comparisons:
        a_str = _fmt_metric(c.value_a)
        b_str = _fmt_metric(c.value_b)
        if c.diff is not None:
            diff_str = f"{c.diff:+.4f}"
        else:
            diff_str = "—"
        if c.noise_std is not None:
            noise_str = f"±{c.noise_std:.4f}"
        else:
            noise_str = "—"

        if c.significant is None:
            judge = "无噪声带"
        elif c.significant:
            judge = "✓ 显著"
        elif c.diff is not None and abs(c.diff) < 1e-6:
            judge = "无差异"
        else:
            judge = "噪声内"

        line = (f"{c.name:<{name_w}} {a_str:<{val_w}} {b_str:<{val_w}} "
                f"{diff_str:<{diff_w}} {noise_str:<{noise_w}} {judge}")
        lines.append(line)

    # Summary judgment
    lines.append("")
    sig_metrics = [c for c in report.comparisons if c.significant is True]
    if report.noise_band is not None and sig_metrics:
        names = [c.name for c in sig_metrics]
        lines.append(f"判定：B 在以下指标上显著不同: {names}")
    elif report.noise_band is not None:
        lines.append("判定：所有指标差异均在噪声带内。")
    else:
        lines.append("判定：无噪声带，仅显示差异不判定显著性。")

    return "\n".join(lines)


def _fmt_metric(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare(
    run_a: Path,
    run_b: Path,
    window: int = 1,
    noise_band_path: Optional[Path] = None,
) -> str:
    """Compare two runs and return the rendered report.

    Args:
        run_a: First run directory.
        run_b: Second run directory.
        window: Number of recent updates to average (default: 1).
        noise_band_path: Path to noise band JSON file (optional).

    Returns:
        Human-readable comparison report.
    """
    run_a = Path(run_a).resolve()
    run_b = Path(run_b).resolve()
    if not run_a.exists():
        raise FileNotFoundError(f"Run A does not exist: {run_a}")
    if not run_b.exists():
        raise FileNotFoundError(f"Run B does not exist: {run_b}")

    noise_band = None
    if noise_band_path is not None:
        noise_band_path = Path(noise_band_path).resolve()
        if not noise_band_path.exists():
            raise FileNotFoundError(f"Noise band file does not exist: {noise_band_path}")
        noise_band = load_noise_band(noise_band_path)

    report = compare_runs(run_a, run_b, window=window, noise_band=noise_band)
    return render_compare_table(report)
