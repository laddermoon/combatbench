"""``whatif`` — offline counterfactual tool (S4).

S4: Re-runs ``ppo_update`` on a captured snapshot with one or more
parameter overrides applied, compares the counterfactual update against
the baseline replay (gradient direction cosine, gradient magnitude,
combined-advantage cosine, influence-share deltas, leg-joint gradient
share), and classifies the change as below or above the S6 noise band.

This is a **falsification tool** — it can reliably reject ineffective
changes (gradient direction change < noise ⇒ the change cannot beat
seed noise in a single update) but cannot prove a change will train
successfully.  It only sees the marginal effect of one update, not the
accumulated dynamics of many updates.  See ``DEBUG_GUIDE.md`` §3.5.

Depends on:
- S2 (``replay_snapshot`` / ``verify_against_log``) for the offline
  re-run and baseline self-verification.
- S6 (``NoiseBand`` / ``load_noise_band``) for the noise-band
  significance threshold.

The override API is experiment-owned (P5): experiments declare
overridable parameters via :meth:`ExperimentPPO.whatif_params` and
implement :meth:`ExperimentPPO.apply_whatif_overrides`.  Generic code
parses ``--set``/``--sweep``, validates keys against the declaration,
parses values to the declared type, and calls the hook on a fresh
experiment instance before re-running the standard replay path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from baseline.framework.ppo.experiment import UpdateStats

from .compare import NoiseBand, load_noise_band
from .replay import (
    ReplayResult,
    VerificationResult,
    _load_experiment,
    _load_manifest,
    replay_snapshot,
    verify_against_log,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Humanoid21 leg-joint action-dim indices (from CONTROLLED_JOINTS):
#   3 hip_x_right,  4 hip_z_right,  5 hip_y_right,
#   6 knee_right,   7 ankle_y_right, 8 ankle_x_right,
#   9 hip_x_left,  10 hip_z_left,  11 hip_y_left,
#  12 knee_left,   13 ankle_y_left, 14 ankle_x_left
DEFAULT_LEG_ACTION_DIMS: Tuple[int, ...] = tuple(range(3, 15))


# ---------------------------------------------------------------------------
# Parameter declaration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WhatifParam:
    """One overridable parameter declared by an experiment.

    Attributes:
        name: Parameter name as it appears in ``--set``/``--sweep``.
        type: Python type for value parsing (``bool``, ``int``,
            ``float``, or ``str``).
        description: Human-readable description shown in error messages
            and the param listing.
        requires_rebuild: ``True`` if applying this parameter requires
            re-running ``build_trajectories`` + ``PPOBuffer`` (e.g.
            reward computation, ``actor_weight`` masks).  ``False`` if
            it only affects ``ppo_update``-time inputs that are read
            inside ``replay_snapshot`` (e.g. the exploration spec via
            ``experiment.exploration(update)``).  This flag is
            informational — ``whatif`` always re-runs the full replay
            path, so the rebuild happens regardless.  It exists so
            experiments and users understand which overrides are
            "cheap" (post-buffer) versus "expensive" (re-build).
    """
    name: str
    type: type
    description: str = ""
    requires_rebuild: bool = True


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WhatifVariant:
    """One counterfactual (or baseline) run result."""
    overrides: Dict[str, Any]
    stats: UpdateStats
    full_grad: Optional[np.ndarray] = None
    """Flattened actor gradient (epoch 0, minibatch 0), or None if
    ``--full-grad`` was not captured."""
    full_grad_param_names: Optional[List[str]] = None
    replay_dir: Optional[Path] = None
    """Where per-variant .npz outputs were written."""


@dataclass
class WhatifComparison:
    """Baseline vs one variant comparison."""
    overrides: Dict[str, Any]
    # Gradient direction (cosine of full flattened actor grad).
    # None if --full-grad not captured for either side.
    grad_cosine: Optional[float] = None
    grad_norm_ratio: Optional[float] = None
    # Combined advantage cosine (from combine.npz combined_adv).
    combined_adv_cosine: Optional[float] = None
    # Per-channel influence share.
    influence_share_baseline: Dict[str, float] = field(default_factory=dict)
    influence_share_variant: Dict[str, float] = field(default_factory=dict)
    # Leg-joint gradient share (action_dim_grad_norms subset).
    leg_grad_share_baseline: Optional[float] = None
    leg_grad_share_variant: Optional[float] = None
    # Verdict inputs.
    grad_change_pct: Optional[float] = None
    """1 - grad_cosine, expressed as a percentage. None if grad_cosine
    is None."""
    noise_std_grad: Optional[float] = None
    """Noise-band std for the gradient-proxy metric, or None if no
    noise band / metric not in band."""
    verdict: str = "no_noise_band"
    """One of: ``incomparable``, ``no_noise_band``, ``below_noise``,
    ``above_noise``."""


@dataclass
class WhatifReport:
    """Full whatif result."""
    snapshot_dir: Path
    update: int
    experiment_name: str
    episodes_mode: str
    baseline: WhatifVariant
    variants: List[WhatifVariant] = field(default_factory=list)
    comparisons: List[WhatifComparison] = field(default_factory=list)
    noise_band: Optional[NoiseBand] = None
    noise_band_path: Optional[Path] = None
    baseline_verification: Optional[VerificationResult] = None
    git_commit: Optional[str] = None
    is_sweep: bool = False
    sweep_key: Optional[str] = None
    full_grad_captured: bool = False
    whatif_dir: Optional[Path] = None
    """Where report.json / report.txt were written."""


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

_TRUE_STRINGS = {"true", "yes", "1", "on"}
_FALSE_STRINGS = {"false", "no", "0", "off"}


def parse_value(raw: str, target_type: type) -> Any:
    """Parse a string value to the declared type.

    Args:
        raw: Raw string from the CLI.
        target_type: One of ``bool``, ``int``, ``float``, ``str``.

    Returns:
        The parsed value.

    Raises:
        ValueError: If the string cannot be parsed to the type.
    """
    if target_type is bool:
        low = raw.strip().lower()
        if low in _TRUE_STRINGS:
            return True
        if low in _FALSE_STRINGS:
            return False
        raise ValueError(
            f"cannot parse {raw!r} as bool (expected one of "
            f"{sorted(_TRUE_STRINGS | _FALSE_STRINGS)})"
        )
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    if target_type is str:
        return raw
    raise ValueError(
        f"unsupported whatif param type {target_type!r}; "
        f"expected bool/int/float/str"
    )


def parse_set_args(
    set_pairs: List[str],
    declared: Dict[str, WhatifParam],
) -> Dict[str, Any]:
    """Parse a list of ``key=value`` strings into a validated override dict.

    Args:
        set_pairs: List of ``"key=value"`` strings from ``--set``.
        declared: Dict of declared params from
            :meth:`ExperimentPPO.whatif_params`.

    Returns:
        Dict mapping param name to parsed value.

    Raises:
        ValueError: On unknown key, malformed pair, or type parse
            failure.  The error message lists declared params when the
            key is unknown.
    """
    if not declared:
        raise ValueError(
            "this experiment declares no whatif params. "
            "Override whatif_params() and apply_whatif_overrides() to "
            "enable `debug.py whatif`."
        )
    overrides: Dict[str, Any] = {}
    for pair in set_pairs:
        if "=" not in pair:
            raise ValueError(
                f"malformed --set pair {pair!r}; expected 'key=value'"
            )
        key, _, raw = pair.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise ValueError(
                f"malformed --set pair {pair!r}; empty key"
            )
        if key not in declared:
            available = ", ".join(sorted(declared.keys()))
            raise ValueError(
                f"unknown whatif param {key!r}. "
                f"Declared params: [{available}]"
            )
        param = declared[key]
        overrides[key] = parse_value(raw, param.type)
    return overrides


def parse_sweep_arg(
    sweep: str,
    declared: Dict[str, WhatifParam],
) -> Tuple[str, List[Any]]:
    """Parse a ``key=v1,v2,...`` sweep string.

    Args:
        sweep: The ``--sweep`` argument string.
        declared: Dict of declared params.

    Returns:
        ``(key, [v1, v2, ...])`` with values parsed to the declared
        type.

    Raises:
        ValueError: On unknown key, malformed string, empty value list,
            or type parse failure.
    """
    if not declared:
        raise ValueError(
            "this experiment declares no whatif params. "
            "Override whatif_params() and apply_whatif_overrides() to "
            "enable `debug.py whatif`."
        )
    if "=" not in sweep:
        raise ValueError(
            f"malformed --sweep {sweep!r}; expected 'key=v1,v2,...'"
        )
    key, _, raw = sweep.partition("=")
    key = key.strip()
    raw = raw.strip()
    if not key:
        raise ValueError(f"malformed --sweep {sweep!r}; empty key")
    if key not in declared:
        available = ", ".join(sorted(declared.keys()))
        raise ValueError(
            f"unknown whatif param {key!r}. "
            f"Declared params: [{available}]"
        )
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(
            f"malformed --sweep {sweep!r}; empty value list"
        )
    param = declared[key]
    values = [parse_value(p, param.type) for p in parts]
    return key, values


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------

def _load_stats_json(replay_dir: Path) -> Dict[str, Any]:
    """Load ``stats.json`` from a replay directory."""
    stats_path = replay_dir / "stats.json"
    if not stats_path.exists():
        return {}
    with open(stats_path) as f:
        return json.load(f)


def _load_combine_adv(replay_dir: Path) -> Optional[np.ndarray]:
    """Load ``combined_adv`` from ``combine.npz`` if present."""
    npz_path = replay_dir / "combine.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    if "combined_adv" in data:
        return np.asarray(data["combined_adv"], dtype=np.float64)
    return None


def _load_full_grad(replay_dir: Path) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Load ``full_grad`` + param names from ``update.npz`` if present."""
    npz_path = replay_dir / "update.npz"
    if not npz_path.exists():
        return None, None
    data = np.load(npz_path, allow_pickle=True)
    grad = data.get("full_grad")
    names = data.get("full_grad_param_names")
    grad_arr = np.asarray(grad, dtype=np.float64) if grad is not None else None
    names_list = (
        [str(n) for n in names.tolist()] if names is not None else None
    )
    return grad_arr, names_list


def _cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Cosine similarity, None if either vector is all-zero or empty."""
    if a is None or b is None:
        return None
    if a.size == 0 or b.size == 0:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def _influence_shares(stats_json: Dict[str, Any]) -> Dict[str, float]:
    """Extract ``influence_share_<channel>`` keys from a stats dict."""
    out: Dict[str, float] = {}
    for k, v in stats_json.items():
        if k.startswith("influence_share_"):
            out[k[len("influence_share_"):]] = float(v)
    return out


def _grad_dims(stats_json: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract ``grad_dim_NN`` keys as an array, or None if absent."""
    dims: List[float] = []
    i = 0
    while f"grad_dim_{i:02d}" in stats_json:
        dims.append(float(stats_json[f"grad_dim_{i:02d}"]))
        i += 1
    if not dims:
        return None
    return np.asarray(dims, dtype=np.float64)


def _leg_grad_share(
    grad_dims: Optional[np.ndarray],
    leg_dims: Tuple[int, ...],
) -> Optional[float]:
    """Fraction of total gradient norm mass on leg joints."""
    if grad_dims is None or grad_dims.size == 0:
        return None
    # Filter out leg dims that are out of bounds (e.g. mock experiments
    # with fewer action dims than the humanoid21 default).
    valid_leg_dims = tuple(d for d in leg_dims if d < grad_dims.size)
    if not valid_leg_dims:
        return None
    total = float(grad_dims.sum())
    if total < 1e-12:
        return None
    leg = float(grad_dims[list(valid_leg_dims)].sum())
    return leg / total


def _grad_norm(full_grad: Optional[np.ndarray]) -> Optional[float]:
    if full_grad is None or full_grad.size == 0:
        return None
    return float(np.linalg.norm(full_grad))


# ---------------------------------------------------------------------------
# Noise-band verdict
# ---------------------------------------------------------------------------

# Metrics from the noise band that proxy for "gradient-direction change".
# The noise band is computed from training-run stats, not from gradient
# cosine directly, so we use approx_kl / grad_norm_actor as proxies.
_GRAD_PROXY_METRICS: Tuple[str, ...] = ("approx_kl", "grad_norm_actor")


def _noise_std_for_grad(noise_band: Optional[NoiseBand]) -> Optional[float]:
    """Pick a representative noise std for the gradient-direction verdict.

    Tries ``approx_kl`` first (most directly tied to policy update
    magnitude), then ``grad_norm_actor``.  Returns None if neither is
    in the band.
    """
    if noise_band is None:
        return None
    for metric in _GRAD_PROXY_METRICS:
        std = noise_band.std(metric)
        if std is not None and std > 0:
            return std
    return None


# ---------------------------------------------------------------------------
# Core: run one variant replay
# ---------------------------------------------------------------------------

def _run_variant(
    snapshot_dir: Path,
    overrides: Dict[str, Any],
    *,
    device: Optional[torch.device],
    full_grad: bool,
    out_subdir: str,
    experiment_factory: Optional[Callable[[], ExperimentPPO]] = None,
) -> WhatifVariant:
    """Reconstruct a fresh experiment, apply overrides, run replay.

    Args:
        experiment_factory: Optional callable that returns a fresh
            experiment instance.  When None, the experiment is
            reconstructed via the registry (``_load_experiment``).
            Used by tests to inject mock experiments.
    """
    if experiment_factory is not None:
        experiment = experiment_factory()
    else:
        manifest = _load_manifest(snapshot_dir)
        experiment = _load_experiment(snapshot_dir, manifest)
    experiment.apply_whatif_overrides(overrides)
    result = replay_snapshot(
        snapshot_dir,
        device=device,
        experiment=experiment,
        include_full_grad_override=full_grad or None,
        out_subdir=out_subdir,
    )
    full_grad_arr, grad_names = _load_full_grad(result.replay_dir)
    return WhatifVariant(
        overrides=dict(overrides),
        stats=result.stats,
        full_grad=full_grad_arr,
        full_grad_param_names=grad_names,
        replay_dir=result.replay_dir,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def whatif(
    snapshot_dir: Path,
    *,
    overrides: Optional[Dict[str, Any]] = None,
    sweep_key: Optional[str] = None,
    sweep_values: Optional[List[Any]] = None,
    noise_band_path: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    full_grad: bool = False,
    device: Optional[torch.device] = None,
    leg_action_dims: Optional[Tuple[int, ...]] = None,
    experiment_factory: Optional[Callable[[], ExperimentPPO]] = None,
) -> WhatifReport:
    """Run an offline counterfactual comparison on a snapshot.

    Either ``overrides`` (for ``--set``) or ``sweep_key`` + ``sweep_values``
    (for ``--sweep``) must be provided; they are mutually exclusive.

    Args:
        snapshot_dir: Path to ``<run_dir>/debug/u{u:05d}/``.
        overrides: Dict of param name → parsed value (for ``--set``).
        sweep_key: Param name to sweep (for ``--sweep``).
        sweep_values: List of parsed values to sweep over.
        noise_band_path: Optional path to an S6 noise-band JSON for
            significance classification.
        run_dir: Optional training run directory for baseline
            self-verification via :func:`verify_against_log`.
        full_grad: Force full-grad capture for gradient-direction cosine.
        device: Torch device (default: CPU).
        leg_action_dims: Action-dim indices counted as "leg" joints for
            the leg-grad-share metric. Defaults to humanoid21 leg joints
            (indices 3–14).
        experiment_factory: Optional callable that returns a fresh
            experiment instance.  When None, the experiment is
            reconstructed via the registry.  Used by tests to inject
            mock experiments.

    Returns:
        :class:`WhatifReport`.
    """
    snapshot_dir = Path(snapshot_dir)
    if leg_action_dims is None:
        leg_action_dims = DEFAULT_LEG_ACTION_DIMS

    is_sweep = sweep_key is not None
    if is_sweep:
        if overrides:
            raise ValueError(
                "whatif: --set and --sweep are mutually exclusive"
            )
        if not sweep_values:
            raise ValueError("whatif: --sweep requires at least one value")
    else:
        if not overrides:
            raise ValueError(
                "whatif: must provide either overrides or sweep_key/values"
            )

    manifest = _load_manifest(snapshot_dir)
    update = manifest["update"]
    experiment_name = manifest.get("experiment_name", "?")
    episodes_mode = manifest.get("episodes_mode", "subset")
    git_commit = manifest.get("git_commit")

    # --- Baseline replay (no overrides) ---
    # If replay/ already exists, reuse it; otherwise run it now.
    baseline_replay_dir = snapshot_dir / "replay"
    if not (baseline_replay_dir / "stats.json").exists():
        replay_snapshot(
            snapshot_dir,
            device=device,
            include_full_grad_override=full_grad or None,
        )
    baseline_stats_json = _load_stats_json(baseline_replay_dir)
    baseline_adv = _load_combine_adv(baseline_replay_dir)
    baseline_grad, baseline_grad_names = _load_full_grad(baseline_replay_dir)
    # Reconstruct a baseline UpdateStats-like object from stats.json for
    # the report.  We don't need the full typed object — only the
    # log-dict fields are used by the comparison.
    baseline_variant = WhatifVariant(
        overrides={},
        stats=UpdateStats.empty(()),  # placeholder; metrics read from json
        full_grad=baseline_grad,
        full_grad_param_names=baseline_grad_names,
        replay_dir=baseline_replay_dir,
    )

    # --- Baseline verification ---
    baseline_verification: Optional[VerificationResult] = None
    if run_dir is not None and episodes_mode == "all":
        try:
            baseline_verification = verify_against_log(snapshot_dir, run_dir)
        except FileNotFoundError:
            baseline_verification = None
    comparable = (
        episodes_mode == "all"
        and (
            baseline_verification is None
            or baseline_verification.verdict == "pass"
        )
    )
    # If --run-dir was given but verification failed, mark incomparable.
    if (
        run_dir is not None
        and episodes_mode == "all"
        and baseline_verification is not None
        and baseline_verification.verdict != "pass"
    ):
        comparable = False

    # --- Noise band ---
    noise_band: Optional[NoiseBand] = None
    if noise_band_path is not None:
        noise_band = load_noise_band(noise_band_path)
    noise_std_grad = _noise_std_for_grad(noise_band)

    # --- Build override list ---
    if is_sweep:
        override_list = [{sweep_key: v} for v in sweep_values]
    else:
        override_list = [dict(overrides)]

    # --- Run variants ---
    variants: List[WhatifVariant] = []
    for i, ov in enumerate(override_list):
        out_subdir = f"whatif/variant_{i:02d}"
        variant = _run_variant(
            snapshot_dir,
            ov,
            device=device,
            full_grad=full_grad,
            out_subdir=out_subdir,
            experiment_factory=experiment_factory,
        )
        variants.append(variant)

    # --- Compare ---
    comparisons: List[WhatifComparison] = []
    baseline_influence = _influence_shares(baseline_stats_json)
    baseline_grad_dims = _grad_dims(baseline_stats_json)
    baseline_leg_share = _leg_grad_share(baseline_grad_dims, leg_action_dims)
    baseline_grad_norm = _grad_norm(baseline_grad)

    for variant in variants:
        variant_stats_json = _load_stats_json(variant.replay_dir)
        variant_adv = _load_combine_adv(variant.replay_dir)
        variant_influence = _influence_shares(variant_stats_json)
        variant_grad_dims = _grad_dims(variant_stats_json)
        variant_leg_share = _leg_grad_share(
            variant_grad_dims, leg_action_dims,
        )
        variant_grad_norm = _grad_norm(variant.full_grad)

        grad_cos = _cosine(baseline_grad, variant.full_grad)
        adv_cos = _cosine(baseline_adv, variant_adv)
        grad_change_pct = (
            (1.0 - grad_cos) * 100.0 if grad_cos is not None else None
        )
        grad_norm_ratio = (
            variant_grad_norm / baseline_grad_norm
            if (variant_grad_norm is not None and baseline_grad_norm
                and baseline_grad_norm > 1e-12)
            else None
        )

        # Verdict
        if not comparable:
            verdict = "incomparable"
        elif noise_std_grad is None or grad_change_pct is None:
            verdict = "no_noise_band"
        elif grad_change_pct < noise_std_grad * 100.0:
            verdict = "below_noise"
        else:
            verdict = "above_noise"

        comparisons.append(WhatifComparison(
            overrides=variant.overrides,
            grad_cosine=grad_cos,
            grad_norm_ratio=grad_norm_ratio,
            combined_adv_cosine=adv_cos,
            influence_share_baseline=baseline_influence,
            influence_share_variant=variant_influence,
            leg_grad_share_baseline=baseline_leg_share,
            leg_grad_share_variant=variant_leg_share,
            grad_change_pct=grad_change_pct,
            noise_std_grad=noise_std_grad,
            verdict=verdict,
        ))

    full_grad_captured = baseline_grad is not None and all(
        v.full_grad is not None for v in variants
    )

    return WhatifReport(
        snapshot_dir=snapshot_dir,
        update=update,
        experiment_name=experiment_name,
        episodes_mode=episodes_mode,
        baseline=baseline_variant,
        variants=variants,
        comparisons=comparisons,
        noise_band=noise_band,
        noise_band_path=noise_band_path,
        baseline_verification=baseline_verification,
        git_commit=git_commit,
        is_sweep=is_sweep,
        sweep_key=sweep_key if is_sweep else None,
        full_grad_captured=full_grad_captured,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_report(report: WhatifReport, whatif_dir: Path) -> None:
    """Persist ``report.json`` + ``report.txt`` under ``whatif_dir``."""
    whatif_dir = Path(whatif_dir)
    whatif_dir.mkdir(parents=True, exist_ok=True)

    # --- report.json ---
    payload: Dict[str, Any] = {
        "snapshot_dir": str(report.snapshot_dir),
        "update": report.update,
        "experiment_name": report.experiment_name,
        "episodes_mode": report.episodes_mode,
        "git_commit": report.git_commit,
        "is_sweep": report.is_sweep,
        "sweep_key": report.sweep_key,
        "full_grad_captured": report.full_grad_captured,
        "noise_band_path": (
            str(report.noise_band_path) if report.noise_band_path else None
        ),
        "baseline_verification": (
            {
                "verdict": report.baseline_verification.verdict,
                "comparable": report.baseline_verification.comparable,
                "n_passed": report.baseline_verification.n_passed,
                "n_failed": report.baseline_verification.n_failed,
            }
            if report.baseline_verification is not None
            else None
        ),
        "comparisons": [
            {
                "overrides": c.overrides,
                "grad_cosine": c.grad_cosine,
                "grad_norm_ratio": c.grad_norm_ratio,
                "combined_adv_cosine": c.combined_adv_cosine,
                "influence_share_baseline": c.influence_share_baseline,
                "influence_share_variant": c.influence_share_variant,
                "leg_grad_share_baseline": c.leg_grad_share_baseline,
                "leg_grad_share_variant": c.leg_grad_share_variant,
                "grad_change_pct": c.grad_change_pct,
                "noise_std_grad": c.noise_std_grad,
                "verdict": c.verdict,
            }
            for c in report.comparisons
        ],
    }
    with open(whatif_dir / "report.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    # --- report.txt ---
    with open(whatif_dir / "report.txt", "w") as f:
        f.write(render_report(report))

    report.whatif_dir = whatif_dir


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_VERDICT_LABEL = {
    "incomparable": "不可比较（子集快照或基线未通过校验）",
    "no_noise_band": "无噪声带，仅显示差异不判定显著性",
    "below_noise": "效果无法与噪声区分。不建议花训练时间验证。",
    "above_noise": "改动效果超过噪声带，可进入候选池。",
}


def _fmt_overrides(overrides: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.1f}%"


def _fmt_cos(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}"


def _fmt_share(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def render_report(report: WhatifReport) -> str:
    """Render a :class:`WhatifReport` as a human-readable string.

    Format follows ``DEBUG_GUIDE.md`` §3.5 example.
    """
    lines: List[str] = []
    lines.append(f"反事实试算   @ update {report.update}")
    lines.append(f"  snapshot: {report.snapshot_dir}")
    lines.append(f"  experiment: {report.experiment_name}")
    lines.append(f"  episodes: {report.episodes_mode}")
    if report.git_commit:
        lines.append(f"  git: {report.git_commit[:12]}")
    if report.baseline_verification is not None:
        v = report.baseline_verification
        lines.append(
            f"  基线校验: {v.verdict.upper()} "
            f"(passed={v.n_passed} failed={v.n_failed})"
        )
    elif report.episodes_mode == "all":
        lines.append("  基线校验: 未运行（未提供 --run-dir，结论可信度降低）")
    lines.append("")

    if not report.full_grad_captured:
        lines.append(
            "  [note] 梯度方向不可用（快照未开 --full-grad）。"
            "用 --full-grad 强制捕获以获得梯度余弦。"
        )
        lines.append("")

    for comp in report.comparisons:
        label = _fmt_overrides(comp.overrides)
        lines.append(f"反事实试算   基线 vs {label}")
        lines.append("")

        # Influence-share table
        all_channels = sorted(
            set(comp.influence_share_baseline.keys())
            | set(comp.influence_share_variant.keys())
        )
        if all_channels:
            lines.append("影响份额")
            for ch in all_channels:
                b = comp.influence_share_baseline.get(ch)
                v = comp.influence_share_variant.get(ch)
                lines.append(
                    f"  {ch:<20s} {_fmt_share(b):<8s} → {_fmt_share(v)}"
                )
            lines.append("")

        # Cosine lines
        lines.append(
            f"combined_adv 与基线余弦相似度  {_fmt_cos(comp.combined_adv_cosine)}"
        )
        lines.append(
            f"梯度方向     与基线余弦相似度  {_fmt_cos(comp.grad_cosine)}"
        )
        if comp.grad_norm_ratio is not None:
            lines.append(
                f"梯度量级比   |variant|/|baseline|  {comp.grad_norm_ratio:.3f}"
            )
        lines.append(
            f"腿部关节梯度占比               "
            f"{_fmt_share(comp.leg_grad_share_baseline)}  →  "
            f"{_fmt_share(comp.leg_grad_share_variant)}"
        )
        lines.append("")

        # Verdict
        lines.append(f"判定：{_VERDICT_LABEL.get(comp.verdict, comp.verdict)}")
        if comp.grad_change_pct is not None and comp.noise_std_grad is not None:
            lines.append(
                f"      梯度方向变化 {comp.grad_change_pct:.2f}%，"
                f"噪声带 std {comp.noise_std_grad:.4f} "
                f"({comp.noise_std_grad * 100:.2f}%)."
            )
        lines.append("")

    # Sweep summary table
    if report.is_sweep and report.comparisons:
        lines.append("扫描汇总")
        header = (
            f"  {report.sweep_key:<24s} "
            f"{'grad_cos':<10s} "
            f"{'adv_cos':<10s} "
            f"{'leg_share':<10s} "
            f"{'verdict':<14s}"
        )
        lines.append(header)
        for comp in report.comparisons:
            v = comp.overrides.get(report.sweep_key)
            lines.append(
                f"  {str(v):<24s} "
                f"{_fmt_cos(comp.grad_cosine):<10s} "
                f"{_fmt_cos(comp.combined_adv_cosine):<10s} "
                f"{_fmt_share(comp.leg_grad_share_variant):<10s} "
                f"{comp.verdict:<14s}"
            )
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "DEFAULT_LEG_ACTION_DIMS",
    "WhatifParam",
    "WhatifVariant",
    "WhatifComparison",
    "WhatifReport",
    "parse_value",
    "parse_set_args",
    "parse_sweep_arg",
    "whatif",
    "save_report",
    "render_report",
]
