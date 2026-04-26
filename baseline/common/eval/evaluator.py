"""``PolicyEvaluator``: per-agent episode metrics with bootstrap CIs.

Design (see ``baseline/DESIGN.md`` §3.7):

  * Reuse the rollout stack: :class:`PolicyEvaluator` is a thin wrapper
    around :class:`baseline.common.rollout.RolloutCollector`. The
    collector handles episode iteration, observer binding, seeding;
    the evaluator just (a) forces ``deterministic=True`` /
    ``store_extras=False`` defaults appropriate for evaluation, and
    (b) folds a list of ``RolloutBatch`` into per-agent statistics.

  * Custom metrics are first-class: pass ``metric_fns={"name": f}``
    where ``f(RolloutBatch) -> float``. Defaults give "return" (sum
    of rewards) and "length" (num steps). Anything else (success rate,
    final pose error, energy, contact frequency) is one closure away.

  * Bootstrap CIs: every metric reports mean + std + percentile
    bootstrap CI. The bootstrap is non-parametric and trivially
    correct for unbiased statistics (mean), which is the only thing
    we actually report. We resample WITH replacement from the per-
    episode metric values ``B`` times and take the empirical
    ``[α/2, 1-α/2]`` quantiles of the bootstrap means. ``B=0``
    disables the CI computation (keep mean / std only).

  * Head-to-head: when two agents are captured, the matching pair of
    return arrays can be fed into :func:`head_to_head_winrate` to get
    self-play win rate + draw rate + bootstrap CI on the win rate.
    This stays a free function rather than a method so callers can
    apply it to any two arrays (e.g. a baseline run vs a candidate run).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from baseline.common.rollout import RolloutBatch, RolloutCollector
from baseline.common.rollout.collector import PolicyFactory, RuntimeFactory


MetricFn = Callable[[RolloutBatch], float]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@dataclass
class MetricStats:
    """Per-metric summary statistics.

    ``ci_lower`` / ``ci_upper`` are ``None`` when bootstrap was
    disabled (``bootstrap_samples=0``). Otherwise they are the
    percentile-bootstrap interval at level ``1 - alpha`` for the *mean*
    of this metric.
    """

    name: str
    mean: float
    std: float
    n: int
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    alpha: Optional[float] = None
    raw: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __repr__(self) -> str:
        if self.ci_lower is None:
            return f"{self.name}={self.mean:.4f}±{self.std:.4f} (n={self.n})"
        return (
            f"{self.name}={self.mean:.4f}±{self.std:.4f} "
            f"(n={self.n}, {(1 - self.alpha) * 100:.0f}% CI "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        )


@dataclass
class EvalReport:
    """Output of :meth:`PolicyEvaluator.evaluate`.

    ``per_agent`` maps ``agent_id -> {metric_name: MetricStats}``.
    Construction is keyed by capture order; missing metrics are simply
    absent from the inner dict. ``num_episodes`` is the actual count
    of episodes whose data made it into ``per_agent`` (may be < ``n``
    if the collector dropped some).
    """

    num_episodes: int
    per_agent: Dict[str, Dict[str, MetricStats]]

    def get(self, agent_id: str, metric: str) -> MetricStats:
        return self.per_agent[agent_id][metric]

    def __repr__(self) -> str:
        lines = [f"EvalReport(num_episodes={self.num_episodes})"]
        for agent, metrics in self.per_agent.items():
            lines.append(f"  [{agent}]")
            for name, stats in metrics.items():
                lines.append(f"    {stats}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default metrics
# ---------------------------------------------------------------------------
def _metric_return(b: RolloutBatch) -> float:
    return float(np.sum(b.rewards))


def _metric_length(b: RolloutBatch) -> float:
    return float(b.num_steps)


_DEFAULT_METRICS: Dict[str, MetricFn] = {
    "return": _metric_return,
    "length": _metric_length,
}


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: np.ndarray,
    *,
    n_samples: int = 1000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Percentile-bootstrap confidence interval at level ``1 - alpha``.

    ``values`` is a 1-D array. We draw ``n_samples`` resamples *with
    replacement* of the same length, compute ``statistic`` on each,
    and return the empirical ``α/2`` and ``1-α/2`` quantiles.

    Edge cases:
      * Length 0 → ``(nan, nan)``.
      * Length 1 → CI is degenerate ``(value, value)``.
      * Both are returned without raising; the caller decides whether
        to treat them as missing.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        v = float(statistic(arr))
        return (v, v)
    rng_ = rng if rng is not None else np.random.default_rng()
    # Vectorized resample: (n_samples, n_values) index array.
    idx = rng_.integers(0, arr.size, size=(n_samples, arr.size))
    if statistic is np.mean:
        # Fast path — vectorize without a python loop.
        boot_stats = arr[idx].mean(axis=1)
    else:
        boot_stats = np.array([statistic(arr[idx[i]]) for i in range(n_samples)])
    low = float(np.quantile(boot_stats, alpha / 2.0))
    high = float(np.quantile(boot_stats, 1.0 - alpha / 2.0))
    return (low, high)


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------
@dataclass
class HeadToHeadResult:
    win_rate: float
    draw_rate: float
    loss_rate: float
    n: int
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    alpha: Optional[float]


def head_to_head_winrate(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    *,
    bootstrap_samples: int = 0,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> HeadToHeadResult:
    """Per-episode win / draw / loss rate for agent A vs agent B.

    ``returns_a[i] > returns_b[i]`` counts as a win for A; equality
    counts as a draw. The bootstrap CI is on the win rate (the only
    statistic that's interesting to bound).
    """
    a = np.asarray(returns_a, dtype=np.float64).reshape(-1)
    b = np.asarray(returns_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(
            f"returns_a and returns_b must align; got shapes "
            f"{a.shape} vs {b.shape}."
        )
    if a.size == 0:
        return HeadToHeadResult(
            win_rate=float("nan"), draw_rate=float("nan"),
            loss_rate=float("nan"), n=0,
            ci_lower=None, ci_upper=None, alpha=None,
        )
    wins = (a > b).astype(np.float64)
    draws = (a == b).astype(np.float64)
    win_rate = float(wins.mean())
    draw_rate = float(draws.mean())
    loss_rate = float((a < b).mean())

    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    if bootstrap_samples > 0:
        rng = np.random.default_rng(seed)
        ci_lower, ci_upper = bootstrap_ci(
            wins, n_samples=bootstrap_samples, alpha=alpha, rng=rng,
        )

    return HeadToHeadResult(
        win_rate=win_rate,
        draw_rate=draw_rate,
        loss_rate=loss_rate,
        n=int(a.size),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha if bootstrap_samples > 0 else None,
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class PolicyEvaluator:
    """Run a fixed set of policies for ``n`` episodes and report stats.

    The constructor signature mirrors :class:`RolloutCollector`'s on
    purpose: an evaluator IS a collector with eval-flavored defaults
    and an aggregation step on top.

    Parameters
    ----------
    runtime_factory / policy_factories / capture_agents /
    obs_observer_template / reward_observer_template / reward_extractor /
    default_reward:
        Same as :class:`RolloutCollector`.
    deterministic:
        Forwarded to ``collector.collect(...)``. Default ``True`` —
        evaluation should not sample stochastic actions unless the
        caller explicitly says so.
    """

    def __init__(
        self,
        runtime_factory: RuntimeFactory,
        policy_factories: Mapping[str, PolicyFactory],
        *,
        capture_agents: Optional[Sequence[str]] = None,
        obs_observer_template: str = "{agent}_obs",
        reward_observer_template: Optional[str] = "{agent}_reward",
        reward_extractor: Optional[Callable[[Any], float]] = None,
        default_reward: float = 0.0,
        deterministic: bool = True,
    ) -> None:
        self._collector = RolloutCollector(
            runtime_factory=runtime_factory,
            policy_factories=policy_factories,
            capture_agents=capture_agents,
            obs_observer_template=obs_observer_template,
            reward_observer_template=reward_observer_template,
            reward_extractor=reward_extractor,
            default_reward=default_reward,
            store_extras=False,  # eval doesn't need log_probs / values
        )
        self._deterministic = bool(deterministic)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        n: int,
        *,
        base_seed: Optional[int] = None,
        options_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
        state_dicts: Optional[Mapping[str, Mapping[str, Any]]] = None,
        metric_fns: Optional[Mapping[str, MetricFn]] = None,
        bootstrap_samples: int = 0,
        bootstrap_alpha: float = 0.05,
        bootstrap_seed: Optional[int] = None,
    ) -> EvalReport:
        """Run ``n`` episodes and return per-agent :class:`EvalReport`.

        ``metric_fns`` is ``{name: callable(RolloutBatch) -> float}``;
        unioned with the defaults (``"return"``, ``"length"``). Pass
        e.g. ``{"return": custom_return_fn}`` to override a default.

        ``bootstrap_samples=0`` (default) skips CI computation and
        returns mean / std only — fast path for sanity checks.
        """
        metrics: Dict[str, MetricFn] = dict(_DEFAULT_METRICS)
        if metric_fns is not None:
            metrics.update(metric_fns)

        batches = self._collector.collect(
            n=n,
            base_seed=base_seed,
            options_fn=options_fn,
            deterministic=self._deterministic,
            state_dicts=state_dicts,
        )

        rng = np.random.default_rng(bootstrap_seed)
        per_agent: Dict[str, Dict[str, MetricStats]] = {}
        total_episodes = 0
        for agent_id, ep_batches in batches.items():
            total_episodes = max(total_episodes, len(ep_batches))
            metric_dict: Dict[str, MetricStats] = {}
            for name, fn in metrics.items():
                values = np.asarray(
                    [fn(b) for b in ep_batches], dtype=np.float64,
                )
                metric_dict[name] = self._summarize(
                    name=name,
                    values=values,
                    bootstrap_samples=bootstrap_samples,
                    alpha=bootstrap_alpha,
                    rng=rng,
                )
            per_agent[agent_id] = metric_dict

        return EvalReport(num_episodes=total_episodes, per_agent=per_agent)

    def close(self) -> None:
        self._collector.close()

    def __enter__(self) -> "PolicyEvaluator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _summarize(
        *,
        name: str,
        values: np.ndarray,
        bootstrap_samples: int,
        alpha: float,
        rng: np.random.Generator,
    ) -> MetricStats:
        n = int(values.size)
        if n == 0:
            return MetricStats(
                name=name, mean=float("nan"), std=float("nan"),
                n=0, raw=values,
            )
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        ci_lower: Optional[float] = None
        ci_upper: Optional[float] = None
        if bootstrap_samples > 0:
            ci_lower, ci_upper = bootstrap_ci(
                values, n_samples=bootstrap_samples, alpha=alpha, rng=rng,
            )
        return MetricStats(
            name=name, mean=mean, std=std, n=n,
            ci_lower=ci_lower, ci_upper=ci_upper,
            alpha=alpha if bootstrap_samples > 0 else None,
            raw=values,
        )
