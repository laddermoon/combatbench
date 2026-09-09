"""Metric verification infrastructure — compare current vs strict metrics.

S5: Metric verifiers answer ⑨环 of the nine-link signal chain: "is
the metric measuring what I think it is?"  They run deterministic eval
episodes and compare the production metric definition with a strict
alternative, plus a contact-jitter count to detect inflated metrics.

This module provides:
- :func:`verify_metric` — run eval episodes and compare current vs strict
- :func:`compute_contact_jitter` — count single-frame contact flips

See ``DEBUG_GUIDE.md`` §3.6 ``metric --verify`` and
``DESIGN_debug_system.md`` §5.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from envs.framework.policy import PolicyBlueprint

from baseline.framework.ppo.experiment import ExperimentPPO
from baseline.framework.rollout import extract_per_step_field


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetricVerification:
    """Result of verifying one metric at one update."""
    metric_name: str
    update: int = -1
    """Update number of the policy that was verified.  -1 if not set."""
    n_episodes: int = 0
    n_agents: int = 0
    current_values: List[float] = field(default_factory=list)
    """Per-agent current (production) metric values."""
    strict_values: List[float] = field(default_factory=list)
    """Per-agent strict metric values."""
    jitter_values: List[float] = field(default_factory=list)
    """Per-agent contact jitter counts."""
    current_mean: float = 0.0
    strict_mean: float = 0.0
    jitter_mean: float = 0.0
    verdict: str = "confirmed"
    """``"confirmed"`` or ``"falsified"``."""


# ---------------------------------------------------------------------------
# Agent discovery
# ---------------------------------------------------------------------------

def _get_episode_agents(episode: Any) -> List[str]:
    """Return the list of agent_ids present in an Episode."""
    obs = getattr(episode, "observations", None)
    if obs is None:
        return []
    return list(obs.keys())


# ---------------------------------------------------------------------------
# Contact jitter
# ---------------------------------------------------------------------------

def compute_contact_jitter(
    episode: Any, agent_id: str, experiment: ExperimentPPO,
) -> float:
    """Count single-frame contact flips for one agent.

    A "contact flip" is a frame where the contact state changes from
    the previous frame (T→F or F→T).  High jitter means the contact
    signal is noisy — the foot is bouncing on and off the ground
    rather than cleanly lifting or landing.

    This is the diagnostic that reveals when ``steps`` is counting
    contact bounces instead of true gait cycles.
    """
    # Find the foot observer key for this agent
    foot_key = _get_foot_key(experiment, agent_id)
    if foot_key is None:
        return 0.0
    T = episode.num_frames
    if T == 0:
        return 0.0
    contact_l = extract_per_step_field(
        episode.observer_outputs, foot_key, "left_foot_contact", T,
    )
    contact_r = extract_per_step_field(
        episode.observer_outputs, foot_key, "right_foot_contact", T,
    )
    if contact_l is None or contact_r is None:
        return 0.0
    contact_l = np.asarray(contact_l[:T], dtype=bool)
    contact_r = np.asarray(contact_r[:T], dtype=bool)
    # Count flips: sum of |diff| over frames
    flips_l = int(np.sum(np.abs(np.diff(contact_l.astype(np.int8)))))
    flips_r = int(np.sum(np.abs(np.diff(contact_r.astype(np.int8)))))
    return float(flips_l + flips_r)


def _get_foot_key(experiment: ExperimentPPO, agent_id: str) -> Optional[str]:
    """Get the foot observer key for an agent from the experiment.

    Uses duck typing: if the experiment has a ``_get_agent_observer_keys``
    method (StandupStepV3 does), use it.  Otherwise return None.
    """
    getter = getattr(experiment, "_get_agent_observer_keys", None)
    if getter is None:
        return None
    try:
        foot_key, _ = getter(agent_id)
        return foot_key
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Current metric computation
# ---------------------------------------------------------------------------

def _compute_current_metric(
    experiment: ExperimentPPO, metric_name: str,
    episode: Any, agent_id: str,
) -> float:
    """Compute the current (production) metric value for one agent.

    For ``steps``: delegates to ``experiment._check_stepping`` if
    available (StandupStepV3 has it).  Returns 1.0 if stepping was
    detected, 0.0 otherwise.
    """
    if metric_name == "steps":
        checker = getattr(experiment, "_check_stepping", None)
        if checker is None:
            return 0.0
        # _check_stepping needs (episode, foot_key, phi4stage_key, T)
        getter = getattr(experiment, "_get_agent_observer_keys", None)
        if getter is None:
            return 0.0
        try:
            foot_key, phi4stage_key = getter(agent_id)
        except (KeyError, TypeError):
            return 0.0
        T = episode.num_frames
        if T == 0:
            return 0.0
        try:
            return 1.0 if checker(episode, foot_key, phi4stage_key, T) else 0.0
        except Exception:
            return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Core: verify a metric
# ---------------------------------------------------------------------------

def verify_metric(
    experiment: ExperimentPPO,
    policy_bp: PolicyBlueprint,
    metric_name: str,
    n_episodes: int = 16,
    base_seed: int = 200_000,
    n_workers: int = 1,
) -> MetricVerification:
    """Run deterministic eval episodes and compare current vs strict metric.

    Steps:
    1. Build eval jobs via ``experiment.build_jobs(..., stochastic=False)``
    2. Collect episodes via :class:`ParallelRollouter`
    3. For each episode/agent:
       a. Compute strict metric via ``experiment.metric_verifiers()[name]``
       b. Compute current metric via :func:`_compute_current_metric`
       c. Compute contact jitter via :func:`compute_contact_jitter`
    4. Report comparison and verdict

    Verdict:
    - ``"falsified"``: strict_mean < current_mean * 0.5 (current metric
      is inflated by >50%)
    - ``"confirmed"``: otherwise

    Args:
        experiment: The experiment instance.
        policy_bp: Deployable policy blueprint.
        metric_name: Metric name to verify (must be in
            ``experiment.metric_verifiers()``).
        n_episodes: Number of eval episodes.
        base_seed: Base seed for eval jobs.
        n_workers: Number of parallel rollout workers.

    Returns:
        :class:`MetricVerification` with per-agent values and verdict.
    """
    verifiers = experiment.metric_verifiers()
    if metric_name not in verifiers:
        available = list(verifiers.keys())
        raise KeyError(
            f"Metric {metric_name!r} has no verifier. "
            f"Available: {available}"
        )
    strict_fn = verifiers[metric_name]

    from baseline.framework.rollout.parallel_rollouter import ParallelRollouter

    jobs = experiment.build_jobs(
        policy_bp, base_seed, n_episodes, stochastic=False,
    )
    with ParallelRollouter(num_workers=n_workers) as rollouter:
        episodes = rollouter.collect(jobs)

    current_values: List[float] = []
    strict_values: List[float] = []
    jitter_values: List[float] = []
    for episode in episodes:
        for agent_id in _get_episode_agents(episode):
            try:
                strict_values.append(float(strict_fn(episode, agent_id)))
            except Exception:
                strict_values.append(0.0)
            current_values.append(
                float(_compute_current_metric(
                    experiment, metric_name, episode, agent_id,
                ))
            )
            jitter_values.append(
                float(compute_contact_jitter(episode, agent_id, experiment))
            )

    current_mean = float(np.mean(current_values)) if current_values else 0.0
    strict_mean = float(np.mean(strict_values)) if strict_values else 0.0
    jitter_mean = float(np.mean(jitter_values)) if jitter_values else 0.0

    # Verdict: if strict_mean < current_mean * 0.5 → "falsified"
    if current_mean > 0 and strict_mean < current_mean * 0.5:
        verdict = "falsified"
    else:
        verdict = "confirmed"

    return MetricVerification(
        metric_name=metric_name,
        n_episodes=len(episodes),
        n_agents=len(current_values),
        current_values=current_values,
        strict_values=strict_values,
        jitter_values=jitter_values,
        current_mean=current_mean,
        strict_mean=strict_mean,
        jitter_mean=jitter_mean,
        verdict=verdict,
    )
