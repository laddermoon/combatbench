"""Probe execution infrastructure — run behavior probe suites.

S5: Probes answer ①环 of the nine-link signal chain: "did the target
behavior ever physically happen?"  They run deterministic rollouts
(``stochastic=False``) from fixed initial states (fixed seeds), so
results are comparable across updates.

This module provides:
- :func:`run_probe_suite` — run one suite on one policy, return pass rates
- :func:`list_available_updates` — find updates with policy exports
- :func:`load_policy_blueprint` — load a specific update's policy
- :func:`load_experiment_from_run` — reconstruct experiment from config.json

See ``DEBUG_GUIDE.md`` §3.7 ``probe`` and ``DESIGN_debug_system.md`` §5.2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from envs.framework.policy import PolicyBlueprint

from baseline.framework.ppo.experiment import (
    ExperimentPPO,
    ProbeSuite,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result of running one probe across all seeds/episodes."""
    probe_name: str
    passed: int
    total: int
    pass_rate: float
    best_episode_idx: int = -1
    """Index of the first passing episode, -1 if none passed."""


@dataclass
class SuiteResult:
    """Result of running one probe suite at one update."""
    suite_name: str
    update: int = -1
    """Update number of the policy that was probed.  -1 if not set."""
    probe_results: List[ProbeResult] = field(default_factory=list)
    n_episodes: int = 0
    """Number of episodes run (one per seed)."""


# ---------------------------------------------------------------------------
# Agent discovery
# ---------------------------------------------------------------------------

def _get_episode_agents(episode: Any) -> List[str]:
    """Return the list of agent_ids present in an Episode.

    Uses ``episode.observations`` keys as the source of truth — every
    agent that produced observations is a valid probe target.
    """
    obs = getattr(episode, "observations", None)
    if obs is None:
        return []
    return list(obs.keys())


# ---------------------------------------------------------------------------
# Core: run a probe suite
# ---------------------------------------------------------------------------

def run_probe_suite(
    experiment: ExperimentPPO,
    policy_bp: PolicyBlueprint,
    suite: ProbeSuite,
    n_workers: int = 1,
) -> SuiteResult:
    """Run a probe suite and return per-probe pass rates.

    Steps:
    1. Build deterministic jobs via ``experiment.build_probe_jobs(policy_bp, suite)``
    2. Collect episodes via :class:`ParallelRollouter`
    3. For each probe, evaluate predicate on each episode (all agents)
    4. Return pass rate per probe

    Args:
        experiment: The experiment instance (provides ``build_probe_jobs``).
        policy_bp: Deployable policy blueprint (from a specific update's
            ``policy_exports/uNNNNN/``).
        suite: The :class:`ProbeSuite` to run.
        n_workers: Number of parallel rollout workers.

    Returns:
        :class:`SuiteResult` with per-probe pass rates.
    """
    from baseline.framework.rollout.parallel_rollouter import ParallelRollouter

    jobs = experiment.build_probe_jobs(policy_bp, suite)
    with ParallelRollouter(num_workers=n_workers) as rollouter:
        episodes = rollouter.collect(jobs)

    probe_results: List[ProbeResult] = []
    for probe in suite.probes:
        passed = 0
        total = 0
        best_idx = -1
        for ep_idx, episode in enumerate(episodes):
            for agent_id in _get_episode_agents(episode):
                total += 1
                try:
                    if probe.predicate(episode, agent_id):
                        passed += 1
                        if best_idx < 0:
                            best_idx = ep_idx
                except Exception:
                    # Predicate errors count as failures, not crashes.
                    # A probe that can't evaluate (missing observer data,
                    # etc.) should not halt the entire suite.
                    pass
        probe_results.append(ProbeResult(
            probe_name=probe.name,
            passed=passed,
            total=total,
            pass_rate=passed / max(total, 1),
            best_episode_idx=best_idx,
        ))
    return SuiteResult(
        suite_name=suite.name,
        probe_results=probe_results,
        n_episodes=len(episodes),
    )


# ---------------------------------------------------------------------------
# Run-dir utilities
# ---------------------------------------------------------------------------

def list_available_updates(run_dir: Path) -> List[int]:
    """Find all update numbers that have a policy export.

    Scans ``run_dir/policy_exports/`` for directories matching
    ``uNNNNN`` (excluding ``uNNNNN_eval``).
    """
    exports_dir = run_dir / "policy_exports"
    if not exports_dir.exists():
        return []
    updates: List[int] = []
    for d in exports_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.endswith("_eval"):
            continue
        if not name.startswith("u"):
            continue
        try:
            updates.append(int(name[1:]))
        except ValueError:
            pass
    return sorted(updates)


def load_policy_blueprint(run_dir: Path, update: int) -> PolicyBlueprint:
    """Load the policy blueprint for a specific update.

    Reads ``run_dir/policy_exports/uNNNNN/policy_blueprint.yaml``.
    """
    bp_path = run_dir / "policy_exports" / f"u{update:05d}" / "policy_blueprint.yaml"
    if not bp_path.exists():
        raise FileNotFoundError(
            f"No policy export at {bp_path}. "
            f"Available updates: {list_available_updates(run_dir)}"
        )
    return PolicyBlueprint.load(bp_path)


def load_experiment_from_run(run_dir: Path) -> ExperimentPPO:
    """Reconstruct an experiment instance from ``run_dir/config.json``.

    Reads the ``experiment.name`` field and uses the PPO experiment
    registry to instantiate it.
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json at {config_path} — cannot reconstruct experiment"
        )
    with open(config_path) as f:
        config = json.load(f)
    exp_name = config.get("experiment", {}).get("name")
    if not exp_name:
        raise KeyError(
            f"config.json missing 'experiment.name' field: {config_path}"
        )
    from baseline.experiments_ppo import get_ppo_experiment
    return get_ppo_experiment(exp_name)
