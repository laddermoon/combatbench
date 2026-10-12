"""R2-5: eval evidence — per-agent detail rows + best-policy meta."""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np

from baseline.framework.sac.experiment import CommonParamsSAC
from baseline.framework.sac.loop import train_sac
from baseline.framework.sac.tests.test_env_integration import (
    _FakeExperiment,
    _FakeRollouter,
)
from baseline.framework.sac.held_out_eval import _resolve_artifact_path
from baseline.experiments_sac.exp_sac_balance import SacBalance
from baseline.experiments_sac.exp_sac_standup import SacStandup


def _ep(episode_seed=42, episode_index=0, num_frames=4, **fields):
    return SimpleNamespace(
        num_frames=num_frames,
        episode_seed=episode_seed,
        episode_index=episode_index,
        observer_outputs=fields.get("observer_outputs", {}),
        agent_termination_reason=fields.get(
            "agent_termination_reason", {"robot_a": "", "robot_b": ""}
        ),
    )


def test_default_eval_episode_details_is_empty() -> None:
    assert _FakeExperiment().eval_episode_details([_ep()]) == []


def test_standup_eval_episode_details_match_on_eval() -> None:
    exp = SacStandup()
    episodes = [
        _ep(
            episode_seed=100,
            episode_index=0,
            observer_outputs={
                "standing_balance_a": {
                    "potential": np.array([0.1, 0.95, 0.4, 0.2]),
                },
                "standing_balance_b": {
                    "potential": np.array([0.3, 0.5, 0.6, 0.4]),
                },
            },
        ),
        _ep(
            episode_seed=101,
            episode_index=1,
            observer_outputs={
                "standing_balance_a": {
                    "potential": np.array([0.9, 0.9, 0.9, 0.9]),
                },
                "standing_balance_b": {
                    "potential": np.array([0.2, 0.3, 0.2, 0.1]),
                },
            },
        ),
    ]
    rows = exp.eval_episode_details(episodes)
    assert len(rows) == 4  # 2 episodes x 2 agents — the real denominator
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["episode_seed"], []).append(r)
    assert set(by_ep) == {100, 101}
    # max_pot and success flag are consistent with the source series
    a0 = next(r for r in rows if r["episode_seed"] == 100 and r["agent_id"] == "robot_a")
    assert a0["max_pot"] == 0.95
    assert a0["final_pot"] == 0.2
    assert a0["success"] is True
    b1 = next(r for r in rows if r["episode_seed"] == 101 and r["agent_id"] == "robot_b")
    assert b1["success"] is False
    # aggregation agrees with on_eval's denominator
    result = exp.on_eval(episodes, env_step=0)
    expected_success = sum(1 for r in rows if r["success"]) / len(rows)
    assert result["info"]["success"] == round(expected_success, 3)


def test_balance_eval_episode_details_match_on_eval() -> None:
    exp = SacBalance()
    episodes = [
        _ep(
            episode_seed=200,
            episode_index=0,
            agent_termination_reason={
                "robot_a": "imbalance",
                "robot_b": "timeout",
            },
        ),
        _ep(
            episode_seed=201,
            episode_index=1,
            agent_termination_reason={"robot_a": "", "robot_b": "timeout"},
        ),
    ]
    rows = exp.eval_episode_details(episodes)
    assert len(rows) == 4
    by_ep = {r["episode_seed"]: r for r in rows if r["agent_id"] == "robot_a"}
    assert by_ep[200]["survived"] is False
    assert by_ep[200]["termination_reason"] == "imbalance"
    assert by_ep[201]["survived"] is True
    result = exp.on_eval(episodes, env_step=0)
    expected_survived = float(sum(1 for r in rows if r["survived"]))
    assert result["info"]["survived"] == expected_survived


def test_resolve_artifact_path_prefers_run_relative(tmp_path) -> None:
    """A bare 'policy' arg must hit <run>/policy even when an unrelated
    'policy/' dir exists under the CWD (the repo has one — real bug)."""
    from pathlib import Path

    run_dir = tmp_path / "run"
    (run_dir / "policy").mkdir(parents=True)
    resolved = _resolve_artifact_path(Path("policy"), run_dir)
    assert resolved == run_dir / "policy"
    other = _resolve_artifact_path(Path("nowhere"), run_dir)
    assert other == run_dir / "nowhere"  # neither exists → run-relative
    abs_path = tmp_path / "abs" / "x"
    assert _resolve_artifact_path(abs_path, run_dir) == abs_path


def test_best_policy_export_writes_meta_json(tmp_path) -> None:
    """R2-W52: policy/ is no longer anonymous — the best export carries
    which eval produced it and the weight fingerprint."""

    class _BestExperiment(_FakeExperiment):
        def common_params(self) -> CommonParamsSAC:
            return dataclasses.replace(
                super().common_params(), eval_interval=2, eval_episodes=1
            )

        def on_eval(self, episodes, env_step):
            return {
                "is_new_best": True,
                "info": {"success": 0.75},
                "stop_training": False,
            }

    run_dir = tmp_path / "run"
    train_sac(_BestExperiment(), run_dir=run_dir, rollouter=_FakeRollouter())

    meta_path = run_dir / "policy" / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["kind"] == "best_of_run"
    assert meta["eval"]["success"] == 0.75
    assert isinstance(meta["env_step"], int) and meta["env_step"] > 0
    assert meta["policy_fingerprint"]
    assert meta["written_unix"] > 0
