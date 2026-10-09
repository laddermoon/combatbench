"""Fake-environment end-to-end integration for the SAC phase-2 loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn as nn

from baseline.framework.sac.checkpoint import load_checkpoint_bundle
from baseline.framework.sac.experiment import (
    CommonParamsSAC,
    DataSource,
    ExperimentSAC,
    SACParams,
    SACRewardChannel,
)
from baseline.framework.sac.metrics import EVENTS_RELATIVE_PATH, load_events
from baseline.framework.sac.loop import train_sac
from baseline.framework.sac.tn_actor import TNActor
from baseline.framework.sac.tests.test_replay import _slice


@dataclass
class _FakeEpisode:
    num_frames: int = 4
    agent_termination_reason: dict = None

    def __post_init__(self) -> None:
        if self.agent_termination_reason is None:
            self.agent_termination_reason = {"robot_a": "timeout"}


class _FakeRollouter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def collect(self, jobs):
        return [_FakeEpisode() for _ in jobs]


class _FakeExperiment(ExperimentSAC):
    name = "fake_sac"

    def common_params(self) -> CommonParamsSAC:
        return CommonParamsSAC(
            name=self.name,
            learning_rate=1e-3,
            critic_learning_rate=1e-3,
            grad_clip_norm=1.0,
            episodes_per_update=1,
            max_env_steps=4,
            eval_interval=10_000,
            eval_episodes=1,
            video_eval_interval=0,
            rollout_workers=1,
            seed=31,
        )

    def sac_params(self) -> SACParams:
        return SACParams(
            replay_buffer_size=8,
            batch_size=2,
            warmup_steps=1,
            utd_ratio=1.0,
            max_grad_steps_per_round=8,
            tau=0.05,
            init_alpha=0.2,
            auto_alpha=True,
            target_entropy=-3.0,
            use_grad_norm=False,
        )

    def reward_channels(self):
        return (SACRewardChannel(name="r", gamma=0.95),)

    def build_actor(self, device):
        return TNActor(3, 2, arch="s01", hidden_dim=16, seed=31).to(device)

    def build_q_critic(self, channel_name: str, device) -> nn.Module:
        raise NotImplementedError("MultiHeadQCritic owns critic construction")

    def data_sources(self):
        return (DataSource(kind="self", agent="robot_a"),)

    def build_jobs(self, *args, **kwargs) -> List[Any]:
        return [object()]

    def build_slices(self, episodes):
        return [_slice(T=4, obs_dim=3, action_dim=2, channels=1)]

    def on_eval(self, episodes, env_step):
        return {"is_new_best": False, "info": {}, "stop_training": False}


def test_fake_env_to_replay_training_metrics_and_checkpoint(tmp_path) -> None:
    run_dir = tmp_path / "run"
    train_sac(
        _FakeExperiment(),
        run_dir=run_dir,
        rollouter=_FakeRollouter(),
    )

    events = load_events(run_dir / EVENTS_RELATIVE_PATH)
    event_types = [event.event_type for event in events]
    assert "config" in event_types
    assert "export" in event_types
    assert "round" in event_types
    assert "checkpoint" in event_types
    round_event = next(event for event in events if event.event_type == "round")
    assert round_event.clocks["collection_round"] == 1
    assert round_event.clocks["env_step"] == 4
    assert round_event.clocks["agent_transition"] == 4
    assert round_event.clocks["critic_tick"] == 4

    bundle = load_checkpoint_bundle(run_dir / "checkpoints" / "checkpoint_s00000004")
    assert bundle.resume_mode == "full"
    assert bundle.replay is not None and bundle.replay.size == 4
    assert bundle.runtime_state["clocks"]["critic_tick"] == 4


def test_unknown_actor_specification_fails_loudly() -> None:
    from baseline.experiments_sac.exp_sac_balance import SacBalance

    try:
        SacBalance(actor_arch="not_an_actor").build_actor(torch.device("cpu"))
    except ValueError as exc:
        assert "unsupported SAC actor_arch" in str(exc)
    else:
        raise AssertionError("unknown actor spec was silently accepted")


def test_loop_rejects_unsupported_declared_data_source(tmp_path) -> None:
    class BadExperiment(_FakeExperiment):
        def data_sources(self):
            return (DataSource(kind="opponent", agent="robot_b"),)

    try:
        train_sac(BadExperiment(), run_dir=tmp_path / "bad", rollouter=_FakeRollouter())
    except ValueError as exc:
        assert "only self data sources" in str(exc)
    else:
        raise AssertionError("unsupported data source was silently accepted")
