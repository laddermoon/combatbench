"""Loop clock/UTD/tick-ring tests for the SAC-native training loop."""
from __future__ import annotations

from baseline.framework.sac.clocks import SACClockState
from baseline.framework.sac.diagnostics import SACTickRing
from baseline.framework.sac.loop import _planned_updates, _tick_metrics


def test_fractional_utd_preserves_credit() -> None:
    planned, dropped, credit = _planned_updates(
        transitions_added=3,
        replay_size=100,
        warmup_steps=1,
        batch_size=4,
        utd_ratio=0.5,
        max_updates=10,
        utd_credit=0.0,
    )
    assert planned == 1
    assert dropped == 0
    assert credit == 0.5
    planned, dropped, credit = _planned_updates(
        transitions_added=3,
        replay_size=100,
        warmup_steps=1,
        batch_size=4,
        utd_ratio=0.5,
        max_updates=10,
        utd_credit=credit,
    )
    assert planned == 2
    assert dropped == 0
    assert credit == 0.0


def test_utd_cap_drops_instead_of_rolling_over() -> None:
    planned, dropped, credit = _planned_updates(
        transitions_added=100,
        replay_size=1000,
        warmup_steps=1,
        batch_size=4,
        utd_ratio=2.0,
        max_updates=50,
        utd_credit=0.0,
    )
    assert planned == 50
    assert dropped == 150
    assert credit == 0.0


def test_utd_does_not_accumulate_before_warmup() -> None:
    planned, dropped, credit = _planned_updates(
        transitions_added=100,
        replay_size=20,
        warmup_steps=50,
        batch_size=4,
        utd_ratio=2.0,
        max_updates=50,
        utd_credit=1.0,
    )
    assert planned == 0
    assert dropped == 0
    assert credit == 1.0


def test_tick_ring_is_bounded_and_preserves_order() -> None:
    clocks = SACClockState()
    ring = SACTickRing(capacity=3)
    for i in range(5):
        clocks.tick_critic()
        ring.append(
            clocks=clocks,
            metrics={"critic.loss": float(i)},
            sample_ids=[i],
        )
    latest = ring.latest()
    assert len(ring) == 3
    assert [item["sample_ids"][0] for item in latest] == [2, 3, 4]
    assert latest[-1]["clocks"]["critic_tick"] == 5


def test_tick_metrics_use_sac_namespaces() -> None:
    metrics = _tick_metrics(
        {
            "critic_loss": 1.0,
            "actor_loss": 2.0,
            "alpha": 0.2,
            "alpha_loss": 0.1,
            "q1_mean_ra": 3.0,
            "q2_mean_ra": 5.0,
            "td_abs_mean_ra": 0.4,
            "q1_loss_ra": 0.2,
            "actor_weight_mean_ra": 0.8,
            "actor_weight_next_mean_ra": 0.7,
            "critic_updated_ra": 1.0,
            "critic_valid_weight_ra": 6.0,
            "target_pair1_frac": 0.25,
            "actor_pair1_frac": 0.75,
            "actor_valid_count": 8.0,
        },
        batch_size=8,
        replay_size=100,
        tau=0.005,
    )
    assert metrics["critic.loss"] == 1.0
    assert metrics["actor.loss"] == 2.0
    assert metrics["temperature.alpha"] == 0.2
    assert metrics["critic.q_mean"] == 4.0
    assert metrics["critic.td_mean"] == 0.4
    assert metrics["critic.q1_loss.ra"] == 0.2
    assert metrics["actor.weight.ra"] == 0.8
    assert metrics["actor.weight_next.ra"] == 0.7
    assert metrics["critic.updated.ra"] == 1.0
    assert metrics["critic.valid_weight.ra"] == 6.0
    assert metrics["target.pair1_frac"] == 0.25
    assert metrics["actor.pair1_frac"] == 0.75
    assert metrics["actor.valid_count"] == 8.0
