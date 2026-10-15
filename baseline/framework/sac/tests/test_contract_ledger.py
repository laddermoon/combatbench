"""Tests for the WP1 contract ledger + diagnostic cuts."""
from __future__ import annotations

import numpy as np
import torch

from baseline.framework.sac.analysis import (
    candidate_twin_analysis,
    critic_fit,
    post_detail,
    recompute_targets,
    rollout_exploration,
    slice_verification,
    update_checks,
    verify_dump_contracts,
)
from baseline.framework.sac.clocks import SACClockState
from baseline.framework.sac.debugkit import (
    begin_critic_tick_dump,
    finish_critic_tick_dump,
)
from baseline.framework.sac.experiment import SACParams
from baseline.framework.sac.tests.test_trainer import _batch, _models
from baseline.framework.sac.trainer import sac_update_v2, trainer_state_dict
from baseline.experiments_sac.contract_verifiers import (
    get_slice_verifier,
    known_semantics,
)


def _make_dump(tmp_path, *, batch=None, sp=None):
    """One real critic_tick dump via the actual capture path."""
    actor, critic, channels = _models(C=2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-3)
    if batch is None:
        batch = _batch(B=16)
    if sp is None:
        sp = SACParams(use_grad_norm=False, reward_scale=2.0)
    clocks = SACClockState(collection_round=3, critic_tick=7)
    tmp = begin_critic_tick_dump(
        dump_root=tmp_path / "debug_dumps",
        critic_tick=8,
        clocks=clocks,
        batch=batch,
        trainer_pre_state=trainer_state_dict(
            actor, critic, actor_optimizer, log_alpha, alpha_optimizer,
        ),
        hypothesis="unit",
    )
    capture = {}
    stats = sac_update_v2(
        actor=actor, critic=critic, actor_optimizer=actor_optimizer,
        log_alpha=log_alpha, alpha_optimizer=alpha_optimizer,
        batch=batch, channels=channels, sp=sp,
        grad_clip_norm=1.0, device=torch.device("cpu"), capture=capture,
    )
    return finish_critic_tick_dump(
        tmp, forward_capture=capture,
        trainer_post_state=trainer_state_dict(
            actor, critic, actor_optimizer, log_alpha, alpha_optimizer,
        ),
        update_stats=stats, actor=actor, critic=critic,
        channels=channels, sp=sp, grad_clip_norm=1.0, critic_lr=1e-3,
    )


def _status(res):
    return {c["id"]: c["status"] for c in res["checks"]}


def test_ledger_no_fails_on_consistent_dump(tmp_path) -> None:
    dump = _make_dump(tmp_path)
    res = verify_dump_contracts(dump)
    fails = [c for c in res["checks"] if c["status"] == "fail"]
    assert fails == [], f"unexpected fails: {[c['name'] for c in fails]}"
    st = _status(res)
    # Math identities must pass on a genuine capture.
    assert st["C4.1"] == "pass"
    assert st["C8.1"] == "pass"
    assert st["C1.1"] == "pass"
    assert st["C3.1"] == "pass"


def test_ledger_flags_injected_provenance_mismatch(tmp_path) -> None:
    batch = _batch(B=16)
    batch["metadata"] = [
        {"versions": {"regularizer_mode": "entropy",
                      "reward_semantics": ""},
         "collection": {}, "behavior": {}}
        for _ in range(16)
    ]
    dump = _make_dump(tmp_path, batch=batch)
    res = verify_dump_contracts(dump)
    st = _status(res)
    assert st["C1.4"] == "fail"
    c14 = next(c for c in res["checks"] if c["id"] == "C1.4")
    assert c14["metric"]["mismatched_rows"] == 16
    assert len(c14["violating_samples"]) == 16


def test_unknown_semantics_warns_not_passes(tmp_path) -> None:
    batch = _batch(B=16)
    batch["metadata"] = [
        {"versions": {"regularizer_mode": "shannon",
                      "reward_semantics": "nonexistent_v9"},
         "collection": {}, "behavior": {}}
        for _ in range(16)
    ]
    dump = _make_dump(tmp_path, batch=batch)
    res = verify_dump_contracts(dump)
    c12 = [c for c in res["checks"]
           if c["id"] == "C1.2" and "nonexistent_v9" in c["name"]]
    assert c12 and c12[0]["status"] == "warn"


def test_polyak_check_detects_tampered_target(tmp_path) -> None:
    actor, critic, channels = _models(C=2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-3)
    batch = _batch(B=16)
    sp = SACParams(use_grad_norm=False, reward_scale=2.0)
    clocks = SACClockState(collection_round=3, critic_tick=7)
    tmp = begin_critic_tick_dump(
        dump_root=tmp_path / "debug_dumps", critic_tick=8,
        clocks=clocks, batch=batch,
        trainer_pre_state=trainer_state_dict(
            actor, critic, actor_optimizer, log_alpha, alpha_optimizer,
        ),
    )
    capture = {}
    stats = sac_update_v2(
        actor=actor, critic=critic, actor_optimizer=actor_optimizer,
        log_alpha=log_alpha, alpha_optimizer=alpha_optimizer,
        batch=batch, channels=channels, sp=sp,
        grad_clip_norm=1.0, device=torch.device("cpu"), capture=capture,
    )
    post = trainer_state_dict(
        actor, critic, actor_optimizer, log_alpha, alpha_optimizer,
    )
    # Corrupt one target network — Polyak identity must now fail.
    gk = next(iter(post["critic_state_dict"]))
    for k in post["critic_state_dict"][gk]["q1_target"]:
        post["critic_state_dict"][gk]["q1_target"][k] += 10.0
    dump = finish_critic_tick_dump(
        tmp, forward_capture=capture, trainer_post_state=post,
        update_stats=stats, actor=actor, critic=critic,
        channels=channels, sp=sp, grad_clip_norm=1.0, critic_lr=1e-3,
    )
    res = update_checks(dump)
    assert max(res["polyak_residuals"].values()) > 1.0
    ledger = verify_dump_contracts(dump)
    assert _status(ledger)["C8.1"] == "fail"


def test_counterfactual_target_recompute(tmp_path) -> None:
    dump = _make_dump(tmp_path)
    base = recompute_targets(dump)
    assert base["closure_ok"] is True
    shifted = recompute_targets(dump, gamma=0.5)
    assert shifted["overrides"] == {"gamma": 0.5}
    for ch, e in shifted["per_channel"].items():
        assert "delta_vs_recorded" in e
        assert abs(e["gamma"] - 0.5) < 1e-9


def test_balance_slice_verifier_recompute_and_fault() -> None:
    fn = get_slice_verifier("basic_balance_phi_cross_v1")
    assert fn is not None
    B = 8
    rng = np.random.default_rng(0)
    phi_post = rng.uniform(0, 1, B)
    phi_pre = rng.uniform(0, 1, B)
    cross = rng.uniform(-1, 1, B)
    w0, w1 = 3.0, 1.0
    gate_pre = np.stack(
        [np.full(B, w0), w1 * phi_pre ** 2], axis=1)
    gate_post = np.stack(
        [np.full(B, w0), w1 * phi_post ** 2], axis=1)
    batch = {
        "sample_ids": np.arange(B),
        "rewards": np.stack(
            [0.01 * np.clip(phi_post, 0, 1), cross], axis=1),
        "actor_weight": gate_pre / gate_pre.sum(1, keepdims=True),
        "actor_weight_next": gate_post / gate_post.sum(1, keepdims=True),
        "task_facts": {"phi_pre": phi_pre,
                       "phi_post_reference": phi_post},
        "reward_features": {"cross_support": cross},
    }
    checks = fn(
        batch, ["r_fall", "r_cross"], None,
        atol=1e-5, contract_source="class_default_fallback",
    )
    assert all(c["status"] == "pass" for c in checks), checks

    # Corrupt one reward row → fail with the violating sample id.
    batch["rewards"][3, 0] = 9.9
    checks = fn(
        batch, ["r_fall", "r_cross"], None,
        atol=1e-5, contract_source="class_default_fallback",
    )
    bad = [c for c in checks if c["status"] == "fail"]
    assert bad
    assert 3 in bad[0]["violating_samples"]


def test_semantics_registry_covers_both_experiments() -> None:
    sems = known_semantics()
    assert "standup_4stage_dense_v2" in sems
    assert "basic_balance_phi_cross_v1" in sems


def test_cut_functions_return_payloads(tmp_path) -> None:
    dump = _make_dump(tmp_path)
    assert rollout_exploration(dump)["available"] is True
    assert "checks" in slice_verification(dump)
    tw = candidate_twin_analysis(dump)
    assert "target" in tw or "actor" in tw or "integ_weight_sum" in tw
    fit = critic_fit(dump)
    assert fit["per_channel"]
    post = post_detail(dump)
    assert "det_action_drift" in post
    upd = update_checks(dump)
    assert upd["polyak_residuals"]
