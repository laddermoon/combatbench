"""P5-CLI-1: permanent tests for the SAC analysis layer + extended CLI."""
from __future__ import annotations

import json

import pytest
import torch

from baseline.framework.sac import analysis
from baseline.framework.sac.clocks import SACClockState
from baseline.framework.sac.debugkit import (
    begin_critic_tick_dump,
    finish_critic_tick_dump,
)
from baseline.framework.sac.experiment import SACParams
from baseline.framework.sac.tests.test_env_integration import (
    _FakeExperiment,
    _FakeRollouter,
)
from baseline.framework.sac.tests.test_trainer import _batch, _models
from baseline.framework.sac.trainer import sac_update_v2, trainer_state_dict
from baseline.framework.sac.loop import train_sac


def _make_dump(tmp_path):
    actor, critic, channels = _models(C=2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-3)
    batch = _batch(B=16)
    sp = SACParams(use_grad_norm=False, reward_scale=2.0)
    tmp = begin_critic_tick_dump(
        dump_root=tmp_path / "debug_dumps",
        critic_tick=5,
        clocks=SACClockState(collection_round=2, critic_tick=4),
        batch=batch,
        trainer_pre_state=trainer_state_dict(
            actor, critic, actor_optimizer, log_alpha, alpha_optimizer,
        ),
        hypothesis="analysis-test",
    )
    capture = {}
    stats = sac_update_v2(
        actor=actor, critic=critic, actor_optimizer=actor_optimizer,
        log_alpha=log_alpha, alpha_optimizer=alpha_optimizer,
        batch=batch, channels=channels, sp=sp, grad_clip_norm=1.0,
        device=torch.device("cpu"), capture=capture,
    )
    return finish_critic_tick_dump(
        tmp,
        forward_capture=capture,
        trainer_post_state=trainer_state_dict(
            actor, critic, actor_optimizer, log_alpha, alpha_optimizer,
        ),
        update_stats=stats,
        actor=actor, critic=critic, channels=channels, sp=sp,
        grad_clip_norm=1.0, critic_lr=1e-3,
        replay_stats={"size": 128, "overwritten": 3},
    )


def test_parse_source_key():
    key = (
        "run_x/r000042/j000003/robot_b/e-7/f000123"
        "/hdeadbeef/vsac_transition_v2"
    )
    p = analysis.parse_source_key(key)
    assert p["parseable"] is True
    assert p["collection_round"] == 42
    assert p["job_index"] == 3
    assert p["agent_id"] == "robot_b"
    assert p["episode_seed"] == -7
    assert p["frame_index"] == 123
    assert p["schema"] == "sac_transition_v2"
    assert analysis.parse_source_key("src:3")["parseable"] is False


def test_dump_inspect_reports_v3_detail(tmp_path):
    dump_dir = _make_dump(tmp_path)
    out = analysis.dump_inspect(dump_dir)
    assert out["schema_version"] == "sac_dump_v3"
    assert out["batch"]["size"] == 16
    assert out["consistency"]["checked"] is True
    assert out["replay_stats"]["size"] == 128
    assert set(out["per_channel"]) == {"r0", "r1"} or out["per_channel"]
    for ch in out["per_channel"].values():
        assert "td_abs_max" in ch
    assert "actor_state_dict" in out["param_delta"]


def test_dump_samples_sorted_by_td(tmp_path):
    dump_dir = _make_dump(tmp_path)
    out = analysis.dump_samples(dump_dir, sort="td_abs", limit=5)
    rows = out["rows"]
    assert len(rows) == 5
    tds = [r["td_abs"] for r in rows]
    assert tds == sorted(tds, reverse=True)
    with pytest.raises(ValueError, match="unknown sort"):
        analysis.dump_samples(dump_dir, sort="bogus")
    with pytest.raises(ValueError, match="not in dump channels"):
        analysis.dump_samples(dump_dir, channel="nope")


def test_dump_trace_full_provenance(tmp_path):
    dump_dir = _make_dump(tmp_path)
    out = analysis.dump_trace(dump_dir, sample_id=3)
    assert out["sample_id"] == 3
    assert out["provenance"]["raw"] == "src:3"
    assert set(out["per_channel"]) != set()
    ch = next(iter(out["per_channel"].values()))
    assert "td_q1" in ch and "target" in ch and "actor_cand_q" in ch
    assert out["actor_side"]["pair_index"] is not None
    assert out["target_side"]["cand_f"] != "unavailable"
    with pytest.raises(ValueError, match="requires"):
        analysis.dump_trace(dump_dir)
    with pytest.raises(ValueError, match="matches=0"):
        analysis.dump_trace(dump_dir, sample_id=99999)


def test_metric_series_runs_and_catalog(tmp_path):
    run_dir = tmp_path / "run"
    train_sac(_FakeExperiment(), run_dir=run_dir, rollouter=_FakeRollouter())
    summary = analysis.run_summary(run_dir)
    assert summary["experiment_name"] == "fake_sac"
    assert summary["event_counts"]["round"] >= 1
    series = analysis.metric_series(
        run_dir, "collection.env_steps", event="round",
    )
    assert series["count"] >= 1
    assert series["points"][0]["value"] > 0
    idx = analysis.runs_index(tmp_path)
    assert any(r["name"] == "run" for r in idx)
    cat = analysis.metric_catalog(prefix="critic.")
    assert cat and all(s["name"].startswith("critic.") for s in cat)


def test_replay_report_offline(tmp_path):
    run_dir = tmp_path / "run"
    train_sac(_FakeExperiment(), run_dir=run_dir, rollouter=_FakeRollouter())
    rep = analysis.replay_report(run_dir)
    assert rep["size"] > 0
    assert rep["sample_age"]["mean"] is not None
    assert rep["collection_round_counts"]
    # Direct replay.pt path also works.
    ckpt = sorted((run_dir / "checkpoints").iterdir())[-1]
    rep2 = analysis.replay_report(ckpt / "replay.pt")
    assert rep2["size"] == rep["size"]


def test_on_demand_dump_request_is_consumed(tmp_path):
    """P5-ONDEMAND-1: dump_request.json sentinel triggers capture."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "dump_request.json").write_text(json.dumps({
        "hypothesis": "on-demand-check",
        "critic_tick": 2,
    }))
    train_sac(
        _FakeExperiment(), run_dir=run_dir, rollouter=_FakeRollouter(),
    )
    dump_dir = run_dir / "debug_dumps" / "critic_tick_00000002"
    assert dump_dir.is_dir()
    request = json.loads((dump_dir / "request.json").read_text())
    assert request["hypothesis"] == "on-demand-check"
    assert not (run_dir / "dump_request.json").exists()


def test_http_server_endpoints_share_analysis_layer(tmp_path):
    """P5-API-1: every endpoint hits the same analysis functions."""
    import urllib.request
    from urllib.error import HTTPError

    from baseline.framework.sac.debugserver import serve

    run_dir = tmp_path / "run"
    train_sac(_FakeExperiment(), run_dir=run_dir, rollouter=_FakeRollouter())
    _make_dump(tmp_path)

    server, _thread = serve(tmp_path, port=0)
    try:
        port = server.server_address[1]

        def get(path):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}"
            ) as resp:
                return json.loads(resp.read())

        runs = get("/api/runs")
        assert any(r["name"] == "run" for r in runs)
        run = get("/api/run?name=run")
        assert run["experiment_name"] == "fake_sac"
        series = get(
            "/api/run/metrics?name=run&key=collection.env_steps"
        )
        assert series["count"] >= 1
        rep = get("/api/run/replay?name=run")
        assert rep["size"] > 0
        dump_dir = tmp_path / "debug_dumps" / "critic_tick_00000005"
        ins = get(f"/api/dump/inspect?path={dump_dir}")
        assert ins["batch"]["size"] == 16
        tr = get(f"/api/dump/trace?path={dump_dir}&sample_id=3")
        assert tr["sample_id"] == 3
        cat = get("/api/catalog?prefix=actor.")
        assert all(s["name"].startswith("actor.") for s in cat)
        with pytest.raises(HTTPError) as exc:
            get("/api/nope")
        assert exc.value.code == 404
    finally:
        server.shutdown()


def test_query_payload_dotpath(tmp_path):
    dump_dir = _make_dump(tmp_path)
    payload = analysis.load_artifact(dump_dir)
    assert analysis.query_payload(payload, "manifest.kind") == "critic_tick"
    assert analysis.query_payload(
        payload, "analysis.consistency.checked",
    ) is True
    with pytest.raises(KeyError):
        analysis.query_payload(payload, "manifest.nope")
