import copy
import json

import numpy as np
import pytest

from envs.batchframework.validation import (
    Unsupported, capture, compare, digest, load, replay, save,
)


class Adapter:
    backend = "test-cpu"
    version = "1"

    def __init__(self, dependency):
        self.dependencies = [dependency]

    def execute(self, case):
        if case["kind"] != "logic":
            raise Unsupported("unknown plugin")
        return {"robot_a": {"observation": np.arange(96, dtype=np.float32),
                            "bootstrap": True, "potential": 0.5}}


@pytest.fixture
def setup_case(tmp_path):
    dep = tmp_path / "source.py"
    dep.write_text("version = 1\n")
    case = {"id": "timeout-7", "kind": "logic", "level": "V2", "seed": 42,
            "episode": 7, "frame": 200, "input": {"reason": "timeout"}}
    adapter = Adapter(dep)
    return case, adapter


def test_capture_roundtrip_replay(setup_case, tmp_path):
    case, adapter = setup_case
    bundle = capture(case, adapter)
    path = tmp_path / "case.json"
    save(path, bundle)
    restored = load(path)
    assert digest(restored) == digest(bundle)
    report = replay(restored, adapter, digest(bundle))
    assert report["status"] == "pass"
    assert report["scope"] == "single-case"
    assert report["case_id"] == "timeout-7"
    assert report["failures"] == []
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mutation,field", [
    ("order", "observation"), ("missing", "potential"),
    ("bootstrap", "bootstrap"), ("shape", "observation"),
    ("nan", "potential"), ("extra", "unexpected"),
    ("dtype", "observation"),
])
def test_corrupt_candidate_is_located(setup_case, mutation, field):
    case, adapter = setup_case
    bundle = capture(case, adapter)
    execute = adapter.execute

    def broken(case):
        output = execute(case)
        agent = output["robot_a"]
        if mutation == "order":
            agent["observation"] = agent["observation"][::-1].copy()
        elif mutation == "missing":
            del agent["potential"]
        elif mutation == "bootstrap":
            agent["bootstrap"] = False
        elif mutation == "shape":
            agent["observation"] = agent["observation"].reshape(1, 96)
        elif mutation == "nan":
            agent["potential"] = float("nan")
        elif mutation == "extra":
            agent["unexpected"] = 0
        else:
            agent["observation"] = agent["observation"].astype(np.float64)
        return output

    adapter.execute = broken
    report = replay(bundle, adapter, digest(bundle))
    assert report["status"] == "fail"
    assert any(field in f["field"] for f in report["failures"])
    assert report["episode"] == 7
    assert report["frame"] == 200
    assert "robot_a" in report["failures"][0]["field"]
    json.dumps(report, allow_nan=False)


def test_stale_source_blocks_execution(setup_case):
    case, adapter = setup_case
    bundle = capture(case, adapter)
    adapter.dependencies[0].write_text("version = 2\n")
    adapter.execute = lambda case: pytest.fail("must not execute stale reference")
    report = replay(bundle, adapter, digest(bundle))
    assert report["status"] == "stale"


def test_digest_protects_oracle_and_tolerance(setup_case):
    case, adapter = setup_case
    bundle = capture(case, adapter)
    approved = digest(bundle)
    bundle["tolerance"]["atol"] = 10
    assert replay(bundle, adapter, approved)["status"] == "stale"


def test_unsupported_and_not_run(setup_case):
    case, adapter = setup_case
    bundle = capture(case, adapter)
    assert replay(bundle, None, digest(bundle))["status"] == "not-run"

    def unsupported(case):
        raise Unsupported("plugin unknown")

    adapter.execute = unsupported
    report = replay(bundle, adapter, digest(bundle))
    assert report["status"] == "unsupported"
    assert "plugin unknown" in report["failures"][0]["actual"]


def test_exception_is_failure_not_skipped(setup_case):
    case, adapter = setup_case
    bundle = capture(case, adapter)

    def broken(case):
        raise KeyError("required")

    adapter.execute = broken
    report = replay(bundle, adapter, digest(bundle))
    assert report["status"] == "fail"
    assert "KeyError" in report["failures"][0]["actual"]


def test_candidate_cannot_mutate_oracle(setup_case):
    case, adapter = setup_case
    bundle = capture(case, adapter)
    original = digest(bundle)
    execute = adapter.execute

    def mutating(case):
        case["input"]["reason"] = "ko"
        return execute(case)

    adapter.execute = mutating
    replay(bundle, adapter, original)
    assert digest(bundle) == original


def test_comparator_tolerance_and_nonfinite():
    assert compare(1.0, 1.000001, {"atol": 1e-5, "rtol": 0.0}) == []
    assert compare(True, 1, {"atol": 0.0, "rtol": 0.0})
    assert compare(float("nan"), float("nan"), {"atol": 0.0, "rtol": 0.0})
    assert compare([1, 2], [1], {"atol": 0.0, "rtol": 0.0})
    with pytest.raises(ValueError):
        compare(1.0, 2.0, {"atol": float("inf"), "rtol": 0.0})


def test_field_tolerances_are_local_and_typos_fail():
    tolerance = {"atol": 0.0, "rtol": 0.0}
    expected = {"potential": 0.5, "stage": 3.0}
    actual = {"potential": 0.500001, "stage": 4.0}
    overrides = {"output.potential": {"atol": 1e-5, "rtol": 0.0}}
    failures = compare(expected, actual, tolerance, overrides)
    assert [f["field"] for f in failures] == ["output.stage"]
    with pytest.raises(ValueError, match="unused"):
        compare(expected, actual, tolerance, {"output.typo": tolerance})


def test_invalid_case_rejected(setup_case):
    case, adapter = setup_case
    del case["seed"]
    with pytest.raises(ValueError):
        capture(case, adapter)


def test_cpu_reward_cases_and_physics_replay(tmp_path):
    from envs.batchframework.validation_cpu import CPUAdapter, reward_cases, physics_case

    adapter = CPUAdapter()
    stages = set()
    for case in reward_cases():
        bundle = capture(case, adapter)
        stages.add(bundle["expected"][case["input"]["agent"]]["stage"])
        assert replay(bundle, adapter, digest(bundle))["status"] == "pass"
    assert stages == {1.0, 2.0, 3.0, 4.0}
    for steps in (1, 25):
        bundle = capture(physics_case(steps), adapter)
        path = tmp_path / f"physics-{steps}.json"
        save(path, bundle)
        report = replay(load(path), adapter, digest(bundle))
        assert report["status"] == "pass", report


def test_reward_oracle_has_independent_known_answers():
    from envs.batchframework.validation_cpu import CPUAdapter, reward_cases

    outputs = {case["id"]: CPUAdapter().execute(case)[case["input"]["agent"]] for case in reward_cases()}
    assert outputs["airborne-stage1"]["potential"] == 0.05
    assert outputs["supported-stage4"]["potential"] == pytest.approx(1.0, abs=1e-7)
    assert outputs["wide-stage3"]["stage"] == 3
    assert outputs["force-0.999"]["stage"] == 1
    assert outputs["force-1.0"]["stage"] == 4
    assert outputs["force-9.999"]["w_foot"] == 0.0
    assert outputs["force-10.0"]["w_foot"] == 1.0
    assert outputs["distance-0.5199"]["stage"] == 4
    assert outputs["distance-0.5201"]["stage"] == 3
    assert outputs["prone-stage2"] == outputs["duplicate-body-not-two-bodies"]
    assert outputs["wall-not-ground"]["stage"] == 1
    assert outputs["f-score-0.7999"]["stage"] == 1
    assert outputs["f-score-0.8001"]["stage"] == 2
    assert outputs["robot-b-supported"] == outputs["supported-stage4"]


def test_real_standup_timeout_contract_detects_bootstrap_error():
    from envs.batchframework.validation_cpu import CPUAdapter, trajectory_case

    adapter = CPUAdapter()
    bundle = capture(trajectory_case(), adapter)
    expected = bundle["expected"]["robot_a"]
    assert expected["is_terminated"] is False
    np.testing.assert_array_equal(expected["last_obs"], np.arange(96, dtype=np.float32) + 1000)
    np.testing.assert_array_equal(expected["reward"], np.array([0.3, 0.9], dtype=np.float32) * 0.01)
    assert replay(bundle, adapter, digest(bundle))["status"] == "pass"
    execute = adapter.execute

    def broken(case):
        output = execute(case)
        output["robot_a"]["is_terminated"] = True
        return output

    adapter.execute = broken
    report = replay(bundle, adapter, digest(bundle))
    assert report["status"] == "fail"
    assert report["failures"][0]["field"] == "output.robot_a.is_terminated"


def test_cpu_unknown_plugin_is_unsupported():
    from envs.batchframework.validation_cpu import CPUAdapter, physics_case

    case = physics_case()
    case["input"]["plugins"] = ["unregistered.plugin:Force"]
    with pytest.raises(Unsupported, match="plugins"):
        CPUAdapter().execute(case)


def test_policy_eval_is_not_accidentally_executed(setup_case):
    from envs.batchframework.validation_cpu import CPUAdapter

    case, _ = setup_case
    case.update(kind="policy_eval", level="V3")
    with pytest.raises(Unsupported, match="policy_eval"):
        CPUAdapter().execute(case)


def test_missing_source_and_environment_change_are_stale(setup_case, monkeypatch):
    from envs.batchframework import validation

    case, adapter = setup_case
    bundle = capture(case, adapter)
    monkeypatch.setattr(validation, "environment", lambda: {"changed": True})
    assert replay(bundle, adapter, digest(bundle))["status"] == "stale"
    adapter.dependencies[0].unlink()
    assert replay(bundle, adapter, digest(bundle))["status"] == "stale"


def test_cli_capture_replay_and_no_overwrite(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "reward.json"
    command = [sys.executable, "-B", "-m", "envs.batchframework.validation"]
    captured = subprocess.run(command + ["capture", str(path), "--case", "reward"],
                              check=True, text=True, capture_output=True)
    sha = json.loads(captured.stdout)["digest"]
    args = ["replay", str(path), "--digest", sha,
            "--candidate", "envs.batchframework.validation_cpu"]
    result = subprocess.run(command + args, check=True, text=True, capture_output=True)
    report = json.loads(result.stdout)
    assert report["status"] == "pass"
    assert str(path) in report["replay"]
    not_run = subprocess.run(command + ["replay", str(path), "--digest", sha],
                             text=True, capture_output=True)
    assert not_run.returncode == 1
    assert json.loads(not_run.stdout)["status"] == "not-run"
    again = subprocess.run(command + ["capture", str(path), "--case", "reward"],
                           text=True, capture_output=True)
    assert again.returncode != 0
    assert digest(load(path)) == sha
    bundle = load(path)
    bundle["case"]["input"]["operation"] = "unknown-plugin"
    unknown = tmp_path / "unsupported.json"
    save(unknown, bundle)
    result = subprocess.run(command + ["replay", str(unknown), "--digest", digest(bundle),
                                       "--candidate", "envs.batchframework.validation_cpu"],
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "unsupported"
    stale = subprocess.run(command + ["replay", str(path), "--digest", "wrong"],
                           text=True, capture_output=True)
    assert stale.returncode == 1
    assert json.loads(stale.stdout)["status"] == "stale"


def test_write_fixtures_and_index_replay(tmp_path):
    from envs.batchframework import validation
    from envs.batchframework.validation_cpu import CPUAdapter, all_cases

    adapter = CPUAdapter()
    index = validation.write_fixtures(adapter, all_cases(), tmp_path)
    index_path = tmp_path / "index.json"
    assert index_path.is_file() and len(index["files"]) == len(all_cases())
    for name, sha in index["files"].items():
        report = replay(load(tmp_path / name), adapter, sha)
        assert report["status"] == "pass", (name, report["failures"][:1])
    persisted = load(index_path)
    assert persisted == index
    report = replay(load(tmp_path / "supported-stage4.json"), adapter,
                    index["files"]["supported-stage4.json"])
    assert report["status"] == "pass"
    wrong = dict(index["files"], **{"supported-stage4.json": "wrong"})
    assert wrong != index["files"]


def test_reward_oracle_rejects_missing_input():
    from envs.batchframework.validation_cpu import CPUAdapter, reward_cases

    case = copy.deepcopy(reward_cases()[0])
    del case["input"]["positions"]["torso"]
    with pytest.raises((ValueError, KeyError)):
        capture(case, CPUAdapter())
