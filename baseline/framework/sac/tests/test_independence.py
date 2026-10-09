"""P2-IND-0 independence tests for the SAC framework boundary.

These tests intentionally run subprocesses so PPO absence is measured in a
fresh interpreter rather than depending on pytest's module cache.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_python(source: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", "-c", source],
        cwd=_PROJECT_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        timeout=60,
    )


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


def test_sac_package_import_does_not_load_ppo():
    result = _run_python(
        """
import sys

import baseline.framework
import baseline.framework.sac
import baseline.experiments_sac

ppo_modules = [
    name for name in sys.modules
    if name == "baseline.framework.ppo"
    or name.startswith("baseline.framework.ppo.")
]
assert ppo_modules == [], ppo_modules

from baseline.experiments_sac import list_sac_experiments
assert "sac_balance" in list_sac_experiments()
assert not [
    name for name in sys.modules
    if name == "baseline.framework.ppo"
    or name.startswith("baseline.framework.ppo.")
]
"""
    )
    _assert_ok(result)


def test_sac_registry_and_cli_listing_do_not_load_ppo():
    result = _run_python(
        """
import sys

import baseline.framework.train as train

sys.argv = [
    "train.py",
    "--algo", "sac",
    "--list-experiments",
]
train.main()

ppo_modules = [
    name for name in sys.modules
    if name == "baseline.framework.ppo"
    or name.startswith("baseline.framework.ppo.")
]
assert ppo_modules == [], ppo_modules
"""
    )
    _assert_ok(result)
    assert "[SAC]" in result.stdout
    assert "sac_balance" in result.stdout
    assert "[PPO]" not in result.stdout


def test_sac_import_survives_ppo_import_block():
    result = _run_python(
        """
import sys

class BlockPPO:
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "baseline.framework.ppo"
            or fullname.startswith("baseline.framework.ppo.")
        ):
            raise ImportError(f"PPO import blocked: {fullname}")
        return None

sys.meta_path.insert(0, BlockPPO())

import baseline.framework.sac
import baseline.experiments_sac

assert "sac_balance" in baseline.experiments_sac.list_sac_experiments()
"""
    )
    _assert_ok(result)


def test_sac_cli_rejects_ppo_only_flags():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "baseline/framework/train.py",
            "--algo", "sac",
            "--experiment", "sac_balance",
            "--param", "actor_lr=1e-4",
        ],
        cwd=_PROJECT_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "PPO-only or not yet supported by SAC" in output
    assert "--param" in output


def test_sac_actor_builds_without_ppo_imports():
    result = _run_python(
        """
import sys
import torch

from baseline.experiments_sac.exp_sac_balance import SacBalance

exp = SacBalance(use_grad_norm=False)
actor = exp.build_actor(torch.device("cpu"))
assert actor.obs_dim == exp.obs_dim
assert actor.action_dim == exp.action_dim
assert actor.policy_arch == "tn_s01"
assert actor.policy_fingerprint()

ppo_modules = [
    name for name in sys.modules
    if name == "baseline.framework.ppo"
    or name.startswith("baseline.framework.ppo.")
]
assert ppo_modules == [], ppo_modules
"""
    )
    _assert_ok(result)


def test_lazy_ppo_exports_and_ppo_listing_still_work():
    result = _run_python(
        """
import baseline.framework.sac
from baseline.framework import CommonParams, TrainablePolicy

assert CommonParams.__name__ == "CommonParams"
assert TrainablePolicy.__name__ == "TrainablePolicy"
"""
    )
    _assert_ok(result)

    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "baseline/framework/train.py",
            "--algo", "ppo",
            "--list-experiments",
        ],
        cwd=_PROJECT_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    _assert_ok(result)
    assert "[PPO]" in result.stdout
    assert "basic_balance" in result.stdout
