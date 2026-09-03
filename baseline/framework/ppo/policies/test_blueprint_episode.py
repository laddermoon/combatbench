"""Test: TanhGaussianMLPPolicy -> to_blueprint -> build -> run one Episode.

End-to-end smoke test that exercises:
  1. Create a TanhGaussianMLPPolicy
  2. Export via to_blueprint()
  3. Rebuild the policy from the blueprint
  4. Build an EnvRuntime from envs/humanoid21/blueprint.yaml
  5. Run one episode with EpisodeRunner
  6. Verify the Episode data is sane

Usage:
    PYTHONPATH=. python3 baseline/framework/ppo/policies/test_blueprint_episode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the combatbench repo root is importable when run directly.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from baseline.framework.ppo.policies.tanh_gaussian_mlp import TanhGaussianMLPPolicy
from baseline.framework.rollout.episode import Episode
from baseline.framework.rollout.episode import blueprint_hash as _bp_hash
from baseline.framework.rollout.episode_recorder import EpisodeRecorder
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint

# ── 1. Create actor & export via to_blueprint ─────────────────────────────

print("[1] Creating TanhGaussianMLPPolicy (obs=96, act=21, hidden=256) ...")
actor = TanhGaussianMLPPolicy(
    obs_dim=96,
    action_dim=21,
    hidden_dim=256,
    device="cpu",
    deterministic=False,
)
print(f"    actor obs_dim={actor.obs_dim}  action_dim={actor.action_dim}")

export_dir = Path(__file__).resolve().parent / "_test_export"
print(f"[2] Exporting to blueprint -> {export_dir} ...")
policy_bp = actor.to_blueprint(dest_path=str(export_dir))
print(f"    blueprint cls = {policy_bp.cls}")
print(f"    blueprint config keys = {list(policy_bp.config.keys())}")

# ── 2. Rebuild policy from blueprint ──────────────────────────────────────

print("[3] Rebuilding policy from blueprint ...")
rebuilt_policy = policy_bp.build(stochastic=True)
print(f"    rebuilt type = {type(rebuilt_policy).__name__}")

# ── 3. Build env from blueprint.yaml ──────────────────────────────────────

bp_path = Path(__file__).resolve().parents[4] / "envs" / "humanoid21" / "blueprint.yaml"
print(f"[4] Loading env blueprint from {bp_path} ...")
env_pb = ParameterizedEnvBlueprint.load(bp_path)
env_bp = env_pb.materialize()  # use all defaults
print(f"    simulator = {env_bp.simulator.cls}")
print(f"    plugins   = {[p.cls for p in env_bp.plugins]}")
print(f"    observers = {list(env_bp.observer_plugins.keys())}")

# ── 4. Attach recorder & build runtime ────────────────────────────────────

recorder = EpisodeRecorder(
    blueprint_hash=_bp_hash(env_bp),
    snapshot_arrays=True,
)
print("[5] Building EnvRuntime with EpisodeRecorder ...")
runtime = env_bp.build(recorders=[recorder])

# ── 5. Run one episode via EpisodeRunner ───────────────────────────────────

# Both agents use the same rebuilt policy.
print("[6] Running EpisodeRunner.run_episode(seed=42) ...")
runner = EpisodeRunner(
    runtime=runtime,
    policy_a=rebuilt_policy,
    policy_b=rebuilt_policy,
)
runner.run_episode(seed=42)
print("    episode finished.")

# ── 6. Verify the recorded Episode ────────────────────────────────────────

print("[7] Inspecting recorded Episode ...")
ep: Episode = recorder.get_last_episode()
print(f"    base_seed       = {ep.base_seed}")
print(f"    term_records    = {dict(ep.agent_termination_proposal_records)}")

for agent in ("robot_a", "robot_b"):
    obs = ep.observations.get(agent)
    acts = ep.actions.get(agent)
    fin = ep.final_observation.get(agent)
    if obs is not None:
        print(f"    {agent} obs shape   = {obs.shape}")
    if acts is not None:
        print(f"    {agent} actions shape= {acts.shape}")
    if fin is not None:
        print(f"    {agent} final_obs    = {fin.shape}")

print(f"    observer outputs keys = {list(ep.observer_outputs.keys())}")
print(f"    episode_options       = {ep.episode_options}")

# Basic sanity checks
assert ep.base_seed == 42, f"Expected base_seed=42, got {ep.base_seed}"
for agent in ("robot_a", "robot_b"):
    obs = ep.observations.get(agent)
    acts = ep.actions.get(agent)
    assert obs is not None, f"{agent} observations missing"
    assert acts is not None, f"{agent} actions missing"
    assert obs.ndim == 2, f"{agent} obs should be 2-D, got {obs.ndim}"
    assert acts.ndim == 2, f"{agent} actions should be 2-D, got {acts.ndim}"
    assert obs.shape[1] == 96, f"{agent} obs dim should be 96, got {obs.shape[1]}"
    assert acts.shape[1] == 21, f"{agent} act dim should be 21, got {acts.shape[1]}"

print("\n[OK] All assertions passed. Test complete.")
