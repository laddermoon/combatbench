"""Parallel branch rollout with arbitrary state injection.

Per update the orchestrator ships the actor state_dict plus a list of
seed core-states to persistent worker processes. Each worker restores
the env to the seed state via ``set_core_state`` (opponent robot is
physically decoupled and stepped with zero actions) and rolls NStep
with the deterministic policy mean + fixed Gaussian noise.

Returned per seed: the full (NStep+1) cs/obs sequences, the NStep
executed (noisy) actions, the deterministic mean actions, and the
per-step termination flags. The first-step state cs[1] is what the
StartSet admits as the wavefront successor.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, List

import numpy as np

AGENT = "robot_a"
ZERO21 = np.zeros(21, dtype=np.float32)

_ENV = None
_POLICY = None
_POLICY_KEY = None


def _worker_init(env_bp_dict: Dict[str, Any]) -> None:
    global _ENV
    import os
    os.environ.setdefault("MUJOCO_GL", "disabled")
    import torch
    torch.set_num_threads(1)
    from envs.framework.blueprint import EnvBlueprint
    bp = EnvBlueprint.from_dict(env_bp_dict)
    _ENV = bp.build()


def _build_policy(sd: Dict[str, np.ndarray]) -> None:
    global _POLICY, _POLICY_KEY
    import torch
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)
    key = tuple(sorted(sd.keys()))
    if _POLICY is not None and _POLICY_KEY == key:
        pass
    else:
        _POLICY = TruncatedNormalPolicy(
            obs_dim=57, action_dim=21, hidden_dim=256, device="cpu")
        _POLICY_KEY = key
    _POLICY.load_state_dict(
        {k: torch.as_tensor(v) for k, v in sd.items()})
    _POLICY.eval()


def _flatten(cs_dict: Dict[str, Any], aid: str) -> np.ndarray:
    from baseline.framework.dynabptt.corestate import flatten_core_state
    return flatten_core_state(cs_dict, aid)


def _run_branches(payload: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    """Run a chunk of branch rollouts in this worker process."""
    global _ENV
    import torch
    from baseline.framework.dynabptt.hybrid_ppo.standup import (
        unflatten_core_state)

    sd = payload["actor_sd"]
    tasks = payload["tasks"]          # list of (seed_idx, cs(55,))
    nstep = int(payload["nstep"])
    sigma = float(payload["sigma"])
    base_seed = int(payload["base_seed"])
    _build_policy(sd)

    rng = np.random.default_rng(base_seed)
    out: List[Dict[str, np.ndarray]] = []
    for seed_idx, cs0 in tasks:
        bseed = int(rng.integers(2**31))
        for pl in _ENV.plugins:
            pl.set_episode_seed(bseed)
        _ENV.reset(seed=bseed, base_seed=bseed)
        _ENV.simulator.set_core_state(
            {AGENT: unflatten_core_state(np.asarray(cs0))})
        obs_seq, cs_seq, act_seq, det_seq, term_seq = [], [], [], [], []
        obs_now = _ENV.get_observation()
        o = np.asarray(obs_now[0], dtype=np.float32)
        cs_seq.append(_flatten(_ENV.simulator.get_core_state(), AGENT))
        obs_seq.append(o.copy())
        terminated = False
        for _ in range(nstep):
            with torch.no_grad():
                det = _POLICY.deterministic_action(
                    torch.as_tensor(o[None], dtype=torch.float32)
                )[0].cpu().numpy().astype(np.float32)
            a = det + rng.standard_normal(21).astype(np.float32) * sigma
            _ENV.step(a, ZERO21)
            o = np.asarray(_ENV.get_observation()[0], dtype=np.float32)
            cs_seq.append(_flatten(_ENV.simulator.get_core_state(), AGENT))
            obs_seq.append(o.copy())
            act_seq.append(a.astype(np.float32))
            det_seq.append(det)
            terminated = bool(_ENV.is_episode_over())
            term_seq.append(terminated)
            if terminated:
                break
        out.append({
            "seed_idx": int(seed_idx),
            "cs": np.stack(cs_seq).astype(np.float32),      # (T+1, 55)
            "obs": np.stack(obs_seq).astype(np.float32),    # (T+1, 57)
            "act": np.stack(act_seq).astype(np.float32),    # (T, 21)
            "det": np.stack(det_seq).astype(np.float32),    # (T, 21)
            "term": np.asarray(term_seq, dtype=bool),       # (T,)
        })
    return out


class BranchCollector:
    """Persistent worker pool for branch rollouts."""

    def __init__(self, env_bp, num_workers: int = 24):
        self.num_workers = int(num_workers)
        ctx = mp.get_context("spawn")
        self._env_bp_dict = env_bp.to_dict()
        self.pool = ctx.Pool(
            processes=self.num_workers, initializer=_worker_init,
            initargs=(self._env_bp_dict,), maxtasksperchild=None)

    def collect(self, actor_sd_cpu: Dict[str, np.ndarray],
                seed_cs: np.ndarray, nstep: int, sigma: float,
                base_seed: int) -> List[Dict[str, np.ndarray]]:
        """Roll branches for all seeds; returns concatenated results."""
        n = len(seed_cs)
        chunks: List[List[Any]] = [[] for _ in range(self.num_workers)]
        for i in range(n):
            chunks[i % self.num_workers].append((i, seed_cs[i]))
        payloads = [{
            "actor_sd": actor_sd_cpu,
            "tasks": [t for t in c],
            "nstep": nstep, "sigma": sigma,
            "base_seed": base_seed + j * 7919,
        } for j, c in enumerate(chunks) if c]
        results = self.pool.map(_run_branches, payloads)
        flat = [b for r in results for b in r]
        flat.sort(key=lambda b: b["seed_idx"])
        return flat

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()
