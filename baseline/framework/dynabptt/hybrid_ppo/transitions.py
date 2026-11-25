"""Time-aligned episode data + model replay buffer.

Frame semantics (verified against env_runtime.py / episode_recorder.py):
    observations[t]  = o_t        — pre-action observation of state s_t
    actions[t]       = a_t        — action taken from s_t
    corestate[t]     = s_{t+1}    — post-action state (observer output)
    potential[t]     = phi(s_{t+1}) → reward r_t = 0.01 * phi[t]
    stage[t]         = stage(s_{t+1})
    final_observation = o_T

Consequences:
  * model transitions (s_t, a_t) -> s_{t+1} exist only for t >= 1
    (s_t = corestate[t-1]); frame 0's transition is skipped for the
    model but kept for PPO (PPO needs only obs/action/reward).
  * obs-head target for predicted s_{t+1} is obs[t+1]; for t = T-1 it
    is episode.final_observation.
  * a model-rollout seed (s_t, o_t, stage_t) = (cs[t-1], obs[t], stage[t-1]),
    t >= 1.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.rollout import extract_per_step_field

from .standup import CS_OBSERVER_NAME, REWARDER_NAMES, phi_per_frame

AGENT_IDS = ("robot_a", "robot_b")


@dataclass
class AlignedTraj:
    """Per-agent aligned arrays for one episode (see module docstring)."""
    cs: np.ndarray          # (T, 55)  s_{t+1}
    obs: np.ndarray         # (T, 57)  o_t
    act: np.ndarray         # (T, 21)  a_t
    phi: np.ndarray         # (T,)     phi(s_{t+1})
    stage: np.ndarray       # (T,)     stage(s_{t+1})
    final_obs: np.ndarray   # (57,)    o_T
    terminated: bool        # True => real terminal (not timeout)
    ep_key: int             # episode identity for split hashing
    update: int             # training update index that produced it
    agent_id: str = "robot_a"
    # Optional per-frame aux channels (task-specific).
    done: Optional[np.ndarray] = None   # (T,) 1 at terminal frame
    aux_rew: Optional[np.ndarray] = None  # (T,) second reward channel

    @property
    def T(self) -> int:
        return int(self.cs.shape[0])

    # --- model-side views -------------------------------------------------
    def seed(self, t: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Seed for imagined rollout at time t (t >= 1):
        (s_t, o_t, stage(s_t))."""
        assert 1 <= t <= self.T - 1
        return self.cs[t - 1], self.obs[t], self.stage[t - 1]

    def next_obs(self, t: int) -> np.ndarray:
        """o_{t+1} — obs[t+1] or final_obs for the last frame."""
        return self.final_obs if t == self.T - 1 else self.obs[t + 1]


def _episode_terminated(episode, agent_id: str) -> bool:
    records = episode.agent_termination_proposal_records.get(agent_id, ())
    return bool(records) and records[0][0] != "timeout"


def extract_aligned(episode, update: int,
                    agent_ids=AGENT_IDS) -> List[AlignedTraj]:
    """Episode → per-agent AlignedTraj (fails loud on missing fields)."""
    cs_node = episode.observer_outputs.get(CS_OBSERVER_NAME)
    if cs_node is None:
        raise KeyError(
            f"observer '{CS_OBSERVER_NAME}' missing — env blueprint must "
            "be built via make_standup_env_bp()"
        )
    T = int(episode.num_frames)
    out: List[AlignedTraj] = []
    for aid in agent_ids:
        cs = cs_node.get(aid)
        obs = episode.observations.get(aid)
        act = episode.actions.get(aid)
        if cs is None or obs is None or act is None:
            raise KeyError(f"episode missing cs/obs/act for {aid}")
        B = int(episode.agent_frame_boundary[aid])
        if B < 2:
            continue
        stage = extract_per_step_field(
            episode.observer_outputs, REWARDER_NAMES[aid], "stage", T,
        )
        if stage is None:
            raise KeyError(
                f"rewarder '{REWARDER_NAMES[aid]}' field 'stage' missing"
            )
        out.append(AlignedTraj(
            cs=np.asarray(cs[:B], dtype=np.float32),
            obs=np.asarray(obs[:B], dtype=np.float32),
            act=np.asarray(act[:B], dtype=np.float32),
            phi=phi_per_frame(episode, aid)[:B],
            stage=np.asarray(stage[:B], dtype=np.float32),
            final_obs=np.asarray(
                episode.final_observation[aid], dtype=np.float32),
            terminated=_episode_terminated(episode, aid),
            ep_key=int(episode.base_seed),
            update=int(update),
            agent_id=aid,
        ))
    return out


def split_is_holdout(ep_key: int, holdout_frac: float = 0.2) -> bool:
    """Deterministic per-episode split — same episode (both agents)
    always lands in the same partition."""
    h = int(hashlib.md5(str(ep_key).encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000.0 < holdout_frac


class ModelReplay:
    """FIFO replay of AlignedTraj for world-model fitting.

    Two pools: ``fit`` (train windows) and ``hold`` (validation episodes,
    never fitted).  ``hold`` keeps only the most recent ``hold_cap``
    episodes — validation must reflect the current policy's dynamics,
    not ancient data.
    """

    def __init__(self, max_transitions: int = 1_000_000,
                 hold_cap_episodes: int = 400,
                 holdout_frac: float = 0.2) -> None:
        self.max_transitions = int(max_transitions)
        self.hold_cap_episodes = int(hold_cap_episodes)
        self.holdout_frac = float(holdout_frac)
        self.fit: List[AlignedTraj] = []
        self.hold: List[AlignedTraj] = []
        self._fit_trans = 0
        self._hold_seen: set = set()

    def add(self, traj: AlignedTraj) -> str:
        if traj.ep_key in self._hold_seen or split_is_holdout(
                traj.ep_key, self.holdout_frac):
            self._hold_seen.add(traj.ep_key)
            self.hold.append(traj)
            # drop the oldest holdout episodes (per episode: both agents)
            while len(self.hold) > self.hold_cap_episodes * len(AGENT_IDS):
                self.hold.pop(0)
            return "hold"
        self.fit.append(traj)
        self._fit_trans += traj.T - 1
        while self._fit_trans > self.max_transitions and self.fit:
            old = self.fit.pop(0)
            self._fit_trans -= old.T - 1
        return "fit"

    def __len__(self) -> int:
        return self._fit_trans

    def _pool_for(self, recent_frac: float, recent_updates: int,
                  cur_update: int, rng: np.random.Generator
                  ) -> List[AlignedTraj]:
        """Choose between recent-pool and full-pool for one draw."""
        if recent_frac <= 0.0:
            return self.fit
        recent = [t for t in self.fit if t.update >= cur_update - recent_updates]
        if not recent:
            return self.fit
        return recent if rng.random() < recent_frac else self.fit

    def sample_windows(self, batch: int, k: int, rng: np.random.Generator,
                       recent_frac: float = 0.8, recent_updates: int = 2,
                       cur_update: int = 0) -> Dict[str, np.ndarray]:
        """Sample autoregressive windows of k steps.

        Window start t must satisfy 1 <= t <= T - k so that all k
        transitions (cs[t-1]+act[t..t+k-1] -> cs[t..t+k-1]) exist and the
        final obs target (t+k-1 -> obs[t+k] or final_obs) is defined.
        """
        if not self.fit:
            raise RuntimeError("ModelReplay: no fit trajectories yet")
        s0 = np.empty((batch, 55), dtype=np.float32)
        acts = np.empty((batch, k, 21), dtype=np.float32)
        cs_t = np.empty((batch, k, 55), dtype=np.float32)
        obs_t = np.empty((batch, k, 57), dtype=np.float32)
        phi_t = np.empty((batch, k), dtype=np.float32)
        stage_t = np.empty((batch, k), dtype=np.float32)
        done_t = np.zeros((batch, k), dtype=np.float32)
        auxr_t = np.zeros((batch, k), dtype=np.float32)
        for i in range(batch):
            for _ in range(500):
                pool = self._pool_for(
                    recent_frac, recent_updates, cur_update, rng)
                tr = pool[int(rng.integers(len(pool)))]
                hi = tr.T - k
                if hi >= 1:
                    t = int(rng.integers(1, hi + 1))
                    break
            else:
                raise RuntimeError(
                    "ModelReplay: could not draw a window — no trajectory "
                    f"long enough for k={k}"
                )
            s0[i] = tr.cs[t - 1]
            acts[i] = tr.act[t:t + k]
            cs_t[i] = tr.cs[t:t + k]
            for m in range(k):
                obs_t[i, m] = tr.next_obs(t + m)
            phi_t[i] = tr.phi[t:t + k]
            stage_t[i] = tr.stage[t:t + k]
            if tr.done is not None:
                done_t[i] = tr.done[t:t + k]
            if tr.aux_rew is not None:
                auxr_t[i] = tr.aux_rew[t:t + k]
        return {"s0": s0, "act": acts, "cs": cs_t, "obs": obs_t,
                "phi": phi_t, "stage": stage_t, "done": done_t,
                "aux_rew": auxr_t}

    def sample_holdout_windows(self, n_windows: int, k: int,
                               rng: np.random.Generator,
                               min_per_stage: int = 0,
                               ) -> List[Tuple[AlignedTraj, int]]:
        """Draw up to n_windows held-out start indices t (1 <= t <= T-k).

        Returns (traj, t) pairs — caller applies model/closed-loop/real
        comparisons.  Stage of a window = traj.stage[t-1] (stage of the
        start state s_t).
        """
        out: List[Tuple[AlignedTraj, int]] = []
        if not self.hold:
            return out
        tries = 0
        while len(out) < n_windows and tries < n_windows * 60:
            tries += 1
            tr = self.hold[int(rng.integers(len(self.hold)))]
            hi = tr.T - k
            if hi < 1:
                continue
            t = int(rng.integers(1, hi + 1))
            out.append((tr, t))
        return out

    def sample_seeds(self, n: int, rng: np.random.Generator,
                     stages: Optional[set] = None,
                     pool: str = "fit") -> List[Tuple[AlignedTraj, int]]:
        """Draw (traj, t) seed pairs, t in [1, T-1].

        ``stages``: restrict to seeds whose start-state stage
        (traj.stage[t-1]) is in the set (admission-gated subsets).
        """
        src = self.fit if pool == "fit" else self.hold
        out: List[Tuple[AlignedTraj, int]] = []
        if not src:
            return out
        tries = 0
        while len(out) < n and tries < n * 60:
            tries += 1
            tr = src[int(rng.integers(len(src)))]
            t = int(rng.integers(1, tr.T))
            if stages is not None and int(tr.stage[t - 1]) not in stages:
                continue
            out.append((tr, t))
        return out
