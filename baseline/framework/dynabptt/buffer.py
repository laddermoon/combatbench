"""TransitionBuffer — per-agent trajectory store for DynaBPTT.

Frame layout per agent trajectory (see Episode semantics):
  frame[j] = (action a_j executed on s_j, post-state cs_j = s_{j+1},
              obs ô_{j+1}, aux scalars, termination flag at s_{j+1})

Transition j (j >= 1): (cs_{j-1}, a_j) -> (cs_j, ô_j, aux_j, done_j).
Valid transition range: j in [1, B-1] where B is the agent's frame
boundary (post-termination frames are garbage for that agent).

k-step fit window starting at transition j0: input cs_{j0-1}, actions
a_{j0..j0+k-1}, targets from frames j0..j0+k-1 → requires j0 in [1, B-k].

Imagination seed frame f in [0, B-2]: state cs_f paired with its own
observation obs_f (terminal frame B-1 excluded).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import numpy as np


class AgentTraj:
    """Frame-aligned arrays for one agent in one episode."""

    __slots__ = ("cs", "obs", "act", "aux", "terminal", "n_trans", "source")

    def __init__(
        self,
        cs: np.ndarray,
        obs: np.ndarray,
        act: np.ndarray,
        aux: Dict[str, np.ndarray],
        terminal: bool,
        n_trans: int,
        source: str = "learner",
    ) -> None:
        self.cs = cs            # (B, state_dim) post-step core states
        self.obs = obs          # (B, obs_dim) post-step observations
        self.act = act          # (B, action_dim) executed actions
        self.aux = aux          # {name: (B,)} per-step aux targets
        self.terminal = bool(terminal)  # frame B-1 is a true terminal state
        self.n_trans = int(n_trans)     # B-1 valid transitions
        self.source = source            # "learner" | "teacher"

    @property
    def n_frames(self) -> int:
        return self.cs.shape[0]


class TransitionBuffer:
    """FIFO buffer of agent trajectories with window/seed sampling."""

    def __init__(self, max_transitions: int = 2_000_000, seed: int = 0) -> None:
        self._trajs: Deque[AgentTraj] = deque()          # learner data
        self._teacher: Deque[AgentTraj] = deque()        # teacher data
        self._n_trans = 0
        self._max_trans = int(max_transitions)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._n_trans

    @property
    def n_trajs(self) -> int:
        return len(self._trajs) + len(self._teacher)

    def add(self, traj: AgentTraj) -> None:
        if traj.n_trans <= 0:
            return
        # Teacher trajectories are unbounded in length (a surviving policy
        # emits ~10x the transitions of a falling learner): if they share
        # the learner FIFO they quickly dominate the fit, and the model
        # loses accuracy exactly on the learner's own action distribution
        # — the actor then exploits that optimism (imagined survival while
        # the real rollout falls). A side-pool with a fixed sampling share
        # keeps coverage without drowning learner data.
        dq = self._teacher if traj.source == "teacher" else self._trajs
        dq.append(traj)
        self._n_trans += traj.n_trans
        while self._n_trans > self._max_trans and self._trajs:
            self._n_trans -= self._trajs.popleft().n_trans

    def _sample_from(self, dq: Deque[AgentTraj],
                     recent_pool: float = 1.0) -> AgentTraj:
        """Weighted-by-length traj draw. ``recent_pool``<1 restricts the
        draw to the most recent ``recent_pool`` fraction of trajectories —
        the FIFO keeps every early falling trajectory, so uniform sampling
        over-trains the model on obsolete fall modes and biases imagined
        rollouts toward falling (model pessimism)."""
        n_total = len(dq)
        n = n_total if recent_pool >= 1.0 else max(
            1, int(n_total * recent_pool))
        lo = n_total - n
        weights = np.asarray(
            [dq[i].n_trans for i in range(lo, n_total)],
            dtype=np.float64)
        weights /= weights.sum()
        idx = lo + int(self._rng.choice(n, p=weights))
        return dq[idx]

    def _draw(self, teacher_share: float, recent_frac: float,
              recent_pool: float) -> AgentTraj:
        if self._teacher and (
                self._rng.random() < teacher_share or not self._trajs):
            return self._sample_from(self._teacher, 1.0)
        pool = recent_pool if self._rng.random() < recent_frac else 1.0
        return self._sample_from(self._trajs, pool)

    def sample_windows(
        self, batch: int, k: int,
        recent_frac: float = 0.0, recent_pool: float = 0.25,
        teacher_share: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Batch of k-step autoregressive windows.

        Returns:
          s0:    (batch, state_dim) window-start states cs_{j0-1}
          act:   (batch, k, action_dim) logged actions a_{j0..j0+k-1}
          cs:    (batch, k, state_dim) target states
          obs:   (batch, k, obs_dim) target observations
          aux:   {name: (batch, k)} aux targets
          done:  (batch, k) float 0/1 terminal flags
        """
        s0_l, act_l, cs_l, obs_l, done_l = [], [], [], [], []
        aux_l: Dict[str, List[np.ndarray]] = {}
        for _ in range(batch):
            traj = self._draw(teacher_share, recent_frac, recent_pool)
            tries = 0
            while traj.n_frames <= k:  # need at least k+1 frames for a k-window
                traj = self._draw(teacher_share, recent_frac, recent_pool)
                tries += 1
                if tries > 500:
                    longest = max(
                        t.n_frames
                        for t in list(self._trajs) + list(self._teacher))
                    raise RuntimeError(
                        f"sample_windows: no trajectory with n_frames>{k} "
                        f"(longest={longest}) after 500 draws — episodes "
                        "too short for the requested window"
                    )
            B = traj.n_frames
            hi = B - k  # j0 <= B-k
            j0 = int(self._rng.integers(1, hi + 1))
            s0_l.append(traj.cs[j0 - 1])
            act_l.append(traj.act[j0:j0 + k])
            cs_l.append(traj.cs[j0:j0 + k])
            obs_l.append(traj.obs[j0:j0 + k])
            done = np.zeros(k, dtype=np.float32)
            if traj.terminal:
                done[-1] = float(j0 + k - 1 == B - 1)
            done_l.append(done)
            for name, arr in traj.aux.items():
                aux_l.setdefault(name, []).append(arr[j0:j0 + k])
        out = {
            "s0": np.stack(s0_l).astype(np.float32),
            "act": np.stack(act_l).astype(np.float32),
            "cs": np.stack(cs_l).astype(np.float32),
            "obs": np.stack(obs_l).astype(np.float32),
            "done": np.stack(done_l).astype(np.float32),
            "aux": {n: np.stack(v).astype(np.float32) for n, v in aux_l.items()},
        }
        return out

    def sample_seeds(
        self, batch: int, start_frac: float = 0.0, start_frames: int = 4,
        recent_frac: float = 0.0, recent_pool: float = 0.25,
        tail_frac: float = 0.0, tail_span: int = 16,
        teacher_share: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(batch, state_dim) seed states + (batch, obs_dim) their observations.

        ``start_frac`` of seeds are drawn from the first ``start_frames`` of
        a trajectory — guaranteed-healthy states (robot still upright) so a
        fraction of every imagination batch retains survival gradient even
        when the buffer is dominated by about-to-fall trajectories.

        ``tail_frac`` of seeds are drawn from the last ``tail_span`` frames —
        for terminal trajectories these are states from which the true done
        boundary is reachable inside a ``tail_span``-step imagined horizon,
        so the alive-product survival term receives a direct gradient
        (uniform seeds on ~30-frame trajectories with H=8 almost never see
        termination inside the window; survival then rides only on the
        indirect critic bootstrap).
        """
        s_l, o_l = [], []
        for _ in range(batch):
            traj = self._draw(teacher_share, recent_frac, recent_pool)
            hi = traj.n_frames - 1  # frame index range [0, B-2]
            u = self._rng.random()
            if u < start_frac:
                f = int(self._rng.integers(0, min(start_frames, hi)))
            elif u < start_frac + tail_frac:
                lo = max(0, hi - tail_span)
                f = int(self._rng.integers(lo, hi)) if hi > lo else 0
            else:
                f = int(self._rng.integers(0, hi))
            s_l.append(traj.cs[f])
            o_l.append(traj.obs[f])
        return np.stack(s_l).astype(np.float32), np.stack(o_l).astype(np.float32)
