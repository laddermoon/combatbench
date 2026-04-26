"""High-level wrappers around :class:`RunningMeanStd`.

Two flavors, matching the PPO / CleanRL / SB3 conventions:

  * :class:`ObservationNormalizer`
        Standardize observations: subtract running mean, divide by
        running std, optionally clip. Applied at policy input time.
        Stats are updated only when ``training=True`` (so eval rollouts
        don't poison the running estimate).

  * :class:`ReturnNormalizer`
        PPO-style reward scaling: keep a running discounted return
        estimate, track its variance, divide rewards by ``sqrt(var)``.
        We do **not** subtract the mean — the sign of a reward signal is
        informative and centering it would distort the policy gradient.
        This matches the SB3 ``VecNormalize(norm_reward=True)`` recipe
        (see Engstrom et al. 2020 "Implementation Matters in Deep RL").

Both classes carry a ``RunningMeanStd`` instance and expose
``state_dict`` / ``load_state_dict`` so PPO checkpoints round-trip the
normalizer state alongside actor/critic weights — otherwise resumed
training would silently destabilize for the first few iterations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .running_mean_std import RunningMeanStd


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class ObservationNormalizer:
    """Standardize observations against a running mean / variance."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        *,
        clip_range: Optional[float] = 10.0,
        epsilon: float = 1e-8,
        rms_epsilon: float = 1e-4,
    ) -> None:
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        self.rms = RunningMeanStd(self.shape, epsilon=rms_epsilon)
        self._clip_range = clip_range
        self._eps = float(epsilon)

    def update(self, obs: np.ndarray) -> None:
        """Feed observations into the running statistics."""
        self.rms.update(obs)

    def __call__(
        self,
        obs: np.ndarray,
        *,
        update: bool = False,
    ) -> np.ndarray:
        """Normalize ``obs``; optionally update running stats first.

        Pass ``update=True`` during training rollouts so the estimate
        keeps tracking the policy-induced distribution. Pass
        ``update=False`` for evaluation / inference so eval data does
        not perturb the training-time normalizer.
        """
        if update:
            self.rms.update(obs)
        out = self.rms.normalize(obs, eps=self._eps, center=True)
        if self._clip_range is not None:
            np.clip(out, -self._clip_range, self._clip_range, out=out)
        return out

    # Checkpoint IO
    def state_dict(self) -> Dict[str, Any]:
        return {
            "shape": tuple(self.shape),
            "rms": self.rms.state_dict(),
            "clip_range": self._clip_range,
            "epsilon": self._eps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        shape = tuple(state["shape"])
        if shape != self.shape:
            raise ValueError(
                f"Saved obs shape {shape} != current {self.shape}."
            )
        self.rms.load_state_dict(state["rms"])
        self._clip_range = state.get("clip_range", self._clip_range)
        self._eps = float(state.get("epsilon", self._eps))


# ---------------------------------------------------------------------------
# Return / reward
# ---------------------------------------------------------------------------
class ReturnNormalizer:
    """PPO-style reward scaling via the running variance of discounted returns.

    Maintains a per-environment scalar accumulator ``return_acc`` updated
    each step:

        return_acc <- gamma * return_acc * (1 - done) + reward

    The variance of ``return_acc`` (across all envs and time) is what
    we track in :class:`RunningMeanStd`; rewards are then divided by
    ``sqrt(var) + eps``. Mean is NOT subtracted (see module docstring).

    Two update entry points:

      * :meth:`update_step` — online use, tick once per env-step with
        per-env ``rewards`` and ``dones`` arrays of shape ``(num_envs,)``.

      * :meth:`update_from_episodes` — batch use, tick over a list of
        completed episodes (whose dones we infer from the
        ``terminated`` / ``truncated`` flags). Convenient when the
        rollout is collected as ``RolloutBatch`` instances rather than
        as a stream.

    Either way, ``__call__(rewards)`` returns the normalized (scaled)
    rewards using the current variance estimate.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        num_envs: int = 1,
        clip_range: Optional[float] = 10.0,
        epsilon: float = 1e-8,
        rms_epsilon: float = 1e-4,
    ) -> None:
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must lie in [0, 1); got {gamma}")
        self._gamma = float(gamma)
        self._clip_range = clip_range
        self._eps = float(epsilon)
        self._return_acc: np.ndarray = np.zeros(int(num_envs), dtype=np.float64)
        self.rms = RunningMeanStd(shape=(), epsilon=rms_epsilon)

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def num_envs(self) -> int:
        return int(self._return_acc.shape[0])

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update_step(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        """Tick the running-return estimate one env-step.

        ``rewards`` / ``dones`` shape ``(num_envs,)``. ``dones`` may be
        ``bool`` or ``{0, 1}``; an episode boundary zeroes the
        accumulator *before* adding the new step's reward.
        """
        rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones_arr = np.asarray(dones).astype(np.float64).reshape(-1)
        if rewards_arr.shape != self._return_acc.shape:
            raise ValueError(
                f"rewards shape {rewards_arr.shape} != num_envs "
                f"{self._return_acc.shape}."
            )
        self._return_acc = self._gamma * self._return_acc * (1.0 - dones_arr) + rewards_arr
        self.rms.update(self._return_acc)

    def update_from_episodes(
        self,
        rewards_per_episode: list,
        dones_per_episode: Optional[list] = None,
    ) -> None:
        """Update from a list of per-episode reward arrays.

        Each entry of ``rewards_per_episode`` is a 1-D float array; the
        accumulator is reset to 0 between episodes (every episode is
        treated as starting fresh — caller-side ``terminated`` /
        ``truncated`` are equivalent for variance-tracking purposes).
        """
        # We pretend ``num_envs=1`` here regardless of self.num_envs:
        # batch updates from full episodes don't model parallel envs
        # interleaving. Tracked variance is what matters, not the
        # accumulator's identity.
        local_acc = 0.0
        flat_returns: list = []
        for ep_idx, rewards in enumerate(rewards_per_episode):
            local_acc = 0.0  # episode boundary
            for r in np.asarray(rewards, dtype=np.float64):
                local_acc = self._gamma * local_acc + float(r)
                flat_returns.append(local_acc)
        if flat_returns:
            self.rms.update(np.asarray(flat_returns, dtype=np.float64))

    # ------------------------------------------------------------------
    # Use
    # ------------------------------------------------------------------
    def __call__(self, rewards: np.ndarray) -> np.ndarray:
        """Return rewards scaled by ``1 / (sqrt(var) + eps)``."""
        out = self.rms.normalize(rewards, eps=self._eps, center=False)
        if self._clip_range is not None:
            np.clip(out, -self._clip_range, self._clip_range, out=out)
        return out

    # ------------------------------------------------------------------
    # Checkpoint IO
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self._gamma,
            "clip_range": self._clip_range,
            "epsilon": self._eps,
            "return_acc": self._return_acc.copy(),
            "rms": self.rms.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._gamma = float(state["gamma"])
        self._clip_range = state.get("clip_range", self._clip_range)
        self._eps = float(state.get("epsilon", self._eps))
        self._return_acc = np.asarray(state["return_acc"], dtype=np.float64).copy()
        self.rms.load_state_dict(state["rms"])
