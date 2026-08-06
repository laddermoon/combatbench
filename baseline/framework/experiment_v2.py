"""ExperimentV2 — clean PPO-only experiment abstraction.

This module defines the new generation of experiment interface, designed
from scratch to work with the V2 ``Trajectory`` data structures.  It
eliminates the legacy baggage accumulated in ``experiment.py`` (dual
PPO/SAC support, three generations of segment APIs, framework-managed
weight scheduling) and provides a single, coherent contract between the
experiment author and the PPO training loop.

Design principles
-----------------

1. **Experiment owns the full data pipeline.**

   The v1 design split reward extraction (``extract_rewards``), episode
   splitting (``prepare_segments``), and weight scheduling
   (``initial_weights`` / ``next_weights``) across separate methods, with
   the framework normalizing weights and passing them back into the
   trajectory builder.  This created a circular dependency: the experiment
   needed to know the framework's normalization to control actor_weight,
   and the framework needed to know the experiment's weight semantics to
   combine advantages.

   In V2, ``build_trajectories(episode)`` is the single source of truth.
   The experiment decides:
   - How to slice the episode into trajectories (phase-based, gating-based,
     or whole-episode).
   - Per-channel rewards (dense shaping, terminal bonuses, penalties).
   - Per-channel termination (``is_terminated`` → V=0 bootstrap, or
     ``truncated`` → bootstrap from critic).
   - Per-channel actor_weight (how much this channel's advantage influences
     the policy gradient).  This can vary per trajectory, enabling
     curriculum scheduling without framework involvement.

2. **Reward channels are first-class.**

   Each ``RewardChannel`` declares its own ``gamma`` and ``gae_lambda``.
   The framework builds one critic per channel and uses the channel's
   parameters for GAE computation.  This replaces the v1 pattern of a
   separate ``gammas`` dict in ``CommonParams`` and a global
   ``gae_lambda`` in ``PPOParams``.

   Per-channel ``gae_lambda`` is a V2 unique capability: sparse terminal
   rewards benefit from high λ (low bias), while dense shaping rewards
   benefit from lower λ (low variance).  The experiment author can now
   tune this per channel without framework changes.

3. **PPO only.**

   SAC support is removed.  No ``SACParams``, no ``build_q_critic``, no
   ``sac_params()``.  This eliminates the algorithm-dispatch complexity
   that complicated v1's abstract methods.  If SAC is needed in the
   future, a separate ``ExperimentV2SAC`` class can be created without
   polluting the PPO interface.

4. **One job builder.**

   ``build_jobs(policy_bp, base_seed, n_episodes)`` replaces the separate
   ``build_rollout_jobs`` and ``build_eval_jobs`` methods.  The caller
   (training loop) controls whether the policy is stochastic (training
   rollout) or deterministic (evaluation) by passing the appropriate
   ``PolicyBlueprint``.  The experiment does not need to know whether it
   is building jobs for training or evaluation — it just builds N jobs
   with the given blueprint and seed.

5. **Coexistence, not replacement.**

   The old ``Experiment`` ABC in ``experiment.py`` and
   ``CombatExperimentBase`` in ``base.py`` remain untouched.  The 40+
   existing v1 experiments continue to work via the legacy training path
   (``train_ppo()``).  The training CLI (``train.py``) dispatches to
   ``train_ppo_v2()`` or ``train_ppo()`` based on whether the experiment
   is an ``ExperimentV2`` instance.

   New experiments should inherit from ``ExperimentV2`` directly (or from
   a future ``CombatExperimentV2Base`` that provides shared combat
   defaults).  Old experiments are not migrated.

Data flow
---------

::

    ActorPolicyBlueprint (from framework)
         │
         ▼
    build_jobs(policy_bp, base_seed, n_episodes)
         │
         ▼  (ParallelRollouter collects)
    List[Episode]
         │
         ▼  (per episode)
    build_trajectories(episode)
         │
         ▼
    List[Trajectory]
    ┌──────────────────────────────────┐
    │ obs, actions, last_obs           │
    │ channels: {                      │
    │   "r_x": ChannelData(            │──▶ Critic "r_x": GAE(γ_x, λ_x) → adv_x, ret_x
    │     reward, is_terminated,       │      → critic loss: MSE(V, ret_x)
    │     actor_weight                 │
    │   ),                             │
    │   "r_y": ChannelData(            │──▶ Critic "r_y": GAE(γ_y, λ_y) → adv_y, ret_y
    │     reward, is_terminated,       │      → critic loss: MSE(V, ret_y)
    │     actor_weight                 │
    │   ),                             │
    │ }                                │
    │ importance, mode                 │
    └──────────────────────────────────┘
         │
         ▼  (PPOBuffer concatenates all trajectories)
    ppo_update(actor, critics, buf, ...)
         │
         ├── Per-channel: normalize advantages (z-score on active frames)
         ├── Combine: combined_adv = Σ_c  actor_weight_c * confidence_c * norm_adv_c
         ├── Critic update: minimize MSE(V_c(s), ret_c) on active frames
         └── Actor update: PPO clipped surrogate on combined_adv

What the Experiment controls vs what the framework handles
----------------------------------------------------------

| Stage              | Experiment                          | Framework                     |
|--------------------|-------------------------------------|-------------------------------|
| Model building     | build_actor, build_critic           | Creates optimizers            |
| Job construction   | build_jobs                          | ParallelRollouter.collect     |
| Episode→Trajectory | build_trajectories (full control)   | Calls it per episode          |
| GAE computation    | reward_channels (γ, λ config)       | Executes compute_gae          |
| Adv normalization  | —                                   | Default: z-score on active    |
| Adv combination    | —                                   | Default: weighted by aw*conf  |
| Critic update      | —                                   | MSE on returns, masked        |
| Actor update       | —                                   | PPO clipped surrogate         |
| Eval & scheduling  | on_eval (full control)              | Runs eval rollouts, exports   |
| Checkpointing      | scheduler_state/training_state      | Save/load model weights       |

Removed from v1
---------------

- ``initial_weights()`` / ``next_weights()``: The experiment manages
  actor_weight internally in ``build_trajectories``.  No external weight
  scheduling loop.
- ``extract_rewards()``: Folded into ``build_trajectories``.
- ``prepare_segments()`` / ``prepare_training_segments()``: Folded into
  ``build_trajectories``.
- ``build_rollout_jobs()`` / ``build_eval_jobs()``: Merged into
  ``build_jobs()``.
- ``video_env_blueprint()``: Deprecated; video uses eval_jobs[0].
- ``Segment`` dataclass: Replaced by ``Trajectory`` + ``ChannelData``.
- ``SACParams``, ``sac_params()``, ``build_q_critic()``: PPO only.
- ``gammas`` dict in ``CommonParams``: Replaced by ``reward_channels()``.
- ``stage_weights`` parameter in ``ppo_update``: Replaced by per-frame
  ``actor_weight`` from ``ChannelData``.
- ``_current_actor_weights`` hack: Experiment owns weight scheduling.
- ``compute_episode_metrics()`` / ``compare_eval()`` / ``scheduler_info()``:
  Merged into ``on_eval()`` — the experiment receives raw episodes and
  handles metrics, best-of-run judgment, and state updates internally.
- ``normalize_advantages()`` / ``combine_advantages()`` /
  ``normalize_sample_weights()``: Removed — framework defaults are not
  customizable.  The experiment controls the pipeline through
  ``reward_channels()`` and ``ChannelData.actor_weight``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint


# ---------------------------------------------------------------------------
# Actor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TrainablePolicy(Protocol):
    """Interface that the PPO trainer requires from an actor.

    The actor must support:
    - ``evaluate_actions``: recompute log_prob and entropy for given
      (obs, actions) pairs.  Used for PPO importance ratio computation.
    - ``to_blueprint``: export a rollout-ready policy blueprint, with
      a ``stochastic`` flag to control sampling vs deterministic mode.
    """

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor,
        *, frame_modes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (log_prob, entropy) for given obs/actions.

        If ``frame_modes`` is provided, the actor should use it to route
        samples to the appropriate sub-network instead of computing mode
        from the observation.  Values are experiment-defined floats.
        """
        ...

    def to_blueprint(
        self, dest_path: str, *, stochastic: bool = False,
    ) -> PolicyBlueprint:
        """Export a rollout-ready policy blueprint.

        Args:
            dest_path: Directory path for the exported blueprint.
            stochastic: If True, the blueprint uses stochastic sampling
                (for training rollouts).  If False (default), it uses
                deterministic mean actions (for evaluation).
        """
        ...


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommonParams:
    """Training parameters shared across all PPO experiments.

    Unlike v1, this does NOT include ``gammas`` — per-channel gamma and
    gae_lambda are declared in ``RewardChannel`` via ``reward_channels()``.
    """

    name: str
    learning_rate: float
    critic_learning_rate: float
    grad_clip_norm: float
    episodes_per_update: int
    max_updates: int
    eval_interval: int
    eval_episodes: int
    video_eval_interval: int
    rollout_workers: int
    eval_workers: int
    seed: int


@dataclass(frozen=True)
class PPOParams:
    """PPO hyperparameters.

    Per-channel ``gae_lambda`` is declared in ``RewardChannel`` via
    ``reward_channels()`` — it is NOT a global parameter here.
    """

    log_std_min: float
    log_std_max: float
    clip_eps: float
    entropy_coef: float
    target_kl: float
    update_epochs: int
    minibatch_size: int


# ---------------------------------------------------------------------------
# Job type alias
# ---------------------------------------------------------------------------

# A rollout job is a tuple of:
#   (policy_a_blueprint, policy_b_blueprint, env_blueprint, seed, episode_options)
# The experiment builds these in build_jobs(); the framework's
# ParallelRollouter.collect() consumes them.
Job = Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]


# ---------------------------------------------------------------------------
# ExperimentV2 ABC
# ---------------------------------------------------------------------------

class ExperimentV2(ABC):
    """Clean PPO-only experiment abstraction.

    This is the new generation of experiment interface, designed to work
    with V2 ``Trajectory`` data structures.  See the module docstring for
    the full design rationale.

    Subclasses must implement all abstract methods.  Optional override
    methods have default implementations (return None → framework default).

    The experiment is a plain Python object (typically a singleton
    ``EXPERIMENT = MyConfig()`` at module level).  It is not re-instantiated
    per update — the framework calls its methods repeatedly.

    Typical subclass structure::

        class MyExperiment(ExperimentV2):
            # Class-attribute configuration
            name = "my_experiment"
            obs_dim = 96
            action_dim = 21
            # ...

            def reward_channels(self):
                return (
                    RewardChannel("r_balance", gamma=0.99, gae_lambda=0.95),
                    RewardChannel("r_posture", gamma=0.99, gae_lambda=0.90),
                )

            def common_params(self):
                return CommonParams(
                    name=self.name,
                    reward_keys=tuple(ch.name for ch in self.reward_channels()),
                    obs_dim=self.obs_dim,
                    # ...
                )

            def build_actor(self, device):
                ...

            def build_critic(self, channel_name, device):
                ...

            def build_jobs(self, policy_bp, base_seed, n_episodes):
                ...

            def build_trajectories(self, episode):
                ...

            def compute_episode_metrics(self, episode):
                ...

            def compare_eval(self, esum, best_esum):
                ...

            def scheduler_info(self):
                ...
    """

    # ==================================================================
    # Phase 0: Configuration & Model Building
    # ==================================================================

    @abstractmethod
    def reward_channels(self) -> Tuple["RewardChannel", ...]:
        """Declare all reward channels for this experiment.

        Returns one ``RewardChannel`` per critic.  The framework builds
        one V(s) critic per channel and uses the channel's ``gamma`` and
        ``gae_lambda`` for GAE computation.

        The order of channels defines ``reward_keys`` — the framework
        extracts ``ch.name`` for each channel to form the keys tuple.
        This replaces v1's separate ``reward_keys`` and ``gammas`` dict.

        Returns:
            Tuple of RewardChannel, e.g.::

                (
                    RewardChannel("r_stand", gamma=0.99, gae_lambda=0.95),
                    RewardChannel("r_balance", gamma=0.99, gae_lambda=0.90),
                )
        """
        ...

    @abstractmethod
    def common_params(self) -> CommonParams:
        """Return training parameters (lr, episodes_per_update, etc.).

        ``reward_keys`` must match the names from ``reward_channels()``.
        ``gammas`` is NOT included — gamma lives in ``RewardChannel``.
        """
        ...

    @abstractmethod
    def ppo_params(self) -> PPOParams:
        """Return PPO-specific hyperparameters."""
        ...

    @abstractmethod
    def build_actor(self, device: torch.device) -> TrainablePolicy:
        """Build and return the actor policy on the given device."""
        ...

    @abstractmethod
    def build_critic(self, channel_name: str, device: torch.device) -> nn.Module:
        """Build a V(s) critic for one reward channel.

        Called once per channel at training start.  The critic must
        accept an observation tensor of shape ``(B, obs_dim)`` and return
        a value tensor of shape ``(B,)`` or ``(B, 1)``.

        Args:
            channel_name: The ``RewardChannel.name`` for this critic.
            device: Torch device to place the model on.
        """
        ...

    # ==================================================================
    # Phase 1: Job Construction
    # ==================================================================

    @abstractmethod
    def build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Job]:
        """Build rollout jobs for training or evaluation.

        This unified method replaces v1's separate ``build_rollout_jobs``
        and ``build_eval_jobs``.  The caller controls whether the policy
        is stochastic (training) or deterministic (eval) by passing the
        appropriate ``policy_bp``.

        Args:
            policy_bp: The actor's exported policy blueprint.  For
                training rollouts, this has ``stochastic=True``.  For
                evaluation, it is deterministic (mean action).
            base_seed: Base random seed for this batch.  Each job should
                use ``base_seed + i`` as its seed.
            n_episodes: Number of episodes to build.

        Returns:
            List of Job tuples:
            ``(policy_a_bp, policy_b_bp, env_bp, seed, episode_options)``
        """
        ...

    # ==================================================================
    # Phase 2: Episode → Trajectory
    # ==================================================================

    @abstractmethod
    def build_trajectories(self, episode: "Episode") -> List["Trajectory"]:
        """Convert an episode into training trajectories.

        This is the single source of truth for:
        - How the episode is sliced into trajectories (phase-based,
          gating-based, or whole-episode).
        - Per-channel rewards (dense shaping, terminal bonuses, penalties).
        - Per-channel termination (``is_terminated`` → V=0, or
          ``truncated`` → bootstrap from critic).
        - Per-channel ``actor_weight`` (how much this channel's advantage
          influences the policy gradient).  This is where curriculum
          scheduling happens — the experiment decides the weight, not the
          framework.
        - Which channels are active on each trajectory (channels absent
          from ``Trajectory.channels`` are inactive).

        Returns an empty list to skip the episode entirely.

        The experiment should NOT fill ``Trajectory.log_prob`` — the
        framework's PPOBuffer does this via a batched
        ``actor.evaluate_actions`` call.
        """
        ...

    # ==================================================================
    # Phase 3 & 4: Critic & Actor Update
    # ==================================================================
    #
    # The framework handles GAE computation, advantage normalization
    # (z-score on active frames), advantage combination (weighted by
    # actor_weight * confidence), and PPO clipped surrogate.  These are
    # not customizable — the experiment controls the pipeline through
    # ``reward_channels()`` (γ, λ) and ``ChannelData.actor_weight``
    # (per-channel influence on the actor).

    # ==================================================================
    # Evaluation
    # ==================================================================

    @abstractmethod
    def on_eval(
        self, episodes: List["Episode"], update: int,
    ) -> Dict[str, Any]:
        """Process evaluation results and update internal state.

        Called once per eval cycle with all raw eval episodes.  The
        experiment is responsible for:

        - Computing per-episode and aggregate metrics (replaces v1's
          ``compute_episode_metrics`` + framework aggregation).
        - Updating internal curriculum/scheduler state based on eval
          results (replaces v1's ``next_weights``).
        - Determining whether this eval is a new best (replaces v1's
          ``compare_eval``).
        - Returning logging info (replaces v1's ``scheduler_info``).

        The framework does NOT interpret any metrics — it only uses
        ``is_new_best`` to decide whether to export the policy, and
        passes ``info`` through to the logging line.

        Args:
            episodes: Raw eval episodes from rollout.
            update: Current update index (0-based).

        Returns:
            Dict with at least::

                {
                    "is_new_best": bool,   # export policy if True
                    "info": Dict[str, Any],  # free-form logging info
                }

            The ``info`` dict is printed by the framework as-is (e.g.
            ``{"phase": "stability", "mean_length": 187.5, ...}``).
        """
        ...

    # ==================================================================
    # State Persistence (for checkpoint resume)
    # ==================================================================

    def state(self) -> dict:
        """Serialize all internal state for checkpointing.

        The framework calls this when saving a checkpoint and passes
        the returned dict to ``load_state()`` on resume.

        Suggested keys (experiment decides what's relevant):

        - ``best_eval``: Best eval result so far (for ``on_eval`` to
          compare against and determine ``is_new_best``).
        - ``curriculum``: Current curriculum phase, stage, or any
          scheduling state that affects ``build_trajectories`` behavior
          (e.g. actor_weight schedule, phase transitions).
        - ``update_count``: Number of updates completed (if the
          experiment tracks this internally).
        - Any other mutable state that influences ``build_jobs``,
          ``build_trajectories``, or ``on_eval``.

        Returns:
            JSON-serializable dict.  Empty dict if no state to persist.
        """
        return {}

    def load_state(self, state: dict) -> None:
        """Restore internal state from a checkpoint.

        Args:
            state: The dict previously returned by ``state()``.
        """
        pass

    # ==================================================================
    # Serialization
    # ==================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable config snapshot.

        Override to add experiment-specific fields.  The framework calls
        this to write ``run_dir/config.json``.
        """
        return {
            "name": getattr(self, "name", ""),
            "reward_keys": [ch.name for ch in self.reward_channels()],
        }

    def save_run_config(
        self, run_dir: Path, *, smoke: bool = False, algo: str = "ppo",
    ) -> None:
        """Save run configuration to ``run_dir/config.json``."""
        import json
        import time

        payload = {
            "experiment": self.to_dict(),
            "algorithm": algo,
            "smoke": smoke,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)
