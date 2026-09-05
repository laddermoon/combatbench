"""CombatExperimentPPOBase — shared base for humanoid21 V2 PPO experiments.

Provides default values for all framework parameters, shared helpers
(self-play job construction, actor/critic building), and state persistence.
PPO-only — no SAC support.

Subclass and override class attributes + abstract methods.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

from baseline.framework.ppo import (
    CommonParams,
    ExperimentPPO,
    ExplorationSpec,
    PPOParams,
    TrainablePolicy,
)
from baseline.framework.critic_mlp import CriticMLP
from baseline.framework.rollout.job import EiSpec, Job


class CombatExperimentPPOBase(ExperimentPPO):
    """Class-attribute style base for humanoid21 combat V2 experiments.

    Subclass and override:
    - Class attributes (name, obs_dim, action_dim, env_blueprint, etc.)
    - ``reward_channels()`` — declare reward channels
    - ``build_trajectories()`` — episode → trajectories
    - ``on_eval()`` — eval processing + best-of-run
    """

    # --- Identity ---
    name: str = ""

    # --- Network shape ---
    obs_dim: int = 96
    action_dim: int = 21
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    # --- Exploration ---
    # explore_intensity: additive exploration strength ∈ [-1, 1].
    #   0 = neutral (no change to policy distribution).
    #   → +1 = maximum added exploration.
    #   → -1 = maximum exploration suppression.
    #   The specific mapping is policy-defined.
    # entropy_floor: training-side entropy floor ∈ [0, 1].
    #   The framework computes relu(entropy_floor - H_norm) to prevent
    #   policy collapse.  Set to 0 to disable.
    # entropy_coef: coefficient for the entropy floor loss.
    explore_intensity: float = 0.0
    entropy_floor: float = 0.3
    entropy_coef: float = 0.01

    # Sigma bounds — normalization reference points for the policy's
    # entropy, not hard clamps.  See DESIGN_migration_tanh_gaussian.md.
    log_std_min: float = -4.0
    log_std_max: float = 0.0

    # --- Temporally correlated exploration (OU) ---
    # Set at build time on policies that support it (e.g.
    # FixedSigmaGaussianMLPPolicy).  0.0 disables OU.
    noise_tau_steps: float = 0.0
    noise_scale: float = 0.0

    # --- Shared training ---
    learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    grad_clip_norm: float = 1.0

    # --- PPO knobs ---
    clip_eps: float = 0.2
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 8192

    # --- Rollout schedule ---
    episodes_per_update: int = 256 * 8
    max_updates: int = 10000
    eval_interval: int = 5
    eval_episodes: int = 16

    # --- Video recording ---
    video_eval_interval: int = 5

    # --- Parallelism ---
    rollout_workers: int = max(1, (os.cpu_count() or 1) // 2)

    seed: int = 42

    # --- Policy blueprint ---
    # Filename of the initial policy blueprint YAML under
    # humanoid21/blueprints/.  Used by build_actor() to construct the
    # actor.  New policy families override this to point to their own
    # init_policy_<family>.yaml.  Default matches the baseline
    # TanhGaussianMLPPolicy blueprint.
    actor_blueprint: str = "init_policy.yaml"

    # --- Rollout / env configuration (subclass overrides) ---
    # These parameters control how build_jobs() constructs rollout jobs.
    # Each job is a tuple:
    #   (policy_a_bp, policy_b_bp, env_bp, seed, episode_options)
    # The framework's ParallelRollouter.collect() consumes these jobs to
    # run parallel environment rollouts and produce Episode objects.
    #
    # env_blueprint: YAML filename under humanoid21/blueprints/ that
    #   defines the ParameterizedEnvBlueprint (env plugins, observers,
    #   termination conditions, etc.).  _env_pb() loads it from this path.
    #
    # agent_used: Controls which agent's perspective the rollout observes.
    #   "random"  — each episode randomly selects robot_a or robot_b as
    #               the observed agent (self-play).  The env_bp is
    #               materialized with agent_id set per-episode.
    #   "both"    — both agents are observed in a single env (dual mode).
    #               The env_bp is materialized without agent_id; the env
    #               itself manages both agents via DualImbalanceTerminationPlugin.
    #   "robot_a" — always observe robot_a (fixed single-agent mode).
    #   "robot_b" — always observe robot_b (fixed single-agent mode).
    #
    # max_steps: Maximum number of environment steps per episode.
    #   Passed to env_bp.materialize(max_steps=...).
    #
    # init_distance_min / init_distance_max: Range for the initial
    #   distance between the two robots at episode reset.  A random
    #   value uniformly sampled from [min, max] is placed in
    #   episode_options["initial_distance"] for each job.
    env_blueprint: str = ""
    agent_used: str = "random"
    max_steps: int = 200
    init_distance_min: float = 1.5
    init_distance_max: float = 3.5

    # ------------------------------------------------------------------
    # Parameter access (ExperimentPPO interface)
    # ------------------------------------------------------------------

    def common_params(self) -> CommonParams:
        return CommonParams(
            name=self.name,
            learning_rate=self.learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            grad_clip_norm=self.grad_clip_norm,
            episodes_per_update=self.episodes_per_update,
            max_updates=self.max_updates,
            eval_interval=self.eval_interval,
            eval_episodes=self.eval_episodes,
            video_eval_interval=self.video_eval_interval,
            rollout_workers=self.rollout_workers,
            seed=self.seed,
        )

    def ppo_params(self) -> PPOParams:
        return PPOParams(
            clip_eps=self.clip_eps,
            target_kl=self.target_kl,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
        )

    # ------------------------------------------------------------------
    # Update feedback & Exploration scheduling
    # ------------------------------------------------------------------

    def on_update(self, stats, update: int) -> None:
        """Default: no-op.  Override to accumulate training stats for
        closed-loop exploration scheduling, e.g.::

            def on_update(self, stats, update):
                self._kl_history.append(stats.approx_kl)

            def exploration(self, update):
                coef = self.entropy_coef
                if len(self._kl_history) >= 3 and all(
                    kl < 0.005 for kl in self._kl_history[-3:]
                ):
                    coef *= 4.0  # KL flat for 3 updates, push exploration
                return ExplorationSpec(entropy_coef=coef)
        """
        pass

    def exploration(self, update: int) -> ExplorationSpec:
        """Static exploration spec built from the class attributes.

        Returns the three-field ``ExplorationSpec``:
        - ``explore_intensity``: additive exploration strength
          (0 = neutral, +1 = expand, -1 = suppress).  Default 0.0.
        - ``entropy_floor``: training-side entropy floor.  Default 0.3.
        - ``entropy_coef``: coefficient for the entropy floor loss.

        Subclasses that want a schedule override ``on_update`` (to absorb
        stats) and this method (to read accumulated state).  See the
        ``on_update`` docstring for a closed-loop example.
        """
        return ExplorationSpec(
            explore_intensity=self.explore_intensity,
            entropy_floor=self.entropy_floor,
            entropy_coef=self.entropy_coef,
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_actor(self, device: torch.device) -> TrainablePolicy:
        blueprint_dir = Path(__file__).resolve().parent.parent / "humanoid21" / "blueprints"
        bp = PolicyBlueprint.load(blueprint_dir / self.actor_blueprint)
        actor = bp.build().to(device)
        # The actor owns its distribution bounds now that they left
        # PPOParams. Both are forced here (previously only log_std_min was),
        # so the class attributes are authoritative and the values baked into
        # init_policy.yaml cannot silently diverge from them.
        # Guarded with hasattr so policy families without scalar
        # log_std_min/max (e.g. mixture, flow) aren't forced to grow
        # attributes they don't use.
        if hasattr(actor, "log_std_min"):
            actor.log_std_min = float(self.log_std_min)
        if hasattr(actor, "log_std_max"):
            actor.log_std_max = float(self.log_std_max)
        # OU exploration params (for FixedSigmaGaussianMLPPolicy).
        # These were previously passed via ExplorationSpec every update,
        # but now that ExplorationSpec no longer carries them, they are
        # set once at build time.
        if hasattr(actor, "_noise_tau_steps"):
            actor._noise_tau_steps = float(self.noise_tau_steps)
            actor._update_ou_params()
        if hasattr(actor, "_noise_scale"):
            actor._noise_scale = float(self.noise_scale)
        return actor

    def build_critic(self, channel_name: str, device: torch.device) -> nn.Module:
        return CriticMLP(
            obs_dim=self.obs_dim, hidden_dim=self.critic_hidden_dim,
        ).to(device)

    # ------------------------------------------------------------------
    # Job construction (unified build_jobs)
    # ------------------------------------------------------------------

    def build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
        *,
        explore_intensity: EiSpec = 0.0,
    ) -> List[Job]:
        """Build self-play rollout jobs.

        ``explore_intensity`` is placed into each :class:`Job`'s
        ``explore_intensity_a`` / ``explore_intensity_b`` fields (not
        into ``episode_options``).  The episode runner resolves it
        per-frame and passes it to ``policy.act``.

        Subclass can override for non-self-play scenarios.
        """
        return self._build_selfplay_jobs(
            self._env_pb(), policy_bp, base_seed, n_episodes,
            explore_intensity=explore_intensity,
        )

    def _env_pb(self) -> ParameterizedEnvBlueprint:
        """Load the ParameterizedEnvBlueprint from ``env_blueprint`` filename.

        Subclass sets ``env_blueprint`` to the yaml filename (relative
        to ``humanoid21/blueprints/``).  Override only for non-standard
        blueprint loading logic.
        """
        if not self.env_blueprint:
            raise ValueError(
                f"{self.__class__.__name__} must set env_blueprint "
                "to a blueprint filename"
            )
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent / "humanoid21" / "blueprints" / self.env_blueprint
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_explore_intensity(episode, agent_id: str, T: int) -> np.ndarray:
        """Extract per-frame explore_intensity for one agent, truncated to T.

        Reads from ``episode.explore_intensities[agent_id]`` — the
        per-frame input that was passed to ``policy.act`` at rollout
        time, recorded by the episode runner.  Returns a ``(T,)``
        float32 array defaulting to 0.0 (neutral) when the episode has
        no recorded explore_intensity.
        """
        ei = episode.explore_intensities.get(agent_id)
        if ei is None:
            return np.full(T, 0.0, dtype=np.float32)
        return np.asarray(ei, dtype=np.float32)[:T]

    @staticmethod
    def _agent_from_rollout_seed(seed: int) -> str:
        rng = np.random.default_rng(int(seed) + 937)
        return "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"

    def _build_selfplay_jobs(
        self,
        env_pb: ParameterizedEnvBlueprint,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
        *,
        explore_intensity: EiSpec = 0.0,
    ) -> List[Job]:
        rng = np.random.default_rng(base_seed)

        if self.agent_used == "both":
            env_bp = env_pb.materialize(max_steps=self.max_steps)
            jobs: List[Job] = []
            for i in range(n_episodes):
                seed = int(base_seed + i)
                initial_distance = float(
                    rng.uniform(self.init_distance_min, self.init_distance_max)
                )
                jobs.append(Job(
                    policy_a_bp=policy_bp,
                    policy_b_bp=policy_bp,
                    env_bp=env_bp,
                    seed=seed,
                    episode_options={"initial_distance": initial_distance},
                    explore_intensity_a=explore_intensity,
                    explore_intensity_b=explore_intensity,
                ))
            return jobs

        # Single-agent modes: robot_a, robot_b, or random
        agent_ids: Tuple[str, ...]
        if self.agent_used == "random":
            agent_ids = ("robot_a", "robot_b")
        else:
            agent_ids = (self.agent_used,)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=self.max_steps, agent_id=aid)
            for aid in agent_ids
        }

        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            if self.agent_used == "random":
                agent_id = self._agent_from_rollout_seed(seed)
            else:
                agent_id = self.agent_used
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=policy_bp,
                env_bp=env_bps[agent_id],
                seed=seed,
                episode_options={"agent_id": agent_id, "initial_distance": initial_distance},
                explore_intensity_a=explore_intensity,
                explore_intensity_b=explore_intensity,
            ))
        return jobs

    # ------------------------------------------------------------------
    # State persistence (ExperimentPPO interface)
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass
