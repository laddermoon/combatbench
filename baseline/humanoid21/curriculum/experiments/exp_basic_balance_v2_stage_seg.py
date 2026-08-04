"""basic_balance_v2_stage_seg: staged reward with segment-based phase control.

Based on basic_balance_v2, adds two-phase reward scheme:
  - **Struggle phase**: r_struggle (terminal-only: +1 recover, -1 fall) +
    r_height (per-step dense shaping: height * 0.01).
  - **Stability phase**: same as basic_balance_v2 (r_cross, r_joint, r_vel,
    r_tilt, r_foot) + r_struggle (terminal: -1 if degrades to struggle).

Uses PhaseObserver (uprightness + height with hysteresis) to determine phase.
Uses prepare_segments (v2 API) to split episodes into per-phase segments with
per-key critic control via Segment.key_weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.experiment import Segment
from baseline.framework.ppo_trainer import _extract_per_step_scalar, _extract_per_step_field
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


class BasicBalanceV2StageSegConfig(CombatExperimentBase):

    name = "basic_balance_v2_stage_seg"
    # r_struggle replaces r_fall: struggle-phase survival + phase transition rewards
    # Stability-phase keys remain the same as basic_balance_v2
    reward_keys = ("r_struggle", "r_height", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    gammas = {
        "r_struggle": 0.99,
        "r_height": 0.99,
        "r_cross": 0.99,
        "r_joint": 0.99,
        "r_vel": 0.99,
        "r_tilt": 0.99,
        "r_foot": 0.99,
    }

    BLUEPRINT = "basic_balance_v2_stage_seg_env.yaml"

    sac_auto_alpha = True

    _survival_rate: float = 0.0

    # Phase reward constants.
    #
    # All r_struggle signal is terminal; there is deliberately NO per-step
    # term in either phase.  With terminal-only rewards the discount factor
    # alone orders every outcome correctly (see extract_rewards).
    #
    # A per-step struggle penalty must NOT be reintroduced.  With a -0.01
    # per-step penalty and a -1.0 fall terminal, the return for falling
    # after k struggle steps collapses to a constant:
    #
    #   G(k) = -0.01 * (1 - g^k)/(1 - g)  -  g^(k-1)
    #        = -(1 - g^k) - g^(k-1)                  [0.01/(1-0.99) = 1]
    #        = -1 - 0.01 * g^(k-1)
    #
    # i.e. the per-step penalty's infinite-horizon discounted sum
    # (-0.01/(1-g) = -1.0) exactly equals the terminal penalty, so the two
    # are interchangeable and all dependence on k cancels.  The return
    # spans only 0.01 across every k, versus 1.0 for the terminal-only
    # scheme -- a 100x loss of learning signal.  This was observed live as
    # r_struggle return std = 0.005 and explained_variance ~ 0, which zeroed
    # the key's contribution to the policy gradient.
    struggle_recover_bonus: float = 1.0
    struggle_fall_penalty: float = -1.0
    stability_to_struggle_penalty: float = -1.0

    def video_env_blueprint(self):
        return self._make_video_blueprint(self._env_pb())

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def build_rollout_jobs(self, policy_bp, base_seed):
        return self._build_selfplay_jobs(self._env_pb(), policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp, base_seed):
        return self._build_selfplay_jobs(self._env_pb(), policy_bp, base_seed, self.eval_episodes)

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("survived", 0.0) > best_esum.get("survived", 0.0)

    def initial_weights(self) -> Tuple[float, ...]:
        return (3.0, 0.3, 1.0, 0.2, 0.2, 0.2, 0.2)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        survival_rate = float(eval_metrics.get("survived", 0.0))
        self._survival_rate = survival_rate
        return (3.0, 0.3, 1.0, 0.2, 0.2, 0.2, 0.2)

    def _extract_phase_info(self, episode) -> Tuple[np.ndarray, np.ndarray]:
        """Extract per-step phase and transition arrays from PhaseObserver output.

        Returns:
            is_struggle: (T,) bool array — True if step is in struggle phase.
            transition: (T,) object array — transition type per step.
        """
        T = episode.num_frames
        phase_node = episode.observer_outputs.get("phase")
        if phase_node is None:
            return np.zeros(T, dtype=bool), np.array(["none"] * T, dtype=object)

        is_struggle = np.zeros(T, dtype=bool)
        transitions = np.array(["none"] * T, dtype=object)

        if isinstance(phase_node, dict):
            # PhaseObserver outputs a dict with per-step fields
            phase_arr = phase_node.get("is_struggle")
            trans_arr = phase_node.get("transition")
            if phase_arr is not None:
                is_struggle = np.asarray(phase_arr, dtype=bool).reshape(-1)
                if is_struggle.shape[0] != T:
                    is_struggle = np.zeros(T, dtype=bool)
            if trans_arr is not None:
                transitions = np.asarray(trans_arr, dtype=object).reshape(-1)
                if transitions.shape[0] != T:
                    transitions = np.array(["none"] * T, dtype=object)
        else:
            # If observer outputs a scalar/list of scalars
            try:
                raw = np.asarray(phase_node, dtype=object).reshape(-1)
                for t in range(min(len(raw), T)):
                    val = raw[t]
                    if isinstance(val, dict):
                        is_struggle[t] = val.get("is_struggle", False)
                        transitions[t] = val.get("transition", "none")
                    elif isinstance(val, str):
                        is_struggle[t] = val == "struggle"
            except Exception:
                pass

        return is_struggle, transitions

    def _phase_runs(self, episode) -> List[Tuple[int, int, bool]]:
        """Decompose the episode into contiguous same-phase runs.

        Returns a list of ``(start, end, is_struggle)`` with ``end`` exclusive,
        covering ``[0, T)`` with no gaps.

        This is the single source of truth for phase segmentation: both
        ``extract_rewards`` and ``prepare_segments`` derive from it, so the
        boundary rewards and the segment boundaries can never disagree.
        """
        T = episode.num_frames
        if T == 0:
            return []

        is_struggle, _ = self._extract_phase_info(episode)

        runs: List[Tuple[int, int, bool]] = []
        seg_start = 0
        current = bool(is_struggle[0])
        for t in range(1, T):
            if bool(is_struggle[t]) != current:
                runs.append((seg_start, t, current))
                seg_start = t
                current = bool(is_struggle[t])
        runs.append((seg_start, T, current))
        return runs

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """Phase-dependent reward extraction.

        Each phase run is a self-contained sub-episode whose entire
        ``r_struggle`` signal is a single terminal reward on its **last
        frame** (``end - 1``).  There is no per-step term in either phase.

        Struggle run ends by:
          - recovering to stability -> +1.0
          - falling                 -> -1.0
          - timeout                 -> nothing (bootstrapped)

        Stability run ends by:
          - degrading to struggle -> -1.0
          - falling               -> -1.0
          - timeout               -> nothing (bootstrapped)

        Discounting alone then orders every outcome correctly.  For a run of
        length k, the return at its first frame is:

          stability -> struggle : -gamma^(k-1)  -> maximized by large k
                                                   (stay stable as long as possible)
          struggle  -> recovery : +gamma^(k-1)  -> maximized by small k
                                                   (recover as fast as possible)
          struggle  -> fall     : -gamma^(k-1)  -> maximized by large k
                                                   (delay falling)
          any       -> timeout  :  0 + V(s_end) -> best possible outcome

        r_cross, r_joint, r_vel, r_tilt, r_foot are identical to
        basic_balance_v2; the framework masks them out on struggle runs via
        ``Segment.key_weights``.
        """
        T = episode.num_frames
        fell = "imbalance" in episode.termination_proposals

        # --- r_struggle ---
        r_struggle = np.zeros(T, dtype=np.float32)

        for start, end, is_struggle in self._phase_runs(episode):
            if end < T:
                # Phase boundary: the run ends because the phase flipped.
                # Terminal reward belongs to the run's LAST frame (end - 1),
                # not the frame where the new phase was first observed.
                if is_struggle:
                    r_struggle[end - 1] += self.struggle_recover_bonus
                else:
                    r_struggle[end - 1] += self.stability_to_struggle_penalty
            elif fell:
                # Final run ended by falling.
                r_struggle[end - 1] += self.struggle_fall_penalty
            # else: final run ended by timeout -> no terminal, bootstrapped.

        # --- r_height: per-step height reward, active only during struggle ---
        # Dense shaping signal: higher torso = closer to recovery.
        # Value = height * 0.01, so at standing height (~1.28m) each step
        # gives ~0.0128, and near-ground (~0.3m) gives ~0.003.
        phase_node = episode.observer_outputs.get("phase")
        height_arr = np.zeros(T, dtype=np.float32)
        if phase_node is not None and isinstance(phase_node, dict):
            h_raw = phase_node.get("height")
            if h_raw is not None:
                height_arr = np.asarray(h_raw, dtype=np.float32).reshape(-1)
                if height_arr.shape[0] != T:
                    height_arr = np.zeros(T, dtype=np.float32)
        r_height = (height_arr * 0.01).astype(np.float32)

        # --- Stability-phase rewards (same as basic_balance_v2) ---
        r_cross = _extract_per_step_scalar(episode.observer_outputs, "cross_support", T)

        joint_dev_arr = _extract_per_step_field(episode.observer_outputs, "posture", "joint_deviation", T)
        joint_vel_arr = _extract_per_step_field(episode.observer_outputs, "posture", "joint_vel", T)
        torso_tilt_arr = _extract_per_step_field(episode.observer_outputs, "posture", "torso_tilt", T)
        foot_height_arr = _extract_per_step_field(episode.observer_outputs, "posture", "foot_height", T)

        if joint_dev_arr is None:
            joint_dev_arr = np.zeros(T, dtype=np.float32)
        if joint_vel_arr is None:
            joint_vel_arr = np.zeros(T, dtype=np.float32)
        if torso_tilt_arr is None:
            torso_tilt_arr = np.zeros(T, dtype=np.float32)
        if foot_height_arr is None:
            foot_height_arr = np.zeros(T, dtype=np.float32)

        excess_joint = np.maximum(0.0, joint_dev_arr - 0.1)
        r_joint = np.where(excess_joint == 0.0, 0.01, 0.01 - 5.0 * excess_joint)

        excess_vel = np.maximum(0.0, joint_vel_arr - 0.1)
        r_vel = np.where(excess_vel == 0.0, 0.01, 0.01 - 1.0 * excess_vel)

        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt)

        excess_foot = np.maximum(0.0, foot_height_arr - 0.10)
        r_foot = np.where(excess_foot == 0.0, 0.01, 0.01 - 5.0 * excess_foot)

        return {
            "r_struggle": r_struggle,
            "r_height": r_height,
            "r_cross": r_cross,
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
        }

    def prepare_segments(self, episode) -> Optional[List[Segment]]:
        """One segment per phase run, derived from the same ``_phase_runs``
        decomposition that ``extract_rewards`` uses.

        Active critics:
          - Struggle run:  ``r_struggle`` + ``r_height``.
          - Stability run: all keys except ``r_height`` (which is a
            struggle-only shaping signal and would be pure noise during
            stability since height is nearly constant).

        Termination:
          - Ends at a phase boundary -> ``"terminated"``.  ``extract_rewards``
            already placed an explicit terminal reward (+1 recovered / -1
            degraded) on the run's last frame, so bootstrapping would
            double-count the boundary value.
          - Final run ended by falling -> ``"terminated"`` (explicit -1).
          - Final run ended by timeout -> ``"truncated"``, bootstrap V(s_end).
        """
        T = episode.num_frames
        if T == 0:
            return []

        fell = "imbalance" in episode.termination_proposals

        segments: List[Segment] = []
        for start, end, is_struggle in self._phase_runs(episode):
            if end < T:
                termination = "terminated"
            else:
                termination = "terminated" if fell else "truncated"

            if is_struggle:
                seg_key_weights = {"r_struggle": 1.0, "r_height": 1.0}
                # r_struggle has an explicit terminal reward → terminated (V=0).
                # r_height is dense shaping with no terminal → must bootstrap
                # (truncated), otherwise V=0 at recovery makes the critic
                # perversely value staying low over standing up.
                # Exception: if the episode ended by falling, r_height is
                # also terminated (no future height rewards after fall).
                if end >= T and fell:
                    seg_key_termination = {"r_struggle": "terminated", "r_height": "terminated"}
                else:
                    seg_key_termination = {"r_struggle": "terminated", "r_height": "truncated"}
            else:
                seg_key_weights = {"r_struggle": 1.0, "r_cross": 1.0, "r_joint": 1.0,
                                   "r_vel": 1.0, "r_tilt": 1.0, "r_foot": 1.0}
                seg_key_termination = None

            segments.append(Segment(
                start=start,
                end=end,
                weight=1.0,
                key_weights=seg_key_weights,
                termination=termination,
                key_termination=seg_key_termination,
            ))

        return segments

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        fell = "imbalance" in episode.termination_proposals
        is_struggle, _ = self._extract_phase_info(episode)
        struggle_steps = int(np.sum(is_struggle))
        total_steps = episode.num_frames
        runs = self._phase_runs(episode)
        recoveries = sum(
            1 for start, end, is_str in runs if is_str and end < total_steps
        )
        longest_stable = max(
            (end - start for start, end, is_str in runs if not is_str),
            default=0,
        )
        return {
            "survived": 0.0 if fell else 1.0,
            "struggle_ratio": float(struggle_steps / max(total_steps, 1)),
            "struggle_steps": struggle_steps,
            "recoveries": float(recoveries),
            "longest_stable": float(longest_stable),
        }

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "survival_rate": round(self._survival_rate, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "survival_rate": self._survival_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._survival_rate = float(state.get("survival_rate", 0.0))


EXPERIMENT = BasicBalanceV2StageSegConfig()
