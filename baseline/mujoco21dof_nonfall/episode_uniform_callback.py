from typing import List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeUniformRewardCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self._rollout_step_rewards: List[np.ndarray] = []
        self._rollout_step_dones: List[np.ndarray] = []

    def _on_rollout_start(self) -> None:
        self._rollout_step_rewards = []
        self._rollout_step_dones = []

    def _on_step(self) -> bool:
        self._rollout_step_rewards.append(np.asarray(self.locals["rewards"], dtype=np.float32).copy())
        self._rollout_step_dones.append(np.asarray(self.locals["dones"], dtype=bool).copy())
        return True

    def _on_rollout_end(self) -> None:
        rollout_buffer = self.locals["rollout_buffer"]
        last_values = self.locals["values"]
        last_dones = np.asarray(self.locals["dones"], dtype=bool).copy()
        num_steps = int(rollout_buffer.pos)
        if num_steps <= 0:
            return

        step_rewards = np.asarray(self._rollout_step_rewards, dtype=np.float32)
        step_dones = np.asarray(self._rollout_step_dones, dtype=bool)
        if step_rewards.shape[0] != num_steps:
            raise RuntimeError(
                f"EpisodeUniformRewardCallback step count mismatch: callback={step_rewards.shape[0]}, buffer={num_steps}"
            )

        rewritten_rewards = np.array(rollout_buffer.rewards[:num_steps], copy=True)
        episode_reward_totals: List[float] = []
        episode_lengths: List[int] = []

        for env_idx in range(rollout_buffer.n_envs):
            episode_start = 0
            for step_idx in range(num_steps):
                if not step_dones[step_idx, env_idx]:
                    continue
                episode_total_reward = float(np.sum(step_rewards[episode_start:step_idx + 1, env_idx]))
                episode_length = step_idx - episode_start + 1
                uniform_step_reward = episode_total_reward / float(episode_length)
                rewritten_rewards[episode_start:step_idx + 1, env_idx] = uniform_step_reward
                episode_reward_totals.append(episode_total_reward)
                episode_lengths.append(episode_length)
                episode_start = step_idx + 1
            if episode_start != num_steps:
                raise RuntimeError(
                    "Episode-uniform reward mode requires rollout boundaries to align with episode boundaries. "
                    "Set --n-steps to a multiple of the episode length and use settings that avoid spillover episodes."
                )

        rollout_buffer.rewards[:num_steps] = rewritten_rewards
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=last_dones)

        if episode_reward_totals:
            self.logger.record("rollout/episode_uniform_total_reward_mean", float(np.mean(episode_reward_totals)))
            self.logger.record("rollout/episode_uniform_total_reward_std", float(np.std(episode_reward_totals)))
            self.logger.record("rollout/episode_uniform_length_mean", float(np.mean(episode_lengths)))
