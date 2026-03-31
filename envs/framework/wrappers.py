import gymnasium as gym
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from .reward import BaseRewardFunction, NullRewardFunction

class SingleAgentCombatWrapper(gym.Wrapper):
    """
    将双人对战环境包装为标准的单人强化学习环境。
    通过固定一个对手策略，使环境仅暴露单一智能体的观测和动作空间。
    并且通过注入 RewardFunction 将客观 metrics 转化为 RL 标量奖励。
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        learning_agent_id: str, 
        opponent_id: str,
        opponent_policy: Any = None,
        reward_function: Optional[BaseRewardFunction] = None
    ):
        super().__init__(env)
        self.learning_agent_id = learning_agent_id
        self.opponent_id = opponent_id
        self.opponent_policy = opponent_policy
        self.reward_function = reward_function or NullRewardFunction()
        
        # 空间提取...
        if hasattr(self.env.observation_space, 'spaces'):
            obs_key = f"{self.learning_agent_id}_obs" if f"{self.learning_agent_id}_obs" in self.env.observation_space.spaces else self.learning_agent_id
            self.observation_space = self.env.observation_space.spaces[obs_key]
            self._obs_key = obs_key
        else:
            raise ValueError("Base environment must use a Dict observation space for agents.")
            
        if hasattr(self.env.action_space, 'spaces'):
            act_key = f"{self.learning_agent_id}_action" if f"{self.learning_agent_id}_action" in self.env.action_space.spaces else self.learning_agent_id
            self.action_space = self.env.action_space.spaces[act_key]
        else:
            raise ValueError("Base environment must use a Dict action space for agents.")
            
        self._last_full_obs = None
        self._last_full_info = None

    def _get_opponent_action(self, full_obs: Dict[str, Any], full_info: Dict[str, Any]) -> Any:
        if self.opponent_policy is None:
            opp_act_key = f"{self.opponent_id}_action" if f"{self.opponent_id}_action" in self.env.action_space.spaces else self.opponent_id
            opp_action_space = self.env.action_space.spaces[opp_act_key]
            return np.zeros(opp_action_space.shape, dtype=opp_action_space.dtype)
            
        opp_obs_key = f"{self.opponent_id}_obs" if f"{self.opponent_id}_obs" in full_obs else self.opponent_id
        opp_obs = full_obs[opp_obs_key]
        action = self.opponent_policy.predict(opp_obs)
        if isinstance(action, tuple) and len(action) == 2:
            action = action[0]
        return action

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> tuple[Any, Dict[str, Any]]:
        self.reward_function.reset()
        full_obs, full_info = self.env.reset(seed=seed, options=options)
        self._last_full_obs = full_obs
        self._last_full_info = full_info
        return full_obs[self._obs_key], full_info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        opp_action = self._get_opponent_action(self._last_full_obs, self._last_full_info)
        
        agent_act_key = f"{self.learning_agent_id}_action" if f"{self.learning_agent_id}_action" in self.env.action_space.spaces else self.learning_agent_id
        opp_act_key = f"{self.opponent_id}_action" if f"{self.opponent_id}_action" in self.env.action_space.spaces else self.opponent_id
        
        joint_action = {
            agent_act_key: action,
            opp_act_key: opp_action
        }
        
        full_obs, _, terminated, truncated, full_info = self.env.step(joint_action)
        
        # 使用独立的奖励函数计算当前学习者的专属奖励
        agent_reward = self.reward_function.compute_reward(
            self.learning_agent_id, 
            self._last_full_info, 
            full_info
        )
        
        self._last_full_obs = full_obs
        self._last_full_info = full_info
        
        agent_obs = full_obs[self._obs_key]
            
        return agent_obs, float(agent_reward), terminated, truncated, full_info


class DualPerspectiveVectorWrapper(gym.Wrapper):
    """
    为了 Self-Play 或数据翻倍设计的向量化包装器。
    它将双人环境“拍扁”伪装成一个 num_envs=2 的 VectorEnv。
    对 PPO 算法而言，这就是两个正在并行跑的独立环境。
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        agent_ids: List[str] = ['robot_a', 'robot_b'],
        reward_function: Optional[BaseRewardFunction] = None
    ):
        """
        Args:
            env: 基础 CombatGymEnv
            agent_ids: 按顺序排列的参与者ID，决定了 batch 张量中 index 0 和 1 的对应关系
            reward_function: 用于分别计算两个视角的标量奖励
        """
        super().__init__(env)
        assert len(agent_ids) == 2, "Only supports 2 agents for now"
        self.agent_ids = agent_ids
        self.reward_function = reward_function or NullRewardFunction()
        
        # 提取单个智能体的空间作为基础原型（假设双方空间是对称的）
        base_obs_key = f"{self.agent_ids[0]}_obs" if f"{self.agent_ids[0]}_obs" in self.env.observation_space.spaces else self.agent_ids[0]
        base_act_key = f"{self.agent_ids[0]}_action" if f"{self.agent_ids[0]}_action" in self.env.action_space.spaces else self.agent_ids[0]
        
        single_obs_space = self.env.observation_space.spaces[base_obs_key]
        single_act_space = self.env.action_space.spaces[base_act_key]
        
        # 因为我们把它伪装成 VectorEnv，真正的 space 应该还是单个的 space，但返回值会带 batch 维度
        self.single_observation_space = single_obs_space
        self.single_action_space = single_act_space
        self.observation_space = single_obs_space
        self.action_space = single_act_space
        self.num_envs = 2
        
        self.is_vector_env = True
        self._last_full_info = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        self.reward_function.reset()
        full_obs, full_info = self.env.reset(seed=seed, options=options)
        self._last_full_info = full_info
        
        # 将双人的 dict 观测组合成 shape (2, obs_dim) 的张量
        batched_obs = []
        for agent_id in self.agent_ids:
            obs_key = f"{agent_id}_obs" if f"{agent_id}_obs" in full_obs else agent_id
            batched_obs.append(full_obs[obs_key])
            
        return np.stack(batched_obs), full_info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        接收 shape=(2, action_dim) 的张量动作，分别应用到两个机器人上
        """
        assert action.shape[0] == 2, f"Expected action batch size 2, got {action.shape[0]}"
        
        joint_action = {}
        for i, agent_id in enumerate(self.agent_ids):
            act_key = f"{agent_id}_action" if f"{agent_id}_action" in self.env.action_space.spaces else agent_id
            joint_action[act_key] = action[i]
            
        full_obs, _, terminated, truncated, full_info = self.env.step(joint_action)
        
        batched_obs = []
        batched_rewards = []
        batched_terminated = []
        batched_truncated = []
        
        for i, agent_id in enumerate(self.agent_ids):
            # 1. 提取独立观测
            obs_key = f"{agent_id}_obs" if f"{agent_id}_obs" in full_obs else agent_id
            batched_obs.append(full_obs[obs_key])
            
            # 2. 计算独立视角奖励
            reward = self.reward_function.compute_reward(agent_id, self._last_full_info, full_info)
            batched_rewards.append(reward)
            
            # 3. 提取终止信号（通常双方的胜负游戏结束是同时的）
            batched_terminated.append(terminated)
            batched_truncated.append(truncated)
            
        self._last_full_info = full_info
        
        return (
            np.stack(batched_obs), 
            np.array(batched_rewards, dtype=np.float32), 
            np.array(batched_terminated, dtype=bool), 
            np.array(batched_truncated, dtype=bool), 
            full_info
        )
