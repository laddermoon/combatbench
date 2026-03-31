import gymnasium as gym
from typing import Any, Dict, Optional
import numpy as np

class SingleAgentCombatWrapper(gym.Wrapper):
    """
    将双人对战环境包装为标准的单人强化学习环境。
    通过固定一个对手策略，使环境仅暴露单一智能体的观测和动作空间。
    
    设计要求：
    1. 完全与底层 humanoid21 正交，不修改原有任何逻辑
    2. 无缝接入 SB3 等标准强化学习库
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        learning_agent_id: str, 
        opponent_id: str,
        opponent_policy: Any = None
    ):
        """
        Args:
            env: 基础的 CombatGymEnv（双人环境）
            learning_agent_id: 正在学习的智能体ID（例如 'robot_a'）
            opponent_id: 对手的智能体ID（例如 'robot_b'）
            opponent_policy: 对手使用的固定策略实例。如果不提供，默认返回零动作。
                           该策略必须有一个 predict(obs, info) -> action 方法。
        """
        super().__init__(env)
        self.learning_agent_id = learning_agent_id
        self.opponent_id = opponent_id
        self.opponent_policy = opponent_policy
        
        # 从双人观测空间中提取单人观测空间
        if hasattr(self.env.observation_space, 'spaces'):
            # 兼容底层键值为 'robot_a_obs' 的情况
            obs_key = f"{self.learning_agent_id}_obs" if f"{self.learning_agent_id}_obs" in self.env.observation_space.spaces else self.learning_agent_id
            self.observation_space = self.env.observation_space.spaces[obs_key]
            self._obs_key = obs_key
        else:
            raise ValueError("Base environment must use a Dict observation space for agents.")
            
        # 从双人动作空间中提取单人动作空间
        if hasattr(self.env.action_space, 'spaces'):
            # 兼容动作空间（通常动作空间的 key 就是 robot_a，但防患于未然）
            act_key = f"{self.learning_agent_id}_action" if f"{self.learning_agent_id}_action" in self.env.action_space.spaces else self.learning_agent_id
            self.action_space = self.env.action_space.spaces[act_key]
        else:
            raise ValueError("Base environment must use a Dict action space for agents.")
            
        # 保存最近一次完整的双人状态，用于对手策略生成动作
        self._last_full_obs = None
        self._last_full_info = None

    def _get_opponent_action(self, full_obs: Dict[str, Any], full_info: Dict[str, Any]) -> Any:
        """获取对手动作。如果提供了策略则调用它，否则返回全零动作。"""
        if self.opponent_policy is None:
            # 需要找到正确的对手动作空间来生成形状
            opp_act_key = f"{self.opponent_id}_action" if f"{self.opponent_id}_action" in self.env.action_space.spaces else self.opponent_id
            opp_action_space = self.env.action_space.spaces[opp_act_key]
            return np.zeros(opp_action_space.shape, dtype=opp_action_space.dtype)
            
        # 提取对手的专属视角
        opp_obs_key = f"{self.opponent_id}_obs" if f"{self.opponent_id}_obs" in full_obs else self.opponent_id
        opp_obs = full_obs[opp_obs_key]
        action = self.opponent_policy.predict(opp_obs)
        
        # SB3 的 predict 通常返回 (action, state)
        if isinstance(action, tuple) and len(action) == 2:
            action = action[0]
            
        return action

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> tuple[Any, Dict[str, Any]]:
        """重置环境并返回单人观测"""
        full_obs, full_info = self.env.reset(seed=seed, options=options)
        
        self._last_full_obs = full_obs
        self._last_full_info = full_info
        
        # 只返回正在学习的智能体的观测和 info
        return full_obs[self._obs_key], full_info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        接收单人动作，注入对手动作，执行环境步进
        """
        # 1. 生成对手动作
        opp_action = self._get_opponent_action(self._last_full_obs, self._last_full_info)
        
        # 2. 组装双人动作字典
        # 再次确认底层需要的 key
        agent_act_key = f"{self.learning_agent_id}_action" if f"{self.learning_agent_id}_action" in self.env.action_space.spaces else self.learning_agent_id
        opp_act_key = f"{self.opponent_id}_action" if f"{self.opponent_id}_action" in self.env.action_space.spaces else self.opponent_id
        
        joint_action = {
            agent_act_key: action,
            opp_act_key: opp_action
        }
        
        # 3. 步进底层环境
        full_obs, full_reward, terminated, truncated, full_info = self.env.step(joint_action)
        
        # 4. 更新内部状态缓存
        self._last_full_obs = full_obs
        self._last_full_info = full_info
        
        # 5. 提取单人视角的观测和奖励
        agent_obs = full_obs[self._obs_key]
        
        # 注意：这里的奖励处理非常关键。目前如果底层直接返回 dict 奖励，就直接提取。
        # 如果底层返回标量，可能需要搭配 RewardWrapper 来解析。
        # 这里假设底层的 rl_adapter 已经返回了按 agent 划分的字典奖励。
        if isinstance(full_reward, dict):
            agent_reward = full_reward.get(self.learning_agent_id, 0.0)
        else:
            # 如果底层只返回标量（不符合双人设计），打印警告
            agent_reward = float(full_reward)
            
        return agent_obs, agent_reward, terminated, truncated, full_info
