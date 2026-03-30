import gymnasium as gym
from typing import Any, Dict, List, Optional
import numpy as np

from .backend import BaseSimulator
from .engine import SimEngine
from .plugin import BasePlugin
from .context import TerminationReason
from .common_plugins import BaseRLAdapter, TimeoutPlugin

class CombatGymEnv(gym.Env):
    """
    Gym 接口适配层。
    极其轻薄，仅负责将底层 SimEngine 转换为标准的 OpenAI Gymnasium API。
    一切业务逻辑（如奖励计算、视频录制、超时判断）均由注入的 Plugins 实现。
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        simulator: BaseSimulator,
        rl_adapter: BaseRLAdapter,
        plugins: Optional[List[BasePlugin]] = None,
        phy_steps_per_action: int = 1,
        max_steps: Optional[int] = None,
    ):
        """
        Args:
            simulator: 物理引擎后端实例
            rl_adapter: 强化学习数据转译器（必须）
            plugins: 附加的生命周期插件列表（可选）
            phy_steps_per_action: 每个控制步对应的物理细粒度步数
            max_steps: 单个 episode 的最大控制步数（如果有，会自动挂载 TimeoutPlugin）
        """
        super().__init__()
        
        self.engine = SimEngine(simulator, phy_steps_per_action)
        self.rl_adapter = rl_adapter
        
        # 挂载基础适配器
        self.engine.attach_plugin(self.rl_adapter)
        
        # 如果设置了最大步数，自动挂载超时插件
        if max_steps is not None:
            self.engine.attach_plugin(TimeoutPlugin(max_steps))
            
        # 挂载用户自定义的插件（如约束、扰动、视频录制等）
        for p in (plugins or []):
            self.engine.attach_plugin(p)
            
        # 根据 RLAdapter 的定义设置 Gym 空间
        self.observation_space = rl_adapter.get_observation_space()
        self.action_space = rl_adapter.get_action_space()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> tuple[Any, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置底层引擎
        self.engine.reset()
        
        # RLAdapter 已经在 pre_episode 算好了初始 obs 和 info
        return self.rl_adapter.latest_obs, self.rl_adapter.latest_info

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, Dict[str, Any]]:
        """执行一个控制步"""
        self.engine.step(action)
        
        obs = self.rl_adapter.latest_obs
        reward = self.rl_adapter.latest_reward
        info = self.rl_adapter.latest_info
        
        # 解析终止原因，区分 Termination 和 Truncation
        terminated = False
        truncated = False
        
        if self.engine.ctx.is_terminated:
            proposals = self.engine.ctx.termination_proposals
            # 如果是因为超时导致的终止，算作 Truncation
            if TerminationReason.TIMEOUT in proposals:
                truncated = True
                # 可选：如果同时发生了犯规和超时，通常以犯规(Termination)优先，或者都在 info 里体现
                # 这里简单处理，只要有超时，就记为 truncated
                # 如果既有 timeout 又有 ko/foul，可以根据业务需求决定是 terminated 还是 truncated
                if len(proposals) > 1 and any(p != TerminationReason.TIMEOUT for p in proposals):
                    terminated = True
                    truncated = False
            else:
                terminated = True
                
            info["termination_reasons"] = proposals
            
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """渲染一帧当前画面（用于离线渲染或可视化）"""
        return self.engine.simulator.get_broadcastview_image()

    def close(self) -> None:
        """释放环境资源"""
        self.engine.close()
