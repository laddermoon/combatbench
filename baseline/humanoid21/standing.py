#!/usr/bin/env python3
"""
GRPO 简化实现 - 让机器人学会站立
用法: python3 standing.py
"""
import os
import sys
from pathlib import Path

# 设置渲染模式
os.environ['MUJOCO_GL'] = 'egl'

# 添加项目根目录到路径
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from scipy.spatial.transform import Rotation as R

from envs.humanoid21 import make_env
from envs.framework.runtime_plugin import BaseObserverPlugin
from envs.framework.context import ReadOnlySimContext


# ============ 站立奖励器 ============
class StandingRewarder(BaseObserverPlugin):
    """站立奖励计算器"""

    # 优先级（与 CombatScoringPlugin 相同）
    priority = 100

    def __init__(self, agent_id='robot_a'):
        self.agent_id = agent_id
        self._output = 0.0
        self._prev_uprightness = 1.0

    @property
    def name(self):
        return "standing_rewarder"

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0
        self._prev_uprightness = self._compute_uprightness(ctx)

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        uprightness = self._compute_uprightness(ctx)
        core_state = ctx.accessor.get_core_state()
        agent_state = core_state[self.agent_id]

        # 直立奖励
        uprightness_reward = uprightness * 2.0

        # 高度奖励
        height_reward = max(0, agent_state['root_pos'][2] - 0.8) * 1.0

        # 速度惩罚
        linear_vel_penalty = -0.01 * np.linalg.norm(agent_state['root_vel_local'])
        angular_vel_penalty = -0.01 * np.linalg.norm(agent_state['root_angular_vel_local'])

        # 关节惩罚
        joint_penalty = -0.01 * np.mean(np.abs(agent_state['joint_pos_norm']))

        self._output = uprightness_reward + height_reward + linear_vel_penalty + angular_vel_penalty + joint_penalty
        self._prev_uprightness = uprightness

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def get_output(self):
        return self._output

    def _compute_uprightness(self, ctx: ReadOnlySimContext) -> float:
        """计算直立程度 [0, 1]"""
        try:
            core_state = ctx.accessor.get_core_state()
            root_rot = core_state[self.agent_id]['root_rot']
            root_quat_xyzw = np.array([root_rot[1], root_rot[2], root_rot[3], root_rot[0]])
            rot = R.from_quat(root_quat_xyzw)
            up_dir = rot.as_matrix()[:, 2]
            return float(max(0.0, up_dir[2]))
        except Exception:
            return 1.0


# ============ 策略网络 ============
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=96, action_dim=21, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            dist = self.forward(obs)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            return torch.tanh(action)


# ============ GRPO 收集器 ============
def collect_episodes(policy_net, device, env, num_episodes, group_size):
    """收集多个episode的数据"""
    episodes = []

    while len(episodes) < num_episodes:
        env.reset(seed=None)

        # 做一个初始步骤来初始化观测
        env.step(
            np.zeros(21, dtype=np.float32),  # robot_a action
            np.zeros(21, dtype=np.float32)   # robot_b action
        )

        # 获取初始观测
        obs = env.get_observer_output('robot_a_obs')
        if isinstance(obs, tuple):
            obs = obs[0]
        elif isinstance(obs, dict):
            obs = obs.get('obs') or obs.get('observation') or obs

        episode_obs = []
        episode_pre_tanh = []
        episode_log_prob = []
        episode_rewards = []

        while True:
            # 采样动作
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            dist = policy_net(obs_tensor)
            pre_tanh = dist.rsample()
            action = torch.tanh(pre_tanh)
            log_prob = dist.log_prob(pre_tanh).sum(dim=-1)

            action_np = action.detach().cpu().numpy()

            # 执行动作
            env.step(
                np.asarray(action_np, dtype=np.float32),  # robot_a action
                np.zeros(21, dtype=np.float32)               # robot_b action
            )

            # 获取新观测
            obs = env.get_observer_output('robot_a_obs')
            if isinstance(obs, tuple):
                obs = obs[0]
            elif isinstance(obs, dict):
                obs = obs.get('obs') or obs.get('observation') or obs

            # 获取奖励
            reward = env.get_observer_output('standing_rewarder')

            episode_obs.append(obs.copy())
            episode_pre_tanh.append(pre_tanh.detach().cpu().numpy())
            episode_log_prob.append(log_prob.detach().item())
            episode_rewards.append(reward)

            # 检查是否结束
            terminated, truncated = env.get_termination_flags()
            if terminated or truncated:
                break

        total_reward = sum(episode_rewards)
        episodes.append({
            'obs': np.array(episode_obs),
            'pre_tanh': np.array(episode_pre_tanh),
            'log_prob': np.array(episode_log_prob),
            'return': total_reward,
            'length': len(episode_rewards)
        })

    # 计算组内优势
    returns = np.array([ep['return'] for ep in episodes])
    advantages = np.zeros_like(returns)
    for i in range(0, len(returns), group_size):
        group = returns[i:i+group_size]
        advantages[i:i+group_size] = (group - group.mean()) / (group.std() + 1e-8)

    # 构建batch
    all_obs = np.concatenate([ep['obs'] for ep in episodes])
    all_pre_tanh = np.concatenate([ep['pre_tanh'] for ep in episodes])
    all_log_prob = np.concatenate([ep['log_prob'] for ep in episodes])
    all_adv = np.concatenate([np.full(len(ep['obs']), advantages[i]) for i, ep in enumerate(episodes)])

    return {
        'obs': all_obs,
        'pre_tanh': all_pre_tanh,
        'old_log_prob': all_log_prob,
        'advantages': all_adv,
        'mean_return': returns.mean(),
    }


# ============ GRPO 优化 ============
def optimize(policy, optimizer, batch, device, clip_range=0.2, ent_coef=0.01):
    obs = torch.tensor(batch['obs'], dtype=torch.float32, device=device)
    pre_tanh = torch.tensor(batch['pre_tanh'], dtype=torch.float32, device=device)
    old_log_prob = torch.tensor(batch['old_log_prob'], dtype=torch.float32, device=device)
    advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=device)

    # 标准化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(4):  # 4 epochs
        dist = policy(obs)
        new_log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
        ratio = torch.exp(new_log_prob - old_log_prob)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = dist.entropy().sum(dim=-1).mean()
        loss = policy_loss - ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'entropy': entropy.item(),
    }


# ============ 保存为 Policy ============
def save_as_policy(policy, save_path):
    """保存为标准Policy格式"""
    save_path = Path(save_path)
    policy_dir = save_path.parent
    policy_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型权重
    torch.save(policy.state_dict(), save_path)

    # 创建 policy.py
    policy_code = '''"""训练好的站立策略"""
import torch
import numpy as np
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from policy.base import BaseCombatPolicy


class StandingCombatPolicy(BaseCombatPolicy):
    ACTION_DIM = 21

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建网络
        self.net = torch.nn.Sequential(
            torch.nn.Linear(96, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 21)
        )
        self.log_std = torch.nn.Parameter(torch.zeros(21))

        # 加载权重
        state_dict = torch.load(Path(__file__).parent / "model.pt", map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()

    def act(self, obs, info=None):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean = self.net(obs_tensor)
            action = torch.tanh(mean)
        return action.cpu().numpy()

    def reset(self):
        pass
'''
    (policy_dir / 'policy.py').write_text(policy_code)


# ============ 主训练循环 ============
def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_episodes = 16
    group_size = 8
    total_iterations = 500
    save_every = 100

    print(f"使用设备: {device}")

    # 创建环境（带站立奖励器）
    from envs.humanoid21.plugins import CombatScoringPlugin
    from envs.humanoid21.observer_plugins import Humanoid21Observer

    plugins = [CombatScoringPlugin(damage_scale=100.0)]
    observer_plugins = {
        'robot_a_obs': Humanoid21Observer('robot_a'),
        'robot_b_obs': Humanoid21Observer('robot_b'),
        'standing_rewarder': StandingRewarder('robot_a'),
    }

    env = make_env(
        match_duration=5.0,
        control_frequency=20,
        plugins=plugins,
        observer_plugins=observer_plugins,
    )

    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")

    # 创建策略
    policy = PolicyNet(obs_dim=96, action_dim=21, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    print("开始训练...")

    for iteration in range(total_iterations):
        # 收集经验
        batch = collect_episodes(policy, device, env, num_episodes, group_size)

        # 优化
        stats = optimize(policy, optimizer, batch, device)

        if iteration % 10 == 0:
            print(f"Iter {iteration}: Loss={stats['policy_loss']:.4f}, Entropy={stats['entropy']:.4f}, Return={batch['mean_return']:.2f}")

        # 保存
        if iteration % save_every == 0 and iteration > 0:
            save_path = Path(__file__).parent / 'standing_policy' / 'model.pt'
            save_as_policy(policy, save_path)
            print(f"已保存模型到: {save_path}")

    # 最终保存
    save_path = Path(__file__).parent / 'standing_policy' / 'model.pt'
    save_as_policy(policy, save_path)
    print(f"训练完成! 模型保存到: {save_path}")


if __name__ == '__main__':
    main()
