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
from torch.utils.tensorboard import SummaryWriter

from envs.humanoid21 import make_env
from policy.base import BaseCombatPolicy


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
class RolloutCollector:
    def __init__(self, env, num_envs=64):
        self.env = env
        self.num_envs = num_envs
        self.obs = env.reset()
        self.episodes = []

    def collect(self, policy, device, group_size=8):
        self.episodes = []
        episode_data = [[] for _ in range(self.num_envs)]

        while len(self.episodes) < group_size:
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist = policy(obs_tensor)
                pre_tanh = dist.rsample()
                action = torch.tanh(pre_tanh)
                log_prob = dist.log_prob(pre_tanh).sum(dim=-1)

            actions = action.cpu().numpy()
            log_probs = log_prob.cpu().numpy()
            pre_tanh_actions = pre_tanh.cpu().numpy()

            next_obs, rewards, dones, truncateds, infos = self.env.step(actions)
            self.obs = next_obs

            for i in range(self.num_envs):
                episode_data[i].append({
                    'obs': self.obs[i].copy(),
                    'action': actions[i].copy(),
                    'pre_tanh': pre_tanh_actions[i].copy(),
                    'log_prob': log_probs[i],
                    'reward': rewards[i],
                })

                if dones[i] or truncateds[i].get('TimeLimit.truncated', False):
                    # Episode finished
                    ep_data = episode_data[i]
                    total_reward = sum(d['reward'] for d in ep_data)
                    self.episodes.append({
                        'obs': np.stack([d['obs'] for d in ep_data]),
                        'pre_tanh': np.stack([d['pre_tanh'] for d in ep_data]),
                        'log_prob': np.array([d['log_prob'] for d in ep_data]),
                        'return': total_reward,
                        'length': len(ep_data)
                    })
                    episode_data[i] = []

        # 计算组内优势
        returns = np.array([ep['return'] for ep in self.episodes])
        advantages = np.zeros_like(returns)
        for i in range(0, len(returns), group_size):
            group = returns[i:i+group_size]
            advantages[i:i+group_size] = (group - group.mean()) / (group.std() + 1e-8)

        # 构建batch
        all_obs = np.concatenate([ep['obs'] for ep in self.episodes])
        all_pre_tanh = np.concatenate([ep['pre_tanh'] for ep in self.episodes])
        all_log_prob = np.concatenate([ep['log_prob'] for ep in self.episodes])
        all_adv = np.concatenate([np.full(len(ep['obs']), advantages[i]) for i, ep in enumerate(self.episodes)])

        return {
            'obs': all_obs,
            'pre_tanh': all_pre_tanh,
            'old_log_prob': all_log_prob,
            'advantages': all_adv,
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
        'mean_return': batch['advantages'].mean(),
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
    policy_code = f'''"""训练好的站立策略"""
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
    num_envs = 64
    group_size = 8
    total_iterations = 1000
    save_every = 100

    print(f"使用设备: {device}")

    # 创建环境
    def env_factory():
        from envs.humanoid21.plugins import CombatScoringPlugin
        plugins = [
            CombatScoringPlugin(damage_scale=100.0),
        ]
        return make_env(
            match_duration=5.0,  # 5秒一回合
            control_frequency=20,
            plugins=plugins,
        )

    base_env = env_factory()
    obs_space = base_env.observation_space
    action_space = base_env.action_space

    print(f"观测空间: {obs_space}")
    print(f"动作空间: {action_space}")

    # 简单的向量化环境（复用单个环境）
    from gymnasium.vector import SyncVectorEnv
    vec_env = SyncVectorEnv([env_factory] * num_envs)

    # 创建策略
    policy = PolicyNet(obs_dim=96, action_dim=21, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # 收集器
    collector = RolloutCollector(vec_env, num_envs)

    # TensorBoard
    writer = SummaryWriter('./runs/standing_grpo')

    print("开始训练...")

    for iteration in range(total_iterations):
        # 收集经验
        batch = collector.collect(policy, device, group_size=group_size)

        # 优化
        stats = optimize(policy, optimizer, batch, device)

        # 记录
        writer.add_scalar('Loss/policy_loss', stats['policy_loss'], iteration)
        writer.add_scalar('Loss/entropy', stats['entropy'], iteration)
        writer.add_scalar('Reward/mean_return', stats['mean_return'], iteration)

        if iteration % 10 == 0:
            print(f"Iter {iteration}: Loss={stats['policy_loss']:.4f}, Entropy={stats['entropy']:.4f}, Return={stats['mean_return']:.2f}")

        # 保存
        if iteration % save_every == 0 and iteration > 0:
            save_path = Path(__file__).parent / 'standing_policy' / 'model.pt'
            save_as_policy(policy, save_path)
            print(f"已保存模型到: {save_path}")

    vec_env.close()
    print("训练完成!")


if __name__ == '__main__':
    main()
