"""GAE (Generalized Advantage Estimation) 算法演示与说明

本文件演示 GAE 算法的核心计算过程，通过构造的示例数据展示：
- GAE 处理前的 Value 和 Reward
- GAE 处理后的 Advantage 和 Return
- 关键超参数 (gamma, lambda) 的作用

GAE 核心思想:
    GAE 是一种在方差和偏差之间做权衡的 Advantage 估计方法。
    通过 λ 参数控制：λ=1 时无偏但方差大(Monte-Carlo)，
    λ=0 时有偏但方差小(TD(0))。

数学公式:
    TD-error:      δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    GAE Advantage: A_t^GAE = Σ_{k=0}^∞ (γλ)^k · δ_{t+k}
    Return target: R_t = A_t^GAE + V(s_t)

递归计算形式（代码实现使用）:
    A_t = δ_t + γλ · A_{t+1}  （从后向前递推）
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def compute_gae_verbose(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    last_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """GAE 计算实现（带详细日志版本）
    
    与普通实现相同，但增加每一步的详细输出帮助理解计算过程。
    
    Parameters
    ----------
    rewards: shape (T,) - 每步的即时奖励
    values:  shape (T,) - Critic 对每个状态的价值估计 V(s_t)
    last_value: float - 最后状态之后的 bootstrap value V(s_T)
    gamma: discount factor [0,1]
    lam: GAE lambda [0,1]，控制偏差/方差权衡
    verbose: 是否打印详细计算过程
    
    Returns
    -------
    (advantages, returns) - 均为 shape (T,)
    """
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    t = rewards_arr.shape[0]
    
    advantages = np.zeros(t, dtype=np.float32)
    
    # 从最后一步向前递推
    next_value = float(last_value)  # V(s_{i+1})
    next_advantage = 0.0            # A_{i+1}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"GAE 计算过程 (gamma={gamma}, lambda={lam}, last_value={last_value:.4f})")
        print(f"{'='*70}")
        print(f"{'Step':>4} | {'Reward':>8} | {'Value':>8} | {'V_next':>8} | "
              f"{'Delta':>8} | {'GA_term':>8} | {'Adv':>8} | {'Return':>8}")
        print(f"{'-'*70}")
    
    for i in range(t - 1, -1, -1):
        # TD-error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        delta = rewards_arr[i] + gamma * next_value - values_arr[i]
        
        # GAE: A_t = δ_t + γλ·A_{t+1}
        next_advantage = delta + gamma * lam * next_advantage
        advantages[i] = next_advantage
        
        # Return = Advantage + Value
        return_i = advantages[i] + values_arr[i]
        
        if verbose:
            gae_term = gamma * lam * (advantages[i+1] if i < t-1 else 0.0)
            print(f"{i:>4} | {rewards_arr[i]:>8.4f} | {values_arr[i]:>8.4f} | "
                  f"{next_value:>8.4f} | {delta:>8.4f} | {gae_term:>8.4f} | "
                  f"{advantages[i]:>8.4f} | {return_i:>8.4f}")
        
        next_value = float(values_arr[i])
    
    returns = advantages + values_arr
    
    if verbose:
        print(f"{'-'*70}")
        print(f"最终状态后的 bootstrap value: {last_value:.4f}")
        print(f"{'='*70}")
    
    return advantages, returns


def demonstrate_simple_episode():
    """演示: 简单 5 步 episode
    
    场景: 智能体移动，中间有奖励，最后到达目标
    """
    print("\n" + "="*70)
    print("示例 1: 简单 5 步 Episode")
    print("="*70)
    print("场景: 智能体向目标移动，第2步获得中间奖励，最后到达目标获得大奖励")
    print()
    
    # 构造数据
    rewards = np.array([0.0, 0.0, 1.0, 0.0, 10.0])  # 即时奖励
    values = np.array([8.0, 9.0, 9.5, 9.8, 10.0])  # Critic 估计的价值
    # 注意: Critic 预测准确，因为它看到最终价值接近10
    
    print("输入数据:")
    print(f"  Rewards: {rewards}")
    print(f"  Values:  {values}")
    print(f"  假设 episode 正常结束 (last_value=0)")
    print()
    
    # 使用不同 lambda 值对比
    for lam in [0.0, 0.5, 0.95, 1.0]:
        print(f"\n>>> lambda={lam} 时的 GAE 结果:")
        adv, ret = compute_gae_verbose(rewards, values, last_value=0.0, 
                                       gamma=0.99, lam=lam, verbose=False)
        print(f"  Advantages: {adv.round(4)}")
        print(f"  Returns:    {ret.round(4)}")
        print(f"  Adv 解释: mean={adv.mean():.4f}, std={adv.std():.4f}")
        if lam == 0.0:
            print("  → λ=0: TD(0)，每步 advantage ≈ r_t + γV_{t+1} - V_t，方差最小")
        elif lam == 1.0:
            print("  → λ=1: Monte-Carlo，Return = 实际累积折扣奖励，无偏但方差大")
        else:
            print(f"  → λ={lam}: 偏差/方差权衡，结合多步信息")


def demonstrate_truncated_episode():
    """演示: 被截断的 episode
    
    展示 terminated vs truncated 的区别，以及 last_value 的作用
    """
    print("\n" + "="*70)
    print("示例 2: 被截断 (Truncated) 的 Episode")
    print("="*70)
    print("场景: 10步 episode 但只跑了前4步就被截断，需要用 bootstrap value")
    print()
    
    rewards = np.array([1.0, 1.0, 1.0, 1.0])  # 每步稳定奖励
    values = np.array([5.0, 6.0, 7.0, 8.0])   # Critic 估计
    
    print("输入数据:")
    print(f"  Rewards: {rewards}")
    print(f"  Values:  {values}")
    print(f"  实际 episode 长度=4 (被截断), Critic 估计第4步后还有价值")
    print()
    
    # 对比: terminated vs truncated
    print(">>> 情况 A: 真实终止 (last_value=0, 比如掉下悬崖)")
    adv_term, ret_term = compute_gae_verbose(rewards, values, last_value=0.0,
                                              gamma=0.99, lam=0.95, verbose=True)
    
    print("\n>>> 情况 B: 被截断 (last_value=15, Critic 估计后续价值)")
    adv_trunc, ret_trunc = compute_gae_verbose(rewards, values, last_value=15.0,
                                                gamma=0.99, lam=0.95, verbose=True)
    
    print("\n对比总结:")
    print(f"  Terminated Returns: {ret_term.round(2)} (偏小，因为没有后续)")
    print(f"  Truncated Returns:  {ret_trunc.round(2)} (包含 bootstrap 后续价值)")
    print("  ⚠️  错误设置 last_value 会导致训练偏差!")


def demonstrate_high_variance_scenario():
    """演示: 高方差场景下不同 lambda 的影响
    
    展示为什么 PPO 通常使用 λ=0.95 而不是 1.0
    """
    print("\n" + "="*70)
    print("示例 3: 高方差场景 (奖励稀疏且有噪声)")
    print("="*70)
    print("场景: 稀疏奖励，随机噪声，展示 λ 对噪声敏感度的影响")
    print()
    
    np.random.seed(42)
    
    # 构造稀疏奖励: 大部分为0，偶尔有正负奖励
    rewards = np.zeros(20)
    rewards[5] = 5.0    # 一个正奖励
    rewards[15] = -3.0  # 一个负奖励 (噪声/惩罚)
    
    # Critic 尝试预测但实际有误差
    true_values = np.array([5 * (0.99 ** (max(0, 5-t))) for t in range(20)])
    values = true_values + np.random.randn(20) * 0.5  # 加入估计噪声
    
    print("输入数据 (稀疏奖励 + 噪声):")
    print(f"  Rewards: {rewards}")
    print(f"  Values:  [{', '.join([f'{v:.2f}' for v in values[:5]])}...]")
    print(f"  (共20步，第5步奖励+5，第15步奖励-3)")
    print()
    
    # 对比不同 lambda
    results = {}
    for lam in [0.0, 0.5, 0.95, 1.0]:
        adv, ret = compute_gae_verbose(rewards, values, last_value=0.0,
                                       gamma=0.99, lam=lam, verbose=False)
        results[lam] = (adv, ret)
    
    print("不同 λ 的 Advantage 分布:")
    print(f"{'Lambda':>8} | {'Adv Mean':>10} | {'Adv Std':>10} | {'Min':>8} | {'Max':>8}")
    print(f"{'-'*60}")
    for lam in [0.0, 0.5, 0.95, 1.0]:
        adv, _ = results[lam]
        print(f"{lam:>8.2f} | {adv.mean():>10.4f} | {adv.std():>10.4f} | "
              f"{adv.min():>8.4f} | {adv.max():>8.4f}")
    
    print(f"\n关键观察:")
    print(f"  - λ=0.0: Advantage 标准差最小 ({results[0.0][0].std():.4f})")
    print(f"  - λ=1.0: Advantage 标准差最大 ({results[1.0][0].std():.4f})")
    print(f"  - 高方差会让策略梯度估计噪声大，训练不稳定")
    print(f"  - PPO 常用 λ=0.95 在偏差和方差间取得平衡")


def demonstrate_ppo_usage():
    """演示: PPO 中 GAE 结果的实际使用
    
    展示 advantage 和 return 在 PPO 更新中的作用
    """
    print("\n" + "="*70)
    print("示例 4: PPO 训练中的 GAE 应用")
    print("="*70)
    print("说明 GAE 输出如何用于 PPO 的 Actor 和 Critic 更新")
    print()
    
    # 构造一个简单的 rollout batch
    T = 10
    rewards = np.array([1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.5, 0.0, 1.0, 2.0])
    values = np.array([10.0, 10.5, 11.0, 10.8, 10.5, 10.0, 10.2, 9.8, 10.0, 11.0])
    
    print("Rollout 收集的数据:")
    print(f"  Steps: {T}")
    print(f"  Rewards: {rewards}")
    print(f"  Values (old critic): {values}")
    
    # 计算 GAE
    adv, returns = compute_gae_verbose(rewards, values, last_value=10.0,
                                        gamma=0.99, lam=0.95, verbose=False)
    
    # Advantage 归一化 (PPO 常用技巧)
    adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    print(f"\nGAE 计算结果:")
    print(f"  {'Step':>4} | {'Reward':>7} | {'Value':>7} | {'Adv':>8} | {'Adv(norm)':>10} | {'Return':>8}")
    print(f"  {'-'*60}")
    for t in range(T):
        print(f"  {t:>4} | {rewards[t]:>7.2f} | {values[t]:>7.2f} | "
              f"{adv[t]:>8.4f} | {adv_normalized[t]:>10.4f} | {returns[t]:>8.2f}")
    
    print(f"\nPPO 更新使用:")
    print(f"  1. Critic Loss: MSE(Value_pred, Return_target)")
    print(f"     - 目标: critic 预测接近 Return={returns.mean():.2f}±{returns.std():.2f}")
    print(f"  2. Actor Loss: -log_prob(action) * Advantage")
    print(f"     - Advantage 范围: [{adv.min():.3f}, {adv.max():.3f}]")
    print(f"     - 正 advantage → 增加该动作概率")
    print(f"     - 负 advantage → 减少该动作概率")
    print(f"  3. 通常会对 Advantage 做归一化 (std={adv_normalized.std():.3f})")


def print_formula_reference():
    """打印 GAE 公式参考"""
    print("\n" + "="*70)
    print("GAE 公式参考")
    print("="*70)
    print("""
【核心公式】

1. TD-error (单步):
   δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
   
   含义: 实际获得的奖励 + 下一状态价值 - 当前状态价值
        正值说明实际比预期好，负值说明比预期差

2. GAE Advantage (多步组合):
   A_t^GAE(λ) = Σ_{k=0}^{T-t-1} (γλ)^k · δ_{t+k}
   
   含义: 对未来所有 TD-error 进行加权求和
        (γλ)^k 是衰减因子，λ 控制衰减速度

3. 递归形式 (代码实现用):
   A_t = δ_t + (γλ)·A_{t+1}
   
   含义: 当前 advantage = 当前 TD-error + 衰减后的下一步 advantage
        从后向前递推，边界条件 A_T = 0

4. Return target (Critic 学习目标):
   R_t = A_t + V(s_t)
   
   含义: Critic 应该输出这个值作为状态价值估计
        等价于 λ-加权的多步回报

【λ 的作用】

λ = 0:  A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)
        → 纯 TD(0)，方差最小，但有偏差
        
λ = 1:  A_t = Σ γ^k r_{t+k} - V(s_t)  
        → 纯 Monte-Carlo，无偏但方差最大
        
0 < λ < 1: 介于两者之间，权衡偏差和方差

【gamma 的作用】

gamma 折扣未来奖励:
- γ=0.99 常用，表示 1 步后的奖励打 0.99 折
- 100 步后的奖励 ≈ 0.99^100 ≈ 0.37 折
- gamma 越小越关注即时奖励
""")


if __name__ == "__main__":
    # 运行所有演示
    demonstrate_simple_episode()
    demonstrate_truncated_episode()
    demonstrate_high_variance_scenario()
    demonstrate_ppo_usage()
    print_formula_reference()
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    print("""
使用建议:
1. 查看 compute_gae_verbose() 函数理解具体实现
2. 修改 demonstrate_* 函数中的参数观察影响
3. 注意 lambda=0.95 是 PPO 的标准选择
4. 正确设置 last_value 对 truncated episodes 很重要
""")
