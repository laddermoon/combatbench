# PPO 实验框架使用指南

本指南面向想要使用本框架训练自己实验的用户。你不需要阅读框架源码就能上手——读完本指南，你就能写出一个完整的 PPO 实验并跑起来。

---

## 1. 这个框架是什么

一个**多 critic PPO 训练框架**，专为 CombatBench 的 humanoid21 机器人对抗环境设计。核心能力：

- **多 reward channel**：每个 reward 组件有独立的 critic、独立的 gamma 和 GAE lambda
- **Trajectory 级控制**：实验决定如何切分 episode、每个 channel 的 reward / 终止标志 / actor_weight
- **Curriculum scheduling**：通过 `build_trajectories` 动态调整 `actor_weight`，无需框架介入
- **探索调度**：`on_update()` + `exploration()` 这对 hook 让实验根据训练统计动态调整探索强度
- **Checkpoint resume**：实验状态自动序列化/恢复

---

## 2. 心智模型：数据怎么流

```
 ┌─────────────────────────────────────────────────────────┐
 │  每个 update 的完整流程                                   │
 └─────────────────────────────────────────────────────────┘

  1. exploration(update) → ExplorationSpec
       │  实验决定本轮探索参数（entropy_coef, temperature 等）
       │  读取 on_update() 累积的内部状态
       ▼
  2. actor.set_exploration(spec)
       │  策略将 spec 翻译成自己分布族的具体参数
       ▼
  3. actor.to_blueprint(stochastic=True) → PolicyBlueprint
       │  导出策略蓝图用于 rollout
       ▼
  4. build_jobs(policy_bp, base_seed, n_episodes) → List[Job]
       │  实验构建 rollout 任务（哪个环境、哪个对手、什么种子）
       ▼
  5. ParallelRollouter.collect(jobs) → List[Episode]
       │  框架并行执行 rollout，收集完整 episode
       ▼
  6. build_trajectories(episodes) → List[Trajectory]
       │  实验把 episode 切成 trajectory，填入每个 channel 的
       │  reward / is_terminated / actor_weight
       ▼
  7. PPOBuffer(trajs, actor, channels)
       │  框架批量调用 actor.evaluate_actions 计算 old log_prob，
       │  为每个 channel 计算 GAE → advantage + return
       ▼
  8. ppo_update(actor, critics, buf, ...)
       │  每 channel: normalize advantage (z-score on active frames)
       │  L1 归一化: 每帧 Σ_c |aw_c| = 1（解耦 aw 与有效学习率）
       │  合并: combined_adv = Σ_c aw_c_normed × confidence_c × norm_adv_c
       │  Critic 更新: MSE(V_c, return_c)
       │  Actor 更新: PPO clipped surrogate on combined_adv
       │  → 返回 UpdateStats (typed)
       ▼
  8b. on_update(stats, update)
       │  实验吸收本轮训练统计到内部状态（如 KL 历史）
       │  下一轮的 exploration() 会读取这些状态
       ▼
  9. (每 eval_interval 轮) build_jobs(det_policy_bp, ...) → eval episodes
       │  on_eval(eval_episodes, update) → {is_new_best, info, stop_training?}
       │  实验计算指标、判断是否 new best、更新 curriculum 状态
       ▼
 10. save checkpoint (每 N 轮): actor/critic/optimizer + experiment.state()
```

**核心分工：**

| 阶段 | 实验负责 | 框架负责 |
|------|---------|---------|
| 模型构建 | `build_actor`, `build_critic` | 创建 optimizer |
| Job 构建 | `build_jobs` | ParallelRollouter 并行执行 |
| Episode→Trajectory | `build_trajectories`（完全控制） | 调用它 |
| GAE 计算 | `reward_channels`（声明 γ, λ） | 执行 compute_gae |
| Advantage 归一化 | — | z-score on active frames |
| Advantage 合并 | — | L1 归一化 aw (Σ|aw|=1) 后加权 by aw × confidence |
| Critic 更新 | — | MSE on returns |
| Actor 更新 | — | PPO clipped surrogate |
| Eval & 调度 | `on_eval`（完全控制） | 跑 eval rollout、导出策略 |
| 训练统计反馈 | `on_update(stats, update)` | 调用它，传入 typed UpdateStats |
| 探索 | `exploration(update)` → ExplorationSpec | 路由 spec → set_exploration |
| Checkpoint | `state()` / `load_state()` | 存模型+config、恢复 |

---

## 3. 关键概念

### 3.1 RewardChannel

每个 reward 组件是一个 channel，对应一个独立的 V(s) critic。

```python
RewardChannel(name="r_balance", gamma=0.99, gae_lambda=0.95)
```

- **name**：唯一标识，用于索引 critic 和 trajectory 中的 channel data
- **gamma**：该 channel 的折扣因子
- **gae_lambda**：该 channel 的 GAE λ。稀疏终端 reward 适合高 λ（低偏差），密集 shaping reward 适合低 λ（低方差）。不同 channel 可以有不同的值。

`reward_channels()` 返回所有 channel 的 tuple，框架为每个 channel 建一个 critic。

### 3.2 ChannelData

每个 trajectory 上、每个 channel 的数据：

```python
ChannelData(
    reward=np.array([0.01, 0.01, ...]),  # (T,) 每步 reward
    is_terminated=True,                   # True→V=0 不 bootstrap; False→从 critic bootstrap
    actor_weight=3.0,                     # 该 channel 的 advantage 权重，可以是标量或 (T,) 数组
)
```

- **is_terminated**：该 channel 在这条 trajectory 上是否"终止"。不同 channel 在同一条 trajectory 上可以有不同的值。例如机器人摔倒：`r_fall` 标记 terminated（V=0），但 `r_cross` 可能标记 truncated（从 critic bootstrap）。
- **actor_weight**：该 channel 的 advantage 对 policy gradient 的影响力。可以是标量或 `(T,)` 数组（实现**逐步权重变化**，curriculum scheduling 的核心机制）。

  **L1 归一化**：合并 advantage 前，框架对每帧的 actor_weight 做 L1 归一化，使 `Σ_c |aw_c| = 1`。这意味着：
  - **只有 channel 间的比例重要，绝对值不重要**。把所有 aw 同时乘以 k 不改变 combined_adv，不会影响有效学习率。
  - **负权重保留方向**。`aw=-1` 表示反转该 channel 的 advantage 方向，归一化后符号不变。
  - **集中权重 = 更强信号**。单 channel（aw=1）的 combined_adv std ≈ 1；两个等权 channel（aw=0.5, 0.5）的 std ≈ 0.71。

  `actor_weight` 在框架中有四个不同层面的作用，理解它们的区别很重要：

  | 层面 | aw=0 的帧 | aw>0 的帧 |
  |------|----------|----------|
  | **Critic 训练** | ✓ 参与。critic 在所有 active 帧上学习 V(s)，与 aw 无关 | ✓ 参与 |
  | **GAE 计算与传播** | ✓ 参与。GAE backward pass 穿过 aw=0 帧把未来 reward 传播到 aw>0 帧 | ✓ 参与 |
  | **Advantage 归一化** | ✗ **不参与**。z-score 的 mean/std 只在 aw>0 帧上计算 | ✓ 参与 |
  | **L1 归一化** | ✗ 不参与（|0|=0，不影响 Σ|aw|） | ✓ 参与。与其他 channel 的 aw 一起按 |aw| 比例分配 |
  | **Actor gradient** | ✗ 不产生（乘以 0） | ✓ 产生 |

  **为什么归一化排除 aw=0 帧**：归一化的目的是让实际驱动 actor 的 advantage 有稳定的尺度。aw=0 帧不产生 gradient，它们的 advantage 分布（可能属于不同 phase，reward 模式不同）如果混入统计，会扭曲 aw>0 帧的归一化结果。

  **为什么做 L1 归一化**：不做归一化时，`combined_adv = Σ_c aw_c × conf_c × norm_adv_c` 的尺度正比于 `Σ|aw_c|`。用户以为自己在调"channel 重要性"，实际同时在调有效学习率。L1 归一化解耦了这两个维度：aw 只控制 channel 间的相对重要性，学习率完全由 optimizer LR 决定。

  **对软过渡无影响**：`aw == 0` 是精确零值判断。软过渡（如 `phi**2` 连续值）几乎不会恰好为 0，所以不会被排除。只有硬切换（布尔 mask 转浮点 → 精确 0.0）才会触发排除。

  **典型用法**：
  - `aw=0.0`（标量）：整个 channel 的 critic 照常训练但不影响 actor——适合 warmup 新 critic
  - `aw` 为 `(T,)` 数组：逐步权重变化，如 `3.0 * standup_mask`（硬切换）或 `3.0 * phi**2`（软过渡）
  - `aw=-1.0`：反转该 channel 的 advantage 方向（归一化后仍为负）

### 3.3 Trajectory

一条 trajectory 是 episode 的一个连续切片，是 PPO 训练的原子单元：

```python
Trajectory(
    obs=obs_array,          # (T, obs_dim)
    actions=act_array,      # (T, act_dim)
    last_obs=final_obs,     # (obs_dim,) 用于 bootstrap
    channels={
        "r_fall": ChannelData(...),
        "r_cross": ChannelData(...),
    },
    importance=1.0,         # 该 trajectory 的样本权重
    mode=None,              # 可选：actor 路由模式
)
```

- 一个 episode 可以切成多条 trajectory（按阶段、按 gating 切换等）
- `channels` 里**不包含**的 channel 在这条 trajectory 上是 inactive 的（不训练 critic，不贡献 advantage）

### 3.4 ExplorationSpec

实验对探索的"意图"，由实验在每轮 update 前返回：

```python
ExplorationSpec(
    temperature=1.0,        # 采样噪声倍数，1.0 = 策略原生尺度
    entropy_coef=0.001,     # 熵正则化系数
    entropy_target=None,    # 或目标熵（与 entropy_coef 互斥）
    clip_eps=None,          # PPO clip eps 覆盖
    target_kl=None,         # PPO KL 早停阈值覆盖
)
```

所有字段都是可选的，`None` 表示"不关心，保持现状"。策略只处理它认识的字段，忽略其余的。

**为什么探索是分离的**：框架不硬编码 `loss -= entropy_coef * entropy`。因为对角高斯有闭式熵，但 mixture / flow / diffusion 没有。探索的**意图**由实验定（`ExplorationSpec`），**机制**由策略定（`set_exploration` 翻译成自己分布族的具体操作）。框架只负责路由。

---

## 4. 如何写一个新实验

继承 `ExperimentPPO`，实现所有 abstract 方法。以下是一个完整的最小示例：

```python
"""exp_my_experiment.py — 我的最小实验"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from baseline.framework.ppo import (
    CommonParams, ExperimentPPO, ExplorationSpec,
    PPOParams, TrainablePolicy,
)
from baseline.framework.ppo.trajectory import (
    ChannelData, RewardChannel, Trajectory,
)
from baseline.framework.rollout import extract_per_step_scalar
from baseline.framework.ppo.policies import CriticMLP

# 你的环境蓝图和策略蓝图
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class MyExperiment(ExperimentPPO):

    # === 身份 ===
    name = "my_experiment"

    # === 网络维度 ===
    obs_dim = 96
    action_dim = 21

    # === 训练参数 ===
    learning_rate = 1e-4
    episodes_per_update = 256
    max_updates = 5000
    eval_interval = 5
    eval_episodes = 16

    # === 内部状态 ===
    _best_score: float = -1.0

    # ------------------------------------------------------------------
    # Phase 0: 配置 & 模型构建
    # ------------------------------------------------------------------

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        """声明所有 reward channel。每个 channel 一个 critic。"""
        return (
            RewardChannel("r_fall", gamma=0.99, gae_lambda=0.95),
            RewardChannel("r_cross", gamma=0.99, gae_lambda=0.95),
        )

    def common_params(self) -> CommonParams:
        return CommonParams(
            name=self.name,
            learning_rate=self.learning_rate,
            critic_learning_rate=3e-4,
            grad_clip_norm=1.0,
            episodes_per_update=self.episodes_per_update,
            max_updates=self.max_updates,
            eval_interval=self.eval_interval,
            eval_episodes=self.eval_episodes,
            video_eval_interval=0,   # 0 = 不录视频
            rollout_workers=48,
            seed=42,
        )

    def ppo_params(self) -> PPOParams:
        return PPOParams(
            clip_eps=0.2,
            target_kl=0.05,
            update_epochs=4,
            minibatch_size=8192,
        )

    def build_actor(self, device: torch.device) -> TrainablePolicy:
        """构建 actor。通常从 blueprint YAML 加载。"""
        bp = PolicyBlueprint.load("path/to/init_policy.yaml")
        actor = bp.build().to(device)
        return actor

    def build_critic(self, channel_name: str, device: torch.device) -> nn.Module:
        """为每个 channel 构建一个 V(s) critic。"""
        return CriticMLP(obs_dim=self.obs_dim, hidden_dim=256).to(device)

    # ------------------------------------------------------------------
    # 探索调度（可选，默认返回 None = 保持现状）
    # ------------------------------------------------------------------

    def exploration(self, update: int) -> ExplorationSpec | None:
        """每轮 update 前调用。读取 on_update 累积的状态来动态调整。"""
        return ExplorationSpec(entropy_coef=0.001, temperature=1.0)

    # ------------------------------------------------------------------
    # Phase 1: Job 构建
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes) -> List[Tuple]:
        """构建 rollout 任务。

        每个 job 是一个 tuple:
        (policy_a_bp, policy_b_bp, env_bp, seed, episode_options)

        - 训练 rollout: policy_bp 是 stochastic 的
        - 评估 rollout: policy_bp 是 deterministic 的
        你不需要区分——框架传什么就用什么。
        """
        env_pb = ParameterizedEnvBlueprint.load("path/to/env.yaml")
        jobs = []
        for i in range(n_episodes):
            seed = base_seed + i
            env_bp = env_pb.materialize(max_steps=200, agent_id="robot_a")
            jobs.append((policy_bp, policy_bp, env_bp, seed, {}))
        return jobs

    # ------------------------------------------------------------------
    # Phase 2: Episode → Trajectory
    # ------------------------------------------------------------------

    def build_trajectories(self, episodes) -> List[Trajectory]:
        """把 episode 转成 trajectory。这是你实验的核心逻辑。

        你在这里决定:
        - 怎么切分 episode
        - 每个 channel 的 reward 是什么
        - 每个 channel 是否终止
        - 每个 channel 的 actor_weight 是多少
        """
        all_trajs = []
        for ep in episodes:
            agent_id = "robot_a"
            T = ep.num_frames
            if T == 0:
                continue

            obs = np.asarray(ep.observations[agent_id], dtype=np.float32)
            acts = np.asarray(ep.actions[agent_id], dtype=np.float32)
            fin = np.asarray(ep.final_observation[agent_id], dtype=np.float32)

            # 从 observer outputs 提取 reward 信号
            r_cross = extract_per_step_scalar(
                ep.observer_outputs, "cross_support_a", T,
            )

            # 自己构造 shaping reward
            phi = extract_per_step_scalar(
                ep.observer_outputs, "height_phi_a", T,
            )
            r_fall = 0.01 * np.clip(phi, 0.0, 1.0)

            # 判断是否摔倒
            term_reason = ep.agent_termination_reason.get(agent_id, "")
            fell = term_reason.startswith("imbalance")

            all_trajs.append(Trajectory(
                obs=obs,
                actions=acts,
                last_obs=fin,
                channels={
                    "r_fall": ChannelData(
                        reward=r_fall.astype(np.float32),
                        is_terminated=fell,
                        actor_weight=3.0,
                    ),
                    "r_cross": ChannelData(
                        reward=r_cross.astype(np.float32),
                        is_terminated=fell,
                        actor_weight=1.0,
                    ),
                },
                importance=1.0,
            ))
        return all_trajs

    # ------------------------------------------------------------------
    # Phase 3: Eval
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        """处理 eval 结果，更新内部状态，返回是否 new best。"""
        survived = 0
        total = 0
        for ep in episodes:
            for aid in ("robot_a", "robot_b"):
                total += 1
                if not ep.agent_termination_reason.get(aid, "").startswith("imbalance"):
                    survived += 1

        is_new_best = survived > self._best_score
        if is_new_best:
            self._best_score = float(survived)

        return {
            "is_new_best": is_new_best,
            "info": {
                "survived": float(survived),
                "survival_rate": round(survived / max(total, 1), 3),
            },
        }

    # ------------------------------------------------------------------
    # State persistence（用于 checkpoint resume）
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {"best_score": self._best_score}

    def load_state(self, state: dict) -> None:
        self._best_score = float(state.get("best_score", -1.0))


# 模块级单例——registry 通过这个属性发现实验
EXPERIMENT_CLASS = MyExperiment
```

### 4.1 需要实现的方法清单

| 方法 | 必须? | 何时被调用 | 你要做什么 |
|------|-------|-----------|-----------|
| `reward_channels()` | **必须** | 训练开始 | 声明所有 channel 的 name/gamma/lambda |
| `common_params()` | **必须** | 训练开始 | 返回训练参数 |
| `ppo_params()` | **必须** | 训练开始 | 返回 PPO 超参 |
| `build_actor(device)` | **必须** | 训练开始 | 构建并返回 actor |
| `build_critic(name, device)` | **必须** | 训练开始（每 channel 一次） | 构建并返回 V critic |
| `build_jobs(bp, seed, n)` | **必须** | 每 update（训练+eval） | 构建 rollout 任务列表 |
| `build_trajectories(episodes)` | **必须** | 每 update | episode → trajectory |
| `on_eval(episodes, update)` | **必须** | 每 eval_interval 轮 | 计算 eval 指标、判断 best、更新状态 |
| `on_update(stats, update)` | 可选 | 每 update 后 | 吸收训练统计到内部状态，默认 no-op |
| `exploration(update)` | 可选 | 每 update 前 | 返回探索参数，默认 None |
| `state()` | 可选 | checkpoint 时 | 序列化内部状态 |
| `load_state(state)` | 可选 | resume 时 | 恢复内部状态 |

### 4.2 模块级单例

实验文件末尾**必须**导出 `EXPERIMENT_CLASS`：

```python
EXPERIMENT_CLASS = MyExperiment
```

registry 通过 `importlib` 自动发现 `exp_*.py` 文件，读取 `EXPERIMENT_CLASS` 属性注册。

### 4.3 运行你的实验

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 baseline/framework/train.py --experiment my_experiment --algo ppo --smoke
```

`--smoke` 跑 2 轮快速验证。确认没问题后去掉 `--smoke` 正式训练，或加 `--background` 后台运行。

---

## 5. 进阶用法

### 5.1 Curriculum scheduling

curriculum 的核心是**动态调整 `actor_weight`**。不需要任何框架支持——在 `build_trajectories` 里根据当前训练阶段返回不同的 `actor_weight` 即可。

```python
class MyExperiment(ExperimentPPO):

    _phase: int = 0  # 0=balance, 1=standup, 2=walk

    def on_eval(self, episodes, update):
        # ... 计算 survival_rate ...
        if survival_rate > 0.8 and self._phase == 0:
            self._phase = 1  # 进入下一阶段
        return {"is_new_best": ..., "info": {"phase": self._phase}}

    def build_trajectories(self, episodes):
        trajs = []
        for ep in episodes:
            # 根据 phase 调整 actor_weight
            if self._phase == 0:
                weights = {"r_fall": 3.0, "r_cross": 1.0, "r_walk": 0.0}
            elif self._phase == 1:
                weights = {"r_fall": 1.0, "r_cross": 1.0, "r_walk": 0.5}
            else:
                weights = {"r_fall": 0.5, "r_cross": 0.5, "r_walk": 2.0}

            # ... 构建 trajectory，用 weights 填 ChannelData.actor_weight ...
        return trajs

    def state(self):
        return {"phase": self._phase}

    def load_state(self, state):
        self._phase = int(state.get("phase", 0))
```

关键点：
- `actor_weight=0.0` 的 channel 仍然训练 critic，但不影响 actor——适合 warmup
- `actor_weight` 可以是 `(T,)` 数组，实现**单条 trajectory 内的逐步权重变化**（例如用 φ² gating）
- `aw=0` 的帧**不参与 advantage 归一化统计**——z-score 的 mean/std 只在 aw>0 帧上计算，避免不同 phase 的 advantage 分布互相干扰（详见 §3.2）
- **L1 归一化**：每帧 `Σ_c |aw_c| = 1`，只有 channel 间的比例重要，绝对值不影响有效学习率。把所有 aw 同时乘以 k 不改变训练结果。
- 阶段切换逻辑放在 `on_eval` 里（基于 eval 结果），状态通过 `state()/load_state()` 持久化

### 5.2 探索调度

探索调度通过 `on_update` + `exploration` 这对 hook 实现，和 curriculum scheduling 的 `on_eval` + `build_trajectories` 完全对称：

- `on_update(stats, update)`：每轮 PPO 更新后调用，吸收 typed `UpdateStats` 到内部状态
- `exploration(update)`：每轮 rollout 前调用，读取内部状态返回 `ExplorationSpec`

```python
class MyExperiment(ExperimentPPO):

    _kl_history: List[float] = []  # 在 __init__ 或类属性初始化

    def on_update(self, stats, update):
        """吸收本轮训练统计。stats 是 typed UpdateStats。"""
        self._kl_history.append(stats.approx_kl)

    def exploration(self, update):
        """根据累积的 KL 历史动态调整探索强度。"""
        coef = 0.001
        if len(self._kl_history) >= 3:
            recent = self._kl_history[-3:]
            if all(kl < 0.005 for kl in recent):
                coef *= 4.0  # KL 连续 3 轮太平，策略卡住了，加大探索
            elif max(recent) > 0.1:
                coef *= 0.5  # KL 太大，策略在乱跳，减小探索
        return ExplorationSpec(entropy_coef=coef, temperature=1.0)
```

`UpdateStats` 的框架保证字段（typed，跨策略族稳定）：
- `approx_kl`, `max_kl`, `early_stop_kl`, `clip_frac`, `ratio_mean`, `ratio_max`
- `policy_loss`, `value_loss`, `grad_norm_actor`
- `epochs_done`, `n_batches`, `n_episodes`, `total_steps`
- `ep_len_mean`, `ep_len_min`, `ep_len_max`
- `critic_losses`, `explained_variance`, `confidence`（per-channel dict，key 是 channel name）
- `adv_mean`, `adv_std`, `ret_mean`, `ret_std`, `critic_grad_norms`（per-channel dict）

`policy_stats` 子 dict（**无跨策略族契约**）：
- Tanh-Gaussian 会贡献 `entropy`, `std_mean`, `std_min`, `std_max`, `tanh_sat_frac`
- 换成 mixture / flow / diffusion 策略后，这些 key 可能不存在或完全不同
- 当作 opaque hints 用，不要依赖具体 key 名

### 5.3 Per-channel GAE lambda

不同 reward 性质不同，适合不同的 bias-variance tradeoff：

```python
def reward_channels(self):
    return (
        # 稀疏终端 reward：高 λ（低偏差，信任远距离 return）
        RewardChannel("r_ko", gamma=0.99, gae_lambda=0.98),
        # 密集 shaping reward：低 λ（低方差，更依赖局部估计）
        RewardChannel("r_balance", gamma=0.99, gae_lambda=0.90),
    )
```

### 5.4 Episode 切分成多条 trajectory

一个 episode 可以切成多条 trajectory。典型场景：gating policy 切换控制权后，前一段属于 fight policy，后一段属于 recover policy。

```python
def build_trajectories(self, episodes):
    trajs = []
    for ep in episodes:
        # 找到 gating 切换点
        gate_step = find_gate_switch(ep)
        if gate_step is None:
            # 没有切换，整条 episode 一条 trajectory
            trajs.append(build_traj(ep, 0, ep.num_frames))
        else:
            # 切成两段，分别属于不同 policy
            trajs.append(build_traj(ep, 0, gate_step, is_terminated=True))
            trajs.append(build_traj(ep, gate_step, ep.num_frames))
    return trajs
```

切分时注意：
- 前一段的 `last_obs` 是 `obs[gate_step]`（下一段的第一帧）
- 前一段标记 `is_terminated=True`（V=0，不 bootstrap），因为后续状态属于不同 policy，OOD bootstrap 会产生错误的乐观值估计

#### ⚠️ 恶龙禁区：重叠 Trajectory

> ** Dragons be here. ** 以下行为框架**不会阻止**你做，但除非你完全理解后果，
> 否则不要碰。

`build_trajectories` 返回的 trajectory 列表中，**框架不检查 frame 是否重叠**。如果你从同一 episode 切出两条 trajectory 且它们的 frame 区间有交集，那些重叠 frame 会被当作独立数据点处理——没有任何去重。

后果的严重程度取决于**重叠的 frame 是否也共享相同的 reward channel**：

**仅 frame 重叠，channel 不重叠**（如 Traj A: frame 0-60 只有 r_fall，Traj B: frame 40-100 只有 r_cross）：

| 后果 | 机制 |
|------|------|
| **双倍采样权重** | 重叠 frame 在 buffer 中出现两次，minibatch 随机采样时被抽中的概率是其他 frame 的两倍。两个 advantage 来自不同 channel，不算"矛盾"而是"不同视角"，但采样权重的翻倍是隐性的 |

这种用法相对可控——每个 critic 只见到该 frame 一次，z-score 归一化也不受影响。主要代价是采样权重翻倍，可以用 `importance` 权重补偿。

**frame 重叠 + channel 也重叠**（如 Traj A 和 Traj B 都有 r_fall，且 frame 40-60 重叠）：

| 后果 | 机制 |
|------|------|
| **双倍梯度权重** | 同上，且更严重：同一个 channel 的 advantage 对同一 (s,a) 产生两份 gradient |
| **advantage 不一致** | 同一 channel 在两条 trajectory 中各自做 GAE backward pass，由于 reward 序列、bootstrap value、累积路径不同，会得到**两个不同的 advantage 值**。PPO 的 ratio 对同一 (s,a) 相同，但 advantage 方向可能矛盾，clipped surrogate 产生互相冲突的 gradient |
| **z-score 统计偏移** | 重叠 frame 的两个 advantage 值都参与该 channel 的 mean/std 计算，等于这些 frame 在归一化统计中获得双倍权重 |
| **critic target 矛盾** | 同一个 s 在同一个 critic 的 loss 中出现两次，但两条 trajectory 的 GAE returns 可能不同，critic 被同时拉向两个方向 |

这种用法**几乎总是错误的**。如果你发现自己需要这种结构，请阅读下面的替代方案。

**为什么框架不去重**：去重后需要决定"保留哪个 advantage"，这本身就是一个语义决策，框架无法替你做。而且如果重叠 frame 的 advantage 一致，那重叠就没有意义——直接用一条 trajectory 即可。

**如果你确实需要重叠**，以下是用前更安全的替代方案：

1. **同一帧需要多个 phase 视角的训练信号** → 用一条覆盖全 episode 的 trajectory + per-step `actor_weight` 数组。不同 channel 在不同 frame 上用不同的 `(T,)` 权重即可实现"同一帧受多个 critic 驱动"，且每帧只出现一次，GAE 连续，无矛盾。

2. **同一 channel 在不同 phase 需要不同 γ/λ** → 拆成两个 channel（如 `r_fall_standup` 和 `r_fall_balance`），各自有独立的 `RewardChannel` 配置，然后用 per-step `actor_weight` 分别 gate。代价是多一个 critic，但行为等价且无副作用。

3. **同一物理轨迹用不同假设重新标注** → 这属于 off-policy relabeling 范畴，on-policy PPO 的 buffer 设计不适合承载。考虑用 importance sampling 或专门的 off-policy buffer。

**如果你经过以上分析仍然决定使用重叠 trajectory**，你需要自行承担以下责任：
- 确认重叠 frame 在两条 trajectory 中的 reward 和 advantage 语义一致，或明确接受矛盾的 gradient
- 用 `importance` 权重补偿双倍采样（例如重叠 frame 的 trajectory 设 `importance=0.5`）
- 在实验文档中记录重叠设计的原因和预期效果

### 5.5 多 agent

一个 episode 里两个机器人都在动。你可以为每个 agent 各建一条 trajectory：

```python
def build_trajectories(self, episodes):
    trajs = []
    for ep in episodes:
        for agent_id in ("robot_a", "robot_b"):
            traj = self._build_agent_trajectory(ep, agent_id)
            if traj:
                trajs.extend(traj)
    return trajs
```

两个 agent 共享同一个 actor 和 critic（self-play），但各自有独立的 obs/action/reward。

### 5.6 Checkpoint resume

实验的内部状态通过 `state()` / `load_state()` 自动持久化。框架在存 checkpoint 时调用 `state()`，在 resume 时调用 `load_state()`。

需要持久化的典型状态：
- `_best_score` / `_best_eval`：best-of-run 判断基准
- `_phase` / `_curriculum_state`：curriculum 阶段
- 任何影响 `build_trajectories` 或 `on_eval` 行为的可变状态

模型权重和 optimizer state 由框架自动处理，你不需要管。

---

## 6. 常用工具函数

从 observer outputs 提取 per-step 信号（定义在 `baseline.framework.rollout`）：

```python
from baseline.framework.rollout import extract_per_step_field, extract_per_step_scalar

# 提取 dict-valued observer 的某个字段
phi = extract_per_step_field(ep.observer_outputs, "height_phi_a", "phi", T)

# 提取 observer 的第一个值（scalar observer）
r_cross = extract_per_step_scalar(ep.observer_outputs, "cross_support_a", T)
```

这些函数会做长度校验——如果 observer 输出长度和 episode 长度不匹配，会抛 `ValueError`，因为那通常意味着 observer 有 timestep 对齐 bug。

---

## 7. 调试建议

1. **先用 `--smoke` 跑**：2 轮 update，快速验证代码能跑通
2. **看 `__RAW_STATS__` 行**：每轮输出的 JSON 包含完整的训练统计，可以 grep 出来分析
3. **`is_terminated` 设错是最常见的 bug**：如果 critic loss 爆炸或 advantage 异常，先检查终止标志
4. **`actor_weight` 全 0 = actor 不学习**：确认至少有一个 channel 的 actor_weight > 0。注意 aw 经过 L1 归一化（`Σ|aw|=1`），绝对值不影响有效学习率——只有比例重要
