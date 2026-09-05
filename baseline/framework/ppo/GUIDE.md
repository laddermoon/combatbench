# PPO 实验框架使用指南

本指南面向想要使用本框架训练自己实验的用户。读完本指南，你就能写出一个完整的 PPO 实验并跑起来。

---

## 1. 这个框架是什么

一个**多 critic PPO 训练框架**，专为 CombatBench 的 humanoid21 机器人对抗环境设计。核心能力：

- **多 reward channel**：每个 reward 组件有独立的 critic、独立的 gamma 和 GAE lambda
- **Trajectory 级控制**：实验决定如何切分 episode、每个 channel 的 reward / 终止标志 / actor_weight
- **Curriculum scheduling**：通过 `build_trajectories` 动态调整 `actor_weight`，无需框架介入
- **探索调度**：`on_update()` + `exploration()` 这对 hook 让实验根据训练统计动态调整探索强度
- **Checkpoint resume**：实验状态自动序列化/恢复

---

## 2. 数据流

```
每个 update 的完整流程:

1. exploration(update) → ExplorationSpec
     实验决定本轮探索参数，读取 on_update() 累积的内部状态
2. actor.to_blueprint(stochastic=True) → PolicyBlueprint
     导出策略蓝图用于 rollout
3. build_jobs(policy_bp, base_seed, n_episodes) → List[Job]
     实验构建 rollout 任务（哪个环境、哪个对手、什么种子）
     explore_intensity 注入到每个 job 的 episode_options
4. ParallelRollouter.collect(jobs) → List[Episode]
     框架并行执行 rollout，收集完整 episode
5. build_trajectories(episodes) → List[Trajectory]
     实验把 episode 切成 trajectory，填入每个 channel 的
     reward / is_terminated / actor_weight
6. PPOBuffer(trajs, actor, channels)
     框架批量调用 actor.evaluate_actions 计算 old log_prob，
     为每个 channel 计算 GAE → advantage + return
7. ppo_update(actor, critics, buf, ...)
     每 channel: normalize advantage (z-score on active frames)
     L1 归一化: 每帧 Σ_c |aw_c| = 1
     合并: combined_adv = Σ_c aw_c_normed × confidence_c × norm_adv_c
     Critic 更新: MSE(V_c, return_c)
     Actor 更新: PPO clipped surrogate on combined_adv
8. on_update(stats, update)
     实验吸收本轮训练统计到内部状态
9. (每 eval_interval 轮) build_jobs(det_policy_bp, ...) → eval episodes
     on_eval(eval_episodes, update) → {is_new_best, info, stop_training?}
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
| Advantage 合并 | — | L1 归一化 aw (Σ\|aw\|=1) 后加权 by aw × confidence |
| Critic 更新 | — | MSE on returns |
| Actor 更新 | — | PPO clipped surrogate |
| Eval & 调度 | `on_eval`（完全控制） | 跑 eval rollout、导出策略 |
| 训练统计反馈 | `on_update(stats, update)` | 调用它，传入 typed UpdateStats |
| 探索 | `exploration(update)` → ExplorationSpec | 路由 explore_intensity 到 policy |
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
- **gae_lambda**：该 channel 的 GAE λ。稀疏终端 reward 适合高 λ（低偏差），密集 shaping reward 适合低 λ（低方差）

### 3.2 ChannelData

每个 trajectory 上、每个 channel 的数据：

```python
ChannelData(
    reward=np.array([0.01, 0.01, ...]),  # (T,) 每步 reward
    is_terminated=True,                   # True→V=0 不 bootstrap; False→从 critic bootstrap
    actor_weight=3.0,                     # 该 channel 的 advantage 权重，标量或 (T,) 数组
)
```

- **is_terminated**：不同 channel 在同一条 trajectory 上可以有不同的值。例如机器人摔倒：`r_fall` 标记 terminated（V=0），但 `r_cross` 可能标记 truncated（从 critic bootstrap）
- **actor_weight**：该 channel 的 advantage 对 policy gradient 的影响力。可以是标量或 `(T,)` 数组（实现逐步权重变化）

  **L1 归一化**：合并 advantage 前，框架对每帧的 actor_weight 做 L1 归一化，使 `Σ_c |aw_c| = 1`。只有 channel 间的比例重要，绝对值不影响有效学习率。负权重保留方向（反转 advantage）。

  `actor_weight` 在框架中的四个层面：

  | 层面 | aw=0 的帧 | aw>0 的帧 |
  |------|----------|----------|
  | Critic 训练 | ✓ 参与 | ✓ 参与 |
  | GAE 计算与传播 | ✓ 参与 | ✓ 参与 |
  | Advantage 归一化 | ✗ 不参与 | ✓ 参与 |
  | Actor gradient | ✗ 不产生 | ✓ 产生 |

  **典型用法**：
  - `aw=0.0`：整个 channel 的 critic 照常训练但不影响 actor——适合 warmup 新 critic
  - `aw` 为 `(T,)` 数组：逐步权重变化，如 `3.0 * standup_mask`（硬切换）或 `3.0 * phi**2`（软过渡）
  - `aw=-1.0`：反转该 channel 的 advantage 方向

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
    explore_intensity=ei,   # (T,) 每帧探索强度
)
```

- 一个 episode 可以切成多条 trajectory（按阶段、按 gating 切换等）
- `channels` 里**不包含**的 channel 在这条 trajectory 上是 inactive 的（不训练 critic，不贡献 advantage）

### 3.4 ExplorationSpec

实验对探索的"意图"，由实验在每轮 update 前返回：

```python
ExplorationSpec(
    explore_intensity=0.0,   # ∈ [-1, 1]: 0=中性, +1=最大探索, -1=最大压制
    entropy_floor=0.3,       # ∈ [0, 1]: 策略归一化熵下界
    entropy_coef=0.01,       # 熵下界损失系数，None=默认联动 explore_intensity
)
```

所有字段都是可选的，`None` 表示"不关心，保持现状"。

- **explore_intensity**：附加探索强度。具体每个值对应什么分布参数的变化，由策略自己定义。框架只规定范围和中性点 0。
- **entropy_floor**：策略归一化熵的下界。0 和 1 的具体含义由策略定义。框架用单向 hinge `relu(floor - H_norm)` 计算损失，只在熵低于下界时产生梯度。

详见 `DESIGN_unified_exploration_control.md`。

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

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class MyExperiment(ExperimentPPO):

    name = "my_experiment"
    obs_dim = 96
    action_dim = 21
    learning_rate = 1e-4
    episodes_per_update = 256
    max_updates = 5000
    eval_interval = 5
    eval_episodes = 16
    _best_score: float = -1.0

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
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
            video_eval_interval=0,
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
        bp = PolicyBlueprint.load("path/to/init_policy.yaml")
        return bp.build().to(device)

    def build_critic(self, channel_name: str, device: torch.device) -> nn.Module:
        return CriticMLP(obs_dim=self.obs_dim, hidden_dim=256).to(device)

    def exploration(self, update: int) -> ExplorationSpec | None:
        return ExplorationSpec(explore_intensity=0.0, entropy_floor=0.3)

    def build_jobs(self, policy_bp, base_seed, n_episodes) -> List[Tuple]:
        env_pb = ParameterizedEnvBlueprint.load("path/to/env.yaml")
        jobs = []
        for i in range(n_episodes):
            seed = base_seed + i
            env_bp = env_pb.materialize(max_steps=200, agent_id="robot_a")
            jobs.append((policy_bp, policy_bp, env_bp, seed, {}))
        return jobs

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs = []
        for ep in episodes:
            agent_id = "robot_a"
            T = ep.num_frames
            if T == 0:
                continue

            obs = np.asarray(ep.observations[agent_id], dtype=np.float32)
            acts = np.asarray(ep.actions[agent_id], dtype=np.float32)
            fin = np.asarray(ep.final_observation[agent_id], dtype=np.float32)

            r_cross = extract_per_step_scalar(
                ep.observer_outputs, "cross_support_a", T,
            )
            phi = extract_per_step_scalar(
                ep.observer_outputs, "height_phi_a", T,
            )
            r_fall = 0.01 * np.clip(phi, 0.0, 1.0)

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

    def on_eval(self, episodes, update) -> Dict[str, Any]:
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

    def state(self) -> dict:
        return {"best_score": self._best_score}

    def load_state(self, state: dict) -> None:
        self._best_score = float(state.get("best_score", -1.0))


EXPERIMENT_CLASS = MyExperiment
```

### 4.1 方法清单

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
| `exploration(update)` | 可选 | 每 update 前 | 返回 ExplorationSpec，默认 None |
| `state()` | 可选 | checkpoint 时 | 序列化内部状态 |
| `load_state(state)` | 可选 | resume 时 | 恢复内部状态 |

### 4.2 运行

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 baseline/framework/train.py --experiment my_experiment --algo ppo --smoke
```

`--smoke` 跑 2 轮快速验证。确认没问题后去掉 `--smoke` 正式训练，或加 `--background` 后台运行。

---

## 5. 进阶用法

### 5.1 Curriculum scheduling

curriculum 的核心是**动态调整 `actor_weight`**。在 `build_trajectories` 里根据当前训练阶段返回不同的 `actor_weight` 即可，不需要任何框架支持。

```python
class MyExperiment(ExperimentPPO):
    _phase: int = 0  # 0=balance, 1=standup, 2=walk

    def on_eval(self, episodes, update):
        # ... 计算 survival_rate ...
        if survival_rate > 0.8 and self._phase == 0:
            self._phase = 1
        return {"is_new_best": ..., "info": {"phase": self._phase}}

    def build_trajectories(self, episodes):
        trajs = []
        for ep in episodes:
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
- `actor_weight` 可以是 `(T,)` 数组，实现单条 trajectory 内的逐步权重变化
- 阶段切换逻辑放在 `on_eval` 里，状态通过 `state()/load_state()` 持久化

### 5.2 探索调度

通过 `on_update` + `exploration` 这对 hook 实现：

```python
class MyExperiment(ExperimentPPO):
    _kl_history: List[float] = []

    def on_update(self, stats, update):
        self._kl_history.append(stats.approx_kl)

    def exploration(self, update):
        if len(self._kl_history) >= 3:
            recent = self._kl_history[-3:]
            if all(kl < 0.005 for kl in recent):
                return ExplorationSpec(explore_intensity=0.5)  # KL 太平，加大探索
            elif max(recent) > 0.1:
                return ExplorationSpec(explore_intensity=-0.3)  # KL 太大，压制探索
        return ExplorationSpec(explore_intensity=0.0)  # 中性
```

`UpdateStats` 的框架保证字段（跨策略族稳定）：`approx_kl`, `max_kl`, `clip_frac`, `policy_loss`, `value_loss`, `grad_norm_actor`, `epochs_done`, per-channel 的 `explained_variance`/`confidence`/`adv_mean`/`adv_std` 等。

`policy_stats` 子 dict 是策略贡献的诊断，**无跨策略族契约**，当作 opaque hints 用。

### 5.3 Per-channel GAE lambda

```python
def reward_channels(self):
    return (
        RewardChannel("r_ko", gamma=0.99, gae_lambda=0.98),      # 稀疏终端：高 λ
        RewardChannel("r_balance", gamma=0.99, gae_lambda=0.90), # 密集 shaping：低 λ
    )
```

### 5.4 Episode 切分成多条 trajectory

一个 episode 可以切成多条 trajectory。典型场景：gating policy 切换控制权后，前一段属于 fight policy，后一段属于 recover policy。

切分时注意：
- 前一段的 `last_obs` 是 `obs[gate_step]`（下一段的第一帧）
- 前一段标记 `is_terminated=True`（V=0，不 bootstrap），因为后续状态属于不同 policy，OOD bootstrap 会产生错误的乐观值估计

**避免重叠 trajectory**：框架不去重 frame。如果两条 trajectory 的 frame 区间有交集且共享相同 channel，同一 (s,a) 会产生两份可能矛盾的 gradient。如果需要同一帧受多个 channel 驱动，用一条覆盖全 episode 的 trajectory + per-step `actor_weight` 数组。

### 5.5 多 agent

一个 episode 里两个机器人都在动。你可以为每个 agent 各建一条 trajectory。两个 agent 共享同一个 actor 和 critic（self-play），但各自有独立的 obs/action/reward。

### 5.6 Checkpoint resume

实验的内部状态通过 `state()` / `load_state()` 自动持久化。需要持久化的典型状态：`_best_score`、`_phase`、任何影响 `build_trajectories` 或 `on_eval` 行为的可变状态。模型权重和 optimizer state 由框架自动处理。

---

## 6. 工具函数

从 observer outputs 提取 per-step 信号（定义在 `baseline.framework.rollout`）：

```python
from baseline.framework.rollout import extract_per_step_field, extract_per_step_scalar

phi = extract_per_step_field(ep.observer_outputs, "height_phi_a", "phi", T)
r_cross = extract_per_step_scalar(ep.observer_outputs, "cross_support_a", T)
```

这些函数会做长度校验——如果 observer 输出长度和 episode 长度不匹配，会抛 `ValueError`。

---

## 7. 调试建议

1. **先用 `--smoke` 跑**：2 轮 update，快速验证代码能跑通
2. **看 `__RAW_STATS__` 行**：每轮输出的 JSON 包含完整的训练统计
3. **`is_terminated` 设错是最常见的 bug**：如果 critic loss 爆炸或 advantage 异常，先检查终止标志
4. **`actor_weight` 全 0 = actor 不学习**：确认至少有一个 channel 的 actor_weight > 0
