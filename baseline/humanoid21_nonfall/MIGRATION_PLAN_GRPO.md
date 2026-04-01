# GRPO 迁移执行方案 (Humanoid21 EnvRuntime)

本文档是针对 `mujoco21dof_nonfall` 下 GRPO 训练链路迁移到新 `EnvRuntime` 架构的一步步执行指南。**该方案已剥离所有与 SB3 相关的逻辑**，专注于 `GRPOActor` 和 `GRPORolloutCollector` 的适配。

## 核心迁移思路
不修改 GRPO 核心算法（`grpo.py`），而是通过在 `env_adapter.py` 中编写新的 Gym Wrapper 来包裹 `envs.humanoid21.make_env()` 产生的 `EnvRuntime`，使输出的 `obs` 和 `info` 字典（如 `reward_terms`, `episode_stats`, `self_play_views`）与原接口保持一致，从而实现无缝对接。

---

## 阶段一：文件骨架初始化

**目标**：建立纯粹的 GRPO 训练目录，脱离 `train_sb3.py` 依赖。

1. **复制核心文件**：
   将以下文件从 `baseline/mujoco21dof_nonfall/` 拷贝到 `baseline/humanoid21_nonfall/`：
   - `grpo.py` (保持原样，无需大改)
   - `reward.py` (保持原样，直接复用奖励配置和计算)
   - `train_grpo.py` (稍后修改)

2. **新建辅助配置模块 `run_config.py`**：
   原 `train_grpo.py` 依赖了 `train_sb3.py` 里的 `build_run_dir`、`save_run_config` 等函数。在当前目录新建 `run_config.py`，把以下无关 SB3 的纯工具函数从旧 `train_sb3.py` 拷过来：
   - `build_run_dir()`
   - `save_run_config()`
   - `build_distance_stage_reward_config()`

3. **修改 `train_grpo.py` 的 import 路径**：
   - 将原导入 `from combatbench.baseline.mujoco21dof_nonfall.grpo import ...` 
   - 改为本地导入 `from .grpo import ...`
   - 将 `from combatbench.baseline.mujoco21dof_nonfall.train_sb3 import ...` 
   - 改为本地导入 `from .run_config import ...`

---

## 阶段二：实现新架构环境适配器 (`env_adapter.py`)

**目标**：把新 `EnvRuntime` 包装成 GRPO 收集器能看懂的单边或自我对弈（Self-Play）Gym 环境。

在当前目录新建 `env_adapter.py`，实现以下内容：

### 1. 实现通用信息提取器 (Info Extractor)
编写辅助函数，从 `EnvRuntime` 抽取数据并归一化为 GRPO 需要的格式：
- **`_extract_metrics(runtime, agent_key)`**:
  从 `runtime.get_shared_info()` 和 observer 提取 `horizontal_distance`, `uprightness`, `damage_dealt`, `clamp_count` 等。
- **`_build_agent_info(metrics, reward_terms, episode_stats)`**:
  组装出 `info["attacker_metrics"]`, `info["reward_terms"]`, `info["episode_stats"]`。

### 2. 实现 `Humanoid21SingleAgentEnv`
继承 `gym.Env`。
- **`__init__`**: 
  调用 `envs.humanoid21.make_env(...)` 创建 `self.runtime`。
  定义 `action_space` 和 `observation_space`。
- **`reset`**:
  调用 `self.runtime.reset()`。
  返回 `self.runtime.get_observer_output("robot_a_obs")` 和组装好的 `info`。
- **`step`**:
  执行 `self.runtime.step(action, opponent_action)`（其中 `opponent_action` 由内建对手策略生成）。
  调用奖励函数 `compute_attacker_reward`。
  检查 `self.runtime.get_termination_flags()` 返回 `terminated`, `truncated`。
  返回 `obs_a, reward, terminated, truncated, info`。

### 3. 实现 `Humanoid21SelfPlayEnv`
继承 `gym.Env`。专门对接 `GRPORolloutCollector` 的 self-play 模式。
- **`reset`**:
  返回 `np.stack([obs_a, obs_b])`。
  构建并返回 `{ "self_play_views": { "robot_a": info_a, "robot_b": info_b } }`。
- **`step(actions)`**:
  切分 `actions` 为 `action_a` 和 `action_b`。
  执行 `self.runtime.step(action_a, action_b)`。
  分别计算两边的 reward 和 stats。
  返回 `stacked_obs, 0.0, terminated, truncated, combined_info`。

---

## 阶段三：适配多进程向量化 (Vectorization)

**目标**：GRPO 需要 `VecEnv` 收集数据，但不使用 `train_sb3.py` 的逻辑。

1. **在 `env_adapter.py` 中补充工厂函数**：
```python
def make_humanoid21_env(args, eval_mode=False, rank=0):
    def _init():
        # 根据 args 创建 Humanoid21SingleAgentEnv 或 Humanoid21SelfPlayEnv
        env = Humanoid21SelfPlayEnv(...) if args.rollout_self_play and not eval_mode else Humanoid21SingleAgentEnv(...)
        return env
    return _init
```

2. **在 `run_config.py` 中补充 VecEnv 构建逻辑**：
```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from .env_adapter import make_humanoid21_env

def build_train_vec_env(args):
    env_fns = [make_humanoid21_env(args, eval_mode=False, rank=i) for i in range(args.n_envs)]
    return SubprocVecEnv(env_fns, start_method=args.subproc_start_method)

def build_eval_env(args):
    env_fns = [make_humanoid21_env(args, eval_mode=True, rank=0)]
    return DummyVecEnv(env_fns)
```
*(注：虽然不使用 SB3 算法，但 `VecEnv` 属于 SB3 的底层并行工具，在 `grpo.py` 中已经被依赖，这里保留其用于环境并行收集。)*

---

## 阶段四：收尾与联调

1. **更新 `train_grpo.py`**：
   - 检查并清理未使用的方法和库。
   - 确保 `build_train_vec_env` 和 `build_eval_env` 导入的是我们自己新写的配置模块。
   - 修改默认的 `--output-dir` 为 `baseline/humanoid21_nonfall/runs`。

2. **运行测试验证**：
   在根目录下执行：
   ```bash
   python things/combatbench/baseline/humanoid21_nonfall/train_grpo.py --total-timesteps 2000 --n-envs 2 --episodes-per-update 4
   ```
   **预期结果**：
   - 不报错崩掉，成功走完一次 `optimize_grpo`。
   - Terminal 正常输出 `Mean episode return` 等 rollouts 统计数据。
   - 正常生成 checkpoint 和 tensorboard 日志。

---

## 避坑指南 (Pitfalls)

- **`episode_stats` 字典名不能改**：`grpo.py` 内部硬编码了去取 `info["episode_stats"]["clamp_count"]`，在抽取新 `EnvRuntime` 状态时，务必填入对应的键值。
- **环境不能自动 reset**：`GRPORolloutCollector` 通过判断 `dones` 手动维护 trajectory 收集。新写的 `env_adapter.py` 中**不需要也不应该**在 `step` 结束时自动调用 `reset()`，交给外部的 `VecEnv`（如 DummyVecEnv/SubprocVecEnv 的自动 reset 机制）或 GRPO Collector 去处理。
- **对旧 reward.py 的兼容**：新架构下可能少了一些字段，提取 `metrics` 字典传给 `compute_attacker_reward()` 前，打印一次 dict 结构比对，看是否有 keyError。
