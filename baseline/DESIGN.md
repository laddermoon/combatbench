# Baseline 策略构建层 — 设计文档（草案 v2）

> 本文是 **设计意向**，不是实现 PR。目标是给 `baseline/` 这一层定一个清晰
> 的边界与点清单，等用户审过、改过之后再分批落地。
>
> **v2 变更**（相对 v1）：
> - rollout collector 定位改为 **`EpisodeRunner` / `ParallelRunner` 的薄包装**，
>   不再自己写 episode loop；
> - collector 从 day 1 **直接支持多 controlled agents**（self-play / 对抗），
>   因为 `EpisodeRunner` 已经原生 dict-of-policy / dict-of-binding；
> - 删除 `OptionsSchedule` / `curriculum/` 点——太浅（≈ `lambda i: {...}`），
>   改成 `examples/` 里的一条 recipe；
> - 删除独立的 `workers/pool.py` 点——ParallelRunner 已经覆盖，没必要再包一层；
> - 新增两个真正"深"的点：**observation / reward 归一化**（running stats，
>   PPO 稳定性依赖）和 **变长 episode → 定长 minibatch 的采样器**；
> - §1 增加一条「点的最低深度门槛」。
>
> 上一层（`envs/framework/`）的 reset / seed / observer / recorder 已经稳定
> （见 `RESET.md` / `SEED.md`），下一层（具体仿真，目前只有 `envs/humanoid21/`）
> 也稳定。这一层是**夹在中间、给 RL 训练用的脚手架**。

---

## 1. 指导原则

> **「点做实，线让用户自己连。」**

| 原则 | 含义 | 反例（不要做） |
|---|---|---|
| **点 ≠ 线** | 提供**小而锋利的可组合块**（actor backbone、rollout collector、PPO update step、checkpoint IO 等）。**不**提供"一键跑通 PPO"的总管脚本。 | 做一个 `train_anything.py`，args 里塞 200 个 flag，里面 if/else 走十几条不同算法分支。 |
| **env-agnostic** | 每个点只依赖 `envs/framework/` 暴露的抽象（`EnvRuntime` / `Policy` / `BaseObserverPlugin` / `EpisodeResult` / `Recorder`），**不**依赖 `humanoid21` 任何 symbol。 | 在通用模块里 `from envs.humanoid21 import ...`、把 21 / `robot_a` / `robot_b` 写死。 |
| **维度参数化** | 所有 actor/critic、obs adapter、action adapter、reward adapter 的 shape 由 ctor 入参或 observation/action space 决定。 | `ACTION_DIM = 21`、`obs.shape == (60,)` 这类硬编码 / 默认值。 |
| **没有"被某条线收购"的点** | 同一个 actor backbone、同一个 rollout collector，应该既能跑 PPO 又能跑 GRPO 又能跑纯 eval。 | "RolloutCollector_for_GRPO" / "PPOActor"。 |
| **可被 duck-typed 替代** | 用 Protocol / 简单基类 + 必需方法签名；用户可以**不继承**任何东西，只要鸭子像就能塞进来。 | 强制继承一棵深类树。 |
| **状态显式** | 训练中需要持久 / 切换的状态（actor weights、optimizer、RNG、obs-stats、replay buffer）都通过显式数据结构 + checkpoint IO 暴露，不藏在闭包/全局/单例里。 | 用 module-level global 当 worker 状态（当前 `base.py:_ROLLOUT_RUNTIME_BUILDER` 就是这种模式，要清掉）。 |
| **复用上层设施** | 上层能做的事，这一层**不重做**。Episode loop 用 `EpisodeRunner`；多进程调度用 `ParallelRunner`；options 派发用 `run_n_episodes(options_fn=...)`；回合比赛用 `MatchRunner`。这一层只补"上层没解决的训练侧问题"。 | 再写一个自己的 `for step in ...: sim.step(...)` 循环，让 `EpisodeRunner` 闲置。 |

### 1.1 点的最低深度门槛

一个东西**值不值得**放到 `baseline/common/` 做成"点"，按下面三条门槛过筛；
**任何一条不过**都应该：降级成 `examples/` 下的一段 recipe、写进 baseline
脚本自己处理、或者干脆不做。

1. **复杂度门槛**：一个"点"的参考实现至少要能给出 > 50 行有实质行为的代码
   （不计 boilerplate / 类型声明）。如果一个 `def f(t): return a + (b-a)*t`
   就能说明白，它不是点，是教程里的一行。
2. **正确性门槛**：这个东西有**容易写错**的角落（数值稳定、masking、bootstrap
   边界、bit-equal 复现、多进程一致性、running stats 的并行合并）。没有可
   踩的坑，就没有做成共享点的价值。
3. **复用跨度门槛**：这个点至少能被 **≥ 2 条独立 baseline 路线**（例：PPO /
   GRPO / self-play / eval）共用，并且换一个仿真环境（例：gym MuJoCo /
   mobilephone）之后**不用改实现**就能继续用。

用这把尺子重审 v1 的点清单——被 v2 砍掉的 `OptionsSchedule` / `workers/pool.py`
都是在门槛 1 栽的。保留和新增的点，每一个都在 §3 末尾打了这三条的回执。

**「线」由谁来连？** —— 用户的训练脚本。每条 baseline 是**一个 < 300 行
的脚本**，它从 `baseline/common/` 取点拼起来。脚本本身不进
`baseline/common/`，它在 `baseline/<task>/` 下。

---

## 2. 模块地形图（目标态）

```
baseline/
├── DESIGN.md                       # 本文档
├── common/                         # ★ 所有点都活在这里；env-agnostic
│   ├── policies/
│   │   ├── tanh_gaussian_mlp.py    # （已存在，dim 已参数化，保留）
│   │   ├── critic_mlp.py           # 抽出来：当前埋在 humanoid21/base.py
│   │   ├── policy_adapter.py       # nn.Module ↔ Policy Protocol 的标准桥
│   │   └── checkpoint.py           # actor/critic/optim 的 save/load + export
│   ├── rollout/
│   │   ├── collector.py            # EpisodeRunner/ParallelRunner 薄包装
│   │   │                           #   → list[RolloutBatch]；含权重注入/对手注入
│   │   ├── batch.py                # RolloutBatch dataclass
│   │   └── sampler.py              # 变长 episodes → 定长 (B,T,*) minibatch + mask
│   ├── normalize/
│   │   ├── running_stats.py        # Welford running mean/var；支持多进程合并
│   │   ├── obs_normalizer.py       # 包成 observer 或 adapter，训练/eval 两种模式
│   │   └── reward_normalizer.py    # return-based running std（PPO 标配）
│   ├── algos/
│   │   ├── value_targets.py        # GAE / RTG / n-step；独立纯函数
│   │   ├── ppo_step.py             # clipped surrogate + value loss + entropy
│   │   └── grpo_step.py            # 组内归一化 advantage
│   └── eval/
│       ├── runner.py               # 包 EpisodeRunner 或 MatchRunner
│       │                           #   → 多 seed + bootstrap CI
│       └── stats.py                # mean/std/CI/distribution helpers
├── humanoid21/                     # ★ 全部是「线」，每个文件一条 baseline
│   ├── ppo_standing.py             # = 拼 common/policies + algos/ppo + rollout
│   ├── grpo_self_play.py
│   ├── eval_combat.py              # 用 common/eval + MatchRunner
│   └── ...                         # 旧脚本逐步迁；不阻塞新东西
└── <other_env>/                    # 将来：cartpole / mujoco gym / mobilephone 等
    └── ...
```

**目的地明示**（v1 砍掉的目录）：

- ~~`rollout/parallel.py`~~ — 合入 `rollout/collector.py`，并行与否就是一个
  `max_workers` 开关（底层转给 `ParallelRunner`）。
- ~~`curriculum/`~~ — 课程化的"点"早在 `envs/framework` 那边已经是
  `ctx.episode_options` + `run_n_episodes(options_fn=...)`。`baseline` 这
  一层不再加任何东西，**课程 schedule 写在训练脚本里**；参考样例在
  `examples/07_curriculum_recipe.py`（待新增）。
- ~~`workers/pool.py`~~ — `ParallelRunner` 已经是这个东西，再包一层没价值。
- ~~`replay/reader.py`~~ — 离线重放读 recorder 的 demo 放在 `examples/05`
  里就够了；做成"点"需要先有至少一条 offline-RL baseline 才值得。

**当前态 vs 目标态的最大缝隙**：`baseline/humanoid21/base.py` 里 `Actor` /
`Critic` / `RolloutCollector` / 多进程 worker 状态全部混在一起，且静默假设
`robot_a` / `robot_b` / `21d action` / 自己的 episode loop。§5 的 B1–B4
就是在拆这个文件。

---

## 3. 点清单（v2 过了深度门槛的 8 个点）

每个点都给出：**契约**（最小签名）+ **深度回执**（过 §1.1 三条门槛的简短说明）。

### 3.1 `TorchPolicyAdapter`：nn.Module ↔ `Policy` Protocol

```python
# baseline/common/policies/policy_adapter.py
class TorchPolicyAdapter(Policy):
    """把任意 nn.Module 封成 envs.framework.policy.Policy。

    Module 必须暴露:
      - act_numpy(obs: np.ndarray, *, deterministic: bool) -> (action, extras)
      - obs_dim, action_dim 属性（int）
    """
    def __init__(
        self,
        module: nn.Module,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
        observation_space: spaces.Space | None = None,  # 仅 sanity check
        action_space: spaces.Space | None = None,
    ): ...
    def act(self, obs) -> np.ndarray: ...
    def act_with_extras(self, obs) -> tuple[np.ndarray, dict]: ...
    def reset(self, seed=None) -> None: ...
    def load_state_dict(self, sd) -> None: ...    # 训练中热更新权重
    def export(self, policy_dir: Path) -> None:   # 一键生成部署用 policy/ 目录
        ...
```

**深度回执**：
- 复杂度：~100 行（device routing、deterministic/stochastic 分流、extras
  收集、权重热更新、export）。
- 正确性：obs/action dtype 与 shape 的 sanity、`torch.inference_mode()` vs
  `no_grad()`、与 `ParallelRunner` pickle 兼容、`act_with_extras` 与
  `act` 返回值一致性。
- 复用跨度：PPO / GRPO / eval / self-play 全都共用。env 换了仍然不改。

### 3.2 Actor / Critic backbone（维度参数化）

- `TanhGaussianMLPPolicy(obs_dim, action_dim, hidden_dim, log_std_min, log_std_max)`
  —— 已存在，参数化完备。
- `CriticMLP(obs_dim, hidden_dim)` —— 从 `humanoid21/base.py:Critic` 抽出。
- `policies/checkpoint.py` —— `save_actor(model, path)` / `load_actor(path)` /
  `export_inference_only(model, policy_dir)`；当前 `tanh_gaussian_mlp.py`
  末尾的 `build_actor_export_payload` / `export_actor_policy_artifacts` 挪
  过来。

**深度回执**：
- 复杂度：actor ~200 行（reparam trick、log_prob、entropy、evaluate_actions）；
  checkpoint IO ~100 行（version-tagged payload、state_dict filtering、与
  `policy/` 部署目录的互转）。
- 正确性：tanh squash 的 log_prob jacobian 校正、log_std clamp、evaluate 时
  `atanh` 数值稳定；checkpoint 的 forward-compat（多一个 critic key 时不炸）。
- 复用跨度：PPO / GRPO / eval / export for deployment 全都共用。

### 3.3 `RolloutBatch` + `RolloutCollector`（天生多 agent）

> **上层支持审计**：`EpisodeRunner` 已经是 dict-of-policy / dict-of-binding /
> 每 agent 一个 `AgentTrajectory`（obs + actions + rewards + extras），
> `RolloutConfig.capture(agent_id)` 控制每个 agent 是否收集，`store_extras=True`
> + `Policy.act_with_extras` 自动收 log_prob / value。`final_obs` 就是
> `traj.observations[-1]`（`store_initial_observation=True` 时 T+1 obs）。
> 所以 collector **不需要任何框架改动**就能支持多 controlled agent。

```python
# baseline/common/rollout/batch.py
@dataclass
class RolloutBatch:
    agent_id:  str
    obs:       np.ndarray          # (T+1, *obs_shape) —— 含 final_obs 在末尾
    actions:   np.ndarray          # (T,   *action_shape)
    rewards:   np.ndarray          # (T,)
    log_probs: np.ndarray | None   # (T,)  由 extras['log_prob'] 转
    values:    np.ndarray | None   # (T,)  由 extras['value'] 转；或后处理填
    terminated: bool
    truncated:  bool
    info: dict                     # seed / termination_reasons / metrics 摘要

    @property
    def final_obs(self) -> np.ndarray:  # obs[-1]，等价于 gym 的 next_obs
        ...


# baseline/common/rollout/collector.py
class RolloutCollector:
    """EpisodeRunner / ParallelRunner 的薄包装。

    只做三件事：
      1) 根据 per-agent observer template 组 `observer_bindings`；
      2) EpisodeResult.trajectories[agent] → RolloutBatch；
      3) max_workers > 1 时，委托 ParallelRunner 做并行。
    **绝不自己写 while is_episode_active: step(...) 循环。**

    多 controlled agent：在 policy_factories 里传 >= 2 个受控 policy，
    collect() 的返回就是 dict[agent_id, list[RolloutBatch]]。
    """
    def __init__(
        self,
        runtime_factory: Callable[[], EnvRuntime],
        policy_factories: Mapping[str, Callable[[], Policy]],
        #   key = agent_id（与 runtime.AGENT_IDS 一一对应）
        #   value = policy factory；controlled/opponent 都从这里传
        *,
        capture_agents: Sequence[str] | None = None,
        #   None → 捕获所有 agent；否则只捕获列表里的（对手不拿 rollout）
        obs_observer_template:    str = "{agent}_obs",
        reward_observer_template: str | None = "{agent}_reward",
        reward_extractor: Callable[[Any], float] | None = None,
        store_extras: bool = True,       # PPO 必开
        max_workers: int = 1,
    ): ...

    def collect(
        self,
        seeds: Sequence[int],
        *,
        options_fn: Callable[[int], dict] | None = None,
        deterministic: bool = False,
        state_dicts: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        #   {agent_id: state_dict}；给受控 policy 热更新权重
    ) -> dict[str, list[RolloutBatch]]:
        # max_workers == 1 → EpisodeRunner.run_n_episodes(options_fn=..., ...)
        # max_workers  > 1 → ParallelRunner.run(..., options_fn=...)
        ...
```

**深度回执**：
- 复杂度：~180 行（binding 装配 + trajectory → RolloutBatch 转换 + per-agent
  capture 路由 + parallel 分支 + state_dict 注入）。
- 正确性：`final_obs` 对齐（依赖 framework `store_initial_observation=True`
  的 T+1 语义，不再自己摸）；`log_prob` / `value` 与 `action` 按步对齐；
  多 controlled 时每个 agent 的 state_dict 路由正确；parallel 时 pickle
  权重与 bit-equal 复现。
- 复用跨度：PPO / GRPO / eval / self-play 全都是它的 caller；单 agent gym
  （只有 `robot_a` 一条策略）与双 controlled（self-play）同一接口。

> **单 agent gym 环境怎么用？** 目前 `EpisodeRunner` 硬编码
> `AGENT_IDS=("robot_a","robot_b")`，单 agent 场景需要在 runtime 侧提供一个
> no-op 的 "robot_b" placeholder（或后续把 `AGENT_IDS` 改成 runtime 属性，
> 这是**框架侧**的小改动，不阻塞本文档 v2.1 的落地）。

### 3.4 `RolloutSampler`：变长 episode → 定长 minibatch

```python
# baseline/common/rollout/sampler.py
class RolloutSampler:
    """把 list[RolloutBatch]（每个长度不同）转成 PPO/GRPO 能消化的
    (num_minibatches, batch_size, *feature) tensor，附 mask。

    支持：
      - padding-based: 所有 episode pad 到同长度，mask 标 valid 位
      - concat-based:  把所有 step 按时间顺序拼接，minibatch 随机采样
        （PPO 默认走这条；长 episode 更高效）
    """
    def __init__(
        self,
        batches: list[RolloutBatch],
        *,
        mode: Literal["concat", "pad"] = "concat",
        minibatch_size: int,
        device: torch.device | str = "cpu",
        fields: Sequence[str] = ("obs", "actions", "log_probs", "advantages", "returns"),
    ): ...

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]: ...
    def __len__(self) -> int: ...              # 每 epoch 的 minibatch 数
```

**深度回执**：
- 复杂度：~120 行。shape 推断、字段裁剪、padding with mask、concat 的
  随机 shuffle、device 搬运。
- 正确性：PPO 多 epoch 训练时 minibatch 必须**每个 epoch 重 shuffle**；
  pad mode 的 loss 要乘 mask；RNN / sequence 模型的 concat 模式要保时序；
  advantage / return 的 dtype 与 reward 对齐。
- 复用跨度：PPO / GRPO / 任何 on-policy 方法都走这里；写过一次的人知道，
  这是 on-policy 训练最容易踩坑的地方之一。

### 3.5 Observation / Reward 归一化（running stats）

```python
# baseline/common/normalize/running_stats.py
class RunningMeanStd:
    """Welford 在线算 mean/var，支持多进程 chunk merge。"""
    def __init__(self, shape: Sequence[int], epsilon: float = 1e-4): ...
    def update(self, x: np.ndarray) -> None: ...          # 单 worker
    def merge(self, other: "RunningMeanStd") -> None: ...  # 跨 worker 合并
    def normalize(self, x: np.ndarray, clip: float | None = 5.0) -> np.ndarray: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, sd: dict) -> None: ...


# baseline/common/normalize/obs_normalizer.py
class ObsNormalizerObserver(BaseObserverPlugin):
    """Observer 侧的归一化：包在原始 obs observer 外面，输出 normalized obs。
    训练期 update=True（见到新样本就更新 stats）；eval 期 update=False。"""
    def __init__(self, wrapped: BaseObserverPlugin, *, update: bool = True,
                 clip: float = 5.0): ...


# baseline/common/normalize/reward_normalizer.py
class ReturnNormalizer:
    """PPO 标配：用 discounted return 的 running std 归一化 reward
    （不减 mean，只除 std）。"""
    def __init__(self, gamma: float = 0.99): ...
    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray: ...
```

**深度回执**：
- 复杂度：~200 行（Welford、多进程 chunk merge、observer 包装、state_dict IO）。
- 正确性：**这是 PPO 在新环境上跑不稳时的首要嫌疑犯**。Welford 的并行合并
  公式要对（朴素 `update` 在并行下会有数值误差）；训练/eval 切换 `update`
  标志；checkpoint 必须带上 stats（否则 resume/deploy 后分布偏移）。
- 复用跨度：几乎所有 continuous-control PPO/GRPO baseline 需要；MuJoCo gym
  / humanoid21 / cartpole 都通用。`ObsNormalizerObserver` 直接插在
  `envs/framework/observer` 系统里，换仿真环境完全不用改。

### 3.6 Advantage / Return 估计器（纯函数）

```python
# baseline/common/algos/value_targets.py
def compute_gae(
    rewards, values, dones,
    *, gamma: float, lam: float,
    bootstrap_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:          # (advantages, returns)
    ...

def compute_rtg(
    rewards, dones, *, gamma: float,
) -> np.ndarray: ...

def compute_n_step_returns(
    rewards, values, dones, *, gamma: float, n: int,
    bootstrap_value: float = 0.0,
) -> np.ndarray: ...

def normalize_advantages(adv: np.ndarray, eps: float = 1e-8) -> np.ndarray: ...
```

**深度回执**：
- 复杂度：~150 行。GAE 的逆序累加、bootstrap 边界、done 处理、dtype。
- 正确性：`done` 与 `truncation` 的区别（truncation 要用 `bootstrap_value`，
  不能当终止）；GAE 的 λ=0/1 极端情况；多 episode 的边界重置。
- 复用跨度：PPO / GRPO / 任何 on-policy actor-critic 通用。

### 3.7 PPO / GRPO update step

```python
# baseline/common/algos/ppo_step.py
def ppo_update(
    actor: nn.Module,
    critic: nn.Module,
    optim_actor, optim_critic,
    sampler: RolloutSampler,         # 已带 advantages/returns
    *,
    clip_range: float,
    value_clip: float | None,
    entropy_coef: float, vf_coef: float,
    epochs: int,
    max_grad_norm: float | None = 0.5,
    kl_target: float | None = None,      # early-stop if approx_kl > target
) -> dict:                               # losses / grad_norms / approx_kl / clip_frac
    ...


# baseline/common/algos/grpo_step.py
def grpo_update(
    actor: nn.Module,
    optim_actor,
    grouped_sampler: RolloutSampler,     # 组内 advantage 归一化后的数据
    *,
    clip_range: float,
    entropy_coef: float,
    epochs: int,
    max_grad_norm: float | None = 0.5,
) -> dict: ...
```

**深度回执**：
- 复杂度：~300 行（含 minibatch 循环、KL early-stop、clip fraction 统计、
  value clipping、梯度裁剪、approx_kl 的 Schulman 估计）。
- 正确性：`evaluate_actions` 与 `act_with_extras` 的 log_prob 一致性；
  ratio 的 numerical stability；value_clip 的正确形式；entropy coef 的
  sign。
- 复用跨度：PPO / GRPO 各自 ≥ 1 条现役 baseline，未来可能再加 A2C / REINFORCE
  变体。

### 3.8 `PolicyEvaluator`：多 seed + 置信区间

```python
# baseline/common/eval/runner.py
class PolicyEvaluator:
    """用 EpisodeRunner（单 episode eval）或 MatchRunner（多回合对战 eval）
    跑 N seed，输出每 metric 的 mean / std / bootstrap CI。"""
    def __init__(
        self,
        mode: Literal["episode", "match"],
        runtime_factory: Callable[[], EnvRuntime],
        policy_factory: Callable[[], Policy],
        opponent_policy_factory: Callable[[], Policy] | None = None,
        *,
        controlled_agent_id: str = "robot_a",
        match_rounds: int = 6,              # mode="match" 时用
    ): ...

    def evaluate(
        self,
        seeds: Sequence[int],
        *,
        options_fn=None,
        deterministic: bool = True,
        bootstrap_iters: int = 1000,
        ci: float = 0.95,
    ) -> EvalReport:
        """EvalReport: dict of metric_name → {mean, std, ci_low, ci_high, samples}"""

    def write_report(self, path: Path, report: EvalReport) -> None: ...
```

**深度回执**：
- 复杂度：~200 行（两种模式分流、bootstrap resampling、metric 聚合、报告
  格式）。不是纯"跑 N 个 seed 取平均"——bootstrap CI、deterministic vs
  stochastic 双跑、match-mode 时自动从 `MatchResult.rounds` 抽指标。
- 正确性：seed 分布要独立（用 `SeedSequence.spawn`）；match-mode 的 KO 局
  要正确归类；deterministic=True 时 policy 不应消耗 RNG；stats 稳定性。
- 复用跨度：任何 baseline 训练完都想跑 eval；单 agent gym（mode="episode"）
  与 combat（mode="match"）共用一套接口。

---

## 4. 显式不做（避免 over-engineering）

- **不**做"统一训练循环框架"。每条 baseline 自己 `for step in range(...)`。
- **不**做配置 DSL。每条 baseline 用 dataclass + argparse / hydra 由用户决定。
- **不**做日志/metric backend 抽象。用户直接 `print` 或自己接 wandb / tb。
- **不**做 curriculum schedule 的库。这东西用 `lambda idx: {...}` 一行就够
  （见 `examples/` 里的 recipe）；做成库只是在制造"查文档才会用"的负担。
- **不**做 replay buffer 的通用形状（off-policy / DQN 不在当前范围）。将来要
  做，单开 `common/replay/buffer.py`。
- **不**做自己的 worker pool。`ParallelRunner` 就是那个 worker pool；再包一层
  没有新信息。
- **不**碰 `policy/` 这一层。`policy/` 是**部署用**的策略目录规范
  （`policy.py` + `model.pt`，给 `load_policy()` 用）；本文谈的是**训练用**
  的 backbone 与 IO。两边通过 `common/policies/checkpoint.py` 的 export
  接口对接。

---

## 5. 当前实现 vs 目标的 gap

| ID | Gap | 现状指针 | 改法 | 影响面 |
|---|---|---|---|---|
| **B1** | `Actor` / `Critic` / `RolloutCollector` 全部住在 `baseline/humanoid21/base.py`，且自己写了 episode loop | `@/data1/mono/things/combatbench/baseline/humanoid21/base.py:249-322` | 拆到 `common/policies/` + `common/rollout/`；collector 改为 `EpisodeRunner` 的薄包装，**删掉** `_collect_actor_episode` 的 `while` 循环 | 大 |
| **B2** | RolloutCollector 用 module-global `_ROLLOUT_*` 传 worker 状态 | `base.py:224-235` | 委托 `ParallelRunner`；`policy_factory` + `controlled_state_dict` 做权重同步 | 中 |
| **B3** | RolloutCollector 假设 `robot_a` / `robot_b` 与固定 observer name | `base.py:_collect_actor_episode` | observer name 参数化（template）；`controlled_agent_id` / `opponent_agent_id` 入参 | 中 |
| **B4** | rollout 输出是 dict，各 baseline 自己摸 key | `base.py:310-322` | 引入 `RolloutBatch` dataclass | 中 |
| **B5** | checkpoint export 耦合在 `tanh_gaussian_mlp.py` 末尾 | `tanh_gaussian_mlp.py:85-204` | 抽到 `common/policies/checkpoint.py`；格式不变 | 小 |
| **B6** | GAE / PPO update 在每条 baseline 里各写一遍 | `baseline/humanoid21/standing_*.py` | 抽 `common/algos/value_targets.py` + `ppo_step.py` + `grpo_step.py` | 大 |
| **B7** | 没有 running-stats 的 obs/reward 归一化，各脚本自己糊或者不用 | `standing_balance_ppo.py` 等 | 上 `common/normalize/`；observer 包装法与 adapter 两种路径 | 中 |
| **B8** | 没有变长 episode → minibatch 的 sampler，各 baseline 自己拼 | `standing_*.py` | 上 `common/rollout/sampler.py`（concat + pad 两种模式） | 中 |
| **B9** | eval 散落在每条 baseline 末尾，无 CI | `standing_*.py` 末尾 + `examples/06_evaluate_policy.py` | 抽 `common/eval/runner.py`，episode/match 双模式，bootstrap CI | 中 |
| **B10** | 课程化写法各异（环境变量 / 内部 counter / hardcoded ramp） | `standing_grpo_rtg_tune*.py` | **不**做 schedule 库；加一条 `examples/07_curriculum_recipe.py` 示范用 `options_fn` | 小 |
| **B11** | `policy/README.md` 仍写着 `ACTION_DIM = 21` | `policy/README.md:43, 156` | 改成"由 `action_space` / ctor 决定"；`BaseCombatPolicy.ACTION_DIM` 标 deprecated | 小 |

---

## 6. 落地顺序（建议）

每一步是一个独立 PR，每一步都该绿测试 + 老脚本不破。

1. **B4 + B5（小、无行为变更）**：`RolloutBatch` dataclass + `checkpoint.py` 搬运。
   `base.py` 里 re-export 旧名字保持兼容。
2. **B1**：`Actor` / `Critic` 搬到 `common/policies/`；`base.py` 改成 re-export。
   老脚本一行不动。
3. **B2 + B3 + §3.1 的 `TorchPolicyAdapter`**：新 `RolloutCollector` 上线，
   **复用 `EpisodeRunner` / `ParallelRunner`**，删 `base.py:_collect_actor_episode`
   的 while loop。老 `RolloutCollector` 保留并行存在，新 baseline 优先用新的。
4. **B8**：`RolloutSampler` 上线。`B6` 会依赖它。
5. **B7**：`common/normalize/` 上线；做一个 toggle 在新 baseline 里验证
   PPO 稳定性改善再默认开。
6. **B6**：抽 `common/algos/`；写 `baseline/humanoid21/ppo_standing.py` 作
   第一条"整洁版" baseline，用 3–5 号点拼出来，对比老 `standing_balance_ppo.py`
   的曲线做回归。
7. **B9**：`common/eval/` 上线，迁 `examples/06_evaluate_policy.py` 复用它。
8. **B10**：`examples/07_curriculum_recipe.py` 新增；`examples/03` 已经示范了
   基础用法，`07` 做一条带 PPO 训练的完整课程化 demo。
9. **B11**：`policy/README.md` 文案修订；`BaseCombatPolicy.ACTION_DIM`
   标 deprecated 但不删。

> **重要**：1–9 都不要求删 `baseline/humanoid21/standing_*.py` 老脚本。
> 新东西不动它们；由用户手动决定哪些可以归档。

---

## 7. 跨环境可移植性自查清单

将来引入第二个 sim env（例：`gymnasium` MuJoCo / `mobilephone`）时，
`baseline/common/` 应该 **0 行改动**就能跑起来。做 PR 时按这个表自查：

| 检查项 | 通过条件 |
|---|---|
| `grep -ri 'humanoid21\|robot_a\|robot_b\|action_dim\s*=\s*21' baseline/common/` | 0 命中 |
| `grep -ri 'from envs.humanoid21' baseline/common/` | 0 命中 |
| 所有 ctor 接受 `obs_dim` / `action_dim` 而不是 default 21 | 类型签名 review |
| Observer name 都是 template / 入参，不是字符串字面量 | code review |
| 新 sim env 的 baseline 脚本 = `from baseline.common import ...` + 自己的 `runtime_factory` | 实测 |
| 新 env 跑 PPO 稳定 | `common/normalize/` 打开后收敛曲线正常 |

---

## 8. 用户拍板结果（2026-04-26）

1. ~~多 controlled agents 接口~~ → **一次做掉**（见 §3.3）。`EpisodeRunner`
   已原生支持 dict-of-policy，collector 直接按 multi-agent 接口落地。
2. ~~`ObsNormalizerObserver` 的 `update` toggle~~ → 走 **observer ctor**
   flag（`update: bool`），简单直接；训练脚本启动时 `update=True`，eval
   构造时 `update=False`。
3. ~~`RolloutSampler` RNN/序列模式~~ → **先不支持**；只做 `concat` + `pad`。
4. ~~`baseline/humanoid21/standing_*.py` 旧脚本~~ → **完全不动**；新点落地
   过程中不迁移、不删除、不归类。

---

## 9. 不在本文档里、但已规划好的相邻改动

- `envs/framework/RESET.md` 的 G1–G6 已落地（commit `96c2701`）。本文
  §3.3 的 `RolloutCollector.collect(options_fn=...)` 直接接 G1。
- `envs/framework/SEED.md` 的 base_seed → per-component 派生方案稳定；
  `PolicyEvaluator.evaluate(seeds=...)` 直接走它。
- `policy/load_util.py` 的 `load_policy(query_string)` 协议不变；本文
  §3.2 `checkpoint.py` 的 export 产物**就是** `policy/` 目录格式。
