# CombatBench: 人形机器人对战基准平台

![CombatBench Hero](assets/images/hero.png)

**在线平台：[www.combatbench.tech](http://www.combatbench.tech)**（域名无法访问时可用 IP：[180.76.152.227](http://180.76.152.227)）— 注册账号、提交策略、观看比赛、查看 Elo 排名。

CombatBench 是一个基于 MuJoCo 的开源人形机器人对战仿真平台：两个 21 自由度的人形机器人在拳击台里对抗，用强化学习训练策略，看谁能把对方的血量先打到零。

它不只是一个单一的基准场景或一层 Gym 封装——它是一个可复用的**环境运行时框架**，用于构建机器人对机器人的任务，`humanoid21` 是基于此框架的第一个完整实现，配套基线策略、训练方法论和公开榜单。整个项目围绕 [combatbench.tech](http://www.combatbench.tech) 在线平台运营（域名不可访问时可用 IP [180.76.152.227](http://180.76.152.227)）：参赛者在网站上注册、提交策略，平台后台自动跑对战并用 Elo 算法排名。

---

## 为什么做这件事

强化学习社区有大量 benchmark，但它们大多集中在三类任务上：单 agent 连续控制（MuJoCo、DM Control、IsaacGym）、离散博弈（Atari、围棋、星际争霸）、机械臂操作（RoboSuite、Meta-World）。

**两个高自由度人形机器人在连续物理下的对抗**这件事，几乎没有一个成熟的、被维护的、带规则和基线的公开 benchmark。

这个空白值得填补，因为双人对抗是强化学习里最硬的任务类型之一：它同时要求全身平衡、接触丰富的控制、对非平稳对手的快速反应、以及攻防策略的权衡。还有一个现实理由——这类任务**不拼算力**，一个中等配置的 GPU 就能训练出有竞争力的策略，起决定作用的是策略的精巧度和训练方法的好坏。

---

## 核心特点

- **框架优先**：`envs/framework` 层将物理后端、世界规则、运行时调度和观测输出分离。
- **世界插件与观测插件正交**：换奖励函数不用碰物理，换观测编码不用碰规则——两个扩展轴完全独立。
- **Blueprint 可序列化**：整个环境（模拟器 + 插件 + 参数）序列化为一个 YAML 文件，实验可复现、可分享。
- **为实验而构建**：同一运行时可支持强化学习训练、脚本基线、消融实验、评估比赛和未来的新机器人。
- **人形对战作为压力测试**：平衡、接触、扰动、非对称策略和自我博弈都在一个环境中展现。
- **不拼算力**：CPU 并行物理 + 课程学习 + 安全门，让消费级硬件也能训练竞技策略。

---

## CombatBench 里有什么

### 1. 仿真环境：Humanoid21

两个 21 自由度的人形机器人在 6.1 米见方的全封闭拳击台里对战。

- **机器人**：21 自由度（腰部 3 + 每腿 6 + 每臂 3），固定增益 PD 伺服控制。
- **观测**：96 维向量，分四块——本体感知 42 维 + 根状态 13 维 + 触觉 2 维 + 对手信息 39 维。所有对手信息转换到自身坐标系，不提供绝对世界坐标以避免位置过拟合。
- **动作**：21 维归一化关节目标位置，[-1, 1] 区间，20Hz 决策频率。
- **物理**：MuJoCo 500Hz，每次决策之间跑 25 个物理子步，双方参数严格对称。

### 2. 框架：可扩展运行时

框架围绕一组显式抽象接口构建。以下是核心接口、用途和已有实例：

**`BaseSimulator`** — 物理引擎的薄封装。只管步进（`physical_step`）、状态读写、力施加。不知道"比赛""得分"等概念。
- 实例：`MujocoCombatSimulator`（humanoid21）、T800 simulator（集成中）

**`IDataAccessor` / `IDataMutator`** — 能力分离的读写契约。Accessor 永远可用（只读：核心状态、派生状态、传感器数据）；Mutator 按需授予（写：设置状态、设置动作、施加外力）。权限边界显式，观测插件拿不到 Mutator，不会误改物理。

**`SimContext`** — 跨插件共享的黑板。存放 `metrics`（血量、伤害、计数）、`events`（命中、越界）、`termination_proposals`（timeout / ko / foul），插件间通过它通信而非直接互调。

**`BasePlugin`**（世界插件）— 世界规则的裁判，拥有 6 个生命周期钩子（episode 前后、action step 前后、physics step 前后），声明 `require_mutator=True` 即可写物理。
- 实例：`CombatScoringPlugin`（HP 扣分）、`NonFallConstraintPlugin`（防摔约束）、`InitialStatePerturbationPlugin`（初始扰动）、`ContinuousWindPlugin`（风力）、`InstantPushPlugin`（瞬时推力）、`TimeoutPlugin`（超时终止）

**`BaseObserverPlugin`**（观测插件）— 只读输出构造器，从 `IDataAccessor` 构建观测、奖励、调试信号。由内部 `_ObserverDispatcherPlugin` 统一批量调度，每个生命周期只做一次上下文转换。
- 实例：96 维观测构造器、8 个奖励模块（`cross_support` / `damage` / `follow_opponent` 等）、平衡分析调试器

**`PostActionRecorder`**（录制器）— 第三类运行时钩子，和插件平级但本质不同：纯副作用，不修改仿真状态也不产出被 runtime 消费的输出。在每步动作后记录 pre-action 观测、action、post-action observer outputs，形成完整的 $(s_t, a_t, s'_{t+1})$ 转移快照。
- 实例：`BaseFrameRecorder`（落盘格式：每步 PNG 图像 + JSON 状态，含 manifest 和 index，足以确定性回放每一次 `IDataAccessor` 读取）、`EpisodeBufferRecorder`（内存缓存，供训练器直接消费）
- 配套 `ReplaySimulator`：实现 `BaseSimulator` 接口，从录制文件回放——observer / plugin / 训练代码无需修改即可在录制数据上重跑，这让"录制 → 逐帧检查 → 定位问题"形成闭环

世界插件和观测插件是**正交的两条扩展轴**：要改规则（比如加犯规系统）加世界插件；要改奖励或观测编码加观测插件，互不影响。录制器则是独立于这两者的第三轴——负责把训练过程中的关键 episode 固化下来供调试和回放。

**`EnvRuntime`** — 开发者的公共 API。`step(action_a, action_b)` 驱动双边动作，`get_observer_output()` 取观测/奖励，`get_shared_info()` 取黑板数据。配套 `RoundRunner`（单回合执行）和 `MatchRunner`（多局比赛 + HP 累积）。

**`EnvBlueprint`** — 整个环境序列化为 YAML：模拟器类+配置、世界插件有序列表、观测插件映射、运行时参数。加载一个 YAML 文件即可完整复现别人的实验。

### 3. 基线：四阶段课程 + 安全门

直接在完整对战任务上端到端训练会失败（机器人活不过前几秒，探索黑洞）。我们的基线用**四阶段课程**分解问题：

| 阶段 | 任务 | 难点 |
|------|------|------|
| 1. 基础站立 | 不加扰动，学会站 | 关节协调 |
| 2. 平衡恢复 | 加初始状态扰动，学会恢复 | 抗倾斜、抗推 |
| 3. 跟随对手 | 一边平衡一边追踪移动的对手 | 平衡 + 移动 + 朝向 |
| 4. 完整对战 | HP 规则下正式打 | 攻防权衡 |

**安全门**（Safety Gate）是基线的核心创新：一个 MLP 分类器预测当前状态是否安全，不安全时把控制权交给冻结的保守恢复策略。用了滞后状态机——宁可多保护一会儿，也不冒险太早交还控制权。

### 4. 平台：[combatbench.tech](http://www.combatbench.tech)

**网站 [www.combatbench.tech](http://www.combatbench.tech)（备用 IP [180.76.152.227](http://180.76.152.227)）是整个项目的对外入口。** 参赛者在网站上注册账号、提交策略，平台后台自动跑对战，用 Elo 算法算排名，可以看到比赛视频和排名榜。配套命令行提交工具 `combat-submit`（支持 macOS / Linux / Windows）。

整个流程：本地用框架和基线训练策略 → 打包提交到 combatbench.tech → 后台自动对战 → Elo 排名实时更新 → 观看比赛回放。

### 5. 方法论：用 AI 训练 AI

除了 benchmark 本身，项目还贡献了一套训练方法论。核心思想和测试驱动开发（TDD）高度相似：**训练前先把"健康的训练"长什么样定义下来**，然后让 AI 在闭环里监控-诊断-修复。

训练过程输出结构化仪表盘（不是给人看的，是给程序看的），日志分析程序提取关键指标并标记警报，AI 定期看健康报告并按诊断协议排查问题（KL 早停、解释方差为负、熵塌缩等），然后调超参或改代码，重训并循环。

---

## 比赛规则

两个机器人各 100 血，打 6 局每局 30 秒，先打到 0 血的输，时间到看谁血多。

- **有效打击**：攻击部位（手、前臂、肘、上臂、足、小腿、膝、大腿）击中目标部位（头扣 3 血、躯干扣 1 血），碰撞相对速度需超过阈值，单次碰撞只扣一次血。
- **无犯规限制**：摔倒、翻滚、地面追打、抱缠、任何姿势都是合法的。AI 不需要模仿人类武术，可以自由演化最优策略。
- **严格对称**：双方质量、运动学、执行器增益、物理参数完全一致，给定相同随机种子的对战逐位可复现。

详见 [`docs/RULE_zh.md`](docs/RULE_zh.md)。

---

## 安装

### 要求

- Python 3.8+
- MuJoCo 3.x
- Gymnasium
- NumPy
- OpenCV (cv2)

### 设置

```bash
# 克隆仓库
# git clone https://github.com/laddermoon/combatbench.git
# cd combatbench

# 安装依赖（确保已安装 mujoco）
pip install mujoco gymnasium numpy opencv-python imageio egl
```

---

## 快速开始

运行一回合并保存视频：

```bash
# 无显式策略运行（默认站立行为）
python envs/humanoid21/run_round.py --duration 10 --video test.mp4

# 使用随机策略运行
python envs/humanoid21/run_round.py --policy-a random --duration 5 --video test.mp4

# 运行两个不同策略
python envs/humanoid21/run_round.py \
  --policy-a random \
  --policy-b standing \
  --duration 15 --video match.mp4
```

用 Python 代码运行：

```python
from combatbench.envs.humanoid21 import Humanoid21Simulator, RoundRunner
from combatbench.policy import load_policy

policy_a = load_policy("random?scale=0.1&seed=42")
policy_b = load_policy("standing")

runner = RoundRunner(
    simulator=Humanoid21Simulator(),
    policy_a=policy_a,
    policy_b=policy_b,
    match_duration=30.0,
    render_mode="rgb_array",
)
result = runner.run(save_video_path="output.mp4")
print(f"Winner: {result.winner}, Steps: {result.steps}")
```

---

## 项目结构

```
combatbench/
├── assets/          # MuJoCo XML 模型、纹理、网格
├── envs/
│   ├── framework/   # 可复用核心框架（后端契约、运行时、插件系统）
│   └── humanoid21/  # 21 自由度人形机器人环境
├── policy/          # 策略接口和参考策略（random、standing）
├── baseline/        # 训练基线（PPO 课程 + 安全门）
│   └── humanoid21/
│       ├── curriculum/   # 四阶段课程训练框架
│       ├── rewards/      # 8 个可组合奖励模块
│       └── runs/         # 125+ 训练记录
├── docs/            # 规则、环境规格、设计文档
├── examples/        # 9 个示例脚本（覆盖完整开发周期）
└── tools/           # combat-submit 提交工具
```

---

## 策略接口

所有对战策略必须继承自 `BaseCombatPolicy` 并实现 `act()` 方法：

```python
from combatbench.policy import BaseCombatPolicy
import numpy as np

class MyPolicy(BaseCombatPolicy):
    ACTION_DIM = 21

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 你的初始化

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        """返回形状为 (21,) 的动作数组，值范围在 [-1, 1]"""
        # 你的动作计算
        return action

    def reset(self) -> None:
        """重置内部状态（可选）"""
        pass
```

策略支持目录式加载和带参数的查询字符串格式：

```python
policy = load_policy("my_policy")                    # 目录路径
policy = load_policy("my_policy.policy.MyPolicy")    # 模块路径
policy = load_policy("my_policy?scale=0.2&seed=42")  # 带参数
```

完整接口定义请参见 [`policy/base.py`](policy/base.py) 和 [`policy/README.md`](policy/README.md)。

---

## 关键文档

设计契约和深度文档：

- **框架架构**：[`envs/framework/DESIGN.md`](envs/framework/DESIGN.md)
- **Humanoid21 观测设计**：[`envs/humanoid21/OBSERVATION_zh.md`](envs/humanoid21/OBSERVATION_zh.md)
- **Humanoid21 数据契约**：[`envs/humanoid21/DATASPEC.md`](envs/humanoid21/DATASPEC.md)
- **Humanoid21 控制契约**：[`envs/humanoid21/CONTROLSPEC.md`](envs/humanoid21/CONTROLSPEC.md)
- **Humanoid21 基线指南**：[`baseline/humanoid21/README.md`](baseline/humanoid21/README.md)
- **训练可观测性契约**：[`baseline/humanoid21/curriculum/OBSERVABILITY.md`](baseline/humanoid21/curriculum/OBSERVABILITY.md)

规则与环境：

- [对战规则](docs/RULE.md) / [中文规则](docs/RULE_zh.md)
- [环境详情](docs/ENVIRONMENT.md) / [中文环境](docs/ENVIRONMENT_zh.md)

---

## 面向谁

- **强化学习研究者**：一个新的对抗连续控制 benchmark，有完整环境、框架和基线。
- **机器人控制研究者**：高自由度人形在对抗压力下的平衡、恢复、接触控制问题。
- **策略/博弈研究者**：HP 制 + 无限制规则下的双 agent 策略演化。
- **想要参与但不一定有大算力的团队和个人**：这个任务门槛低的是算力要求，高的是策略和方法论的巧妙程度。

---

## 路线图

- 更多机器人平台（T800 人形部分集成中、Unitree G1 规划中）
- 纯视觉感知变体（去掉对手关键点，只用主观视角图像）
- AI-in-the-loop 训练方法论泛化到更多 RL 任务族
- 社区贡献：榜单上的策略越多，越能看到 HP-only 规则鼓励的策略多样性涌现出来

---

## 贡献

我们欢迎贡献！请遵循标准开源拉取请求工作流。

---

## 链接

- **在线平台（注册 / 提交策略 / 排名 / 比赛视频）：[www.combatbench.tech](http://www.combatbench.tech)（备用 IP [180.76.152.227](http://180.76.152.227)）**
- GitHub 仓库：[github.com/laddermoon/combatbench](https://github.com/laddermoon/combatbench)
