
Constraint / Intervention 插件：

[接口]
on_reset(env, state)
before_physics_step(env, action_dict)
after_physics_step(env, state)
modify_state(env, state) 或 apply(env)
[实现例子]
NonFallOrientationClamp
RootHeightClamp
JointSafetyClamp
ActionRateLimiter


每个机器人都有一个 control_mode：

[policy]
正常接收 policy 动作
[frozen_zero]
动作恒为 0
[hold_pose]
用 reference pose 持续站住，不只是“零动作”
[scripted]
用脚本动作
[replay]
回放动作序列



为什么 hold_pose 很重要
“打也不动”如果只是零动作，未必真的稳定。
你可能真正需要的是：

站桩对手
静态靶子
姿态锁定对手
所以建议把“静止”拆成两类：

[zero action] 不发力
[pose hold] 主动维持姿态



4. 随机扰动 / 推一下
这应该是单独的 Disturbance Injector，不要混在 step 逻辑里。

建议接口
on_step_begin(env, step_idx)
on_substep(env, physics_step_idx)
maybe_apply_force(env, rng)
配置维度
[作用对象]
robot_a
robot_b
both
[作用部位]
torso
pelvis
arm
random body
[触发方式]
固定时间点
随机泊松触发
进入某姿态区间后触发
[力的参数]
大小
方向
持续时间
世界系 / 局部系
推荐用途
walk balance
push recovery
robustness training
adversarial disturbance benchmark


第一层：稳定的原始事实 raw_info
只包含客观事实：

robot states
joint states
root pose / vel
collisions
contacts
hit records
hp / damage
relative pose
clamp/intervention records
external force records


第二层：按需计算的 metrics
通过 Metric Collectors 动态生成：

distance metrics
facing metrics
upright metrics
action smoothness metrics
episode aggregates
curriculum-specific metrics
建议接口
MetricCollector.compute(prev_info, info, actions, context) -> dict
并且允许注册多个 collector：

DistanceMetricsCollector
BalanceMetricsCollector
CombatMetricsCollector
ActionMetricsCollector
InterventionMetricsCollector
重要原则
Env 核心只保证 raw_info 可靠
训练 wrapper / reward wrapper 再选择自己要的 metrics
不要把 reward 需要的所有字段都塞回 CombatGymEnv 本体



envs/training_wrappers.py
single-agent view
symmetric self-play view
reward wrapper
episode stats wrapper



CombatEnvConfig
不要让 CombatGymEnv.__init__ 继续堆参数了。
建议改成 config dataclass。

例如拆成：

[base config]
dt
control_frequency
match_duration
render_mode
[spawn config]
initial_distance
orientation
resetter
[control config]
kp / kd
action scale
reference pose
[runtime modules]
constraints
disturbances
metric_collectors
[agent configs]
robot_a control mode
robot_b control mode
这样比 20 个裸参数更可维护。

对外 API 的建议
为了让别人开发方便，我建议把 API 设计成“声明式”：

python
env = CombatGymEnv(
    config=CombatEnvConfig(...),
    resetter=RandomizedPoseResetter(...),
    constraints=[
        NonFallOrientationClamp(...),
    ],
    disturbances=[
        RandomPushInjector(...),
    ],
    metric_collectors=[
        DistanceMetricsCollector(),
        BalanceMetricsCollector(),
    ],
)



[整理 info schema]
统一成：
raw_state
events
interventions
metrics
debug



可复现性与实验版本化
[seed 一致性]
reset、初始化姿态、扰动、对手脚本、domain randomization 都要统一受 seed 控制。
[scenario/version 标识]
每次训练都应能明确记录：
env 版本
激活了哪些插件
参数快照
[deterministic/debug mode]
方便复现某一次异常 rollout。
这是平台型项目非常重要的一层。



2. Domain Randomization
[物理参数随机化]
质量
摩擦
阻尼
关节强度
[观测噪声]
位置、速度、接触信号噪声
[执行噪声]
action delay
action scaling drift
action dropout
如果以后希望策略更稳健，这一层很关键。


3. 观测系统的模块化
[可选 observation blocks]
proprioception
opponent relative state
contact
history stack
privileged info
[训练 / 部署观测分离]
训练时可以用 privileged info
部署时只保留可感知信息
[可裁剪观测]
不同算法、不同任务不需要固定 127 维模板



4. Action / Controller 抽象
你现在 CombatGymEnv 里已经有 controller 参数和 target position 逻辑，这块以后很容易继续膨胀。

[不同控制模式]
torque-like
target position
delta target
pose tracking
[动作后处理链]
clip
scale
smooth
deadzone
action latency
[per-agent controller config]
两个机器人不一定同配置
这个建议尽早单独抽象，否则训练逻辑和控制逻辑会缠在一起。

5. 终止条件系统
[termination plugins]
time limit
出界
跌倒
双方长时间无接触
达成目标
[终止原因标准化]
所有 done/truncated 都要有统一 reason code
这对 reward、统计、debug 都很重要。

6. 事件系统 / Trace 系统
[event log]
hit
clamp
disturbance
reset source
out-of-bound
[step trace 开关]
只在 debug 模式下记录详细 trace
[episode summary 自动生成]
episode 结束时输出聚合统计
以后你做分析工具时会非常方便。

7. 状态快照与回放
[snapshot / restore]
保存当前仿真状态
从中间状态恢复
[trajectory replay]
重播某个 episode
[counterfactual debug]
同一个状态下测不同动作
这是调 reward、调扰动、调初始化时很有价值的能力。

8. 多任务 / 多场景接口
[task layer]
combat
approach
balance
recovery
target reaching
[scenario presets]
combat_nonfall_v1
walk_balance_push_v1
static_target_strike_v1
否则未来任务一多，env 参数会变成一坨。

9. 安全保护与数值稳定
[NaN / Inf 检查]
[姿态异常保护]
[关节极限保护]
[非法状态自动标记]
[仿真爆炸时 graceful fail]
平台化项目要考虑“别人乱用时也能给出好错误”。

10. 评估友好的能力
[评估模式与训练模式分离]
关闭噪声
固定 reset
固定扰动计划
[benchmark suites]
一组固定测试 case
[统一 summary schema]
方便自动对比算法


1. Core Env
[职责]
物理仿真
两机器人生命周期
基础观测
基础事件
[文件]
envs/combat_gym.py
2. Runtime Modules
[职责]
resetter
constraints
disturbances
control modes
termination rules
[文件]
envs/resetters.py
envs/constraints.py
envs/disturbances.py
envs/control_modes.py
envs/terminations.py
3. Metrics / Data Layer
[职责]
从 raw state / events 计算可扩展指标
[文件]
envs/metrics.py
4. Task / Training Wrappers
[职责]
single-agent
self-play
reward wrapper
episode aggregation
[文件]
envs/training_wrappers.py





仿真的核心业务流程：
1. 设置初始位置。
2. 不断执行动作Step 
3. 在每个动作步中执行几个物理仿真步。
4. 判断结束。

数据分类：
1. 仿真计算需要的数据， 这又可以分成：
1.1 场景数据（仿真资产）、配置参数（PD参数等） 静态的
1.2 状态数据 动态的，会有一个初始状态，然后会通过外部输入和内部仿真计算进行更新 （物理Step级）
1.3 外部输入 （动作Step级）
2. 不是仿真计算必须的数据，理论上的衍生的，用于渲染、计算Reward等
2.1 
数据粒度： 
1. 物理步级
2. 动作步级
3. Episode级

CombatGymEnv 只完成上面的基础功能，不需要实现Gym接口。
在上面的过程中留下嵌入点。可以通过这些嵌入点，执行数据操控。
这些嵌入点可以是标准化的： PreActionStep 、 PostActionStep 、 PrePhyStep 、 PostPhyStep
PreEpisode \ PostEpisode

就只保留这一种标准化的嵌入方式，把constraints control modes disturbances metrics resetters 都归入这个框架下是否可行 ， 有什么是不能覆盖的吗




class BasePlugin:
    def pre_episode(self, ctx: CombatContext): pass
    def pre_action_step(self, ctx: CombatContext): pass
    def pre_phy_step(self, ctx: CombatContext): pass
    def post_phy_step(self, ctx: CombatContext): pass
    def post_action_step(self, ctx: CombatContext): pass
    def post_episode(self, ctx: CombatContext): pass
```

**最终结论**：
**完全可行**。只要引入一个携带生命周期缓存和分区权限的 `Context`，就可以废弃现有的多套插件接口，用一种标准化 Hook 囊括所有功能。


CombatCore # 维护数据， 执行仿真 ， 仿感器数据的获取。 在指定位置执行插件。
CombatCore的实现中：必须清晰定义自己所维护的数据的定义
1. 仿真计算需要的数据， 这又可以分成：
1.1 场景数据（仿真资产）、配置参数（PD参数等） 静态的
1.2 状态数据 动态的，会有一个初始状态，然后会通过外部输入和内部仿真计算进行更新 （物理Step级）
1.3 外部输入 （动作Step级）

插件的基本设计
class CombatContext:
    static_data: Any      # 静态配置，只读
    state: Any            # 物理状态，Resetters/Constraints 可改
    cameras: Dict[Str][Callable] # 用来获取摄像头观测数据 ， 只读， 可调用来获取数据
    action_input: Any     # 最近一次的Action动作， 可改
    terminate_flag: bool  # 任何插件可置为 True


class BasePlugin:
    def pre_episode(self, ctx: CombatContext): pass
    def pre_action_step(self, ctx: CombatContext): pass
    def pre_phy_step(self, ctx: CombatContext): pass
    def post_phy_step(self, ctx: CombatContext): pass
    def post_action_step(self, ctx: CombatContext): pass
    def post_episode(self, ctx: CombatContext): pass


把CombatCore 和Plugin做为底层抽象。




CombatGymEnv 
->  Simulator 


OpenSimulator   执行仿真，并且向外开放数据，让外部可以访问和修改仿真状态数据，以便实现多种能力（比如观测、扰动、数据记录、Reward计算等）。
实现的功能： 
接收动作指令
物理步推进
获取传感器数据
获取静态数据： 比如机器人、场景、配置参数等
获取状态数据： 比如关节角度 速度 受力等
修改状态数据： 对于数据进行修改



things/combatbench/envs/humanoid21/envs。py
在这里面实现几种环境：
第一类，只有机器人A可控（只关注机器人A），机器人B采用固定策略：
1.机器人B保持初始姿态始终不动，像雕塑一样（违背物理规律也没问题），受到攻击也不动（如果做不到就算了）。 
2.机器人B站立不动，使用站立策略 things/combatbench/policy/standing.py
3.机器人B使用某种策略 BaseCombatPolicy的实现（2是这个的特例而己）。
第一类中，Reward、Observation之类只关注机器人A
第二类， 两个机器人都受控
1. 不关注Reward， 只是为了跑比赛。
2. 用同一策略SelfPlay，跑完一个Episode之后，分别从两个机器人的视角记录两条Episode数据
通过这两几种情况验证框架的可用性。 注意一点是，   

Nonfall这种约束作为一种通用的Hook， 与上面的各种正交，每一种都可以加Nonfall Hook， 就可以让机器人不倒。
为了实现上面的目标，需要持续优化things/combatbench/envs/humanoid21 目录下的所有代码实现。
完成功能之后至少优化三轮。
优化目标是代码抽象简洁合理， 代码清晰易懂，没有冗余的代码。 


things/combatbench/envs/humanoid21
尝试对于以上目录下的核心代码进行测试， 以功能和集成测试为主。 同时留意设计缺陷。