# CombatBench 项目分析报告

> 生成日期：2026-10-20
> 重点：`baseline/humanoid21/hierarchy`（分层控制路线）训练现状

## 1. 项目概览

**名称**：CombatBench — 基于 MuJoCo 的双人形机器人格斗仿真 + RL 训练基准。

- **类型**：研究/实验项目（非生产系统），自建 RL runtime，不用 gym.Env / SB3
- **语言**：Python 3.10（MuJoCo、PyTorch、NumPy、SciPy）
- **核心场景**：两台 21-DOF 人形机器人在圆形场地中对战，含碰撞、计分、HP、摔倒/出界判定

## 2. 技术栈与框架

| 层 | 内容 |
|---|---|
| 物理 | MuJoCo（`envs/humanoid21/battle_circular_v2.xml` 当前场地） |
| 观测 | 96 维（42 本体 + 13 root + 2 足底力 + 39 对手） |
| 运行时 | `envs/framework/`：`EnvRuntime` + 插件系统（BasePlugin / BaseObserverPlugin），SimContext 黑板 + accessor/mutator 权限分离 |
| 训练 | `baseline/framework/`：自研多 critic PPO/SAC；每奖励通道一独立 critic；`actor_weight` 逐帧调制梯度；实验注册表自动发现 `exp_*.py` |
| 复现 | 每个 run 自动建 `exp/*` git 分支做代码快照，run 目录含 `REPRODUCE.md` |

## 3. 目录结构（关键部分）

```
combatbench/
├── envs/
│   ├── framework/          # 核心运行时（EnvRuntime、插件、RoundRunner/MatchRunner）
│   ├── batchframework/     # MJX 批量仿真（实验性）
│   └── humanoid21/         # 21-DOF 机器人环境 + 场地 XML
├── policy/                 # 策略参考实现（目录式：policy.py + blueprint + model.pt）
├── mocap/humanoid21/       # 动作捕捉重定向/切割/跟踪工具链
├── baseline/
│   ├── framework/          # 训练框架（ppo/sac/rollout、code_snapshot、debug 工具）
│   ├── experiments_ppo/    # PPO 实验注册表（exp_*.py 自动发现）
│   ├── experiments_sac/    # SAC 实验注册表
│   ├── humanoid21/         # 项目特定训练代码
│   │   ├── rewards/  plugins/  blueprints/  curriculum/(legacy)
│   │   └── hierarchy/      # ★ 分层控制大本营（本报告重点）
│   └── runs/               # 训练输出（gitignored，很大）
├── docs/  tests/  examples/  scripts/
└── CLAUDE.md  README.md  AUDIT.md  REGULARIZATION.md 等
```

## 4. hierarchy 路线：分层格斗（Code-as-Policy + 通用底层执行器）

### 4.1 核心思想（`bootstrip.md` / `COMMAND_DESIGN.md`）

上层战术代码（未来 LLM/VLM 生成或自博弈策略）只在固定低维语义指令空间发命令；
低层策略负责全部 21-DOF 物理控制。

- **观测契约**：`CommandSimulator` 把 96 维观测裁为 `obs[0:57]` 本体（去对手块）+ `obs[57:66]` 9 维指令 = **66 维**
- **9 维指令**（v2 定稿，即时语义、无动作宏）：身体通道 `(vx, vy, w, h)` + 末端通道 `(eff_type, eff_side, tx, ty, tz)`
  - `vx,vy` 目标线速度（÷1.5 m/s）、`w` 角速度（÷2π）、`h` 身高偏移（÷0.4）
  - `eff_type`：0 无 / +1 手 / −1 脚；`txyz` ego 系目标点（÷1.5 m），可随时间变化
  - 全零指令 = 原地标准动态踱步
- **"指令表达意图，不编动作序列"**：移动只给方向不给终点；打击是"快速打到点并回架"的完整动作单元，发力形态由 mocap 参考 clip 塑形
- **部署接口**：`set_motion / set_effector / clear_effector` 写同一指令契约；训练期用 `move_schedule` 批量脚本展开成逐帧指令表

### 4.2 标准动态动作（路线 B）

站立归宿不是静态 pose，而是 **5 段双脚循环踱步片段**（从已收敛抗推策略 `balance_step_s42i` 采集，`assets/standard_stepping_clips.npz`，5×40 帧）。
`StandardClipObserver` 用短窗状态转移对 `[f_{t-4}, f_t]` 对片段库取 max 相似度 —— 静止木桩拿不到分，只有顺着某一片段继续踱步才得满分。

### 4.3 训练主线（五步 → 当前执行态）

```
S1 倒地起身→标准踱步   ✅ hier_s1_s42i_scratch（66 维从零，~1h，success=1.0）
S2 12级推力抗扰→回踱步 ✅ hier_s2_balance_s42i_b（L11 封顶，recovery=0.948，u1635）
   │   ↑ 这条线是所有后续训练的初始化基座
   ▼
阶段一 11 技能复合训练  ✅ skill_train_mbs 系（u9860 收敛，quality 0.881）
   │   move×4 / turn×2 / step / jab×2 / heavy×2
   │   land_t: heavy ~4-5 步、jab ~7 步（0.2-0.4s 出拳）
   ▼
阶段二 技能衔接        ✅ skill_trans_mbs（warmup 泛化起手覆盖全部技能切换面）
   ▼
★ 当前 skill_full_mbs  进行中 —— 步界结算移动 + launch gate 打击 + warmup，
                        经 cooperative MRO 合并为一个实验
后续 PRECISION_STRIKE_PLAN（阶段三）：参数化目标点 + 结果奖励（d_min / t*），方案待审
```

### 4.4 skill_full_mbs 机制（`SKILL_TRAINING_PLAN.md` + 7 奖励通道）

| 通道 | 挂载 | 语义 |
|---|---|---|
| `r_potential` | 全部 | 平衡势能 phi，奖励截断 0.90（≈1.12m 躯干即满分，允许屈膝前倾运动站姿） |
| `r_fall` | 全部（strike 豁免 actor_weight） | 摔倒惩罚 |
| `r_move_step` | move×4 | **按步结算**：同脚两次落地界定步周期，结算径向净速度 |
| `r_drift_step` | move + idle | 同机制造罚侧漂/偏航；idle 全位移都是漂移 |
| `r_turn` | turn×2 | 稠密角速度跟踪（已收敛不动） |
| `r_strike` | strike×4 | 两段式：段 A launch gate（纯关节姿态判据，17 关节 <量程5% 过门；Lorentzian 均值 − qmax 塑形）+ 段 B 单调进度指针跟踪 + post-gate 进度定价 W_PROG·(k/k_max) |
| `r_std_clip` | idle | 标准片段匹配 |

关键设计：
- **warmup 泛化起手**：episode 先跑随机先序技能（≤35 步）再切目标技能——高层策略衔接技能的起点永远是上一技能中段；显式排除"同名打击→同名打击"（部署语义：继续打完）
- **idle 双 env 互斥**：`IDLE_FALLEN_PROB=0.5` 分倒地 env（RandomFallenStatePlugin）/ warmup env（WarmupSteppingPlugin）
- **打击动态目标**：`strike_dyn_target=true` —— 段首锚定参考手世界轨迹，逐帧重投影 ego 系覆写 txyz，目标点携带出拳相位信息
- **打击平衡豁免**：strike ep 内 r_potential/r_fall actor_weight=0——"怕摔"不能压制出拳梯度
- **多 critic**：每通道独立 critic + γ=0.99/λ=0.95（GAE 有效回授 ~16.8 步 ≈0.84s）
- **超参**：lr actor 3e-5 / critic 3e-4、1536 ep/update、96 workers、KL 0.03、`max_updates=10000`、patience 1000

### 4.5 当前运行状态（2026-10-20 上午）

- **在跑**：`train_skill_full_mbs_ppo_20261020_104315`（PID 2993307，GPU 后台），从 `20261020_095807/checkpoint_u06115.pt` 热恢复，当前 ~u6137，~11.8 s/update
- **当日续训链**：`085927`(←u5810) → `092125` → `093115` → `095807`(u6115) → `104315`
- **最新 eval（u6135）**：falls 1.6-2.6%；move v_along fwd 0.52/back 0.35/left 0.55/right 0.35 m/s；turn ~5 rad/s；step stood=1.0 漂移 ~0.15m
- **⚠️ 打击现状**：`ungated≈0`（所有拳种几乎不过 launch gate），`land_t` 顶格 24.5-25，gate_qmax ~0.50 距阈值 0.10 尚远；`gate_frac` 自适应课程 0.26-0.29 缓降。正处于"对齐段攻坚"阶段——r_strike 均值 ~1.3 主要来自对齐段 Lorentzian 塑形分

### 4.6 实验文件映射

| 文件 | 作用 |
|---|---|
| `experiments_ppo/exp_skill_full_mbs.py` | 当前实验：SkillStepMBS × SkillGateMBS（继承 SkillTransMBS）MRO 合并 |
| `exp_skill_train_mbs.py` | 阶段一基类（技能表/通道/蓝图挂接，647 行） |
| `exp_skill_step_mbs.py` | 步界结算移动奖励 |
| `exp_skill_gate_mbs.py` | 打击 launch gate + 自适应 frac 课程 |
| `exp_skill_trans_mbs.py` | warmup 技能衔接起手 |
| `exp_skill_precise_mbs.py` | 阶段三精确打击（参数化目标锥） |
| `hierarchy/command_simulator.py` | 66 维封装 + 指令链路 + 打击动态目标锚定 |
| `hierarchy/strike_track_observer.py` | 打击跟踪 observer（542 行，gate + 单调指针） |
| `hierarchy/standard_clip_observer.py` | 标准片段匹配 |
| `hierarchy/move_track_observer.py` | 运动学记录（离线步界结算用） |
| `hierarchy/blueprints/skill_*.yaml` | 四族 + idle_warm 五套环境蓝图 |
| `mocap` clips | jab=`hdm05_3146_88_2895_127(_m)` k_land=11；heavy=`cmu_144_20(_m)_seg01` k_land=6 |

## 5. 运维要点 / 踩坑记录（TRAINING_LOG.md）

1. 训练必须 `cd combatbench && PYTHONPATH=.` 启动；`--background` 自带 fork+setsid，**不可再套 nohup**
2. 运行中的 run **不可改其引用的 yaml/构造参数**——fork worker 读新配置会 TypeError 静默死（已发生 3 次）
3. `--smoke` 会把 `max_updates` 覆写成绝对值 2——`--smoke --resume-from` 在高 update 续训会秒退，表象像挂死
4. `max_updates` 到顶正常退出无提示——run 消失先查 `max_updates`/early_stop
5. 监控：`grep "\[eval" baseline/runs/<run>/train.log | tail -3`；调试工具 `baseline/framework/ppo/debug.py`
6. S2 曾发生"空心推力课程"：站立触发阈值与动态步态身高冲突导致推力几乎从未施加——指标晋级了但没训练到

## 6. 观察与待办（文档明示）

- launch gate 过门率是当前主要瓶颈（ungated≈0），自适应 frac 课程 + 逐关节 Lorentzian−qmax 塑形 + 进度定价是最新一组应对（今日提交）
- `quality` 复合分不含 dyaw/gate 等关键量，**不能作验收依据**；最终验收以渲染视频定性为准（`hierarchy/videos/`）
- P2-3/P2-4 待验证：周期净位移≠正确步态；clip 形状匹配≠真实发力（验收需补手速/伸展/身体-手相对运动指标）
- 阶段三（精确打击）方案已写待审；S5 踢腿、S6 高度控制、self-play 高层策略在更后

---
*由 project-analyzer 技能生成*
