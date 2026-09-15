# experiments_ppo

本目录存放 PPO 实验定义。`__init__.py` 自动发现本目录**顶层**的 `exp_*.py`
文件并注册（`archive/`、`todo/` 子目录与 `test_*.py` 不会被扫描）。

完整的框架机制文档见 [`../framework/ppo/GUIDE.md`](../framework/ppo/GUIDE.md)。

---

## 1. 如何启动一个实验

所有命令都从 `things/combatbench/` 目录运行，且必须设置 `PYTHONPATH=.`，
否则 `import baseline` 会失败（`--background` 模式下错误只写进日志，不可见）。

```bash
cd /data1/mono/things/combatbench

# 列出所有已注册实验及其 reward channels
PYTHONPATH=. python3 baseline/framework/train.py --list-experiments

# 冒烟测试：2 轮 update、8 episodes，快速验证能跑通
PYTHONPATH=. python3 baseline/framework/train.py --experiment minimal --smoke

# 正式训练（前台，输出到控制台 + run_dir/train.log）
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance

# 后台训练（fork + setsid，输出只进 run_dir/train.log）
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance --background
```

`--background` 会打印 run 目录、日志路径和 PID，然后退出：

```
[run] monitor: tail -f baseline/runs/<run_name>/train.log
[run] stop: kill <pid>
```

不要再用 `nohup ... &` 包一层——`--background` 自己已 detach。

```bash
# 断点续训
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance \
  --resume-from baseline/runs/<run_name>/checkpoints/checkpoint_u01000.pt
```

### 输出位置

每次训练输出到 `baseline/runs/train_<exp>_ppo_<timestamp>/`：

```
config.json      # 实验配置快照
train.log        # 完整训练日志
pid              # 后台模式 PID
checkpoints/     # 周期 checkpoint（checkpoint_uNNNNN.pt）
policy/          # best-of-run 导出的策略
videos/          # eval 视频
```

### 常用 flag

| Flag | 作用 |
|------|------|
| `--smoke` | 冒烟：max_updates=2、episodes=8，先跑它验证 |
| `--background` | 后台运行（fork + setsid） |
| `--resume-from <ckpt>` | 从 checkpoint 续训 |
| `--reset-update` | 续训时把 update 计数归零 |
| `--run-name <name>` | 自定义 run 名 |
| `--run-dir <path>` | 自定义输出目录 |
| `--seed <n>` | 覆盖实验自带 seed |
| `--set K=V` | 传实验构造函数参数，可重复 |
| `--no-snapshot` | 跳过 git 代码快照（默认开启） |
| `--no-confidence` | 关掉 EV-based confidence 加权 |

---

## 2. 如何添加一个实验

1. 在本目录新建 `exp_<name>.py`（**必须 `exp_` 前缀且在顶层**）。
2. 定义一个满足 `ExperimentPPO` 接口的类（`baseline/framework/ppo/experiment.py`）。
   两种继承方式：
   - **直接继承 `ExperimentPPO`**：实现全部抽象方法（`reward_channels`、
     `common_params`、`ppo_params`、`build_actor`、`build_critic`、
     `build_jobs`、`build_trajectories`、`on_eval`），适合完全自定义场景。
   - **继承 `CombatExperimentPPOBase`**（`from .base import ...`，本目录实验的
     惯例做法，非必须）：已提供 actor/critic 构建、self-play job 构建、
     params 等默认实现，只需覆盖 class attributes（`name`、`env_blueprint`、
     `agent_used`、超参等）并实现 `reward_channels()`、
     `build_trajectories()`、`on_eval()`。
3. 文件末尾导出类（注意是类不是实例，`name` 即注册名）：

   ```python
   EXPERIMENT_CLASS = MyExperiment
   ```

最小骨架（继承 `CombatExperimentPPOBase` 的情形）：

```python
from .base import CombatExperimentPPOBase

class MyExperiment(CombatExperimentPPOBase):
    name = "my_experiment"
    env_blueprint = "xxx_env.yaml"   # baseline/humanoid21/blueprints/ 下的文件名
    agent_used = "both"              # random / both / robot_a / robot_b

    def reward_channels(self): ...      # 声明 channel 的 name/gamma/gae_lambda
    def build_trajectories(self, episodes): ...  # episode → List[Trajectory]
    def on_eval(self, episodes, update): ...     # 返回 {"is_new_best": ..., "info": {...}}

EXPERIMENT_CLASS = MyExperiment
```

可选 hook：`state()` / `load_state()`（checkpoint 时持久化内部状态）、
`on_update()` / `exploration()`（探索调度）。详见 GUIDE.md §4–5。

### 验证

```bash
# 注册名应出现在列表里
PYTHONPATH=. python3 baseline/framework/train.py --list-experiments

# 冒烟跑通
PYTHONPATH=. python3 baseline/framework/train.py --experiment my_experiment --smoke
```

### 参考

- [`exp_minimal.py`](exp_minimal.py) —— 最小可运行示例，新实验从这里抄
- [`../framework/ppo/GUIDE.md`](../framework/ppo/GUIDE.md) —— 框架机制完整文档
- [`base.py`](base.py) —— `CombatExperimentPPOBase` 全部可覆盖的 class attributes
