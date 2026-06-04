# 课程学习框架设计 (Curriculum Learning Framework)

## 目标
- 统一训练入口脚本
- 每个实验独立 Python 文件（奖励处理 + 课程配置）
- 支持跨实验导入 Checkpoint（Dict 关键字匹配）
- 不破坏现有 `train_curriculum*.py`，用新文件实现

---

## 1. 整体架构

```
curriculum/
├── framework/                      # 核心框架（通用，不随实验变化）
│   ├── config.py                   # ExperimentConfig ABC
│   ├── ppo_trainer.py              # PPOBuffer, PPO update
│   ├── checkpoint.py               # CheckpointManager（跨实验加载）
│   ├── registry.py                 # 实验注册中心
│   └── train_loop.py               # 通用训练循环
│
├── experiments/                    # 实验定义（每个实验一个文件）
│   ├── __init__.py                 # 实验注册表
│   ├── exp_v1_relation.py          # V1: 4-reward (r_relation)
│   ├── exp_v2_follow.py            # V2: 6-reward (r_hold/r_radial/r_tangential)
│   ├── exp_v3_combat_only.py       # V3: 纯战斗（从其他实验导入checkpoint）
│   └── ...                         # 后续实验
│
├── train_unified.py                # 统一入口脚本（命令行驱动）
└── DESIGN.md                       # 本设计文档
```

---

## 2. 核心接口

### 2.1 ExperimentConfig (已存在，微调)

位置：`framework/config.py`

```python
class ExperimentConfig(ABC):
    """实验配置抽象基类。"""
    
    # 基础配置
    name: str = ""                          # 实验标识
    version: str = "1.0"                  # 版本（用于checkpoint兼容性）
    reward_keys: Tuple[str, ...] = ()       # 奖励分量名称
    gammas: Dict[str, float] = {}           # 各奖励的 discount
    env_blueprint: str = ""                 # 环境蓝图文件名
    
    # --- 新增：跨实验迁移配置 ---
    # 描述本实验的state_dict与checkpoint中key的映射关系
    # 用于从不同实验的checkpoint导入权重
    checkpoint_key_mapping: Dict[str, str] = field(default_factory=dict)
    
    @abstractmethod
    def initial_weights(self) -> Tuple[float, ...]:
        """返回初始 stage 权重。"""
        ...
    
    @abstractmethod
    def next_weights(self, eval_metrics: Dict[str, float], 
                     current_weights: Tuple[float, ...]) -> Tuple[float, ...]:
        """根据评估指标返回下一 stage 权重。"""
        ...
    
    @abstractmethod
    def extract_rewards(self, observer_outputs: dict, T: int,
                        termination_proposals: Tuple[str, ...]) -> Dict[str, np.ndarray]:
        """从 observer 输出提取各奖励分量的 per-step 值。"""
        ...
    
    @abstractmethod
    def compute_episode_metrics(self, observer_outputs: dict, T: int,
                                termination_proposals: Tuple[str, ...]) -> Dict[str, float]:
        """计算 episode 级别的评估指标（用于课程判断）。"""
        ...
    
    # --- 新增：模型结构描述（用于跨实验加载） ---
    def model_structure_signature(self) -> Dict[str, Any]:
        """
        返回模型结构签名，用于验证 checkpoint 兼容性。
        例如：{"obs_dim": 96, "action_dim": 21, "actor_hidden": 256}
        跨实验加载时，只要结构兼容，就允许导入。
        """
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor_hidden_dim": self.actor_hidden_dim,
            "critic_hidden_dim": self.critic_hidden_dim,
        }
    
    # --- 新增：状态迁移适配 ---
    def adapt_checkpoint_state(self, state_dict: Dict[str, Any], 
                                 source_experiment: str) -> Dict[str, Any]:
        """
        适配来自其他实验的 checkpoint state_dict。
        子类可覆盖此方法处理特殊的权重迁移逻辑。
        
        默认实现使用 checkpoint_key_mapping 进行键名转换。
        """
        adapted = {}
        for target_key, source_key in self.checkpoint_key_mapping.items():
            if source_key in state_dict:
                adapted[target_key] = state_dict[source_key]
        # 未映射的键保持原样（如果键名相同）
        for k, v in state_dict.items():
            if k not in adapted:
                adapted[k] = v
        return adapted
```

### 2.2 实验注册中心 (新增)

位置：`experiments/__init__.py`

```python
# experiments/__init__.py
from typing import Dict
from baseline.humanoid21.curriculum.framework.config import ExperimentConfig

# 实验注册表
_EXPERIMENT_REGISTRY: Dict[str, ExperimentConfig] = {}

def register_experiment(config: ExperimentConfig) -> None:
    """注册实验配置。"""
    _EXPERIMENT_REGISTRY[config.name] = config

def get_experiment(name: str) -> ExperimentConfig:
    """获取实验配置。"""
    if name not in _EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown experiment: {name}. "
                        f"Available: {list(_EXPERIMENT_REGISTRY.keys())}")
    return _EXPERIMENT_REGISTRY[name]

def list_experiments() -> Dict[str, ExperimentConfig]:
    """列出所有注册的实验。"""
    return _EXPERIMENT_REGISTRY.copy()

# 自动导入并注册所有实验
from . import exp_v1_relation
from . import exp_v2_follow
# ... 其他实验
```

### 2.3 CheckpointManager (新增)

位置：`framework/checkpoint.py`

```python
@dataclass
class CheckpointMetadata:
    """Checkpoint 元数据。"""
    experiment_name: str
    experiment_version: str
    update_step: int
    reward_keys: Tuple[str, ...]
    model_structure: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    created_at: str

class CheckpointManager:
    """跨实验兼容的 Checkpoint 管理器。"""
    
    def __init__(self, experiment: ExperimentConfig, device: torch.device):
        self.experiment = experiment
        self.device = device
    
    def save(self, path: Path, actor, critics, optimizer_state, 
             scheduler_state, step: int) -> None:
        """保存 checkpoint（包含完整元数据）。"""
        meta = CheckpointMetadata(
            experiment_name=self.experiment.name,
            experiment_version=self.experiment.version,
            update_step=step,
            reward_keys=self.experiment.reward_keys,
            model_structure=self.experiment.model_structure_signature(),
            scheduler_state=scheduler_state,
            created_at=datetime.now().isoformat(),
        )
        checkpoint = {
            "metadata": meta,
            "actor_state": actor.state_dict(),
            "critics_state": {k: v.state_dict() for k, v in critics.items()},
            "optimizer_state": optimizer_state,
        }
        torch.save(checkpoint, path)
    
    def load(self, path: Path, actor, critics, strict: bool = True) -> Dict[str, Any]:
        """
        加载 checkpoint，支持跨实验导入。
        
        流程：
        1. 读取 checkpoint
        2. 检查元数据兼容性
        3. 如果是不同实验，使用 experiment.adapt_checkpoint_state() 适配
        4. 加载模型权重
        5. 返回恢复的 scheduler_state 和 step
        """
        checkpoint = torch.load(path, map_location=self.device)
        meta = checkpoint.get("metadata")
        
        if meta is None:
            # 旧格式 checkpoint（无元数据），尝试直接加载
            return self._load_legacy(checkpoint, actor, critics)
        
        # 验证模型结构兼容性
        current_structure = self.experiment.model_structure_signature()
        if meta.model_structure != current_structure:
            print(f"[WARN] Model structure mismatch:")
            print(f"  Checkpoint: {meta.model_structure}")
            print(f"  Current:    {current_structure}")
            # 结构不兼容时仍可尝试加载（strict=False）
        
        # 如果是不同实验，进行适配
        if meta.experiment_name != self.experiment.name:
            print(f"[INFO] Cross-experiment load: {meta.experiment_name} -> {self.experiment.name}")
            actor_state = self.experiment.adapt_checkpoint_state(
                checkpoint["actor_state"], meta.experiment_name
            )
            critics_state = {}
            for k, v in checkpoint["critics_state"].items():
                # 根据 reward_keys 映射 critic 名称
                mapped_key = self._map_critic_key(k, meta.reward_keys, self.experiment.reward_keys)
                if mapped_key:
                    critics_state[mapped_key] = self.experiment.adapt_checkpoint_state(
                        v, meta.experiment_name
                    )
        else:
            actor_state = checkpoint["actor_state"]
            critics_state = checkpoint["critics_state"]
        
        # 加载权重
        actor.load_state_dict(actor_state, strict=strict)
        for key, state in critics_state.items():
            if key in critics:
                critics[key].load_state_dict(state, strict=strict)
        
        return {
            "step": meta.update_step,
            "scheduler_state": meta.scheduler_state,
            "source_experiment": meta.experiment_name,
        }
    
    def _map_critic_key(self, key: str, source_keys: Tuple[str, ...], 
                        target_keys: Tuple[str, ...]) -> Optional[str]:
        """
        映射 critic 名称。例如：
        - source: (r_fall, r_cross, r_relation)
        - target: (r_fall, r_cross, r_damage, r_hold, r_radial)
        - r_fall -> r_fall (直接匹配)
        - r_relation 在 target 中不存在，返回 None（跳过）
        """
        if key in target_keys:
            return key
        return None
```

---

## 3. 统一训练入口

位置：`train_unified.py`

```python
#!/usr/bin/env python3
"""统一课程学习训练入口。

用法示例：
    # 从头训练 V2 实验
    python train_unified.py --experiment v2_follow
    
    # 从同实验 checkpoint 恢复
    python train_unified.py --experiment v2_follow --resume runs/v2_follow_xxx/checkpoint_u01000.pt
    
    # 跨实验导入：从 V1 导入到 V2（actor 权重迁移，critic 重新初始化）
    python train_unified.py --experiment v2_follow --import-from runs/v1_relation_xxx/checkpoint_u01000.pt
    
    # 列出所有可用实验
    python train_unified.py --list-experiments
"""

import argparse
import sys
from pathlib import Path

from baseline.humanoid21.curriculum.framework.registry import list_experiments, get_experiment
from baseline.humanoid21.curriculum.framework.train_loop import TrainingLoop


def main():
    parser = argparse.ArgumentParser(description="Curriculum Learning Training")
    parser.add_argument("--experiment", "-e", type=str, help="Experiment name")
    parser.add_argument("--list-experiments", action="store_true", help="List available experiments")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--import-from", type=str, dest="import_from", help="Import actor from other experiment checkpoint")
    parser.add_argument("--run-dir", type=str, help="Custom run directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    
    args = parser.parse_args()
    
    if args.list_experiments:
        experiments = list_experiments()
        print("Available experiments:")
        for name, exp in experiments.items():
            print(f"  {name:20s} - reward_keys: {exp.reward_keys}")
        return
    
    if not args.experiment:
        parser.error("--experiment is required (or use --list-experiments)")
    
    # 获取实验配置
    experiment = get_experiment(args.experiment)
    
    # 初始化训练循环
    loop = TrainingLoop(
        experiment=experiment,
        device=args.device,
        run_dir=args.run_dir,
    )
    
    # 处理 checkpoint 导入
    if args.resume:
        loop.resume_from(args.resume)
    elif args.import_from:
        loop.import_from(args.import_from)  # 跨实验导入
    
    # 开始训练
    loop.run()


if __name__ == "__main__":
    main()
```

---

## 4. 实验文件示例

### 4.1 V1: r_relation 方案 (exp_v1_relation.py)

```python
from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from baseline.humanoid21.curriculum.framework.registry import register_experiment


class V1RelationConfig(ExperimentConfig):
    """V1: 4-reward 课程 (r_fall, r_cross, r_damage, r_relation)。"""
    
    name = "v1_relation"
    version = "1.0"
    reward_keys = ("r_fall", "r_cross", "r_damage", "r_relation")
    gammas = {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_damage": 0.80,
        "r_relation": 0.93,
    }
    env_blueprint = "curriculum_env_v1.yaml"
    
    # 跨实验迁移：如果要导入到 V2，actor 权重可以直接复用
    # V2 的 mapping 中需要指定如何映射 r_relation critic
    
    def initial_weights(self):
        return (3.0, 1.0, 0.0, 0.0)  # Stage 1: balance
    
    def next_weights(self, metrics, current):
        len_ratio = metrics.get("mean_length", 0.0) / 200.0
        if len_ratio < 0.98:
            return (3.0, 1.0, 0.0, 0.0)  # Stage 1
        elif metrics.get("in_zone", 0.0) < 0.5:
            return (2.0, 1.0, 0.0, 1.0)  # Stage 2: approach
        else:
            return (2.0, 1.0, 1.0, 1.0)  # Stage 3: combat
    
    def extract_rewards(self, observer_outputs, T, termination_proposals):
        # ... 具体提取逻辑
        pass
    
    def compute_episode_metrics(self, observer_outputs, T, termination_proposals):
        # ... 计算指标
        pass


register_experiment(V1RelationConfig())
```

### 4.2 V2: r_hold/r_radial 方案 (exp_v2_follow.py)

```python
from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from baseline.humanoid21.curriculum.framework.registry import register_experiment


class V2FollowConfig(ExperimentConfig):
    """V2: 6-reward 课程 (几何方向系数构造的 r_radial)。"""
    
    name = "v2_follow"
    version = "2.0"
    reward_keys = ("r_fall", "r_cross", "r_damage", "r_hold", "r_radial", "r_tangential")
    gammas = {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_damage": 0.80,
        "r_hold": 0.98,
        "r_radial": 0.97,
        "r_tangential": 0.95,  # 已废弃，保持兼容性
    }
    env_blueprint = "curriculum_env_v2.yaml"
    
    # --- 跨实验迁移配置 ---
    # 从 V1 导入时的键映射：V1 的 critic 权重可以直接加载到 V2 的同名 critic
    # r_relation 没有对应项，将被跳过
    checkpoint_key_mapping = {
        # V2 target key -> V1 source key（相同则省略）
    }
    
    def initial_weights(self):
        return (3.0, 1.0, 0.0, 0.0, 0.0, 0.0)  # Stage 1
    
    def next_weights(self, metrics, current):
        len_ratio = metrics.get("mean_length", 0.0) / 200.0
        if len_ratio < 0.98:
            return (3.0, 1.0, 0.0, 0.0, 0.0, 0.0)  # Stage 1: balance
        elif metrics.get("in_zone", 0.0) < 0.5:
            return (2.0, 1.0, 0.0, 0.5, 0.5, 0.0)  # Stage 2: hold + radial
        else:
            return (2.0, 1.0, 1.0, 0.5, 0.5, 0.0)  # Stage 3: combat
    
    def extract_rewards(self, observer_outputs, T, termination_proposals):
        # ... 使用 compute_approach_rewards 进行 trainer-side 后处理
        pass


register_experiment(V2FollowConfig())
```

---

## 5. 跨实验 Checkpoint 导入机制

### 5.1 场景示例

用户先训练了 V1 实验（4-reward），现在想尝试 V2（6-reward）：
- **Actor**: V1 的 actor 可以直接迁移到 V2（结构相同，都是 96->256->21 的策略网络）
- **Critics**: 
  - `r_fall`, `r_cross`, `r_damage` 的 critic 可以直接复用
  - `r_relation` 的 critic 没有对应项（V2 使用 r_hold/r_radial 替代），应跳过
  - `r_hold`, `r_radial` 的 critic 需要重新初始化（或从相近的 critic 热启动）

### 5.2 导入流程

```bash
python train_unified.py \
    --experiment v2_follow \
    --import-from runs/v1_relation_20250601/checkpoint_u01000.pt
```

流程：
1. 创建 V2 实验的模型（actor + 6 critics）
2. 加载 V1 checkpoint，解析元数据
3. Actor 权重直接导入（结构相同）
4. Critics 按名称匹配导入：
   - `r_fall` -> `r_fall` ✓
   - `r_cross` -> `r_cross` ✓
   - `r_damage` -> `r_damage` ✓
   - `r_relation` -> 无匹配，跳过
5. 未匹配的 critics (`r_hold`, `r_radial`) 保持初始化状态或从 `r_damage` 热启动
6. 开始 V2 训练（stage 1 或根据 V1 的 scheduler 状态决定）

---

## 6. 实现计划

### Phase 1: 框架基础设施（不破坏现有代码）
1. 创建 `framework/checkpoint.py` - CheckpointManager
2. 创建 `framework/registry.py` - 实验注册中心
3. 创建 `framework/train_loop.py` - 通用训练循环（从 V2 抽象）
4. 修改 `framework/config.py` - 补充跨实验迁移接口

### Phase 2: 统一入口
1. 创建 `train_unified.py` - 命令行入口
2. 更新 `experiments/__init__.py` - 自动注册

### Phase 3: 迁移现有实验
1. 迁移 `train_curriculum.py` -> `exp_v1_relation.py`
2. 迁移 `train_curriculum_v2.py` -> `exp_v2_follow.py`
3. 验证跨实验导入功能

### Phase 4: 新实验开发
1. 创建新实验只需：新增 `exp_xxx.py` + 注册
2. 复用已有 checkpoint 只需：`--import-from`

---

## 7. 与现有代码的关系

```
现有代码                新框架
-----------------      -----------------
train_curriculum.py  ->  exp_v1_relation.py  (实验定义)
train_curriculum_v2 ->  exp_v2_follow.py    (实验定义)
common.py            ->  framework/config.py (基类)
common_v2.py         ->  framework/config.py (合并)
                       framework/checkpoint.py (新增)
                       framework/train_loop.py (新增)
                       train_unified.py (统一入口)
```

**不破坏现有代码**：现有 `train_curriculum*.py` 保持可用，仅将公共逻辑下沉到 `framework/`。

---

## 8. 关键设计决策

| 决策 | 理由 |
|------|------|
| Dict 关键字匹配 | state_dict 的 key 天然携带语义（如 `actor.layers.0.weight`），无需额外映射配置 |
| 实验注册表 | 避免硬编码实验列表，新增实验只需创建文件并调用 `register_experiment` |
| Checkpoint 元数据 | 保存完整上下文（实验名、版本、reward_keys、scheduler_state），使跨实验加载成为可能 |
| Critic 按名称匹配 | 简单可靠：同名复用，无名跳过，无需复杂启发式 |

