# BLUEPRINTS — 蓝图 YAML Schema 契约

> 类型：契约

蓝图是环境的**可序列化身份**：simulator 类 + 构造配置、插件列表、
observer 映射、runtime 旋钮。加载器在 `blueprint.py` /
`parameterized_blueprint.py` / `policy.py`（policy 侧）。

**身份边界**：蓝图只含环境本体——recorder、video、
`BLUEPRINT_EXCLUDE=True` 的调试插件**不入蓝图**（`to_blueprint` 静默滤除，
包括自动挂载的 `TimeoutPlugin`——它由 `max_steps` 重新推导）。

## 1. EnvBlueprint schema

```yaml
version: 1                        # 必填；仅接受 BLUEPRINT_VERSION=1
runtime:                          # 可选节
  phy_steps_per_action: 25        # 每动作步物理子步数，默认 1
  max_steps: 600                  # 最大动作步；null/省略 = 不自动加 TimeoutPlugin
  strict: true                    # 插件钩子异常策略，默认 true
simulator:                        # 必填节
  cls: "pkg.module:QualName"      # 必填；兼容点号 "pkg.module.QualName"
  config: {initial_distance: 2.0} # 可选，默认 {}
plugins:                          # 可选，有序列表（按 priority 执行）
  - cls: "envs.humanoid21.plugins:CombatScoringPlugin"
    config: {}
observer_plugins:                 # 可选，名→spec 映射（名 = get_observer_output 键）
  robot_a_reward:
    cls: "envs.humanoid21.observer_plugins:CombatScoringObserver"
    config: {}
```

**组件协议**：每个 `cls` 类可选实现 `to_blueprint() -> dict` /
`from_blueprint(config)`；缺省 `to_blueprint` 返回 `{}`，
`from_blueprint` 即 `cls(**config)`。

## 2. PolicyBlueprint schema

```yaml
version: 1                        # 必填；POLICY_BLUEPRINT_VERSION
cls: "policy.random.policy:RandomCombatPolicy"
config:                           # 可选，转 policy.__init__ 的 kwargs
  scale: 0.1
  action_dim: 21
```

- `cls` 两种形态：`pkg.module:Class`（包内类）或
  **`file:${DIR}/policy.py:Class`**（自包含导出策略——训练导出物用此形式，
  文件路径相对于蓝图所在目录解析）。
- `build(**overrides)`：`config` 与 overrides 合并（overrides 赢）后传
  `__init__`；结果为 duck-type 校验——只要有 `act` 方法（P0-6：自包含
  导出不 import 框架 Policy ABC 也能加载）。

## 3. ParameterizedEnvBlueprint

与 EnvBlueprint 同构 + 顶层 `parameters` 节 + 任意值位允许 `${name}` 占位：

```yaml
version: 1
parameters:
  initial_distance:
    default: 2.0                          # 有默认 → materialize 可不传
    description: "Spawn distance (m)."    # 可选文档字段
  max_steps: {}                           # 空 mapping = 必填，无默认
runtime:
  max_steps: "${max_steps}"
simulator:
  config:
    initial_distance: "${initial_distance}"
```

**替换规则**：
- `"${name}"` **独占字符串** → 替换为参数**原值**，保留类型（传 int/list/bool）。
- 字符串内**内嵌** `${name}`（如 `"path/${run_id}/x"`）→ 字符串化拼接。
- 递归进 dict/list；引用未声明参数 `materialize` 时 `KeyError`；
  传未声明 override 同样 raise。
- `materialize(**overrides)` 产出一个普通 `EnvBlueprint`。

## 4. `${DIR}` 占位符

`load()` 时把蓝图文件**父目录的绝对路径**替换进原文所有 `${DIR}`——
蓝图引用同目录资产（state bank、model.pt、嵌套蓝图）不依赖部署路径：

```yaml
config:
  state_bank_path: "${DIR}/state_pool.npz"
```

EnvBlueprint / ParameterizedEnvBlueprint / PolicyBlueprint 三个 load 一致生效。

## 5. 实例参考

| 用途 | 实例 |
|---|---|
| Env 蓝图（参数化、完整插件栈） | `baseline/humanoid21/blueprints/*_env.yaml` |
| Policy 蓝图（包内类） | `policy/blueprints/random.yaml` |
| Policy 蓝图（自包含导出） | 训练产出的 `runs/*/policy_exports/` |
| Round/Match 直接构建 | `envs/framework/tests/` 内蓝图用法 |
