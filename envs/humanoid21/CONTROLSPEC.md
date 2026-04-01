# Humanoid21 控制规范 (Control Specification)

## 1. 控制模式：归一化位置控制 (Normalized Position Control)
策略层与底层物理引擎之间，通过**归一化关节目标位置**进行通信。所有关节的指令统一被压缩并映射到 `[-1, 1]` 的无量纲区间内。

### 1.1 动作接口 (Action Interface)
- **输入**: `action`，形状为 `(21,)` 的 numpy 数组。
- **取值范围**: `[-1, 1]`
- **物理含义**:
  - `Action =  0.0` → 关节期望到达其中位 (Reference)。
  - `Action =  1.0` → 关节期望到达其正向物理限位上限 (Up)。
  - `Action = -1.0` → 关节期望到达其负向物理限位下限 (Down)。

---

## 2. 映射与计算逻辑 (Mapping & Calculation)

### 2.1 静态参数准备
每个受控关节在初始化时，必须从 MuJoCo 模型中提取以下参数：
- `Down` = `jnt_range[i][0]` (下限, rad)
- `Up` = `jnt_range[i][1]` (上限, rad)
- `Reference` = `(Down + Up) / 2.0` (中位, rad)
- `Scale` = `(Up - Down) / 2.0` (半量程缩放因子, rad)

### 2.2 PD 力矩计算流程
底层环境在接收到策略层的 `action` 后，按以下步骤严格执行：

1. **反归一化 (Un-normalization)**
   ```python
   Target_rad = action * Scale + Reference
   ```
2. **PD 控制律 (PD Control Law)**
   ```python
   Torque = KP * (Target_rad - qpos) - KD * qvel
   ```
3. **输出限幅 (Clamping)**
   ```python
   Ctrl = clip(Torque / Gear, ctrl_range_min, ctrl_range_max)
   ```
4. **底层写入 (Actuation)**
   将 `Ctrl` 数组写入 MuJoCo 内部对应的 `data.ctrl` 内存片。

---

## 3. KP 与 KD：固化的内在属性 (Fixed Intrinsic Properties)

**核心原则：`KP` (比例增益) 和 `KD` (微分增益) 是机器人底层系统的物理属性，绝不允许在环境初始化或运行时作为可配参数暴露。**

- **参数形态**：`KP` 和 `KD` 应为形状等于 `(21,)` 的固定数组。不同部位（如腿部承重关节与手臂轻量关节）必须拥有不同的刚度设定。
- **获取方式**：这两套参数是在环境开发期，依据 `ACCEPTANCE_CRITERIA.md` 中的量化指标，经过严格测试和“摸索”后硬编码到系统内部的。
- **设计初衷**：确保策略层面对的是一个物理响应固定、反馈行为一致的本体系统，从而保证强化学习过程的收敛性与稳定性。
