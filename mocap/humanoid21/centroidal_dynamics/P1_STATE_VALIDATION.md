# P1 验证报告：质心状态提取

状态：完成  
范围：`state.py`、`test_state.py`

## 实现路径

`CentroidalStateExtractor` 提供两条独立路径。

### 手工刚体求和路径

对每个 robot subtree body：

\[
m=\sum_i m_i,\qquad
c=\frac{\sum_i m_i x^{COM}_i}{m}
\]

线动量和角动量使用 `mj_jacBodyCom(model,data)` 得到的 body COM 世界速度：

\[
v_i=J^{lin}_{COM,i}\dot q,\qquad
\omega_i=J^{rot}_{COM,i}\dot q
\]

\[
l=\sum_i m_i v_i
\]

\[
L=\sum_i \left[(x^{COM}_i-c)\times(m_i v_i)+R^{inertial}_i I_i R^{inertial,T}_i\omega_i\right]
\]

其中惯性姿态使用 `data.ximat`，而不是 `data.xmat` 或 `data.xquat`。

### MuJoCo subtree 路径

使用根 body：

- `model.body_subtreemass[root]`
- `data.subtree_com[root]`
- `data.subtree_linvel[root]`
- `data.subtree_angmom[root]`

`mj_subtreeVel()` 负责刷新 `subtree_linvel/subtree_angmom`。

`state()` 返回手工求和结果，并要求 subtree 路径在容差内一致；任一不一致都抛出 `StateNotConsistent`。

## 关键审计发现

不能直接用 `data.cvel[body, 3:6]` 作为“世界系 body COM 点速度”参与全身动量求和。受控探针显示它与 `mj_subtreeVel`/`mj_jacBodyCom` 的世界系 COM 速度不一致。当前手工路径使用 `mj_jacBodyCom`，与 MuJoCo subtree 路径在任意 qvel 下一致。

## 已验证性质

测试覆盖：

- reset 状态下 `robot_a`/`robot_b` 的手工与 subtree 状态一致；
- 任意 root + joint 速度下两条路径一致；
- `robot_a` qvel 改变不影响 `robot_b` 状态；
- root 平移使 COM 同步平移，动量不变；
- 绕 root frame origin 的 yaw 旋转使 COM 按对应刚体旋转变化；
- 全身 COM 不等于 root `xpos`，也不等于 root `xipos`；
- 未知 `robot_id` 显式报错。

当前结果：

```text
13 passed
```

## 仍未保证的内容

- 该状态是完整子树的质心状态，不保证对应的身体姿态处于可恢复域；
- `set_core_state()` 后的静态状态不代表连续 rollout 的接触求解历史；
- 本阶段不判断 `CentroidalState` 是否物理可行，只保证它被正确提取。
