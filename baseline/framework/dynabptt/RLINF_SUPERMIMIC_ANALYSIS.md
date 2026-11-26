# RLinf SuperMimic 集成解读报告

来源：[RLinf 文档站](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/supermimic.html)
+ [算法参考页](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/reference/algorithms/supermimic.html)
（2026-11 更新）。RLinf 明确致谢 CuboiLeo 与 zita-ch——即我们已 clone
的两个社区复现（`/tmp/SuperMimic-Reproduced`、`/tmp/supermimic_vibed`），
本集成是在它们之上的工程化版本（FSDP2 多卡 + CUDA Graph + bf16）。

---

## 一、"一次 Update"是什么

Update = **一次全局同步的三阶段训练循环**（多卡间同步执行，每卡
env/rollout/actor 三个 worker，梯度 all-reduce）：

```
阶段1 采集（真实物理）
  4096 个 MuJoCo-Warp 并行 env，policy 叠加探索噪声 ε~N(0,0.1²) 执行
  动作，实际执行的带噪动作写入各 rank 本地 replay buffer。
  回合结束 → 按回报重置采样器优先从回报低的动作片段区间重启
  （0.2s 分箱，50%探索/40%低回报聚焦/10%其他 的混合分布）。
  每 update 约推进 2.5 个控制步/env（6000 update ≈ 15000 总控制步）。

阶段2 拟合世界模型
  从 replay 状态出发，按记录动作自回归展开 k=8 步；
  每 update 训 10 批 × 全局 2048 窗口（4卡时每卡 512 窗口）；
  Bryson 通道加权 MSE（通道组 d_c·σ_c⁻² 归一，如 root_pos σ=0.01m、
  joint_vel σ=3.14 rad/s）。

阶段3 优化 policy + critic
  从 replay 采 4096 个起点（合法 k 步窗口内不跨回合），在**冻结**
  世界模型中想象 H=32 步：
    - 每步用可微正运动学算 tracking 奖励 r_t
    - 平滑终止概率 d̂_t 累加存活权重 c_{t+1}=c_t(1-d̂_t)
    - J = Σ γᵗc_t r_t + γᴴc_H V(s_H)，γ=0.99
    - 一阶梯度穿过 r_t、d̂_t、V 直接反传给 actor
  同批想象轨迹上算 GAE 目标（γ=0.99, λ=0.95）拟合 critic：
  5 epoch × 4 minibatch，价值裁剪 ≤ r_max/(1-γ)=10。
```

**时长**：单卡 A800 0.383s/update（官方单卡 0.515s，1.34×）；
4 卡 0.227s（2.27×）→ **6000 update ≈ 23 分钟**。

**数据量级（修正版理解）**：每 update 每 env 仅推进 ~2.5 控制步
→ 4096 env ≈ **10k 新转移/update**（与我们量级相当！）。真正的
规模差异在 buffer 深度：`replay_length=1000`×4096 env = **4M
转移滑动窗口**，覆盖 ~400 个 update 的策略历史；每 update 从其中
采 10×2048 窗口 + 4096 想象起点，一条转移生命周期内平均被抽
16 次（`replay_ratio=16`）。即"每步新数据"不比我们大，但**拟合
分布的历史多样性差 1-2 个数量级**。

## 二、任务是什么

**Unitree G1（29 DOF）全身动作跟踪**，40 条 LAFAN1 片段（2.45h，
Ubisoft retarget 数据）。控制频率 **50Hz**（Δt=0.02s，4 物理子步）。

| 要素 | 内容 |
|---|---|
| 观测 | 当前状态编码（GenCoord 68d / MaxCoord 214d）+ **10 帧未来参考窗口**（每 5 控制步取 1 帧 = 0.5s 前瞻），含关节角/速度与相对当前根的 6d 根旋转 |
| 动作 | 29 维 PD 目标关节角；训练加 σ=0.1 高斯噪声，评测确定性 |
| 奖励 | r_t = Δt(Σ₆wᵢexp(-eᵢ/σᵢ²) − 0.1‖a_t−a_{t−1}‖² − 2·软限位惩罚)；6 跟踪核=根位置/朝向/部位位置/朝向/线速度/角速度；Σw=5，r_max=0.1/步 |
| 终止（仿真） | 硬规则：\|z_pelvis−z_ref\|>0.25m 或 \|g_z−g_z,ref\|>0.8 |
| 终止（想象） | 同一误差规则的**平滑版** sigmoid((max(归一化误差)−1)/0.05)——全程解析，无学习式终止头 |
| 域随机化 | 33 维物理上下文 ξ：足底摩擦(1)+躯干质心偏移(3)+29 关节编码器偏置——喂给世界模型与 critic |

**结果**：归一化回报 **0.856**（论文 0.82），40 片段全零摔倒。

## 三、训练的模型是什么

三个网络，统一 `Linear→LayerNorm→GELU` 模块堆叠：

| 网络 | 输入 | 结构 | 输出 |
|---|---|---|---|
| **Dynamics（世界模型）** | 68d 编码状态 + 29d PD目标 + 33d ξ | `[2048,2048,2048]` | **仅 35d 速度增量** [Δv_b,Δw_b,Δq̇]；位置/朝向按子步半隐式权重解析积分 `p_{t+1}=p_t+Δt(v_t+(D+1)/2D·Δv)` |
| **MotionPolicy（actor）** | 目标编码器 640d→`[2048,1024,512,256]`→32d tanh 潜变量；解码器 (32+68d)→`[2048,1024,512,256]` | encoder-decoder | 29d PD 目标 |
| **Critic** | 68d+670d目标+33d ξ | `[256,128]` | V(s)，想象轨迹 GAE 拟合 |

两种状态表示可切换：
- **GenCoord**（默认）：存储 71d sim 状态，编码 68d `[z, g_b, v_b, w_b, q, qd]`，部位由**可微正运动学**恢复
- **MaxCoord**：存储 182d 14 部位世界位姿/速度，编码 214d，84d `[dv,dw]` 增量，链式 IK 恢复

**优化器**：隐藏层矩阵用 **Muon**（momentum 0.95 Nesterov，wd 0.1），
首尾层/归一化/偏置用 fused Adam；lr 恒定 2e-4；全局范数裁剪 1.0 +
**actor 逐元素裁剪 0.1**（非有限梯度直接终止训练）。

## 四、工程加速（RLinf 的增量贡献）

| 技术 | 做法 |
|---|---|
| **FSDP2** | Dynamics/Policy/Critic 三个独立 FSDP 单元；默认 `no_shard`（模型仅 ~64MB）+ all-reduce；逐阶段 unshard 仅当前训练的网络 |
| **CUDA Graph** | 前 3 update 常规执行，第 4 次把**三阶段录成单张图**持续回放（官方是 2 张图） |
| **bf16** | FSDP2 mixed_precision 只把权重转 bf16 做矩阵乘；主权重/优化器/运动学/奖励/损失全 fp32 |

## 五、对我们的增量信息（vs 已读官方源码）

1. **重置采样器精确化**：0.2s 分箱 + 50/40/10 混合（探索/低回报/
   其他）——"按回报重采样"的官方细节首次量化
2. **参考目标窗口量化**：10 帧 × 间隔 5 控制步（0.5s 前瞻）；actor
   目标 640d（不含根位置，防泄漏绝对位置），WM/critic 目标 670d
   （含相对根位置）——**goal 输入有明确的防泄漏设计**
3. **奖励公式完整**：6 核权重/σ + 动作平滑惩罚 −0.1‖Δa‖² + 软限位
   −2.0·δ；**单步上限 0.1，正奖励**（再次印证我们 balance 任务
   负奖励反转问题是场景特有）
4. **终止全解析**：想象中的软终止与仿真硬终止**共享同一误差规则**
   ——done 从不学习，从不是数据依赖
5. **优化器组合**：Muon+Adam 混合（我们纯 Adam）；actor 逐元素
   0.1 裁剪（我们只有全局范数）
6. **量级确认**：每 update 新增 ~10k 转移（与我们同量级），但
   滑动 buffer 4M 覆盖 ~400 update 历史——**差距在采样池深度
   （几十倍）而非采集速度**；4096 想象条/update，23 分钟收敛
7. **已知限制**（RLinf 自认）：未实现速度跟踪；推力恢复评测姿态
   与论文不同；多节点未验证

## 六、立即可用的资源

- RLinf repo：`github.com/RLinf/RLinf`，`examples/embodiment/` 下
  `supermimic_lafan1_gencoord/maxcoord` 配置可直接跑（需 HF 下
  LAFAN1 G1 重定向数据，CC BY-NC-ND 许可）
- 单机 4×A800 23 分钟出基准——**我们机器有 8×4090，可复现此
  基准做"参照系"**：先在我们的 BPTT 框架上把 tracking 任务跑通
  （该任务已被 4 家独立实现验证可行），再回到 balance 边界问题
