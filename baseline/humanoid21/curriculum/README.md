Train:
cd /data1/mono/things/combatbench
PYTHONPATH=. nohup python3 baseline/humanoid21/curriculum/train_curriculum.py &> train.log & 


/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_20260602_012824/checkpoints/checkpoint_u02155.pt


Resume:
cd /data1/mono/things/combatbench
PYTHONPATH=. nohup python3 -u baseline/humanoid21/curriculum/train_curriculum.py \
    --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_20260531_172059/checkpoints/checkpoint_u03895.pt &> train_resume10.log & 


PYTHONPATH=. nohup python3 -u baseline/humanoid21/curriculum/train_curriculum_v2.py \
    --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_20260602_161330/checkpoints/checkpoint_u02925.pt &> train_v2_resume2.log & 


Watch Video:
python3 -m http.server 8999 --bind 0.0.0.0 --directory /data1/mono/things/combatbench/baseline/humanoid21/runs


Gen Video:
cd /data1/mono/things/combatbench
python3 -m envs.framework.round_runner \
  --env-blueprint envs/humanoid21/blueprint.yaml \
  --policy-a-blueprint baseline/humanoid21/runs/curriculum_20260528_175538/policy_exports/u00539/policy_blueprint.yaml \
  --policy-b-blueprint baseline/humanoid21/runs/curriculum_20260528_175538/policy_exports/u00539/policy_blueprint.yaml \
  --video out1.mp4

每隔一段时间，按照Rollout一样的配置，生成一个视频。



DEBUG:
cd /data1/mono/things/combatbench
python3 -m envs.framework.round_runner \
  --env-blueprint baseline/humanoid21/blueprints/curriculum_env.yaml \
  --policy-a-blueprint baseline/humanoid21/runs/curriculum_20260529_112738/policy_exports/u01925/policy_blueprint.yaml \
  --policy-b-blueprint baseline/humanoid21/runs/curriculum_20260529_112738/policy_exports/u01925/policy_blueprint.yaml \
  --recorder envs.framework.recorder:BaseFrameRecorder?output_dir=_debug/run03

python3 -m envs.framework.recorder_viewer --no-browser _debug/run03


things/combatbench/baseline/humanoid21/runs/curriculum_20260527_175448/policy_exports/u00837/policy_blueprint.yaml
总体的处理流程：
Rollout得到原始数据。

进行奖励计算。 计算4种奖励。 使用原始奖励。




我现在得到了一个结论，想把机器人策略训练好很难一蹴而就，当前的工程工具在效率上不足以支持这个目录。
我想做一个支持PPO算法训练的课程学习框架，最终实现如下的效果：
1， 统一的训练入口脚本
2， 对于每个实验，有一个Python文件，里面有奖励处理和课程配置。
3， 支持跨实验导入Checkpoint，这个是必须的。通过Dict关键字匹配。
实现要求，不破坏现有的训练代码。用新文件实现。
things/combatbench/baseline/humanoid21/curriculum/train_curriculum_v2.py
things/combatbench/baseline/humanoid21/curriculum/train_curriculum.py
以上面这两个文件为抽象目标， 最终就是要取代这两个文件。以及 commom.py和commom_v2.py




我给你 一份绝对完整、可直接丢给编程Agent一次性实现的终极工程指令。
包含：完整架构定义、全部变量、全部训练阶段、全部规则、全部边界、全部伪代码逻辑、全部约束、全部踩坑规避。
上下文100%齐全，不需要你补充任何东西。
 
我用纯指令、条目式、强制约束式写法，AI可以直接逐行落地编码。

机器人分层策略 P0（平衡自救）+ G（门控判别器）完整训练工程指令
 
全局总架构（Agent必须首先熟读并严格遵守）
 
本系统包含 三个完全独立、原子可插拔模块
 
1. P0+（满血平衡自救策略）：最终上线兜底，训练后永久冻结
2. P0-（弱化平衡策略）：仅用于采集G数据集，永不训练、永不上线
3. G（状态危险判别门控网络）：纯监督学习，只判断「当前状态弱化P0-是否救不住」
 
全局硬性约束（最高优先级，代码强制遵守）
 
1. P0+、G 训练完成后永久冻结，后续所有P1/P2训练只推理、不更新。
2. P0- 完全由P0+复制改造，不训练、不优化、不回传梯度、不更新权重。
3. G 只依赖P0体系数据训练，与P1、格斗、前进任务完全无关，保证原子性。
4. 所有策略切换、数据采集、轨迹截断、GAE计算 必须严格按分段规则，禁止状态跨断裂拼接。
5. 所有P0接管帧 禁止进入P1经验池、禁止参与P1梯度更新。
 
阶段1：训练满血平衡策略 P0+（PPO强化学习）
 
1.1 环境初始化规则（每Episode）
 
1. 机器人躯干倾角、角速度 随机初始化失衡状态
2. 全程随机施加瞬时外力 Push（随机时间、随机方向、随机大小）
3. 初始关节姿态随机轻微偏移
 
1.2 倒地判定（严格唯一）
 
任意非足部身体部位接触地面 → 直接 terminal = True（倒地失败）
持续最大时长 T_max（3s）未倒地 → 平衡成功
 
1.3 P0+ 奖励函数
 
plaintext
  
r = 
- k1 * abs(躯干倾角) 
- k2 * abs(躯干角速度)
+ r_survive（存活满T_max）
+ r_fall（倒地大额负奖励）
 
 
1.4 P0+ 训练终止条件
 
随机扰动下 85%以上Episode可稳定存活
满足后：
 
- 保存 P0+ 权重
- 设置 requires_grad = False 永久冻结
 
阶段2：构造弱化版平衡策略 P0-（无训练、纯工程构造）
 
2.1 构造方式（固定方案，Agent直接实现）
 
基于已冻结P0+，做动作幅度硬限幅弱化
 
plaintext
  
a_raw = P0+.forward(obs)
a_clipped = clip(a_raw, -A_limit, A_limit)
A_limit = 原最大动作幅值的 65%~75%
 
 
2.2 P0- 固定属性
 
1. 权重完全等同于P0+，不训练
2. 仅前向推理
3. 专门用于制造“稍微失衡就救不住”的脆弱平衡策略
 
2.3 弱化程度调参终止标准（必须达到）
 
大批量随机仿真：
 
- P0- 倒地率：30%~40%
- P0- 存活率：60%~70%
 
保证 G 的正负样本比例均衡。
 
阶段3：使用 P0- 采集 G 门控网络训练数据集
 
3.1 数据采集环境
 
与 P0+ 训练环境 完全一致
 
- 随机初态失衡
- 全程随机Push扰动
- 最大时长一致
 
3.2 全程强制规则
 
数据采集全程 控制器固定 P0-
禁止出现任何 P0+、P1、格斗策略
 
3.3 轨迹标签规则（核心、必须严格执行）
 
对每一条完整Episode：
 
1. 最终存活没倒地 → 本条轨迹 所有帧 label = 0
语义：弱化P0-能救 → 状态安全，无需切入P0+
2. 最终倒地终止 → 本条轨迹 所有帧 label = 1
语义：弱化P0-救不住 → 状态危险，必须切入P0+
 
可选高精度优化（推荐实现）
 
采用 5步前瞻打标
每一帧往后看5帧：
 
- 未来5帧内倒地 → label=1
- 未来5帧持续存活 → label=0
 
3.4 数据集标准
 
- 总数据量：150万帧以上
- 正负样本比例：3:7 ~ 4:6
- 训练集:测试集 = 9:1
 
阶段4：训练门控判别网络 G（监督学习）
 
4.1 网络结构
 
输入：连续4帧历史观测序列（关节角、IMU、角速度）
结构：MLP + 浅层GRU（保证时序连续）
输出：Sigmoid(0~1)
 
4.2 训练配置
 
1. 标签平滑：
- 原始0 → 0.1
- 原始1 → 0.9
2. 损失函数：
- MSE回归损失（主损失）
- 时序平滑正则： loss_smooth = |G[t] - G[t-1]| 
- 总损失 = loss_mse + 0.05 * loss_smooth
3. 禁止二分类CE，必须回归连续输出（防止0/1跳变）
 
4.3 训练目标
 
让G输出连续、平滑、单调趋势
 
- 轻微失衡：G低
- 中度失衡：G中
- 重度失衡：G接近1
 
4.4 训练完成后
 
G权重永久冻结，不再训练
 
阶段5：上线推理规则（P1训练时使用）
 
5.1 双滞回阈值（彻底消除乒乓切换）
 
plaintext
  
G > 0.7  → 切入 P0+ 平衡< 0.45 → 切回 P1 前进/格斗模式
中间区间：保持当前控制器不变
 
 
5.2 每帧运行逻辑
 
1. 输入时序观测到 G，输出危险度 g
2. 根据滞回阈值选择控制器
3. P0+ 接管期间所有帧不进P1样本池
 
阶段6：重点：P1训练时的轨迹截断 & GAE 断裂处理（最关键）
 
6.1 硬性规则（彻底解决状态不连续问题）
 
一旦切换 P0+，立刻截断当前P1轨迹，标记 done=True，独立计算GAE
切回P1时新建轨迹，绝不拼接前后断裂状态
 
6.2 伪代码（Agent必须严格复刻）
 
plaintext
  
初始化 empty_trajectory_buffer
每帧循环：
    if 当前控制器 == P1:
        存入buffer
    else:
        if buffer非空:
            对当前buffer执行 GAE、return、advantage 计算
            送入样本池
            清空buffer
        丢弃所有P0帧
episode结束：
    处理剩余buffer
 
 
6.3 核心原理
 
- P0介入 = 动力学断裂点
- 断裂点必须终止轨迹
- GAE、回报、优势函数 禁止跨断裂传播
 
阶段7：P1防投机奖励设计（杜绝碰瓷P0、晃着前进）
 
7.1 奖励组成
 
plaintext
  
r_total = 
    r_closer （拉近对手）
  - 1.0 * switch_punish（发生P1→P0切换瞬间扣-1）
  - 小系数 * 躯干倾角惩罚
 
 
7.2 核心惩罚规则
 
只有切换首帧扣一次-1，P0持续期间不重复扣罚
目的：
让P1学会：
 
1. 尽量不失衡
2. 尽量不触发P0救援
3. 不能依靠P0反复倒地扶正前进
 
最终整体系统特性（Agent必须保证最终达成）
 
1. P0、G 完全原子化、与P1无关、可插拔
2. G 判断的是「弱化P0能不能救」，不是P1能不能救
3. 上线满血P0存在固定裕度：
- G判断救不住时 ≈ 弱化P0已崩
- 满血P0依然能救，完美提前预警
4. 无人工姿态阈值、无人工经验参数
5. G输出连续平滑、无帧间跳变
6. P1训练无状态断裂GAE错误
7. P1不会投机碰瓷平衡策略
 
最终交付文件清单
 
1. P0_plus.pt（冻结满血平衡）
2. P0_minus_config（固定弱化配置，不存权重）
3. G_gate.pt（冻结门控网络）
4. 完整数据集文件
5. PPO分段轨迹GAE代码
6. 双阈值切换逻辑模块

这份是可直接交付工程落地的完整全流程指令，上下文零缺失、逻辑闭环、所有坑提前封堵。
 
需要我接着给你写 配套的P1训练完整Agent指令（完全对接这套系统） 吗？



With New API Train V1:
cd /data1/mono/things/combatbench
PYTHONPATH=. nohup python3 baseline/humanoid21/curriculum/train.py --experiment  --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_20260528_175538/checkpoints/checkpoint_u00610.pt &> newframe.log & 


PYTHONPATH=. python3 -m baseline.humanoid21.curriculum.train --experiment v1_relation &> v1_relation.log & 

PYTHONPATH=. python3 -m baseline.humanoid21.curriculum.train --experiment balance_recover --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_20260608_010630/checkpoints/checkpoint_u01380.pt &> balance_recover2.log & 



生成最大扰动的视频
python3 -m envs.framework.round_runner \
  --env-blueprint baseline/humanoid21/blueprints/balance_recover_env.yaml \
  --policy-a-blueprint /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_20260608_010630/policy_exports/u01385/policy_blueprint.yaml \
  --policy-b-blueprint /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_20260608_010630/policy_exports/u01385/policy_blueprint.yaml \
  --video video1.mp4


接下来要做的一个事情是。打印训练的细节训练里面的细节的东西，比如说做了多少个update。嗯方便对于训练的细节进行深入的掌控。然后另外的话就是max mini batch size，这个如何去设置这个问题，可能也需要嗯去考虑。嗯就是训练参数的自动化自动优化，这个问题，如何去。就是如何是最优化，如何不是盲目的去调参，如何能够更好的去有理论支撑的去调参。然后目标是让在有限时间内能够让这个模型训练的更快啊，然后现在是有点太盲目了。