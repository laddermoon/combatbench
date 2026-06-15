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

PYTHONPATH=. python3 -m baseline.humanoid21.curriculum.train --experiment balance_recover --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_20260608_010630/checkpoints/checkpoint_u01380.pt &> balance_recover9.log & 



生成最大扰动的视频
python3 -m envs.framework.round_runner \
  --env-blueprint baseline/humanoid21/blueprints/balance_recover_env.yaml \
  --policy-a-blueprint /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_20260608_010630/policy_exports/u01385/policy_blueprint.yaml \
  --policy-b-blueprint /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_20260608_010630/policy_exports/u01385/policy_blueprint.yaml \
  --video video1.mp4


接下来要做的一个事情是。打印训练的细节训练里面的细节的东西，比如说做了多少个update。嗯方便对于训练的细节进行深入的掌控。然后另外的话就是max mini batch size，这个如何去设置这个问题，可能也需要嗯去考虑。嗯就是训练参数的自动化自动优化，这个问题，如何去。就是如何是最优化，如何不是盲目的去调参，如何能够更好的去有理论支撑的去调参。然后目标是让在有限时间内能够让这个模型训练的更快啊，然后现在是有点太盲目了。



python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py balance_recover9.log  --watch

#最终的策略用这个
/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_20260611_104207/policy_exports/u06865


继续训练加强的扰动恢复：
PYTHONPATH=. python3 -m baseline.humanoid21.curriculum.train --experiment balance_recover_plus --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_20260611_104207/checkpoints/checkpoint_u06870.pt &> balance_recover_plus1.log & 

python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py balance_recover_plus1.log  --watch

PYTHONPATH=. python3 -m baseline.humanoid21.curriculum.train --experiment balance_recover_plus --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_20260612_103559/checkpoints/checkpoint_u10000.pt &> balance_recover_plus2.log & 


PYTHONPATH=. python3 -m baseline.humanoid21.curriculum.train --experiment balance_recover_plus_refine --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_20260612_103559/checkpoints/checkpoint_u10000.pt &> balance_recover_plus_refine.log & 

python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py balance_recover_plus_refine.log  --watch



[eval 10045] [ep mean_length=167.609 survived=0.820 level=4.000]  [new_best]
[eval 10060] [ep mean_length=172.805 survived=0.852 level=4.000]  [new_best]
[eval 10110] [ep mean_length=174.281 survived=0.859 level=4.000]  [new_best]
[eval 10255] [ep mean_length=177.195 survived=0.875 level=4.000]  [new_best]
[eval 10350] [ep mean_length=178.648 survived=0.883 level=4.000]  [new_best]
[eval 10595] [ep mean_length=180.320 survived=0.891 level=4.000]  [new_best]

/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_refine_20260614_004027/policy_exports/u10595


# 使用弱化版的扰动恢复策略生产数据，用来训练状态是否可恢复的判别模型。

@IDEA.md#L1-8 接下来我要进行第三步了， #最终的策略用这个
/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_20260611_004703/policy_exports/u03275  
现在有两个问题要处理：1， 如何轻微弱化这个策略
2. 如何生成数据
关于2我的想法是复用训练过程中的Rollout的配置， 使用随机初始状态，不需要Level，直接用一个比较大的范围（应该要比最大的Disturb等级还要大的），然后成功不成功的数据都记录下来。 成功的是正样本，不成功的是负样本 
关于1，请给我一些输入， 对于整体流程也给我一些思路


# 推荐收集 2000 个 Episodes，使用 12 个并行进程加速
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 10000 \
  --noise-std 0.08 \
  --workers 12 \
  --output-dir baseline/humanoid21/curriculum/gating_data

--noise-std (默认 0.08)：如果您想让安全裕度（Safety Buffer）更保守（让门控网络更早、更敏感地在刚倾斜时就触发介入），可以增大此噪声值（例如 0.10 ~ 0.12）；如果您想让门控网络尽可能“极限压榨接近策略，直到千钧一发时才切恢复”，可以减小此噪声（例如 0.05）。
成功/失败样本均衡度：运行完后，请看输出的 Safe (Label 1) 与 Unsafe (Label 0) 的帧数比例。接近 5:5 或 4:6 是最利于二分类器收敛的黄金比例。


#生成零扰动的数据对比差异
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 2000 \
  --noise-std 0.0 \
  --workers 12 \
  --output-dir baseline/humanoid21/curriculum/gating_data_without_noise


#生成
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 10000 \
  --noise-std 0.08 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_plus_u06865 \
  --policy-path /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_20260611_104207/policy_exports/u06865

训练时Eval成功率100% ， 怎么掉怎么这么厉害？

💾 Formatting and saving collected dataset... Done!
======================================================================
🎉 Dataset Collection Successfully Completed!
   - Saved .npz Path:  baseline/humanoid21/curriculum/gating_data_plus_u06865/gating_data.npz
   - Saved JSON Path: baseline/humanoid21/curriculum/gating_data_plus_u06865/summary.json
   - Total Frames:     639,993
     - Safe (Label 1):  531,400 (83.0%)
     - Unsafe (Label 0): 108,593 (17.0%)
   - Episode stats:
     - Total:          10000
     - Safe Stands:    2657 (26.6% survival rate)
     - Fallen Runs:    7343 (73.4% fall rate)
     - Average Length: 64.0 ± 82.4 steps
   - Total Execution Time: 200.1 seconds (3.3 minutes)



PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 10000 \
  --noise-std 0.0 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_plus_u06865_nonoise \
  --policy-path /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_20260611_104207/policy_exports/u06865



======================================================================
🎉 Dataset Collection Successfully Completed!
   - Saved .npz Path:  baseline/humanoid21/curriculum/gating_data_plus_u06865_nonoise/gating_data.npz
   - Saved JSON Path: baseline/humanoid21/curriculum/gating_data_plus_u06865_nonoise/summary.json
   - Total Frames:     671,067
     - Safe (Label 1):  566,600 (84.4%)
     - Unsafe (Label 0): 104,467 (15.6%)
   - Episode stats:
     - Total:          10000
     - Safe Stands:    2833 (28.3% survival rate)
     - Fallen Runs:    7167 (71.7% fall rate)
     - Average Length: 67.1 ± 84.1 steps
   - Total Execution Time: 199.9 seconds (3.3 minutes)
======================================================================



# recover_plus:
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 10000 \
  --noise-std 0.08 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_plus \
  --policy-path /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_20260612_103559/policy_exports/u10000

======================================================================
🎉 Dataset Collection Successfully Completed!
   - Saved .npz Path:  baseline/humanoid21/curriculum/gating_data_plus/gating_data.npz
   - Saved JSON Path: baseline/humanoid21/curriculum/gating_data_plus/summary.json
   - Total Frames:     944,540
     - Safe (Label 1):  860,400 (91.1%)
     - Unsafe (Label 0): 84,140 (8.9%)
   - Episode stats:
     - Total:          10000
     - Safe Stands:    4302 (43.0% survival rate)
     - Fallen Runs:    5698 (57.0% fall rate)
     - Average Length: 94.5 ± 92.0 steps
   - Total Execution Time: 238.6 seconds (4.0 minutes)
======================================================================

# recover_plus:
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 10000 \
  --noise-std 0.0 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_plus_nonoise \
  --policy-path /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_20260612_103559/policy_exports/u10000


💾 Formatting and saving collected dataset... Done!
======================================================================
🎉 Dataset Collection Successfully Completed!
   - Saved .npz Path:  baseline/humanoid21/curriculum/gating_data_plus_nonoise/gating_data.npz
   - Saved JSON Path: baseline/humanoid21/curriculum/gating_data_plus_nonoise/summary.json
   - Total Frames:     982,992
     - Safe (Label 1):  903,000 (91.9%)
     - Unsafe (Label 0): 79,992 (8.1%)
   - Episode stats:
     - Total:          10000
     - Safe Stands:    4515 (45.1% survival rate)
     - Fallen Runs:    5485 (54.9% fall rate)
     - Average Length: 98.3 ± 92.5 steps
   - Total Execution Time: 255.4 seconds (4.3 minutes)
======================================================================

# recover_plus_refine:
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 10000 \
  --noise-std 0.08 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_plus_refine \
  --policy-path /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_refine_20260614_004027/policy_exports/u10595


======================================================================
🎉 Dataset Collection Successfully Completed!
   - Saved .npz Path:  baseline/humanoid21/curriculum/gating_data_plus_refine/gating_data.npz
   - Saved JSON Path: baseline/humanoid21/curriculum/gating_data_plus_refine/summary.json
   - Total Frames:     906,683
     - Safe (Label 1):  817,600 (90.2%)
     - Unsafe (Label 0): 89,083 (9.8%)
   - Episode stats:
     - Total:          10000
     - Safe Stands:    4088 (40.9% survival rate)
     - Fallen Runs:    5912 (59.1% fall rate)
     - Average Length: 90.7 ± 91.2 steps
   - Total Execution Time: 239.7 seconds (4.0 minutes)
======================================================================



# mix_level
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data_refine.py \
  --num-episodes 10000 \
  --noise-std 0.08 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_plus_refine_mix_level \
  --policy-path /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_refine_20260614_004027/policy_exports/u10595


======================================================================
🎉 Dataset Collection Successfully Completed!
   - Saved .npz Path:  baseline/humanoid21/curriculum/gating_data_plus_refine_mix_level/gating_data.npz
   - Saved JSON Path: baseline/humanoid21/curriculum/gating_data_plus_refine_mix_level/summary.json
   - Total Frames:     1,651,945
     - Safe (Label 1):  1,618,800 (98.0%)
     - Unsafe (Label 0): 33,145 (2.0%)
   - Episode stats:
     - Total:          10000
     - Safe Stands:    8094 (80.9% survival rate)
     - Fallen Runs:    1906 (19.1% fall rate)
     - Average Length: 165.2 ± 71.9 steps
   - Per-level breakdown:
     Level |  Scale |    Eps |  Falls |   Surv |   Frames |  Surv%
     ----- | ------ |  ----- |  ----- |  ----- |   ------ |  -----
         0 |   0.10 |   1500 |      0 |   1500 |  300,000 | 100.0%
         1 |   0.20 |   1500 |      7 |   1493 |  298,814 |  99.5%
         2 |   0.35 |   1400 |     30 |   1370 |  274,895 |  97.9%
         3 |   0.50 |   1400 |     82 |   1318 |  265,468 |  94.1%
         4 |   0.70 |   1400 |    334 |   1066 |  220,292 |  76.1%
         5 |   0.85 |   1400 |    581 |    819 |  173,620 |  58.5%
         6 |   1.00 |   1400 |    872 |    528 |  118,856 |  37.7%
   - Total Execution Time: 365.8 seconds (6.1 minutes)
======================================================================

PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data_refine.py \
  --num-episodes 10000 \
  --noise-std 0.08 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_plus_mix_level \
  --policy-path /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_20260612_103559/policy_exports/u10000



======================================================================
🎉 Dataset Collection Successfully Completed!
   - Saved .npz Path:  baseline/humanoid21/curriculum/gating_data_plus_mix_level/gating_data.npz
   - Saved JSON Path: baseline/humanoid21/curriculum/gating_data_plus_mix_level/summary.json
   - Total Frames:     1,660,868
     - Safe (Label 1):  1,628,800 (98.1%)
     - Unsafe (Label 0): 32,068 (1.9%)
   - Episode stats:
     - Total:          10000
     - Safe Stands:    8144 (81.4% survival rate)
     - Fallen Runs:    1856 (18.6% fall rate)
     - Average Length: 166.1 ± 71.2 steps
   - Per-level breakdown:
     Level |  Scale |    Eps |  Falls |   Surv |   Frames |  Surv%
     ----- | ------ |  ----- |  ----- |  ----- |   ------ |  -----
         0 |   0.10 |   1500 |      0 |   1500 |  300,000 | 100.0%
         1 |   0.20 |   1500 |     10 |   1490 |  298,355 |  99.3%
         2 |   0.35 |   1400 |     51 |   1349 |  271,480 |  96.4%
         3 |   0.50 |   1400 |     90 |   1310 |  264,144 |  93.6%
         4 |   0.70 |   1400 |    321 |   1079 |  221,932 |  77.1%
         5 |   0.85 |   1400 |    539 |    861 |  181,434 |  61.5%
         6 |   1.00 |   1400 |    845 |    555 |  123,523 |  39.6%
   - Total Execution Time: 365.8 seconds (6.1 minutes)
======================================================================


# 增大 batch-size 拟合 10k 大数据，配置 [512, 256, 128] 的超深网络
PYTHONPATH=. python3 baseline/humanoid21/curriculum/train_gating_network.py \
  --epochs 80 \
  --batch-size 4096 \
  --hidden-dims 512 256 128 \
  --lr 5e-4 \
  --data-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_data_plus


PYTHONPATH=. python3 baseline/humanoid21/curriculum/train_gating_network.py \
  --epochs 500 \
  --batch-size 4096 \
  --hidden-dims 512 256 128 \
  --lr 5e-4 \
  --data-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_data_plus \
  --output-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model_plus

# 使用 Mix数据进行训练
PYTHONPATH=. python3 baseline/humanoid21/curriculum/train_gating_network.py \
  --epochs 500 \
  --batch-size 4096 \
  --hidden-dims 512 256 128 \
  --lr 5e-4 \
  --data-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_data_plus_mix_level \
  --output-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model_plus_mix_level


扰动恢复策略可能需要做一定的改进，主要是比如现在按level来取，这个最大上限是零点。动作的上限是0.4，但实际上这说明就有些动作范围就覆盖不了，你的动作范围覆盖不了，这个恢复策略也没见过，然后。恢复策恢复策略没见过，然后这个这个这个判别模型也没见过，就有很多会有很多这样的，就是这个如何让恢复策略能见到更多有意义的这种初始场景，让他进行去恢复，然后以及。如何让这个。以及如何让这个如何让这个模型。就是判别模型见到更多有意义的场景进行恢复，这个是一个是一个问题。就初始状态的这个准备，其实这个状态库，对，可能得做一想办法做一个状态库，就基于什么来做一个这种初始状态库。然后就训练的时候，这状态库是可以有一些难度判定，也可以没有，就是然后在训练和这个进行采样的时候，这个去从这个状态库进行抽取，那如何去制备这个状态库是一个问题。今天就先这样，没必要一次把这个做得很完美，这个先问题先有待后面去解决。


接下来要做接近策略，
接近策略最大的问题是如何在Rollout的过程中接入这个安全盾，
有一种方式是做一个组合策略的收口，它来决定当前使用哪个策略，问题是如何接入现有的框架，以及如何对于收集到的数据进行切分， 因为我们要对于一个Episode把中间恢复模型介入的部分切掉（或者说把这个也收集起来用来进一步强化恢复模型？），从切开的地方开始就是另一段Episode，不做拼接，因为不连续（这是合理的做法吧？）。 
要做这些事情是否能用当前的Curriculum的框架来做，有哪些对不上的地方，如果有，Propose一版方案



Follow 策略的做法：
Follow策略把对手当成一个点，在模型的输入中可以只保留自身的观测状态，以及对手这个点的位置的信息作为输入，避免信息干扰。
Follow训练时的对手策略可以是这样的， 一个Scripted的策略，它会在可移动范围内到处走（但是会避免走到这个被训练的机器人附近，以免发生碰撞）
两个机器人都是以站立姿态来开始（不加初始随机扰动了）。
Reward就看机器人跟随对手移动的能力。


在 /data1/mono/things/combatbench/baseline/humanoid21/plugins/random_move.py
Env: /data1/mono/things/combatbench/baseline/humanoid21/blueprints/follow_env.yaml

环境验证：
python3 -m envs.framework.round_runner \
  --env-blueprint baseline/humanoid21/blueprints/follow_env.yaml \
  --policy-a-blueprint /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_20260612_103559/policy_exports/u10000/policy_blueprint.yaml \
  --policy-b-blueprint /data1/mono/things/combatbench/policy/blueprints/random.yaml \
  --video video-1.mp4

在/data1/mono/things/combatbench/baseline/humanoid21/curriculum/mixed_policy.py
核心逻辑就是当Gate判断要倒时，切到正式的恢复模型， 恢复之后再切回来。 这个要能接受一个PolicyBlueprint作为创建时的输入。 

#验证 mix policy
/data1/mono/things/combatbench/baseline/humanoid21/blueprints/mixed.yaml
MIXED_POLICY_DEBUG=1 python3 -m envs.framework.round_runner \
  --env-blueprint baseline/humanoid21/blueprints/follow_env.yaml \
  --policy-a-blueprint /data1/mono/things/combatbench/baseline/humanoid21/blueprints/mixed.yaml \
  --policy-b-blueprint /data1/mono/things/combatbench/policy/blueprints/random.yaml \
  --video video8.mp4


ExpConfig：/data1/mono/things/combatbench/baseline/humanoid21/curriculum/experiments/exp_follow.py 
几个关键点：
Episode中 primary与fallback动作数量的统计和对比。
SubEpisode的长度。

几个问题：
从哪个Checkpoint开始训练？ 从Fallback模型吧？
是否同步收集Recover的数据进行训练？ 先不节外生枝吧






PYTHONPATH=. python3 -m baseline.humanoid21.curriculum.train --experiment follow --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_20260611_104207/checkpoints/checkpoint_u06870.pt &> follow.log & 

python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py follow.log  --watch


希望这种扰动恢复策略，不能因为随机策略做了一个动作就无法恢复了，我觉得这个能力有点弱，可能有问题，这个再训练一下试一下。

然后这个。这个跟随策略的话，应该就从扰动策，最强的扰动策略开始训练吧。在这个基础上去训练。

几种奖励，我先想一下，第1个是。第1个是摔倒。对，要切成episode的，对，要切先切episode的，然后切episode的之后，对于每一个这个切出来的episode的，嗯就是跟这个相关的，如果是直接摔倒了，那这个肯定是要惩罚的，但这种情况应该不会多见，但这种情况要有。还有一个是如果。还有是一个是如果那个如果被如果被接管了，如果对如果被接管了，这个也是要惩罚的，这个跟摔倒的惩罚逻辑其实是一样的，就认为他是。就这个应该是比较多见的，就认为他是摔倒了。然后这个的话，这种情况下，就是让机器人要避免走到那种极端的情况下，就是被被那个接管。这是两种最基本的奖励，然后。呃，第3种就是。这第3种就是跟随跟随对手的这个奖励了，跟随对手。这个。这个其实得有一个有一个是这样这样的，应该有一个对手在对手范围内的一个边界，然后再比如说对手在对手范范围，比如说在这种一米之内就没有惩罚。这是就是，对，这也是一种保持吧，也是一种保持，就是一米之内就没有惩罚，然后超出一米就开始有惩罚。然后为了避免这种在这种边界的情况上来回乱跳，就应该让就是让机器人有冲到对手在80比如说就是前如果前面是一米的话，可能就。有要有冲到这种80这个距离的这种势能。也就是比如说跟对手的这种距离的。呃距离的这种线性惩罚应该是在80之内没有，然后过了80就线性增加，这个是让呃机器人去靠近。靠近对手，然后这个其实这个其实是一个是函数，是函数，就是机器人与对手之间的这个距离。然后可以考虑用势函数的这种处处理方式，就是把它变成要变成这种德尔塔的方式。看看这种行不行？然后另外的话，另外的话就是这种应该那只奖励机器人朝向朝对手移动的，我觉得就是在平滑之后，平滑之后的这种速度，只奖励朝向对手移动的，然后所有的横向的移动都应该被惩罚。然后这个向对手移动的越快，就是奖励越大，然后那个横向移动越越快，呃惩罚越大。然后如果是有这种向负方向的这个移动的分量，那这个就是这个也是直接就是惩罚。对，接近对手可能就按这个大概就是这个逻辑来。

我再梳理一下，说如果如果机器人离离另另外一个机器人这个离另外这个机器人啊这个距离大于0.8米，然后这时候然后这时候就奖励他越快奖奖励他就是呃就是这时候啊不按距离奖励，就按这个速度奖励，就是朝向这个机器人，对手移动的越快，就呃这个奖励越大。然后机器人在这个对手的范围在一米之内，就没有这种，没有这种状态距离状态惩罚，然后在一米之外就有这种距离状态惩罚。这个80跟一米有一个区别，这样是不是可以更好的帮助他保持在一米之内呢？还是说这个一米其实就没有什么实际的作用呢？就不需要这个一米的这种状态惩罚。有的可能就不需要这个一米的状态惩罚，就不要这个一米的状态惩罚，因为就这个一米的状态惩罚应该是给呃这不是一个状态惩罚，这是一个。这是一个状态判定。唉对，这是一个状态判定，我我想清楚了，这是一个状态判定，就是当机器人嗯，距离在这个一米之内时。跟这个对手在一米之内时，这时候就启动那个启动这个启动那个打击策略了，嗯当然这个打击策略现在还没有。然后然后当机器人嗯离开这个一米的范围之内时，就启动这个移动策略，向对手移动靠近的策略，然后这个策略呢其实朝向的这个目标是，朝下移动的这个目标是在到80公分就到80公分之内才停止继续向对手靠近。这里面有一个差值，就是为了防止说。呃机器人刚移动到一米之内，比如如果那个8不是80是一米的话，那刚移动到一米之内，然后就切成这个打击策略了，然后他那个嗯打击策略一下子又移出一米之外了，然后又回到这个移动策略，就这么导致这个策略来回切换，就相当于这是一个策略切换之间，就是移动策略和打击策略之间的切换的一个一个这个。延迟迟滞触发器，类似这样，为了消除这种中间的这种震荡。

对，可能还有一个这个机器人朝向的问题，机器人也要朝向对手，不能那个是背对着移动，那个不行，但是我觉得这个问题不大，可以先不加，就是机器人只要向对手移动，大概率就不是背对着移动，因为那样是更难的，强化学习一定会让机器人正对着朝向对手移动。

啊对，还有一个就是。就是怎么判断这个移动模型的能力，就是他当前是在什么样的水平，然后是否要继续加大难度，也就是这个在训练移动模型的时候，这个呃curriculum的设置。就是依靠什么？呃靠什么这个指标去驱动这个curriculum这个学习的进展。比如什么指标到什么程度了，就认为这个阶段学学完了，然后应该到下一个阶段了。嗯这个的话。目前的一个想法是，我的一个想法是。就是呃让。就是要跟踪一个，对，跟踪一个就唉对，其实就是呃就是就是刚才说的这个那个机器人的这个在一米范围之内的这个这个概率，或者是这个。呃这个呃在一比如说一个episode的长度是10秒钟，然后它有9秒钟之之内都在这个一米之内，就说明它这个跟随性。啊跟跟着这个对手的这个能力，就现在已经跟着这个阶段他他这个阶段移动速度对手的能力就已经很强了，然后就可以进入到把让对手移动的更快一点，然后让他继续能够跟得上。对，就是要卡这么一个嗯，可以卡这么一个比例，就是呃在它这个机器人能在呃呃多少呃有效多少时间范围之内，在这个保持在有效攻击范围之内啊。然后超过一个比例值就升到下一个阶段，那这一定是一个比例，而不是说一定是要达到100%，因为。啊机器人在初始的时候，嗯双双方是有一个距离的，然后这个距离呢，就是它机器人移动过去的时候是需要时间的。对，机器人移动过去是需要需要一段时间的，所以一定不能是100%。然后现在我觉得可以考虑的是把这个呃在在训练这个移动的时候，把这个啊对，把这个呃把这个呃环境里面的这个两个机器人之间的初始距离的随机化的这个功能给去掉了，啊我觉得不需要随机化，这样会引入了这个就是这个比百分比，前面说的这个百分比。他没法去统一的去看，因为有的是离这3.5米，有的是离这1.5米，然后这中间。比如说3.5米走到另一个机器人的一米范围之内需要一段时间，然后1.5米需要的这个时间就会短一点，然后他们在这个一米范围之内的这个时间就天然就会啊是差了很多。对，这个是这个呃机器人的就是这个阶段学习的一个重要的指标。然后呃同时要用这个指标去启动驱动这个强化学习驱动这个课程的进展，比如这些参数的话需要。还需要进一步的去看用什么样的参数来去做，现在还不太好凭脑袋拍板，比如说这个在这个范围之内达到了百分之多少？在一米范围之内的时间达到了100%百分之多少？在一阶段，嗯达到百分之多少，然后就算成功，二阶段达到百分之多少就算成功。这个现在嗯还不好说，然后每一阶段这个机器人，就是对手机器人随机移动的这个速度，然后也也不好说，也还不太好设置。

对，然后对手机器人移动的这个呃呃这个可能有一个。可能有一个bug吧，这个可能需要去看一下，就是现在设置的时候是让机器人不能再就是随机移动的那个插件random move那个插件。在移动机器人的时候，不会让它移动到靠近另一个机器人，就是我们在训练的这个机器人，应该是保持了1米2的距离，还是多少的距离，那如果是在那边做了一个这种硬限制的话，那这边呃当然这个我们训练的这个机器人怎么移动也不可能移动到那个机器人的旁边去。这个要去检查一下。


机器人移动就是上一次一直没有过去的一个点，就是一直不知道让他怎么移动，因为上次训练的时候是这个呃对称策略，对，训练的是对称策略，然后那个可能是有一个天然的问题，然后现在可能用的是这个对手是随机策略的，我觉得这个这现在这一关有可能是能够过去的，那我就是控制机器人的移动的方向，然后让它去嗯靠近这个对手。呃我觉得有一定成功的可能性。现在能想到想到的问题就是这么多，明天早上再看吧。



PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --v2 --experiment follow --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_20260612_103559/checkpoints/checkpoint_u10000.pt &> follow.log & 






PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --v2 --experiment follow --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_20260612_103559/checkpoints/checkpoint_u10000.pt &> follow.log & 
我现在用上面这个命令开始了模型训练, 请帮我检查一下这个模型训练有没有问题，就是训练代码有没有问题。因为也是刚刚写完的训练代码有没有问题。然后训练的逻辑有没有什么问题？对，帮我大概先check一下
/data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py
对，然后另外我要做的一个事情是，参照上面的这个，之前训练这种扰动恢复模型时候用到的这个日志分析的这个工具。然后但是现在这个，那现在这个新的这个这个这个现在这个是目标跟随的这个训练，这个要分析的内容和监测的指标都不太都很不一样。然后这个呢，需要重新可能这方面需要重新实现，然后请参照上面这个脚本，然后可能要实现一个新的这种日志分析，就是监控的脚本，然后这个脚本用来追踪训练的过程。然后如果在代码实现中本身这个日志加的不够的话，日志加的有问题的话，就那就可以先把这个日志给加好，对，先把这个日志给加好，然后，我的最终的目标是通过这个监控，然后去发现问题，然后解决问题。通过这种方式来推动这个训练的进展，就是通过相当于是TDD的模式，先那个发现问题，然后解决问题，就不去检查所有的代码，先去刻意一定要把所有代码都发觉得是正确的。然后只是通过一些指标去监测问题，然后如果指标有问题的时候，再去找问题，对通过这种方式

/data1/mono/things/combatbench/baseline/humanoid21/curriculum/OBSERVABILITY.md
/data1/mono/things/combatbench/baseline/humanoid21/curriculum/SYSTEM_MODEL.md
然后，上面这两个文档也是比较重要的，可以去参照的。就这个文档是那个这两个文档说明了当前的系统模型。当然，这个是参照以前的那个，现在这个在系统上加了这个萨episode segmentation的功能，然后之前是没有的，这个稍微有一点不一样，但大总体上应该是差不多的

/data1/mono/things/combatbench/baseline/humanoid21/curriculum/PPOTrainDirection.md

然后，上面这个文档是我整理的一个PP的训练PPO训练的一个指南，或者说说指导手册。然后，就是这个是可以作为一个参照


# 1. 启动健康的 PPO 追逐策略训练
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train \
  --v2 \
  --experiment follow \
  --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_follow_20260615_094945/checkpoints/checkpoint_u10220.pt \
  &> follow.log &

# 2. 使用我们为您全面专业化升级的工具进行实时追踪与诊断：
python3 baseline/humanoid21/curriculum/analyze_follow_logs.py follow.log --watch




PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --v2 --experiment follow --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_20260612_103559/checkpoints/checkpoint_u10000.pt &> follow.log & 


python3 baseline/humanoid21/curriculum/analyze_follow_logs.py follow.log --watch