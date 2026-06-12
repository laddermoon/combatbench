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


#生成Plus数据
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 10000 \
  --noise-std 0.08 \
  --workers 12 \
  --output-dir baseline/humanoid21/curriculum/gating_data_plus


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


💾 Formatting and saving collected dataset... Done!
======================================================================
🎉 Dataset Collection Successfully Completed!
   - Saved .npz Path:  baseline/humanoid21/curriculum/gating_data/gating_data.npz
   - Saved JSON Path: baseline/humanoid21/curriculum/gating_data/summary.json
   - Total Frames:     476,676
     - Safe (Label 1):  346,800 (72.8%)
     - Unsafe (Label 0): 129,876 (27.2%)
   - Episode stats:
     - Total:          10000
     - Safe Stands:    1734 (17.3% survival rate)
     - Fallen Runs:    8266 (82.7% fall rate)
     - Average Length: 47.7 ± 71.0 steps
   - Total Execution Time: 566.0 seconds (9.4 minutes)
======================================================================

en: 49.1 steps

💾 Formatting and saving collected dataset... Done!
======================================================================
🎉 Dataset Collection Successfully Completed!
   - Saved .npz Path:  baseline/humanoid21/curriculum/gating_data_plus/gating_data.npz
   - Saved JSON Path: baseline/humanoid21/curriculum/gating_data_plus/summary.json
   - Total Frames:     483,683
     - Safe (Label 1):  356,600 (73.7%)
     - Unsafe (Label 0): 127,083 (26.3%)
   - Episode stats:
     - Total:          10000
     - Safe Stands:    1783 (17.8% survival rate)
     - Fallen Runs:    8217 (82.2% fall rate)
     - Average Length: 48.4 ± 71.7 steps
   - Total Execution Time: 578.3 seconds (9.6 minutes)
======================================================================

两者的对比，Plus还是更好的


# 增大 batch-size 拟合 10k 大数据，配置 [512, 256, 128] 的超深网络
PYTHONPATH=. python3 baseline/humanoid21/curriculum/train_gating_network.py \
  --epochs 80 \
  --batch-size 4096 \
  --hidden-dims 512 256 128 \
  --lr 5e-4 \
  --data-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_data


PYTHONPATH=. python3 baseline/humanoid21/curriculum/train_gating_network.py \
  --epochs 500 \
  --batch-size 4096 \
  --hidden-dims 512 256 128 \
  --lr 5e-4 \
  --data-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_data_plus \
  --output-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model_plus


接下来要做接近策略，
接近策略最大的问题是如何在Rollout的过程中接入这个安全盾，
有一种方式是做一个组合策略的收口，它来决定当前使用哪个策略，问题是如何接入现有的框架，以及如何对于收集到的数据进行切分， 因为我们要对于一个Episode把中间恢复模型介入的部分切掉（或者说把这个也收集起来用来进一步强化恢复模型？），从切开的地方开始就是另一段Episode，不做拼接，因为不连续（这是合理的做法吧？）。 
要做这些事情是否能用当前的Curriculum的框架来做，有哪些对不上的地方，如果有，Propose一版方案



Follow 策略的做法：
Follow策略把对手当成一个点，在模型的输入中可以只保留自身的观测状态，以及对手这个点的位置的信息作为输入，避免信息干扰。
Follow训练时的对手策略可以是这样的， 一个Scripted的策略，它会在可移动范围内到处走（但是会避免走到这个被训练的机器人附近，以免发生碰撞）
两个机器人都是以站立姿态来开始（不加初始随机扰动了）。
Reward就看机器人跟随对手移动的能力。

Env已经改好: /data1/mono/things/combatbench/baseline/humanoid21/blueprints/follow_env.yaml
ExpConfig：/data1/mono/things/combatbench/baseline/humanoid21/curriculum/experiments/exp_follow.py 



在 /data1/mono/things/combatbench/baseline/humanoid21/plugins/random_move.py
参照 /data1/mono/things/combatbench/envs/humanoid21/disturbance_plugins.py 


已经实现MixedPolicy，用来实现策略的组合，并且通过extra 也输出Gate信息
在/data1/mono/things/combatbench/baseline/humanoid21/curriculum/mixed_policy.py
核心逻辑就是当Gate判断要倒时，切到正式的恢复模型， 恢复之后再切回来。 这个要能接受一个PolicyBlueprint作为创建时的输入。 



[eval 6865] [ep mean_length=200.000 survived=1.000 level=6.000]
  | time: total=23.3s export=0.00s jobs=0.02s rollout=20.6s buffer=0.51s ppo=0.48s eval=1.7s
  | time: total=21.9s export=0.00s jobs=0.02s rollout=20.9s buffer=0.51s ppo=0.48s eval=0.0s
  | time: total=21.7s export=0.00s jobs=0.02s rollout=20.7s buffer=0.52s ppo=0.48s eval=0.0s
  | time: total=21.9s export=0.00s jobs=0.02s rollout=20.9s buffer=0.50s ppo=0.49s eval=0.0s
  | time: total=21.7s export=0.00s jobs=0.02s rollout=20.7s buffer=0.51s ppo=0.48s eval=0.0s