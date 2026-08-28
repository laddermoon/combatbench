端到端的方案

总体的思路是条件奖励加分段训练。

第一步，机器人从倒地开始训练站立模型。
这样机器人就学会了从倒地到站立，这个目前已经完成。
/data1/mono/things/combatbench/baseline/experiments_v2/exp_standup.py

第二步，训练机器人迈步。还是从倒地开始，然后在战力模型的基础上加上这个迈步奖励。
让预期得到的是一个能够会站起来之后会迈步的一个策略。 迈部的奖励也算是条件奖励吧，只有在机器人已经完全站立起来之后才触发这个奖励
/data1/mono/things/combatbench/baseline/experiments_v2/exp_standup_step_v3.py
/data1/mono/things/combatbench/baseline/experiments_v2/exp_step.py
看起来第一个效果好。

上面两个实验也考虑合并成一个。那就是直接从第二步开始训练
合并之后效果暂时不成功，参见 standup_step系列的实验。
#TODO 分析不行的原因以及改进。
/data1/mono/things/combatbench/baseline/experiments_v2/exp_standup_step_v3.py
似乎训练不出来。


后续看，如果有必要的话，可能需要在这中间加一个平衡强化的阶段。在机器人站立起来之后，在加入随机扰动，然后让机器人尽量再保持平衡。
/data1/mono/things/combatbench/baseline/experiments_v2/exp_balance_v2.py
正在训练

第三步，训练机器人跟随。还是从倒地开始，因为这个倒地的这个是一直要强化训练的。然后加上移动靶，就跟现在的这个跟随策略是这个训练是一样的奖励。跟随奖励也是条件奖励，只有在机器人完全站立起来之后才有跟随奖励。
/data1/mono/things/combatbench/baseline/experiments_v2/exp_standup_follow.py

第四步，训练机器人准备攻击，在与目标机器人到了一定距离之后，然后开始准备攻击姿态，主要是要朝向对手在一个范围之内，不能侧对着对手或者背对着对手。 这个奖励只有在跟对手到了一定范围之内才生效。条件奖励。
/data1/mono/things/combatbench/baseline/experiments_v2/exp_standup_face.py

说明，这个步骤3和步骤4存在潜在的合并的可能性，因为实验发现加了face之后follow的更好.
/data1/mono/things/combatbench/baseline/experiments_v2/exp_follow_v2.py
使用合并的方案

这上面的这些步骤完成之后，就会得到一个能够从倒地站立起来，并且走到对手跟前的这样一个策略。


然后第5步是在上面的策略的基础之上训练，再加上攻击的奖励，然后训练攻击模型。
攻击模型的训练还是从倒地开始，就是机器人倒地之后站起来，然后走近对手，然后开始攻击。训练的起点都是这种随机摔倒的姿态。
然后智能推理的时候，机器人是从站立的姿态开始的，这个对于从摔倒姿态开始的这种训练出来的策略肯定是没有难度的。


/data1/mono/things/combatbench/baseline/experiments_v2/exp_standup_face.py
在上面的这个实验的基础上实现Fight。 
1. 在上面这个实验的基础上添加新的Reward，与Fight相关的
2. 对于不再使用移动靶，而是使用策略池中选择。
3. 从上面这个实验训练好的Checkpoint继续训练。
4. 策略池的做法：用上面那个实验的策略作为初代策略；然后每训练一轮得到一代新的策略；从历史策略中从近到远做权重采样，近的权重大，远的权重小。Rollout时随机选择一个策略做为对手策略。
5. 每一轮训练的目标：Eval的指标超过某个指定值。 待定。
6. 新加的Reward： 第一版：得分和失分两个。通过调节得分和失分的Reward的权重比例来调节策略攻守平衡。
7. 可能可用的资源 /data1/mono/things/combatbench/baseline/humanoid21/fight/damage.py


参照/data1/mono/things/combatbench/baseline/experiments_v2/exp_standup_step_v3.py
中的Step奖励，进行细化奖励，引导机器人进行打斗。
通过状态机来看
两个手分开来算
一共有两种状态：
1. 攻击状态，准备攻击状态
初始先检测手臂的状态：
1. 如果在朝对手的受击区域运动。则在攻击过程中。
2. 如果没有，就是在准备过程中。
状态转换：
1. 在准备状态下：肘关节到最大限位（加一个小阈值），则进入攻击状态。
2. 在攻击状态下：肘关节己到最大限位（加一个小阈值）或者已经没有机会打到对手，则进入准备状态

奖励：
1. 准备状态下：奖励肘关节收回
2. 攻击状态下：奖励肘关节伸直，并且拳头速度朝向对手头部

另外需要考虑的是，双手的状态耦合的问题：
一个最简单的解决办法是：
两个手臂的状态一定是反的。左手在进攻，右手一定在准备。左手进入准备，则右手进入进攻状态。 以进攻那只手是否进攻完成做为状态切换的触发器，而不是以准备的那只手准备是否完成。

初始状态的安排是，哪只手离对手近，哪只手进入攻击状态。