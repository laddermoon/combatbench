things/combatbench/envs/framework
在上面的目录中是关于如何支持构造仿真环境，然后如何？让仿真环境能够易用。嗯以及通用的插件体系。是框架性的代码。
things/combatbench/envs/framework/DESIGN.md
可以通过上面的文档了解整个框架的设计和实现。具体的细节可能还是要看代码。
things/combatbench/envs/humanoid21
我上面的目录中是嗯一个仿真环境的具体实现。是某周口里面的21个数控自由度的，这个胶囊机器人，两个胶囊机器人在一个环境里面，这样的一个仿真环境。最终的目标是两个机器人可以互相嗯互相击打，然后这种。仿真的，仿真搏击。
比较重要的三个文档在下面。
things/combatbench/envs/humanoid21/OBSERVATION_zh.md
things/combatbench/envs/humanoid21/DATASPEC.md
things/combatbench/envs/humanoid21/CONTROLSPEC.md
可以通过上面三个文档了解仿真环境的设计和实现。具体的细节可能还是要看代码。
上面的两个目录中的代码都已经实现完成，并且做了嗯相当程度的验证。可以认为没目前可以认为是没有问题的，能直接用的。但是也不排除会有问题，如果发现里面有问题，请不要直接修改，而是要给我提反馈。我来确认。不要轻易改上面两个目录的代码。

接下来我要做的事情是，在下面的目录中实现实现baseline。然后目前专注于做的事情是先实现站立，让一个机器人学会站立。
things/combatbench/baseline/humanoid21
有几个要求:
第一。要用最少的代码来实现，所有的代码都放在同一个文件里面就叫standing.py。用简最简单最符合直觉的代码，不做复杂的封装和工程化。
第二，直接运行用Python运行standing点py可以直接运行，直接启动训练。然后不需要呃各种繁琐的配置，直接写死在代码里面。
第三，用最简单的grpo算法。
第四, 结果要保存成标准Policy（按照 things/combatbench/policy/README.md）。
目前我最直接的想法是嗯，对于每个episode给一个episode级的奖励。然后。嗯，然后把整把这个奖励均匀的分配到嗯平均分配的，或者直接把这个i配送的奖励给到episode中的每一步。就是不对，呃，不做复杂的奖励拆分，比如说哪一步做对了，哪一步做错了，不这样做，然后只是通过ipc的级的奖励来。嗯，来作为训练信号。啊需要注意的一点是这样做方差可能极大，然后要把这个bat size子做到非常高。对batch size子做要非常高才有意义。

请结合我的目标，以及我给出的初始的想法。嗯分析一下其中的可行问，分析一下可行性，以及其中可能存在的问题以及难点。然后给我先出一个综合的方案。






things/combatbench/baseline/humanoid21_nonfall
things/combatbench/baseline/mujoco21dof_nonfall
上面这两个目录都是已经过时的代码，不要参照，不要管他们。
然后我们要做的不是在这种不会倒的环境中去学站立，而是在正常的环境中。

NonFallConstraintPlugin 
FrozenRobotPlugin 
上面这两个插件不要用，然后机器人b不用管。

policy/README.md 里提到内置 standing policy , 这个是文档错误，现在我已经把这个文档更新了，不用管这个问题。




嗯下面请帮我开始实现。然后这个实现要注意一点是嗯符合框架设计的这个本意。就是对框架的使用方式，要符合设计的意图。比如要添加reward，应该通过observer插件来添加等等，这只是举个例子。另外所有的代码都放在standing.py里面，包括实现的插件。


python3 -m envs.humanoid21.run_round --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260406_173405/policy --video test1.mp4


python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260406_180347/policy_final_reexport \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260406_180347/policy_final_reexport \
  --video best_model2.mp4


python3 - <<'PY'
from pathlib import Path
from baseline.humanoid21.standing import export_policy_artifacts

run_dir = Path("/data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260406_180347")
model_path = run_dir / "checkpoints/update_3400.pt"
policy_dir = run_dir / "policy_final_reexport"

export_policy_artifacts(model_path, policy_dir)
print(policy_dir)
PY


python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_000128/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_000128/policy \
  --video standingtest.mp4

/data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_000128/policy



python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260406_191013/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260406_191013/policy \
  --video standingtest1.mp4

/data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260406_191013/policy



STANDING_TURBULENCE_INIT_MODEL=/data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_000128/policy \
nohup python3 /data1/mono/things/combatbench/baseline/humanoid21/standing_with_turbulence.py > turbulence.log &




STANDING_TURBULENCE_INIT_MODEL=/data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_000128/policy nohup python3 /data1/mono/things/combatbench/baseline/humanoid21/standing_with_turbulence.py > nohup_turbulence.out 2>&1 &



/data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_084615/policy


python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_084615/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_084615/policy \
  --video standingtest2.mp4

  /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_ppo_20260407_182646/policy


python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_ppo_20260407_182646/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_ppo_20260407_182646/policy \
  --video standingtest4.mp4



  /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_182313/policy



python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_182313/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_20260407_182313/policy \
  --video standingtest5.mp4


PPO, second round  
python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_ppo_20260409_225537/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_ppo_20260409_225537/policy \
  --video standingtest7.mp4


/data1/mono/things/combatbench/baseline/humanoid21/runs/standing_ppo_20260409_225537/policy

python3 standing_ppo.py --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_ppo_20260409_084802/best_model.pt


python3 standing_ppo.py --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_ppo_20260409_225537/best_model.pt



python3 /data1/mono/things/combatbench/baseline/humanoid21/standing_grpo.py --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_20260410_084850/best_model.pt



python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_20260410_101336/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_20260410_101336/policy \
  --video standingtest8.mp4


python3 -m envs.humanoid21.run_round_turbulence \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_20260410_101336/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_20260410_101336/policy \
  --video standingtest19.mp4


nohup python3 /data1/mono/things/combatbench/baseline/humanoid21/standing_with_turbulence_grpo.py --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_20260410_101336/best_model.pt > standing_with_turbulence_grpo.log & 


python3 -m envs.humanoid21.run_round_turbulence \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_20260412_002901/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_20260412_002901/policy \
  --video standingtest22.mp4


nohup python3 /data1/mono/things/combatbench/baseline/humanoid21/standing_with_turbulence_grpo_rtg_tune.py --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_20260412_002901/best_model.pt > standing_with_turbulence_grpo_mid_rtg_tune.log & 



# 用RTG Tune训练出来的，可以稳稳站立
/data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_rtg_20260413_000056/policy


things/combatbench/baseline/humanoid21/runs/standing_grpo_rtg_20260413_000056/policy

python3 -m envs.humanoid21.run_round_turbulence \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_rtg_20260413_000056/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_rtg_20260413_000056/policy \
  --video standingtest24.mp4


nohup python3 standing_grpo_rtg_tune_turbulence.py --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_grpo_rtg_20260413_000056/best_model.pt > standing_grpo_rtg_tune_turbulence.log &


things/combatbench/baseline/humanoid21/runs/simple_standing_20260428_001634/policy


python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/simple_standing_20260428_001634/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/simple_standing_20260428_001634/policy \
  --video standingsimple.mp4


/data1/mono/things/combatbench/baseline/humanoid21/runs/perturbed_standing_20260428_003519/policy

# 可以平衡
python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_turbulence_stage1_ppo_penalty_20260414_181518/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_turbulence_stage1_ppo_penalty_20260414_181518/policy \
  --video perturbed_ppo_penalty.mp4

# 效果不好
python3 -m envs.humanoid21.run_round_turbulence \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_turbulence_stage1_ppo_balance_dense_20260415_004751/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/standing_turbulence_stage1_ppo_balance_dense_20260415_004751/policy \
  --video perturbed_ppo_penalty2.mp4






python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/simple_standing_20260428_002039/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/simple_standing_20260428_002039/policy \
  --video standingsimple3.mp4


python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/stage1_20260430_093352/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/stage1_20260430_093352/policy \
  --video stage12.mp4

  /data1/mono/things/combatbench/baseline/humanoid21/runs/stage1_20260430_093352/policy







我总体的目标是训练一个Baseline策略来实现两个机器人搏击。
总体的思路是通过课程学习式的强化学习。
课程设计如下：
课程一. 【扰动下的】交叉支撑平衡  最基本的 ; 失衡的判断是是否有另外的支撑点
课程二. 在一的基础上，接近对手【到一个范围内，在这个范围内损失不变】
课程三. 在一二的基础上，净胜伤害【越大越好】
/data1/mono/things/combatbench/baseline/humanoid21/common.py
中已经实现了一些基础的东西，涉及到stage1.py的已经经过了基本的验证。其它的还有没。

当前通过
  /data1/mono/things/combatbench/baseline/humanoid21/stage1.py
这个脚本训练的模型，可以实现双脚走动并保持平衡（姿势比较怪异）

通过
python3 -m envs.humanoid21.run_round \
  --policy-a /data1/mono/things/combatbench/baseline/humanoid21/runs/stage1_20260430_093352/policy \
  --policy-b /data1/mono/things/combatbench/baseline/humanoid21/runs/stage1_20260430_093352/policy \
  --video stage12.mp4
可以生成视频，看到模型训练的效果。

我想实现一个统一的训练脚本，直接实现三个课程。
大概的思路就是从奖励函数下手。如果。如果还没有学会第一阶段，然后这时候基本上就只会关注到第一阶段的。这个奖励或者惩罚。同样如果呃第1阶段学会了，还没有学会第2阶段，那就只会关注到第2阶段的奖励或者惩罚。然后顺延到第三阶段。之所以这样做，是因为。如果我通过课程学习的方式去。课程学习的方式去训练，然后训练完第一阶段之后，然后用这个模型去学第2阶段，很可能。他学第2阶段的时候，就把第一阶段的东西给忘了。然后连第一阶段的这种交叉腿支支撑都不会了。所以我希望设计这样一种统一的，就是课程学习式的这种奖励的方式，然后。来进行训练。然后这个奖励我觉得也许是可以比较骇客的。就是。可能可以这个三种奖励的信息在rollout的时候都收集。然后具体用哪一种呢，就用，根据数据来决定，就是通过数据来判断当前应该是学习哪一阶段。这样的话就不用去管他究竟是在第几阶段，就。就不用去刻意安排它是在第几阶段，然后就只根据他当前的实际的情况来判断就可以了。
然后请帮我做一个计划。分析一下可行性和风险。




再添加一个Observer奖励插件，功能是对于每一步做出交叉支撑的奖励，原理是：                                        
  1。 初始双脚同时着地，设置一个容忍时间，过了容忍时间如果没有一只脚离开地面，则开始惩罚；                        
  2。 当有一个只脚离开地面后，开始计时，有一个最短离地时间和最长离地时间，                                        
  在这个时间之间不惩罚，不在这个时间之内落地， 都进行惩罚，偏离越大，惩罚越重                                     
  3。 这只脚落地后会有一个双脚同时着地的最短和最长时间，不在这个时间段进行惩罚，时间差的越多，惩罚越重。          
  4。 记录上次抬起的是哪一只脚， 如果下次抬起的脚不对，进行惩罚。 