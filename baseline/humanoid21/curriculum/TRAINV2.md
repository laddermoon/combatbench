

PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --experiment basic_balance_v2 &> basic_balance_v2.log & 

python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py basic_balance_v2.log  --watch


/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_v2_20260617_111918/checkpoints/checkpoint_u00685.pt
用时大概7小时。


PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --experiment balance_recover_v2 --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_v2_20260617_111918/checkpoints/checkpoint_u00685.pt &> balance_recover_v2.log & 



python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py balance_recover_v2.log  --watch


/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_v2_20260617_195446/checkpoints/checkpoint_u03480.pt
训练大概12小时。  一个可能的改进是把MaxStep调小。


PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --experiment balance_recover_plus_v2 --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_v2_20260618_131501/checkpoints/checkpoint_u04875.pt &> balance_recover_plus_v2.log & 


python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py balance_recover_plus_v2.log  --watch


撑地后站立
http://180.76.152.227:8999/curriculum_balance_recover_plus_v2_20260618_225956/videos/u07245.mp4

[eval 8845] [ep mean_length=88.898 survived=0.844 level=5.000]  [new_best]



#生成Gate训练数据
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 100000 \
  --noise-std 0.08 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_v2_u08845_10w \
  --policy-path /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_v2_20260618_225956/policy_exports/u08845



#下面这样得到的数据比例相差太悬殊
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data_refine.py \
  --num-episodes 100000 \
  --noise-std 0.08 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_v2_u08845_mixlevel \
  --policy-path /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_v2_20260618_225956/policy_exports/u08845


#训练Gate模型
PYTHONPATH=. python3 baseline/humanoid21/curriculum/train_gating_network.py \
  --epochs 500 \
  --batch-size 4096 \
  --hidden-dims 512 256 128 \
  --lr 5e-4 \
  --data-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_data_v2_u08845_10w \
  --output-dir /data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model_v2_u08845_10w


#训练Follow模型
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --v2 --experiment follow_v2 --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_v2_20260618_225956/checkpoints/checkpoint_u08845.pt &> follow_v2.log & 



python3 baseline/humanoid21/curriculum/analyze_follow_logs.py follow_v2.log --watch



#训练Fight模型
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --v2 --experiment fight_v2 --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_follow_v2_20260620_132447/checkpoints/checkpoint_u09236.pt &> fight_v2.log & 

9236
备用 9244, 9168


python3 baseline/humanoid21/curriculum/analyze_fight_logs.py fight_v2.log --watch


export COMBAT_SCORE_DEBUG_FILE=/data1/mono/things/combatbench/logs/combat_debug.jsonl
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --v2 --experiment fight_v2 --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_fight_v2_20260620_185210/checkpoints/checkpoint_u10058.pt &> fight_v2_new.log & 

python3 baseline/humanoid21/curriculum/analyze_fight_logs.py fight_v2_new.log --watch


得到
/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_fight_v2_20260621_014809/checkpoints/checkpoint_u10516.pt

看一下可积分的逻辑是否正确，作为一个to do就是积分里面应该结合击打的轻重.  还有就是积分的这个分数是否正确，对于一个总总共100分的对我来说是不是一个重击，一下子就打没了20分.
Done

Platform里面如何在前端能够看到视频，这个应该要解决掉.让能够在首页上面看到视频
Done


然后还有一个是起身站立的这个模型.




在上面的基础上添加对手池
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --v2 --experiment fight_v2_oppopool --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_fight_v2_20260621_014809/checkpoints/checkpoint_u10516.pt &> fight_v2_oppopool.log & 

