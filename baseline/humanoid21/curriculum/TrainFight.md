参照跟随策略的做法：

Env使用：
/data1/mono/things/combatbench/baseline/humanoid21/blueprints/fight_env.yaml
与Follow不同之处，添加计分模块。去掉了RandomMove。

有要有一方倒地，Episode就结束。先不去打击一个倒地的对手。 TODO： 需要拓展Imbalance Plugin的能力，支持双手同时监控。


Rollout策略：
使用组合策略，Fight、Follow和Recover三个策略组合。
可以Fallback到Follow 或者 Recovery ， 如果同时满足优先到Recovery。
当距离超过1.3米时，Fallback到Follow， 当距离达到1米时，再切回到Fight策略。中间有个缓冲带。
在Follow策略时，也可以Fallback到Recover策略。
训练时只有Fight策略的数据，切成SubEpisode。

对手策略：
对手使用训练好的Follow模型（先用这个）。 
后续考虑：Self Play 或者 对手池。


奖励设计：
在跟随奖励之上加一个攻击奖励。
还有一个Fallback到Follow策略的惩罚， 与Fall或者Fallback到Recover策略的逻辑一致。


训练起点：
从跟随策略开始训练。

