
输入：
环境的Blueprint，带参
baseline/humanoid21/blueprints/curriculum_env.yaml
初始的Policy的Blueprint
baseline/humanoid21/blueprints/init_policy.yaml


整体流程：
1. 初始化准备：
   根据初始的PolicyBlueprint构建可训练的Policy
   初始Stage 1
2. 如果有resumefrom , 则加载数据， 可以更新Step或者模型权重等等
3. 进入训练迭代
每次训练迭代流程
   1. 把Policy导出成PolicyBlueprint (for rollout)
   2. 准备Rollout的Jobs
   3. rollout得到Rollout数据, 使用ParallelRollouter
   4. 数据处理，不同Stage奖励处理不一样。得到每一步的训练奖励
   5. 更新Policy。
   6. Eval，判断下一步进行哪一个Stage
   7. 如果是最优则Checkpoint，定期也要Checkpoint，Checkpoint要保存训练过程中的所有信息





