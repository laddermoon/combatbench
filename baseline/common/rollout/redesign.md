主要分成三块：
Episode：一个Episode数据的定义。
EpisodeCollection：一个可以管理多个Episode的数据的组件。 支持Save和Load。

EpisodeRecorder： 基于FrameWork的Recorder机制，每次生成一个Episode。 实现上要参照 envs/framework/recorder.py 中 EpisodeBufferRecorder
ParallelRollouter: 一个并行收集大量Episode的工具。 输入EnvBlueprint， Policy， Seeds. 得到Episode Collection。
       


典型应用方式的设想：
1. 每次Rollout最重要的是有一个EnvBlueprint文件。 
2. 当需要调试时，用户可以直接用 envs/framework/round_runner.py + BaseFrameRecorder 进行Rollout复现，来观察结果。这样就为训练和调试留好了标准的切口和工具。
3. Rollout和调试只有Recorder不同，这样就保证了数据的一致性。
4. 训练时拿到的是EpisodeCollection，这也是训练的Rollout之间的边界。
5. EpisodeCollection可以序列化， 这样训练的调试也可以单独进行。



