仿真的核心业务流程：
1. 设置初始位置。
2. 不断执行动作Step 
3. 在每个动作步中执行几个物理仿真步。
4. 判断结束。

数据分类：
1. 仿真计算需要的数据， 这又可以分成：
1.1 场景数据（仿真资产）、配置参数（PD参数等） 静态的
1.2 状态数据 动态的，会有一个初始状态，然后会通过外部输入和内部仿真计算进行更新 （物理Step级）
1.3 外部输入 （动作Step级）
2. 不是仿真计算必须的数据，理论上的衍生的，用于渲染、计算Reward等
2.1 
数据粒度： 
1. 物理步级
2. 动作步级
3. Episode级


整体设计逻辑

# OpenSimulator 
抽象类， 所有的仿真环境都实现这个抽象类
执行仿真，并且向外开放数据，让外部可以访问和修改仿真状态数据，以便实现多种能力（比如观测、扰动、数据记录、Reward计算等）。
接口方法： 
set_action 接收动作指令
physical_step 物理步推进
get_sensors  获取传感器数据
get_static_data   获取静态数据： 比如机器人、场景、配置参数等
get_state 获取状态数据： 比如关节角度 速度 受力等
set_state 修改状态数据： 对于状态数据进行修改(同时修复 物理引擎内部的运动学缓存（正向运动学结果）、动力学缓存（雅可比矩阵、惯性矩阵）、碰撞检测的 Bounding Box 等， 避免不一致)
get_broadcastview_image 获取当前状态下转播视角的观测图片


# Hook 
实现如下的功能： 做为Hook在某个指定时间点被调用
class Hook:
    def invoke(self, 
    invoke_type(以下6种之一 ：pre_action_step , post_action_step , pre_phy_step , post_phy_step , pre_episode , post_episode) , 
    f_get_action, 
    f_get_static_data ,  
    f_get_sensor_data , 
    f_get_core_state , 
    f_get_derived_state , 
    f_set_core_state (如果不想让修改， 给None), 
    f_set_action (如果不想让修改， 给None)): -> bool (teminate flag)
       

# SimRunner
框架类，整合OpenSimulator和Hooks（对于每个Hook注明其Invoke时机）形成完整的Env。 关键一点，它并不负责实现Gym接口，没有Reward。 
这只是一个工具架构。


# GymBuilder
创建可供强化学习训练的环境。
计算Reward和Observation等。
