

OpenSimulator 抽象类， 所有的仿真环境都实现这个抽象类
执行仿真，并且向外开放数据，让外部可以访问和修改仿真状态数据，以便实现多种能力（比如观测、扰动、数据记录、Reward计算等）。
接口方法： 
set_action 接收动作指令
physical_step 物理步推进
get_sensors  获取传感器数据
get_static_data   获取静态数据： 比如机器人、场景、配置参数等
get_state 获取状态数据： 比如关节角度 速度 受力等
set_state 修改状态数据： 对于状态数据进行修改(同时修复 物理引擎内部的运动学缓存（正向运动学结果）、动力学缓存（雅可比矩阵、惯性矩阵）、碰撞检测的 Bounding Box 等， 避免不一致)
get_broadcastview_image 获取当前状态下转播视角的观测图片