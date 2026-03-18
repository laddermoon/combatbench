受控自由度为21 DoF的仿真环境。

仿真环境的包括：
1. 机器人所处环境：
   一个完全封闭的房间，由地面、四面墙、 天花板围合而成。
   房间高度：6.10米
   房间长宽：奥运会/业余比赛 (AIBA标准)：正式比赛规定为 6.10米 (20英尺) 见方。
   房间墙面，用整张图片进行贴图（拉伸缩放图片以适配墙面），things/combatbench/simulator_humanoid21/textures/wall.png
   房间地面，用整张图片进行贴图（拉伸缩放图片以适配墙面）, things/combatbench/simulator_humanoid21/textures/floor.png
   天花板，用整张图片进行贴图（拉伸缩放图片以适配墙面）, things/combatbench/simulator_humanoid21/textures/ceiling.png
2. 两个机器人
   与 things/combatbench/simulator_humanoid21/humanoid.xml（来自https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid.xml） 中的定义保持一致，除了颜色。
   一个红色，一个蓝色
   两个机器人处于房间中线上，分别距离中点一米。面对面站立。
   机器人初始状态，保持竖直站立。
3. 灯光
   四个光源，每个墙角一个，高度5米，
4. 固定摄像机
   每个墙角一个，高度4米， 朝向地面中心点（共4个）
   每面墙中间一个，高度3米， 朝向地面中心点（共4个）
   屋顶正中心一个，向下俯拍

