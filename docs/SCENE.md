# 场景规范

## 坐标规范
地面中心点的坐标为0，0，0
Z轴正方向为向上。
X轴正方向为向东。
Y轴正方向为向南。

## 墙面、地面、天花板规范
使用PLANE。
定义方式： 四个角的x,y,z + 贴图路径， 定义的是内表面。
自动计算另一面的位置，要保证让内表面面更接近 （0，0，1） 

## 帖图规范
当要用某一张图片覆盖在某一个平面时，需要的做法是，只用一张图片（而不是重复拼接），覆盖完整平面，并且保持图片完整（不剪裁），可以缩放，可以比例变形。


## 相机规范
除了朝上朝下的相机，其余相机只沿一个水平轴做俯仰。
相机画面一律横屏，分辨率640*480
相机用两个参数定义：
1. 所在位置（x,y,z）。
2. 看向位置(x,y,z)。


## 使用JSONL文件定义各个元素
示例如下
{"type": "plane", "name": "地面", "corners": [[-3.05, -3.05, 0], [3.05, -3.05, 0], [3.05, 3.05, 0], [-3.05, 3.05, 0]], "texture": "textures/floor.png"}
{"type": "plane", "name": "天花板", "corners": [[-3.05, -3.05, 6.10], [3.05, -3.05, 6.10], [3.05, 3.05, 6.10], [-3.05, 3.05, 6.10]], "texture": "textures/ceiling.png"}
{"type": "plane", "name": "南墙", "corners": [[-3.05, -3.05, 0], [3.05, -3.05, 0], [3.05, -3.05, 6.10], [-3.05, -3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "plane", "name": "北墙", "corners": [[-3.05, 3.05, 0], [3.05, 3.05, 0], [3.05, 3.05, 6.10], [-3.05, 3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "plane", "name": "西墙", "corners": [[-3.05, -3.05, 0], [-3.05, 3.05, 0], [-3.05, 3.05, 6.10], [-3.05, -3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "plane", "name": "东墙", "corners": [[3.05, -3.05, 0], [3.05, 3.05, 0], [3.05, 3.05, 6.10], [3.05, -3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "light", "name": "西南角灯", "position": [-3, -3, 5]}
{"type": "light", "name": "东南角灯", "position": [3, -3, 5]}
{"type": "light", "name": "西北角灯", "position": [-3, 3, 5]}
{"type": "light", "name": "东北角灯", "position": [3, 3, 5]}
{"type": "camera", "name": "西南角摄像机", "position": [-3, -3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "东南角摄像机", "position": [3, -3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "西北角摄像机", "position": [-3, 3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "东北角摄像机", "position": [3, 3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "南墙摄像机", "position": [0, -3, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "北墙摄像机", "position": [0, 3, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "西墙摄像机", "position": [-3, 0, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "东墙摄像机", "position": [3, 0, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "天花板摄像机", "position": [0, 0, 6], "look_at": [0, 0, 0]}