# 场景规范 (Scene Specification)

## 1. 坐标系规范
- 地面中心点的坐标为：`(0, 0, 0)`
- **Z轴：** 正方向为向上。
- **X轴：** 正方向为向东。
- **Y轴：** 正方向为向南。

## 2. 墙面、地面和天花板
- 使用 MuJoCo 的 `plane` 几何体（geom）构建。
- **定义方式：** 通过指定四个角的 `x,y,z` 坐标以及贴图路径来定义（这定义了内表面）。
- 系统会自动计算外表面的位置和法线，以确保内表面的法线始终朝向房间内部。

## 3. 贴图规范
- 当要用图片覆盖平面时，做法是：只用**一张**图片被拉伸（不平铺、不剪裁）以覆盖整个平面。
- 图片会根据平面的长宽比例进行自适应缩放。

## 4. 摄像机规范
- 除了垂直朝上/朝下的摄像机，其他摄像机只沿着一个水平轴做俯仰运动。
- 摄像机画面一律设定为横屏，标准分辨率为 `640x480`。
- 摄像机由两个参数定义：
  1. 所在位置 `(x, y, z)`。
  2. 看向的目标位置 `(x, y, z)`。

## 5. 使用 JSONL 定义场景元素
竞技场是通过读取 `scene_elements.jsonl` 文件以程序化的方式构建的。

定义示例：

```jsonl
{"type": "plane", "name": "floor", "corners": [[-3.05, -3.05, 0], [3.05, -3.05, 0], [3.05, 3.05, 0], [-3.05, 3.05, 0]], "texture": "textures/floor.png"}
{"type": "plane", "name": "ceiling", "corners": [[-3.05, -3.05, 6.10], [3.05, -3.05, 6.10], [3.05, 3.05, 6.10], [-3.05, 3.05, 6.10]], "texture": "textures/ceiling.png"}
{"type": "plane", "name": "south_wall", "corners": [[-3.05, -3.05, 0], [3.05, -3.05, 0], [3.05, -3.05, 6.10], [-3.05, -3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "plane", "name": "north_wall", "corners": [[-3.05, 3.05, 0], [3.05, 3.05, 0], [3.05, 3.05, 6.10], [-3.05, 3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "plane", "name": "west_wall", "corners": [[-3.05, -3.05, 0], [-3.05, 3.05, 0], [-3.05, 3.05, 6.10], [-3.05, -3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "plane", "name": "east_wall", "corners": [[3.05, -3.05, 0], [3.05, 3.05, 0], [3.05, 3.05, 6.10], [3.05, -3.05, 6.10]], "texture": "textures/wall.png"}

{"type": "light", "name": "sw_light", "position": [-3, -3, 5]}
{"type": "light", "name": "se_light", "position": [3, -3, 5]}
{"type": "light", "name": "nw_light", "position": [-3, 3, 5]}
{"type": "light", "name": "ne_light", "position": [3, 3, 5]}

{"type": "camera", "name": "sw_camera", "position": [-3, -3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "se_camera", "position": [3, -3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "nw_camera", "position": [-3, 3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "ne_camera", "position": [3, 3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "south_camera", "position": [0, -3, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "north_camera", "position": [0, 3, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "west_camera", "position": [-3, 0, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "east_camera", "position": [3, 0, 3], "look_at": [0, 0, 0]}
```
