# M5 审计 —— 库扩充、派生工具与标准接口

日期：2026-10-15。目标（README §9 M5）：扩充动作类别、派生/切分工具、
标准查询接口——验证"数据和接口复用成立，而非每任务重做"。

## 事件记录：数据管线被误删与恢复

`baseline/humanoid21/mocap/`（ASF/AMC 解析器 + 29 个原始数据文件）
在 `531a45d8` 中被整目录删除——一次只打算改 README 措辞的提交混入
了删除。恢复方式：解析器与 legacy FK 从 `53554054` 迁入
`mocap/humanoid21/parsers/`（管线自有位置），全部 import 改指；
raw 数据从 git 对象库解出到 `/data1/dev/combatbench_mocap/raw/`
（43 文件）。**教训：数据/工具与代码共仓时，`git add -A` 级别的
清扫会无声吞掉整条管线；mocap 工具链现已全部自包含在
`mocap/humanoid21/` 下。**

## W1 镜像工具（`retarget.py: mirror_frame` + `export_library --mirror`）

- 语义：CMU 系矢状面反射 x→−x（= MuJoCo y→−y，前进方向不变，
  左右对调）；位置取反 + l/r 骨骼语义对调（配对判定 = l/r 前缀
  双方都存在于骨骼表，中线骨如 lowerback 不受影响）+ 旋转帧
  MRM 共轭。镜像发生在 **IK 目标层**，非导出数据逐通道符号变换。
- 验证：`cmu_13_17_m` 与原片段逐关键点比对 = 精确镜像（残差 0.0000，
  hip_y 相关性 1.0）；56 个镜像片段 IK 残差与原片段逐位一致
  （对称性保证）。
- 产物：`cmu_*_m` clip，index `derivation: mirror_sagittal_x`，
  `source: derived`，不重复写 `_src` npz。

## W2 切分工具（`segment.py`）

- `--auto`：低活动度静默期（关节 |dq|<0.15rad/s 且根速度<0.25m/s
  持续 ≥0.4s）做切点；`--segments S-E:label` 手工区间。
- 输出 `*_segNN.npz`（数组切片、时间戳归零、lead_stable_frame
  重算）+ index `derivation: segment` + `frame_range`。
- 实测：`cmu_13_17` 全程连续活跃无静默段 → 自动切段正确地返回
  单段（拳击片段本就是持续动作）；手工模式用于语义切分。

## W3 库扩充：17 → 56 原始片段

新入库 39 段（全部通过 M1 管线：FK→IK→运动学检查→index）：

| 类别 | 片段 | 来源 |
|---|---|---|
| 踢（前踢/回旋/侧踢/膝击/足球踢） | 135_04/07/11, 144_05/06/09, 10_01-03, 111_19, 113_13, 141_14, 86_01/06 | CMU 武术/足球 subject |
| 拳（组合拳/左拳序列/弓步拳/打击） | 144_13/14/20/21, 135_09, 02_05, 02_07 | CMU 144/135/02 |
| 格挡（低格挡/刀手/左右 blocks） | 135_05/10, 144_07/26 | 空手招式 |
| 防守/后撤/闪避 | 76_01/02/03/04/09/11 | defense vignettes |
| 跳跃（前跳/侧跳/转体跳/跳远） | 13_11/19, 141_04/05/06 | |
| 无道具起身（地板） | 111_06, 113_08 | 替代有凳子的 13_01-03 |
| 弓步 | 144_17 | |

合计 index：168 条目 = 56 原始 + 56 镜像 + 56 源骨架。
标签分布见 `library.py summary`（kick 36、punch 33、walk 24、
boxing 21、block 12、defense/retreat/dodge/evade 各 3-12……）。

**已知低质量片段**（index `issues` 已标记，未剔除）：
`141_14`（腿残差 1.72，踢腿超出机器人髋活动度）、`141_04-06`
（跳跃踝残差）、`111_06`/`113_08`（地板起身腿残差 0.4-0.5）、
`135_01/06`（kata 手臂残差 0.3）。判定：保留作"L0+标注"，
物理验证前不升 L2。

## W4 标准接口（`library.py`）

`MotionLibrary`：`list_clips(labels=/kind=/level=)`、
`load(clip_id)`、`info(clip_id)`、`summary()`——标签/类型/验收
等级检索 + npz 加载 + 元信息。这是外部任务取数的入口。

## 判别器 v5（扩库重训）

- 正例 42,454 transitions（112 机器人片段）/ 负例合成族不变。
- acc 0.877；stress 报告 `prior/disc_v5_stress.json`。
- 训练日志 loss 显示 nan（仅打印路径问题，acc 曲线正常 0.66→0.88，
  待修：GP/BCE 聚合的日志行）。

## 验收对照

| 判据 | 结果 |
|---|---|
| 更多动作类别 | ✓ 7 个新类别方向（kick/punch/block/defense/jump/getup/lunge） |
| 过渡/起身 | ✓ 无道具地板起身 ×2（替代凳子依赖） |
| 镜像/切分工具 | ✓ 56 镜像片段精确对称；segment.py 两段模式 |
| 标准接口复用 | ✓ library.py 标签检索 + prior/observer 无需改代码即吃扩库 |
| 条件先验 | ⏸ 留作后续（数据面已备好 labels/segments） |

待用户复核：compare_m5/ 新片段渲染（135_07 回旋踢、76_01 边退
边打、111_06 起身、135_09 弓步拳、144_13 左拳序列、141_14 存疑）。
