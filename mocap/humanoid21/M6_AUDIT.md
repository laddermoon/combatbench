# M6 Audit —— CMU 定向扩充 + HDM05 接入

日期：2025-10-15
范围：运动库扩容（数据库方向，非任务训练）

## 结果总览

| 项 | M5 末 | M6 末 |
|---|---|---|
| 库条目 | 168 | **1255** |
| 机器人片段 | 112 | **565**（cmu 126 + hdm05 439 + 镜像按 derived 计 125） |
| 原始片段时长 | ~17.8 min | ~35 min（cmu ~18 + hdm05 ~17） |
| 标签 | 31 种，长尾 | 56 种，punch 144 / kick 150 / footwork 195 / stand_up 87 |
| 数据源 | CMU | CMU + **HDM05**（第二个源接入，管线可复用验证） |

## W1 CMU 定向扩充（69 片段）

扫了 CMU 全库 109 subjects / 2128 trial 的官方描述（固化到
`cmu_motion_descriptions.json`），定向补齐最薄的 combat 类：
- feint/防守步法 +18（subject 78/102 篮球防守式步法）
- 起身变体 +12（140 趴/侧/仰、139、111/113）
- 滚翻 +5（127/128，新类 roll）
- 格挡/突进 +10（144）、踢打 +8（143/74/02）、闪避 +6（77 duck、76 补齐）
- 跳踢/真摔 +6（88/90，新类 fall）、跑/太极 +4

质量：中位残差 0.137 与库存持平；`cmu_76_07` 为 20 帧坏数据标 rejected。
用户视觉验收通过（compare_m6，18 条代表）。

## W2 规则切分器

`segment.py --detect <event>`：kick=脚速 / punch=手速 / jump=双脚离地 /
block=手高 / ground=骨盆贴地。干跑验证 13_17→18 拳、135_07→5 踢、
141_04→3 跳、144_07→6 格挡。（commit 3731d465）

## W3 HDM05 接入

### 数据源

- 官方 MPI 站 403 + HF 私有 + Wayback 下线 → 用户提供的 GitHub release
  `fabiocarrara/mocap` `hdm05-mocap-data.tar.gz`（1.1GB，进 `/data1/dev/hdm05/`）
- 内容是**预处理 normPOS 坐标流**（非 ASF/AMC）：`HDM05-15`/`HDM05-122`
  两套 `.data`，每条记录 = 已切好的带标签片段（key = `take_cls_start_n`），
  每帧 31 个 3D 标记点，120Hz
- HDM05-122-specific：**2345 片段 / 130 细类**；本次导出战斗相关 ~70 类 439 条

### 格式语义（实测验证）

- pt0 恒为原点 = 骨盆锚点；pt1-30 按 ASF 骨序（骨名与 CMU 完全一致）
- 坐标系同 CMU：y 上、+x 左、z 前（kickL/kickR 标签左右互验确认无翻轴）
- **normPOS = 根相对 + 偏航归一化**：世界位移和朝向已剥离
  → 走路/转身/转圈类只剩"原地姿势"；打击/起身/跳跃类语义完整

### 适配管线（`hdm05_retarget.py`）

位置流无旋转角，需合成 IK 输入。踩过的坑（都是标记点语义≠关节中心）：

| 坑 | 症状 | 修法 |
|---|---|---|
| lfemur 是大腿中段标记 | 髋锚点错，腿目标 0.38m vs 机器人 0.79m | 髋锚点改用 `lhipjoint` |
| ltibia 是踝上 1.4 单位标记 | "小腿方向"实际是脚面方向 → 膝锁死 | `leg_dir='thigh'`：用大腿方向（中点-髋）对比 robot thigh→knee |
| 骨盆落差标定 k | 腿目标短 9%，全程蹲 | 改用髋线→足底落差标定腿长 |
| lfoot 是贴地脚跟标记 | 踝被压地面 | 踝目标抬 +0.85 缩放单位 |
| 只抬踝不抬趾 | 脚尖向下（踮脚） | distal(toes) 同步抬 |
| lhumerus 是三角肌标记 | 手-肩目标偏高 | 肩锚点向锁骨回移 30% |
| 躯干/骨盆无旋转 | F 缺失 | 肩线+脊柱估计 F[thorax]，髋线+脊柱代理 F[root] |
| 骨盆固定原点 | root_pos 无高度 | 骨盆高 = 最低足落差合成 |

### 质量

- 正常类中位残差 0.09-0.5（punch 0.18-0.43、kick 0.24-0.47、walk/hop 0.09-0.3）
- **48 条标 `floor_pose_degenerate_position_only` + low confidence**：
  lieDown/standUpLieFloor/sitDownFloor/standUpSitFloor 躺姿下骨盆高度与
  躯干朝向不可恢复（格式固有缺陷，非 bug）
- 全部 hdm05 片段带 `inplace_source_no_root_motion` issue
- 用户视觉验收通过（compare_hdm05_v5，22 条代表）→ 415 条 L1

### index.json 事故记录

`hdm05_export` 初版把条目写到 index 顶层，而 `export_library` 用
`index["clips"]` 命名空间 → 误判"CMU 条目丢失"。实际只是命名空间
不一致，已合并修复并统一 `hdm05_export` 写入路径。
教训：**跨脚本共享 index.json 时先确认 schema 根键**。

## 使用方式

```python
from mocap.humanoid21.library import MotionLibrary
lib = MotionLibrary()
lib.list_clips(labels=["kick"])     # 150 条（cmu + hdm05 混编）
lib.list_clips(kind="original")     # 565
d = lib.load("hdm05_3146_88_2895_127")
```

新增导出：
```bash
PYTHONPATH=. python3 -m mocap.humanoid21.hdm05_export \
    --classes kickL kickR --per-class 6 --jobs 8 [--mirror]
PYTHONPATH=. python3 -m mocap.humanoid21.hdm05_export --keys <take_cls_start_n>
```

## 待办（backlog）

- HDM05 镜像导出未做（数据源自带 L/R 变体，价值低；`--mirror` 已实现备用）
- 剩余 ~1900 条 HDM05 片段未导出（grab/deposit/climb/stairs/日常类）
- HDM05 片段的物理跟踪验收（position-only 源对接触语义弱，谨慎期待）
- 判别器可用 hdm05 正样本重训（AMP 特征本来就去偏航/骨盆相对——即插即用）
- floor-transition 类如需可用，得换有旋转角的原始 HDM05 AMC 源
