# M0 资产审计报告

> 状态：进行中（随工作包推进增量更新）
>
> 对应大纲：`README.md` §9 M0「资产审计与契约」。
> 退出标准：来源可追溯，片段真实内容与关节语义明确。

## W1 机器人模型事实核查（battle_circular_v2.xml）

事实来源：MuJoCo 3.8.0 编译后模型（`mocap/humanoid21/robot_spec.py` 提取，
机器可读结果见 `robot_spec.json`，经 `Humanoid21Meta.validate()` 通过）。

### 控制语义（已核实，与 CONTROLSPEC.md 一致）

- action ∈ [-1,1]²¹ → 目标角 = `mid + action·half_range`（mid/half 来自 jnt_range）
- 固定 KP/KD 的 PD 力矩 → `ctrl = clip(torque/gear, ctrlrange)`
- 物理步 0.002 s（500 Hz），`phy_steps_per_action=25` → **控制频率 20 Hz**
- 自检：standing 初始 action 解码后与 `joint_pos=0` 一致（max err 4.4e-5 rad）

### 关节轴事实（世界系，standing 姿态；此时 root 朝 +x）

| 语义组 | 关节 | 世界轴 | 限位(°) | gear | 备注 |
|---|---|---|---|---|---|
| 躯干 | abdomen_z/y/x | z / y / x | [-45,45] [-75,30] [-35,35] | 40 | 干净主轴 |
| 右腿 | hip_x / hip_z / hip_y | +x / +z / +y | [-30,10] [-60,35] [-150,20] | 40/40/120 | 干净主轴 |
| | knee | **−y** | [-160,2] | 80 | 负轴 |
| | ankle_y / ankle_x | +y / **(x,z)混合(0.89,0,0.45)** | [-50,50] ×2 | 20 | ankle_x 非纯轴 |
| 左腿 | hip_x / hip_z | **−x / −z** | 同右 | 40 | 镜像负轴 |
| | knee / ankle_y | −y / +y | 同右 | 80/20 | |
| | ankle_x | **(−0.89,0,−0.45)** | [-50,50] | 20 | 混合轴+镜像 |
| 右臂 | shoulder1 | **(0.82,0.41,0.41)** | [-85,60] | 20 | 混合轴，非纯屈伸 |
| | shoulder2 / elbow | **(0,−0.71,0.71)** | [-85,60] / [-100,50] | 20/40 | 同轴两关节 |
| 左臂 | shoulder1 | (−0.82,0.41,−0.41) | [-85,60] | 20 | 镜像 |
| | shoulder2 / elbow | (0,−0.71,−0.71) | 同右 | 20/40 | |

### 关键发现

1. **Hamstring 肌腱耦合**：`0.5·hip_y − 0.5·knee ∈ [−0.3, 2] rad`，即
   `hip_y − knee ∈ [−34.4°, +229.2°]`。髋部深屈（hip_y 负向）+ 膝伸直组合违反约束
   ——例如"直腿高抬"类人动作物理不可行。重定向必须逐帧检查，而不是只查单关节限位。
2. **手臂不是直链**：shoulder 轴为混合轴且 upper_arm→lower_arm→hand 的
   偏移是折线（0.18,−0.18,−0.18)→(0.18,0.18,0.18)。**逐关节角度复制（retarget_v2
   思路）在手臂上必然错误**；M1 必须用基于世界系姿态/位置的 IK 式重定向。
3. **语义标签纠正**：retarget_v2 把 CMU femur ry(abduction) 映射到 hip_z，
   但实际 hip_x（绕世界 x）才是侧展/roll，hip_z 是扭转/yaw——语义错位。
4. **JOINT_LIMITS/JOINT_ORDER 与当前模型完全一致**（v1→v2 限位未变；
   与 meta.CONTROLLED_JOINTS 顺序一致，max diff < 1e-14 deg）。旧表的数值可复用，
   但映射语义需按真实轴重推导。
5. 体形：总质量 40.8 kg，腿长 thigh≈0.40 m / shin≈0.39 m，与人比例明显不同；
   root 为 freejoint（root_a/root_b），nq=56, nv=54, nu=42（双机）。
6. contact excludes 仅 waist_lower↔thigh（左右×双机）——其余自碰撞均生效，
   重定向姿态需防自碰。
7. 旧代码勘误：`baseline/humanoid21/mocap/visualize.py` 引用
   `Humanoid21Meta.ARENA_XML`——该属性实际在 `Humanoid21Simulator.ARENA_XML`，
   旧可视化脚本在当前代码下 import 即报错，属过时件。

## W2 存量资产盘点

工具：`mocap/humanoid21/audit_assets.py`；结果：`assets_index.json`（sha256/帧数/质量指标全量）。

### 原始数据（baseline/humanoid21/mocap/data/raw/，已在 git）

- 4 个 ASF（subject 13/14/15/17）：31 bones、56 DOF、units_length=0.45（inch 系）、
  units_angle=deg、root order `TX TY TZ RX RY RZ`——四者骨架定义一致。
- 29 个 AMC：全部解析成功，无缺帧缺骨、无非有限值。
  时长 3.5–77.8 s（按 CMU 标准 120 Hz 假定值折算，待 W4 官方描述交叉确认）。
- 时长分布：13_01~24（9–48 s）、14_01~03（~46 s）、15_13（78 s）、17_10（23 s）。

### 重定向产物（data/retargeted/，已在 git）

- 29 组 `(T,21)` 角度 npy + 29 个 `_norm` 归一化版本，全部有限、零限位违例
  （容差 1e-4°，最大超界 7e-15 即浮点噪声）、无冻结帧。
- **6.5–15.6% 的关节值恰好压在限位上**——DOF-copy 重定向的裁剪痕迹，
  人体关节角超出机器人限位时被静默截断（主要是肩/髋大幅动作帧）。
- 数据只有关节角：无根部轨迹、无关键点、无接触、无速度。
  **结论：可视为"低可信度的运动学近似"，不满足大纲对参考运动的最低信息要求。**

### 旧代码可复用性裁定

| 文件 | 裁定 | 依据 |
|---|---|---|
| `asf_parser.py` | 复用 | 4/4 ASF 解析正确，字段完整 |
| `amc_parser.py` | 复用 | 29/29 AMC 解析正确；依赖"AMC 行序=ASF dof 序"约定（CMU 成立） |
| `fk.py` | **弃用**（W3 复核证伪） | 三处约定与官方 viewer 不符（axis 序/direction 语义/父子几何归属），渲染怪异实锤；由 `mocap/humanoid21/ref_fk.py` 取代并经静止位型验证 |
| `retarget.py` | 参考实现候选 | 走"FK→相对旋转→分解到真实关节轴"路线，含混合轴处理，思路正确；未经数值验证 |
| `retarget_v2.py` | 过时 | DOF 直拷+符号猜测；肩/踝混合轴与 hip_x/hip_z 语义错位；当前 npy 均由它生成 |
| `visualize.py` | 损坏 | `Humanoid21Meta.ARENA_XML` 不存在（实际在 simulator），且依赖交互 viewer |
| `prepare_data.py` | 模式参考 | 写死 subject 列表与代理地址；下载流程可参考 |

## W3 片段内容核验

### 官方描述（证据：`cmu_motion_descriptions.json`，2026-10-13 抓取）

- subject 13 = "various everyday behaviors"：**仅 13_17、13_18 是 boxing**
- subject 14 = "various everyday behaviors"：14_01/02/03 是 boxing ✓
- subject 15 = "everyday + dance"：仅 15_13 是 boxing ✓
- subject 17 = "different walking styles"：仅 17_10 是 boxing ✓（可能是行进间拳击）
- CMU 页面帧率列均为 **120** ——确认 AMC 采样率 120 Hz。

**纠正 prepare_data.py 的隐含假设**：`BOXING_SUBJECTS` 把 subject 13 的 24 个片段全部当
拳击——实际只有 2 个。真实拳击片段共 7 个：`13_17, 13_18, 14_01, 14_02, 14_03, 15_13, 17_10`。
其余 22 个是日常动作（坐站/喝水/跳/笑/擦窗/扫地/指挥交通）——对先验库仍有价值
（非拳击的自然动作多样性），但不能再标错类别。

### 特征扫描（`content_scan.py` → `content_scan.json`，v2 修正 FK 后重算）

**尺度说明**：CMU 单位暂按髋高≈0.95 m 校准为 **0.11 m/unit**（M1 精标）。
与官方描述交叉核对一致：

- 7 个 boxing 片段手 p99 峰速 **5–11 m/s** + 小范围脚步移动——符合拳击
- 行走片段根速度 1.0–1.7 m/s、位移 2.9–3.8 m ✓
- 13_11/13/19 "forward jump"：位移 ~1.2–1.4 m ✓
- **02_05（官方 punch/strike）修正后 p99 仍仅 1.1/2.5 m/s** → 目测裁决保留
- 13_14 "laugh" 修正后 R 手 p99 7.9 m/s——剧烈甩臂属实；标签核验留给目测
- foot-contact 启发式沿用远点高度阈值，足底贴地正常（min ≈ -0.05 m）

### 预览素材（v2 —— 修正 FK 后重渲）

**2026-10-13 FK 重大修正**：旧 `baseline/.../fk.py` 的三处约定与 CMU 官方
viewer 不符，早期预览视频 `preview/*.mp4` 全部作废：

1. `axis` 欧拉序写反（应为 `Rz·Ry·Rx`，x 先作用；旧代码用 `Rx·Ry·Rz`）
2. ASF `direction` 是全局向量，需 `dir_local = C⁻¹·dir` 转局部（旧代码直接当局部量）
3. **子关节应继承父骨的 dir/len 与父骨 DOF 后姿态系**（旧代码用了子骨几何——
   导致 lfemur 起点被错放整个股骨长度之外，骨架断裂怪异）

依据：CMU 官方 asf_amc_viewer 源码（`skeleton.cxx`/`display.cxx`/`transform.cxx`，
`glMultMatrixd` 列主序语义 + `rot_parent_current` 转置存储已核对）。
数值验证（`fk_check.py`）：静止位型下所有骨骼世界方向与 ASF 全局 direction
误差 ≤1e-15；父子关节连续性误差 0。修正后 `13_01` 末帧（站直）：髋 7.6 /
踝 0.4 / 头 13.6（CMU 单位），解剖比例正确——修正前手/脚位置漂移 1–4 个单位。

修正后实现在 `ref_fk.py`；`preview.py`/`content_scan.py` 已切换到它。
**W3 早期特征表的手速等数字全部基于错误 FK，已用 `content_scan.json` v2 重算**
（拳击 p99 手速 5–11 m/s 合理；绝对米标度为髋高校准 0.11 m/unit 的暂定值，
M1 再定标）。

新预览：`/data1/dev/combatbench_mocap/preview_v2/*.mp4`（不入库）。

## W4 许可核验

CMU Graphics Lab Motion Capture Database（mocap.cs.cmu.edu），2026-10-13 在线核验：

- **"This dataset of motions is free for all uses."**
- **"may be copied, modified, or redistributed without permission"**
- 免责声明：不保证数据质量；分类标签仅供参考（→ 我们的逐片段内容核验确实必要）
- 请求（非强制）：发表时注明 "obtained from mocap.cs.cmu.edu ... NSF EIA-0196217"，
  并将引文发给 jkh+mocap@cs.cmu.edu
- 结论：**许可不构成障碍，可自由使用与再分发**，index 中记录来源 URL + sha256 + 抓取日期

## W5 数据契约

见 `DATASPEC.md`（待执行）

## W6 最小动作集（提案 → 待用户确认）

依据：官方描述 + 修正后 FK 特征 + 预览视频（`/data1/dev/combatbench_mocap/preview_v2/`）。
对大纲 §5.1 各类的覆盖映射：

| 大纲类别 | 候选片段 | 说明 |
|---|---|---|
| 站立/准备姿态 | 拳击片段内的 guard 段；13_01–03 末段（起身到站立） | 由切片获得 |
| 直拳（左/右） | **13_17, 13_18, 14_01, 14_02, 14_03, 15_13, 17_10**（7 clips，4 个 subject） | 官方标注 boxing；手峰速 2.0–5.7 m/s |
| 收拳/换重心/恢复 | 同上拳击片段内部 | 待 M1 切片+阶段标注 |
| 走路/停止 | **07_01, 07_06, 07_12**（走/快走）; **02_01, 02_02**（走） | 新增下载，位移 1.4–1.9 m |
| 行进间拳击 | **17_10** | 官方分类 walking-styles 下的 boxing |
| 出拳变体 | ~~02_05~~ | 目测无法辨认+低速 → 剔除最小集 |
| 风格化步态 | **17_01**（愤怒走）、17_03（潜行） | 已确认纳入 |
| 起身 | **13_01–03**（sit→stand） | 已确认纳入 |
| 明确排除/低优先 | 13_07–09（喝水）、13_20–22（擦窗）、13_23–24（扫地）、13_14–16（笑，13_14 手速异常待目测）、13_10/12（踮脚取物）、13_04–06（坐凳） | 非目标动作但可先验多样性用 |

新增下载（许可已核验，存 `/data1/dev/combatbench_mocap/raw/`）：
02.asf+02_01/02_02/02_05.amc；07.asf+07_01/07_06/07_12.amc；17_01/17_03.amc（ASF 复用旧目录）。

### 用户复核清单 v2（预览 `/data1/dev/combatbench_mocap/preview_v2/<clip>.mp4`，修正 FK 后）

方括号内为手部 p99 稳健峰速时刻附近建议跳看（视频 30fps、源 120Hz 抽 4 帧）。
**注：v1 时刻表基于错误 FK，以下数字已重算**（峰值为瞬时尖峰含噪声，p99 更代表真实出拳速度）：

| clip | 时长 | 官方描述 | 手速 p99（L / R m/s） | 看点 |
|---|---|---|---|---|
| 13_17 | 40s | boxing | 5.7 / 5.7 | 直拳/抱架质量；R 峰 @2.0s、L 峰 @17.1s |
| 13_18 | 25s | boxing | 5.5 / 7.2 | R 峰 @12.0s |
| 14_01 | 47s | boxing | 5.1 / **10.4** | R 峰 @15.6s |
| 14_02 | 45s | boxing | 6.6 / 9.3 | R 峰 @31.8s、L 峰 @43.4s |
| 14_03 | 46s | boxing | 5.1 / **10.3** | R 峰 @25.3s |
| 15_13 | 78s | boxing | 6.2 / **10.3** | 最长拳击片段；R 峰 @48.2s |
| 17_10 | 23s | boxing | 6.4 / **11.4** | 行进间拳击；R 峰 @18.3s |
| 02_05 | 15s | punch/strike | **1.1 / 2.5** | ⚠修正后仍很慢——目测判是出拳还是格挡 |
| 07_01/06/12 | 2–3.5s | walk | ~4–6 | 普通走/快走；步速 1.0–1.7 m/s ✓ |
| 02_01/02_02 | ~2.6s | walk | ~2–4 | 普通走（另一个 subject） |
| 17_01 | 53s | walk angry | 4.5 / 4.2 | 甩手走；R 峰 @33.4s |
| 17_03 | 53s | walk stealthy | 2.4 / 2.8 | 潜行走 @48.2s |
| 13_01 | 19s | sit→stand | 2.4 / 1.6 | 起身恢复 |
| 13_14 | 20s | laugh | 5.9 / 7.9 | ⚠甩臂剧烈；确认非击打类标签 |

### 用户已答复（2026-10-13，基于 preview_v2）

- 13_17 可辨认拳击动作；其余抽查片段动作正常 ✓
- **02_05：看不出是什么动作（只见扭动）→ 判为低速手势/格挡类，不进最小集**；
  保留在资产库中，标签 `ambiguous_slow_strike`（official: punch/strike）
- **13_14：官方 laugh 属实（剧烈甩臂非击打）→ 不进最小集**，标签 `laugh`
- 可选类别已确认纳入：起身（13_01–03）+ 风格化步态（17_01/17_03）

### 最小动作集定稿（17 clips）

| 类别 | clips |
|---|---|
| 拳击 | 13_17, 13_18, 14_01, 14_02, 14_03, 15_13, 17_10 |
| 普通/快速行走 | 07_01, 07_06, 07_12, 02_01, 02_02 |
| 起身（sit→stand） | 13_01, 13_02, 13_03 |
| 风格化步态 | 17_01（angry）, 17_03（stealthy） |

站立/准备、收拳/换重心恢复由拳击片段内切片获得（M1 阶段标注）。

## M0 退出标准自查

| 标准 | 状态 | 证据 |
|---|---|---|
| 来源可追溯 | ✅ | assets_index.json（sha256）+ cmu_motion_descriptions.json + 许可记录 |
| 片段真实内容明确 | ✅ | 官方描述+修正 FK 特征一致；用户目测确认（preview_v2）；最小集 17 clips 定稿 |
| 关节语义明确 | ✅ | robot_spec.json：真实轴/限位/肌腱/PD 语义，含旧假设勘误 |
