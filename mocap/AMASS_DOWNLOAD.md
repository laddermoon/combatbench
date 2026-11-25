# AMASS 数据下载说明

> 类型：操作指南
>
> 配套脚本：`mocap/download_amass.sh`

## 1. AMASS 是什么

AMASS（Archive of Motion Capture as Surface Shapes, ICCV 2019, MPI-IS）把 24 个
分散的光学动捕数据集统一 MoSh 到 SMPL 人体模型参数空间：约 40+ 小时、1.1 万+
段动作、300+ 受试者。每个序列是一个 `.npz`（SMPL 参数），而非骨架关节角。

**注意格式差异**：本项目现有管线消费的是骨架动捕（CMU ASF/AMC、HDM05
normPOS）。AMASS npz 需要 `smplx` 库 + SMPL body model 先 FK 成关节位置，
才能接入现有重定向流程。若只需要 CMU 来源，可直接抓原始 ASF/AMC
（mocap.cs.cmu.edu，免注册、许可自由）。

## 2. 前提：注册账号

到 <https://amass.is.tue.mpg.de/register.php> 注册（科研邮箱，勾选同意各子集
license）。MPI 各站点账号互相独立：

| 站点 | 用途 |
|---|---|
| amass.is.tue.mpg.de | 下载 AMASS npz 数据（本脚本所需） |
| smpl-x.is.tue.mpg.de | 下载 SMPL-X body model（使用 smplx 子集时需要，见 §6.1） |
| mano.is.tue.mpg.de | 下载 SMPL+H body model 素材（使用 smplh 子集时需要） |

三个站账号互相独立，需各注册一次；下载走同一个凭据网关
`download.is.tue.mpg.de`，只是 `domain` 参数不同。

## 3. 用法

```bash
cd /data1/mono/things/combatbench/mocap   # 或任意工作目录

# 全量（24 个子集，SMPL+H gender_specific）
AMASS_USER='you@example.com' AMASS_PASS='***' ./download_amass.sh

# 只下指定子集
AMASS_USER=... AMASS_PASS=... ./download_amass.sh CMU KIT WEIZMANN

# 列出全部子集名
./download_amass.sh --list
```

凭据只从环境变量读取（不落盘、不进命令行参数）。脚本先用小文件预检
凭据：网关返回 HTML/403 而非 bz2 时退出，避免白下几十 GB。

**后台长跑**（全量要下数小时~数天，SSH 断了不碍事）：

```bash
AMASS_USER='...' AMASS_PASS='...' \
nohup ./download_amass.sh > amass_download.log 2>&1 &

tail -f amass_download.log   # 看进度
```

Ctrl+C 中断是安全的：重跑同一命令，已完成的子集自动跳过、半拉的 tar 续传。

## 4. 可选环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `AMASS_OUT` | `./amass`（当前运行目录下） | 输出根目录 |
| `AMASS_FORMAT` | `smplh` | `smplh` 或 `smplx`；smplh 是 PHC/MaskedMimic 常用输入 |
| `AMASS_BETA` | `gender_specific` | 或 `neutral` |
| `AMASS_EXTRACT` | `1` | `0` = 只下载 tar.bz2 不解压 |
| `AMASS_KEEP_TAR` | `0` | `1` = 解压成功后仍保留压缩包 |
| `AMASS_FORCE` | `0` | `1` = 忽略完成记录强制重下 |
| `AMASS_PROXY` | 不挂 | 显式代理（覆盖 `http_proxy`/`https_proxy`）。**实测走机房代理 ~3–10 MB/s，远快于直连**，但偶发 403，失败时去掉该变量直连重试 |

**断点与跳过**：半拉的 tar 由 `wget --continue` 续传；manifest 中已有完成
记录的子集自动跳过；有记录但解压产物缺失时，tar 还在就只补解压、不在则
重下。可随时中断重跑同一命令。

**网关行为（2026-11 实测）：**

- 凭据 POST 后网关返回 302 → `download2.php`，认证随 `PHPSESSID` cookie
  走；wget 自动处理，无需手工 cookie。
- **单文件高频请求会被临时 403**（实测 ~10 次/15min 即触发），只影响该
  文件、不影响其它子集，一段时间后自动恢复。脚本预检因此会连试两个
  文件，单文件下载失败也跳过续跑而不是终止整批。
- 速度**按线路差异巨大**：直连 25–205 KB/s 波动（按 IP 限速，并发无效）；
  走机房代理 `192.168.16.76:18000` 实测 **3–10 MB/s**（BMLrub 2.1GB 用
  ~11min），是首选通道，但网关偶发对代理 IP 403，失败时回退直连。
- **耗时预期**（2026-11 全量实测）：smplh 20 子集 tar 合计 ~30 GB、解压后
  ~44 GB；smplx 4 子集 tar ~10 GB、解压后 ~21 GB。**最大包是 GRAB 10.9G**
  （手部抓取，对本项目价值低，别误当小文件补下）、其次 WEIZMANN(smplx)
  9.8G、KIT 2.2G、BMLrub 2.1G、CMU 2.0G、EyesJapan 1.4G，其余均 <1G。
  走代理全量约 1–2 小时；直连按 25–200 KB/s 波动要数天，**建议按需选子集**。

## 5. 产物

```
$AMASS_OUT/                                    # 当前 = mocap/amass/（~65 GB，已 gitignore）
├── download_manifest.tsv                      # dataset / sfile / bytes / sha256 / npz 段数 / 时间 / 顶层目录
├── amass_download_<format>_<beta>.log         # 每次运行自动 tee 的全程日志
├── smplh_gender_specific/                     # 20 子集（44 GB）
│   └── <DATASET>/<subject>/<motion>*.npz
└── smplx_gender_specific/                     # 4 子集：CNRS/WEIZMANN/DFaust/SSM（21 GB）
    └── <DATASET>/<subject>/<motion>*_stageii.npz
```

npz 字段两种格式见 §6.2。文件命名多数子集是 `*_poses.npz`（smplh）/
`*_stageii.npz`（smplx 及 GRAB/SOMA），脚本一律按 `*.npz` 统计。

## 6. 使用数据：完整链路配置

npz 里只有 SMPL 参数（`poses`/`betas`/`trans`），拿到关节位置需要
`pip install smplx` + **body model 文件**。body model 和 AMASS 是不同站点
的独立账号。

### 6.1 body model 下载（已验证 2026-11）

| 数据 | 注册站点 | sfile | 产物 |
|---|---|---|---|
| SMPL-X（CNRS/WEIZMANN/DFaust/SSM 用） | smpl-x.is.tue.mpg.de | `models_smplx_v1_1.zip` (870MB) | `SMPLX_{FEMALE,MALE,NEUTRAL}.{pkl,npz}` |
| SMPL+H（其余 20 个子集用） | mano.is.tue.mpg.de | `smplh.tar.xz` + `mano_v1_2.zip` | `model.npz` + `MANO_L/R.pkl` |

```bash
# SMPL-X：一个包就够
curl -L -c jar -b jar --data-urlencode "username=$U" --data-urlencode "password=$P" \
 'https://download.is.tue.mpg.de/download.php?domain=smplx&resume=1&sfile=models_smplx_v1_1.zip' \
 -o models_smplx_v1_1.zip && unzip -q models_smplx_v1_1.zip

# SMPL+H：model.npz + MANO 手部参数，官方 merge 脚本合成完整 pkl
# （mano 包里的"现成 SMPLH_*.pkl"是坑：缺 hands_* 键、shapedirs 是 chumpy
#  懒算子，smplx 库加载会挂——详见 AMASS_DOWNLOAD_DETAIL.md）
curl ... 'domain=mano&...&sfile=smplh.tar.xz'   -o smplh.tar.xz   && tar -xf smplh.tar.xz
curl ... 'domain=mano&...&sfile=mano_v1_2.zip'  -o mano_v1_2.zip  && unzip -q mano_v1_2.zip
python3 merge_smplh_mano.py --smplh model.npz --mano-left MANO_LEFT.pkl \
  --mano-right MANO_RIGHT.pkl --gender {female,male,neutral}
```

当前落盘布局（已 `.gitignore`，FK 三性别均已验证）：

```
mocap/smpl_models/
├── smplh/SMPLH_{FEMALE,MALE,NEUTRAL}.pkl      # 覆盖 20 个 smplh 子集
├── smplx/SMPLX_{FEMALE,MALE,NEUTRAL}.{pkl,npz} # 覆盖 4 个 smplx 子集
└── *.tar.xz / *.zip                            # 原始包，溯源保留
```

**下载坑**：`wget --post-data` 偶发拿到登录页 HTML 而非文件——用
`curl -L -c jar -b jar --data-urlencode` + 302 跳转；直连 ~130KB/s 慢，
挂代理 `192.168.16.76:18000` 实测 ~2MB/s。

### 6.2 两种 npz 的字段差异（接代码前必看）

| | smplh（20 子集） | smplx（4 子集） |
|---|---|---|
| 姿态字段 | `poses` (N,156)：root3+body63+hands90 | `root_orient`/`pose_body`/`jaw_pose`/`eye`/`hand` 分立 |
| 体形 | `betas` (16,) | `betas` (10,) 或 `num_betas` |
| 平移 | `trans` (N,3) | `trans` (N,3) |
| 帧率 | `mocap_framerate`（有 i） | `mocap_frame_rate`（无 i） |
| 典型 fps | 120 | 120 |
| body model | SMPLH_*.pkl | SMPLX_*.pkl |

`poses[:, :66]`（root+21 body joints）两侧语义一致——SMPLH/SMPLX 前 22
关节运动树相同，**手部 90 维 humanoid21 用不到直接丢**。因此
smplh→smplx 只需字段重命名/切片（`gmr/scripts/smpl_to_smplx.py`），
运动内容无损，**不必为格式重下几十 GB**。

### 6.3 快速验证 FK 可用

```python
import torch, numpy as np, smplx
m = smplx.create('mocap/smpl_models', model_type='smplx', gender='neutral',
                 use_pca=False, flat_hand_mean=True)
d = np.load('mocap/amass/smplx_gender_specific/SSM/20160330_03333/walking_01_stageii.npz')
out = m(betas=torch.tensor(d['betas'][:10],dtype=torch.float32)[None],
        global_orient=torch.tensor(d['root_orient'][:1],dtype=torch.float32),
        body_pose=torch.tensor(d['pose_body'][:1],dtype=torch.float32),
        jaw_pose=torch.zeros(1,3), leye_pose=torch.zeros(1,3), reye_pose=torch.zeros(1,3),
        left_hand_pose=torch.zeros(1,45), right_hand_pose=torch.zeros(1,45),
        transl=torch.tensor(d['trans'][:1],dtype=torch.float32))
print(out.joints.shape)   # (1,127,3) → 关节位置
```

### 6.4 接入 retarget

本项目用 **GMR**（`mocap/gmr/`，独立 clone）做 SMPL-X→humanoid21 运动学
重定向，已跑通走路/拳击/踢腿，配置方法、坐标约定、踩坑记录全在
`mocap/GMR_RETARGET.md`。批量跑：`mocap/retarget_smplx_batch.py`，产物在
`mocap/retarget_out/`（.pkl + .mp4）。

## 7. 许可与引用

- 仅限科研用途；各子集 license 见下载页链接，使用时须遵守其中限制性条款
- 引用 AMASS 本体：<https://download.is.tue.mpg.de/amass/amass.bib>
- 引用全部子集：下载页逐行的 citation.bib

## 8. 与已有数据的关系

| 已有 | AMASS 中的对应 | 说明 |
|---|---|---|
| `/data1/dev/combatbench_mocap/raw/` CMU ASF/AMC | `CMU` 子集 | 同源，AMASS 是 SMPL 化版本 |
| `/data1/dev/hdm05/data/` normPOS | `HDM05` 子集 | 同源不同表示 |

AMASS 对本项目的增量主要在 KIT、WEIZMANN、BMLmovi、BMLrub、EyesJapan、
TotalCapture 等其它来源的动作多样性。

**子集选择建议（面向 combatbench 动作多样性）：**

- **优先**：`ACCAD`（体育动作含 MartialArtsPunches/Kicks 武术拳脚全套，
  对本项目价值最高、才 104MB）、`KIT`（动作种类最全 4232 段）、
  `WEIZMANN`（步态变体 2227 段，仅 smplx）、`BMLmovi`/`BMLrub`（体育类
  大幅度动作）、`SSM`（含 punching/punch_kick/squat/jumping_jacks，
  已验证可 retarget）
- **冗余可缓**：`CMU`（已有 ASF/AMC 原始版）、`HDM05`（已有 normPOS 版）
- **对本项目价值低**：`GRAB`（手部抓取 **且是最大包 10.9G**）、
  `TCDHands`、`DanceDB`（舞蹈）、`PosePrior`/`MoSh`/`DFaust`（极小或
  标定/极限姿势用）
- **只提供 SMPL-X 无 SMPL+H**：`CNRS`、`WEIZMANN`（2024 新增子集）。
  `DFaust`/`SSM` 的 smplh 路径被网关限流 403 时也走了 smplx 路径成功
  （限流按 sfile 路径算，换路径即绕开）。当前已入库 smplx 子集：
  CNRS 81 段、WEIZMANN 2227 段、DFaust 139 段、SSM 33 段。
  smplh↔smplx 字段转换见 §6.2，**格式不构成数据缺口**。
