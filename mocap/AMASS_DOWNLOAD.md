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
| smpl-x.is.tue.mpg.de | 下载 SMPL+H / SMPL-X body model（使用数据时需要） |

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

凭据只从环境变量读取（不落盘、不进命令行参数）。脚本先用最小的 SSM 包
预检凭据：网关返回 HTML 错误页而非 bz2 时立即退出，避免白下几十 GB。

## 4. 可选环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `AMASS_OUT` | `./amass`（当前运行目录下） | 输出根目录 |
| `AMASS_FORMAT` | `smplh` | `smplh` 或 `smplx`；smplh 是 PHC/MaskedMimic 常用输入 |
| `AMASS_BETA` | `gender_specific` | 或 `neutral` |
| `AMASS_EXTRACT` | `1` | `0` = 只下载 tar.bz2 不解压 |
| `AMASS_KEEP_TAR` | `0` | `1` = 解压成功后仍保留压缩包 |
| `http_proxy`/`https_proxy` | 不挂 | MPI 德国服务器一般直连可达；慢时再挂 |

下载支持断点续传（`wget --continue`），中断后重跑同一命令即可续。

## 5. 产物

```
$AMASS_OUT/
├── download_manifest.tsv          # dataset / sfile / bytes / sha256 / npz 段数 / 时间
└── smplh_gender_specific/
    └── <DATASET>/<subject>/<motion>_poses.npz
```

每个 `*_poses.npz` 字段：`poses`（156 维 SMPL+H pose）、`betas`、`trans`、
`gender`、`mocap_framerate`、`dmpls`。`*_stageii.npz` 为中间产物，一般用前者。

## 6. 使用数据还需要

```bash
pip install smplx
```

外加 SMPL+H body model 文件（smpl-x.is.tue.mpg.de 单独注册下载）。npz 里只有
参数，关节位置/网格要靠 `smplx` FK 出来。

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
