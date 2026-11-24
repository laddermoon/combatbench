# AMASS 下载过程 DETAIL 日志

> 2026-11-24，GPU 服务器 instance-1f1igpaq，`mocap/download_amass.sh`。
> SUMMARY 版见 `AMASS_DOWNLOAD.md`。

## 执行流水

**11:2x 首次运行**：凭据预检失败退出。
根因：网关流程是 POST → `302 → download2.php` + `PHPSESSID` cookie；
预检 curl 未 `-L` 跟随也没带 cookie jar，拿到 302 空页。
wget 自带跳转+cookie，主流程本身没问题。修复预检。

**11:45 用户前台跑全量**：ACCAD 104MB 直连 ~205KB/s 用时 8m41s；
BMLhandball 398MB 掉到 ~25KB/s（ETA 4h）被 Ctrl+C。
发现：直连按 IP 限速且波动巨大，并发无效。

**12:0x 起 nohup 后台批跑（挂机房代理 192.168.16.76:18000）**：
代理提速到 **3–10 MB/s**（BMLrub 2.1GB 仅 ~11min）。
20/24 完成后脚本在 BMLrub 处退出（exit 1）。

**BMLrub bug**：tar 解压顶层目录是 `BioMotionLab_NTroje/` 而非 `BMLrub/`。
`find $DL_DIR/BMLrub` 返回非零 → `set -e + pipefail` 杀死脚本，manifest
未记录。修复：顶层目录改为 `tar -tjf | head -1` 实测，并写入 manifest 第 7
列。同类命名不一致：MoSh→MPI_mosh、PosePrior→MPI_Limits、
TCDHands→TCD_handMocap、HDM05→MPI_HDM05、EyesJapan→Eyes_Japan_Dataset、
Transitions→Transitions_mocap。

**GRAB/SOMA npz_count=0**：这俩子集只发 `*_stageii.npz`、没有 `*_poses.npz`，
数据本身完整（1350/71 段）。计数 pattern 放宽为 `*.npz`，manifest 行已更正。

**CNRS/WEIZMANN 返回 HTML 错误页**：这两个 2024 新增子集**只提供
SMPL-X，没有 SMPL+H 版本**。改用 `AMASS_FORMAT=smplx` 成功。

**WEIZMANN (smplx) 巨大**：9,835,311,668 B（9.2GB），全套最大单文件，
下载 ~1.5h，解压 ~15min，2227 个 npz。

**SSM/DFaust 长期 403**：调试期间对这两个文件高频请求触发网关**按文件
限流**，持续 7h+ 未恢复（直连/代理、有无凭据均 403——按 sfile 全局封）。
解法：限流按 sfile 路径算，换 `AMASS_FORMAT=smplx` 路径直接绕过，
DFaust 213M + SSM 36M 顺利拿到。

## 最终结果（2026-11-24 18:35 全部完成）

- **24/24 子集齐全**：20 × smplh + 4 × smplx（CNRS/WEIZMANN 仅有 smplx；
  DFaust/SSM 因 smplh 路径被限流也走的 smplx）
- 解压后总数据 ~64G，`amass/download_manifest.tsv` 有全部 sha256/npz 计数
- 日志：`amass/amass_download_{smplh,smplx}_gender_specific.log`

## 坑位速查

| 坑 | 解法 |
|---|---|
| 预检拿不到数据 | POST 必须跟 302 + cookie（curl: `-L -c jar -b jar`） |
| 单文件被 403 限流 | 按 sfile 限：换 smplx/smplh 路径或等几小时 |
| 直连极慢 | 机房代理 3–10MB/s；代理偶发 403 就直连 |
| 包名≠解压目录名 | 用 `tar -tjf | head -1` 实测，别猜 |
| *_poses.npz 计数为 0 | 部分子集只有 *_stageii.npz，按 *.npz 统计 |
| CNRS/WEIZMANN 404 HTML | 只有 smplx 版本 |

## 补记：SMPL-X body model 下载（2026-11-25 01:05）

**Why:** AMASS npz 只存 SMPL 参数，FK 出关节位置需要对应 body model。
smpl-x.is.tue.mpg.de 注册后，`domain=smplx&sfile=models_smplx_v1_1.zip`
（870M）经同一凭据网关下载，代理线路 ~4MB/s。

**Result:** `mocap/smpl_models/smplx/SMPLX_{F,M,NEUTRAL}.{pkl,npz}` 就位，
`unzip -t` 无损，`smplx.create(model_type='smplx')` 对 DFaust 真实 npz
FK 成功（127 joints）。注意 **v1_1 包只含 smplx 模型**——smplh 数据仍需
mano.is.tue.mpg.de 的 SMPL+H（`domain=mano&sfile=smplh.tar.xz` +
`mano_v1_2.zip`，merge_smplh_mano.py 合成 pkl）。

**坑:** wget --post-data 首次拿到登录页 HTML（网关抽风/状态问题），
curl -L + cookie jar 正常；百度 pypi 镜像无 smplx 包，须 `-i
https://pypi.org/simple` + 代理；pip 顺带把 cudnn 降到 torch pin
的 9.5.1.17（已验证 torch cuda 正常）。

**Also:** `mocap/smpl_models/` 已加入 .gitignore（licensed 资产不入 git）。

## 补记2：SMPL+H / MANO body model（2026-11-25 01:25）

**Why:** 20 个 smplh 子集需要 SMPL+H 模型。mano.is.tue.mpg.de
账号（独立于 amass/smpl-x）下 `domain=mano&sfile=smplh.tar.xz`
（113M）+ `mano_v1_2.zip`（175M）。

**Result:** `smpl_models/smplh/SMPLH_{FEMALE,MALE,NEUTRAL}.pkl` 就位；
`smplx.create('smplh')` 三性别加载 OK，ACCAD 真实帧 FK 出 73 joints
（Z-up 米制：pelvis 0.96 / head 1.54 / foot 0.07）。

**关键做法：** 不用 mano 包里"现成"的 SMPLH_*.pkl——它不含 hands_*
PCA 键且 shapedirs 是 chumpy 懒求值 Select（数据藏在 `.a.x`）。正确路径：
`smplh_src/<g>/model.npz`（纯 numpy）+ MANO_L/R.pkl 的手部键，经
`tools/merge_smplh_mano.py` 合成完整 pkl。

**坑:**
- chumpy 已从 PyPI 死绝（build 失败）；装了手写 stub 到
  site-packages/chumpy/（Ch/Select = 容错 ndarray，dict-state 兼容），
  仅用于 unpickle MANO pkl 的手部键，不跑 chumpy 计算图
- mano_v1_2 的"现成"SMPLH pkl 缺 hands_* 键，直接用 smplx 库会挂
- AMASS smplh: betas=(16,) poses=(N,156) fps=120；smplx 版字段名不同
  （mocap_frame_rate 无 i、分立的 root_orient/pose_body/pose_hand）
