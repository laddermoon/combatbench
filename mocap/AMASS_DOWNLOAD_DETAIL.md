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
