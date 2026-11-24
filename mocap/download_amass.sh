#!/usr/bin/env bash
# AMASS 批量下载脚本（download.is.tue.mpg.de 凭据式下载网关）。
#
# 前提：已在 https://amass.is.tue.mpg.de/register.php 注册并同意 license。
# 凭据只从环境变量读取，不落盘不进命令行参数（避免进 shell history / git）。
#
# 用法：
#   AMASS_USER='you@example.com' AMASS_PASS='***' ./download_amass.sh           # 全部 24 个子集
#   AMASS_USER=... AMASS_PASS=... ./download_amass.sh CMU KIT                   # 只下指定子集
#   ./download_amass.sh --list                                                 # 列出全部子集名
#
# 可选环境变量：
#   AMASS_OUT      输出根目录（默认 /data1/dev/combatbench_mocap/amass）
#   AMASS_FORMAT   smplh | smplx                  （默认 smplh，PHC/MaskedMimic 常用输入）
#   AMASS_BETA     gender_specific | neutral      （默认 gender_specific）
#   AMASS_EXTRACT  1=下载后解压（默认）；0=只保留 tar.bz2
#   AMASS_KEEP_TAR 1=解压成功后仍保留压缩包（默认 0 删除）
#   http_proxy/https_proxy  MPI 德国服务器一般直连可达，慢时再挂代理
#
# 产物布局：
#   $AMASS_OUT/<format>_<beta>/<DATASET>/<subject>/<motion>_poses.npz
#   $AMASS_OUT/download_manifest.tsv   来源 sfile + sha256 + 解压段数（资产可追溯）

set -euo pipefail

AMASS_OUT="${AMASS_OUT:-/data1/dev/combatbench_mocap/amass}"
AMASS_FORMAT="${AMASS_FORMAT:-smplh}"
AMASS_BETA="${AMASS_BETA:-gender_specific}"
AMASS_EXTRACT="${AMASS_EXTRACT:-1}"
AMASS_KEEP_TAR="${AMASS_KEEP_TAR:-0}"

BASE='https://download.is.tue.mpg.de/download.php?domain=amass&resume=1'

# 2024 license 页的 24 个子集（按官方表顺序）
DATASETS=(
  ACCAD BMLhandball BMLmovi BMLrub CMU CNRS DanceDB DFaust EKUT
  EyesJapanDataset GRAB HDM05 HUMAN4D HumanEva KIT MoSh PosePrior
  SFU SOMA SSM TCDHands TotalCapture Transitions WEIZMANN
)

if [ "${1:-}" = "--list" ]; then
  printf '%s\n' "${DATASETS[@]}"
  exit 0
fi

: "${AMASS_USER:?需要环境变量 AMASS_USER（amass.is.tue.mpg.de 注册邮箱）}"
: "${AMASS_PASS:?需要环境变量 AMASS_PASS}"

if [ $# -gt 0 ]; then
  WANTED=("$@")
  for d in "${WANTED[@]}"; do
    [[ " ${DATASETS[*]} " == *" $d "* ]] || { echo "未知子集: $d（--list 查看全部）" >&2; exit 2; }
  done
else
  WANTED=("${DATASETS[@]}")
fi

urlenc() { python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
U=$(urlenc "$AMASS_USER"); P=$(urlenc "$AMASS_PASS")

SF() { echo "amass_per_dataset/${AMASS_FORMAT}/${AMASS_BETA}/mosh_results/$1.tar.bz2"; }

DL_DIR="$AMASS_OUT/${AMASS_FORMAT}_${AMASS_BETA}"
TAR_DIR="$DL_DIR/_tars"
MANIFEST="$AMASS_OUT/download_manifest.tsv"
mkdir -p "$DL_DIR" "$TAR_DIR"
[ -f "$MANIFEST" ] || printf 'dataset\tsfile\ttar_bytes\tsha256\tnpz_count\tdate\n' > "$MANIFEST"

# 预检：取最小的 SSM 包前 1KB 验证凭据（凭据错误时网关返回 HTML 而非 BZh2 流）
probe="$(mktemp)"
curl -sf -r 0-1023 --data-urlencode "username=$AMASS_USER" \
     --data-urlencode "password=$AMASS_PASS" "$BASE&sfile=$(SF SSM)" -o "$probe" || true
if [ "$(head -c 3 "$probe" 2>/dev/null)" != "BZh" ]; then
  echo "凭据校验失败：网关未返回 bz2 数据（可能是账号/密码错误或站点变更）。返回内容：" >&2
  head -c 500 "$probe" >&2; rm -f "$probe"; exit 1
fi
rm -f "$probe"
echo "凭据校验通过。输出目录: $DL_DIR"

for D in "${WANTED[@]}"; do
  sfile="$(SF "$D")"; tar_path="$TAR_DIR/$D.tar.bz2"
  echo "=== [$D] $sfile"

  wget --continue --tries=5 --waitretry=10 --progress=dot:mega \
    --post-data "username=$U&password=$P" \
    "$BASE&sfile=$sfile" -O "$tar_path"

  if [ "$(head -c 3 "$tar_path")" != "BZh" ]; then
    echo "!! $D 下载内容不是 bz2（疑似网关错误页），已删除，停止。" >&2
    head -c 300 "$tar_path" >&2; rm -f "$tar_path"; exit 1
  fi

  bytes=$(stat -c %s "$tar_path"); sha=$(sha256sum "$tar_path" | cut -d' ' -f1)
  npz_count=0
  if [ "$AMASS_EXTRACT" = "1" ]; then
    tar -xjf "$tar_path" -C "$DL_DIR"
    npz_count=$(find "$DL_DIR/$D" -name '*_poses.npz' 2>/dev/null | wc -l)
    [ "$AMASS_KEEP_TAR" = "1" ] || rm -f "$tar_path"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$D" "$sfile" "$bytes" "$sha" "$npz_count" "$(date -Iseconds)" >> "$MANIFEST"
done

echo "完成。manifest: $MANIFEST"
