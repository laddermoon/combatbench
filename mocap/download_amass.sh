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
#   AMASS_OUT      输出根目录（默认 ./amass，即当前运行目录下的 amass/）
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

AMASS_OUT="${AMASS_OUT:-$PWD/amass}"
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

# 预检：取小文件前 1KB 验证凭据（凭据错误时网关返回 HTML/403 而非 BZh2 流）。
# 网关对凭据 POST 返回 302 → download2.php，会话认证走 PHPSESSID cookie，
# 因此 curl 必须 -L 跟跳转并带 cookie jar（wget 默认即如此，主循环无需特殊处理）。
# 注意：网关会对单文件的高频请求临时 403——预检依次试两个文件，
# 第一个被限流时换第二个，避免把"单文件限流"误判成"凭据错误"。
preflight() {
  local D="$1" probe jar code
  probe="$(mktemp)"; jar="$(mktemp)"
  code=$(curl -sSL -r 0-1023 -c "$jar" -b "$jar" \
       --data-urlencode "username=$AMASS_USER" \
       --data-urlencode "password=$AMASS_PASS" \
       "$BASE&sfile=$(SF "$D")" -o "$probe" -w '%{http_code}') || code="curl_error"
  if [ "$(head -c 3 "$probe" 2>/dev/null)" = "BZh" ]; then
    rm -f "$probe" "$jar"; return 0
  fi
  echo "  预检 $D → HTTP $code（非 bz2）" >&2
  rm -f "$probe" "$jar"; return 1
}
if ! preflight SSM && ! preflight DFaust; then
  echo "凭据校验失败：两次探测均未返回 bz2（账号/密码错误或站点变更）。" >&2
  exit 1
fi
echo "凭据校验通过。输出目录: $DL_DIR"

FAILED=()
for D in "${WANTED[@]}"; do
  sfile="$(SF "$D")"; tar_path="$TAR_DIR/$D.tar.bz2"
  echo "=== [$D] $sfile"

  if ! wget --continue --tries=5 --waitretry=10 --progress=dot:mega \
      --post-data "username=$U&password=$P" \
      "$BASE&sfile=$sfile" -O "$tar_path" \
     || [ "$(head -c 3 "$tar_path" 2>/dev/null)" != "BZh" ]; then
    echo "!! $D 下载失败或非 bz2（可能该文件被临时限流），跳过，稍后重跑即可续传。" >&2
    FAILED+=("$D"); sleep 5; continue
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
  sleep 2   # 网关有频率限制，降低请求间隔触发限流的概率
done

echo "完成。manifest: $MANIFEST"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "以下子集本次失败，重跑同一命令可续传: ${FAILED[*]}" >&2
  exit 1
fi
