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
#   AMASS_FORCE    1=忽略完成记录强制重下（默认 0，已完成的子集自动跳过）
#   AMASS_PROXY    显式代理，如 http://192.168.16.76:18000；设置后覆盖
#                  http_proxy/https_proxy（注意：本网关曾对代理 IP 直接 403）
#
# 已完成判定：manifest 中存在该 (dataset, sfile) 记录。tar 仍在但尚未解压的
# 子集重跑时只补解压、不重新下载。
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
AMASS_FORCE="${AMASS_FORCE:-0}"

if [ -n "${AMASS_PROXY:-}" ]; then
  export http_proxy="$AMASS_PROXY" https_proxy="$AMASS_PROXY"
fi

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

# 全程同时写日志文件（每个 format/beta 一份，跨批次追加）
LOG="$AMASS_OUT/amass_download_${AMASS_FORMAT}_${AMASS_BETA}.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== run started $(date -Iseconds) log=$LOG ====="
[ -f "$MANIFEST" ] || printf 'dataset\tsfile\ttar_bytes\tsha256\tnpz_count\tdate\ttop_dir\n' > "$MANIFEST"

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
# 网关按单文件限流，轮询多个小文件，全灭才判定凭据/站点问题
_ok=0
for _cand in SSM DFaust HumanEva SFU; do
  preflight "$_cand" && { _ok=1; break; }
done
if [ "$_ok" != "1" ]; then
  echo "凭据校验失败：多次探测均未返回 bz2（账号/密码错误或站点变更）。" >&2
  exit 1
fi
echo "凭据校验通过。输出目录: $DL_DIR"

# manifest 中该 (dataset, sfile) 的完成记录 → 打印记录里的解压顶层目录并返回 0。
# 注意：tar 内部顶层目录名不一定等于包名（BMLrub→BioMotionLab_NTroje、
# MoSh→MPI_mosh 等），一律以记录/tar 内容为准。旧记录无第 7 列则回退包名。
have_row() { awk -F'\t' -v d="$1" -v s="$2" '$1==d && $2==s{t=($7!=""?$7:d); f=1} END{if(f)print t; exit !f}' "$MANIFEST"; }
# tar 包内容的真实顶层目录（head 提前关闭管道会让 tar 收 SIGPIPE，|| true 兜底）
tar_top() { tar -tjf "$1" 2>/dev/null | head -1 | cut -d/ -f1 || true; }
# 指定目录中是否已有 npz（参数为完整路径）
has_npz() { find "$1" -name '*.npz' -print -quit 2>/dev/null | grep -q .; }

FAILED=()
for D in "${WANTED[@]}"; do
  sfile="$(SF "$D")"; tar_path="$TAR_DIR/$D.tar.bz2"
  echo "=== [$D] $sfile"

  # 已完成的子集直接跳过（AMASS_FORCE=1 可强制重下）
  if [ "$AMASS_FORCE" != "1" ] && topdir=$(have_row "$D" "$sfile"); then
    if [ "$AMASS_EXTRACT" = "0" ] || has_npz "$DL_DIR/$topdir"; then
      echo "    已完成，跳过（AMASS_FORCE=1 强制重下）"
      continue
    fi
    # 有记录但 npz 缺失：tar 还在本地就只补解压，tar 也不在则重新下载
    if [ "$(head -c 3 "$tar_path" 2>/dev/null)" = "BZh" ]; then
      echo "    tar 已在本地，只补解压"
      topdir="$(tar_top "$tar_path")"
      tar -xjf "$tar_path" -C "$DL_DIR"
      npz_count=$(find "$DL_DIR/$topdir" -name '*.npz' 2>/dev/null | wc -l || true)
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$D" "$sfile" "$(stat -c %s "$tar_path")" \
        "$(sha256sum "$tar_path" | cut -d' ' -f1)" "$npz_count" "$(date -Iseconds)" "$topdir" >> "$MANIFEST"
      continue
    fi
    echo "    记录存在但产物缺失，重新下载"
  fi

  if ! wget --continue --tries=5 --waitretry=10 --progress=dot:mega \
      --post-data "username=$U&password=$P" \
      "$BASE&sfile=$sfile" -O "$tar_path" \
     || [ "$(head -c 3 "$tar_path" 2>/dev/null)" != "BZh" ]; then
    echo "!! $D 下载失败或非 bz2（可能该文件被临时限流），跳过，稍后重跑即可续传。" >&2
    FAILED+=("$D"); sleep 5; continue
  fi

  bytes=$(stat -c %s "$tar_path"); sha=$(sha256sum "$tar_path" | cut -d' ' -f1)
  topdir=""; npz_count=0
  if [ "$AMASS_EXTRACT" = "1" ]; then
    topdir="$(tar_top "$tar_path")"
    tar -xjf "$tar_path" -C "$DL_DIR"
    npz_count=$(find "$DL_DIR/$topdir" -name '*.npz' 2>/dev/null | wc -l || true)
    [ "$AMASS_KEEP_TAR" = "1" ] || rm -f "$tar_path"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$D" "$sfile" "$bytes" "$sha" "$npz_count" "$(date -Iseconds)" "$topdir" >> "$MANIFEST"
  sleep 2   # 网关有频率限制，降低请求间隔触发限流的概率
done

echo "完成。manifest: $MANIFEST"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "以下子集本次失败，重跑同一命令可续传: ${FAILED[*]}" >&2
  exit 1
fi
