#!/bin/bash
# Humanoid21 Simulator 测试运行脚本

set -e

# 设置颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "======================================"
echo "Humanoid21 Simulator 测试套件"
echo "======================================"
echo ""

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 设置 PYTHONPATH
export PYTHONPATH=/data1/mono/things/combatbench

echo "运行数据接口完整测试..."
echo ""

if PYTHONPATH=/data1/mono/things/combatbench python3 "$SCRIPT_DIR/test_data_interfaces.py"; then
    echo ""
    echo -e "${GREEN}======================================"
    echo "✓ 所有测试通过！"
    echo "======================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}======================================"
    echo "✗ 测试失败"
    echo "======================================${NC}"
    exit 1
fi
