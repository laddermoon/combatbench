# CombatBench 提交工具

本目录包含 CombatBench 提交工具的预编译二进制文件。

## 下载

您也可以从 [GitHub Releases](https://github.com/laddermoon/combatbench/releases) 下载最新版本的二进制文件。

## 可用的二进制文件

| 平台 | 二进制文件 | 下载链接 |
|------|-----------|----------|
| **Linux (amd64)** | `combat-submit-linux-amd64` | [下载](https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-linux-amd64) |
| **macOS (Intel)** | `combat-submit-darwin-amd64` | [下载](https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-darwin-amd64) |
| **Windows (amd64)** | `combat-submit-windows-amd64.exe` | [下载](https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-windows-amd64.exe) |

## 快速开始

### 1. 下载适合您平台的二进制文件

```bash
# Linux
wget https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-linux-amd64
chmod +x combat-submit-linux-amd64

# macOS (Intel)
wget https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-darwin-amd64
chmod +x combat-submit-darwin-amd64

# Windows
# 通过浏览器下载，直接使用
```

### 2. 设置您的 API Key

登录 [CombatBench Web 平台](https://github.com/laddermoon/combatbench) 后获取您的 API Key。

```bash
export COMBAT_API_KEY="your-api-key-here"
```

### 3. 提交您的策略

```bash
./combat-submit-linux-amd64 submit \
  --leaderboard-id 1 \
  --name "我的策略" \
  --dir ../my_policy \
  --desc "我的战斗策略"
```

## 文档

完整的使用说明请参阅 [用户指南](../../submisiontool/USER_GUIDE_EN.md) 或 [中文指南](../../submisiontool/USER_GUIDE_CN.md)。

## 源代码

提交工具的源代码位于：[github.com/laddermoon/combatbench/tree/main/submisiontool](https://github.com/laddermoon/combatbench/tree/main/submisiontool)

## 安全性

- 运行前请务必验证二进制文件的校验和
- 仅从官方 GitHub Releases 下载
- 向维护者报告安全问题

## 许可证

详情请参阅 [LICENSE](../../LICENSE)。
