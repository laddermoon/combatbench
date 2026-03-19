# CombatBench 算法提交工具 - 用户指南

本指南面向参与 CombatBench 排行榜竞赛的用户，介绍如何使用 `combat-submit` 工具提交您的算法代码。

## 目录

- [前提条件](#前提条件)
- [快速开始](#快速开始)
- [获取 API Key](#获取-api-key)
- [准备算法代码](#准备算法代码)
- [提交算法](#提交算法)
- [查看排行榜](#查看排行榜)
- [提交状态查询](#提交状态查询)
- [常见问题](#常见问题)
- [故障排除](#故障排除)

---

## 前提条件

在使用提交工具之前，请确保：

1. **已注册账号**: 您需要在 CombatBench 平台注册一个账号
2. **已获取 API Key**: 提交代码需要 API Key 进行身份验证
3. **已安装提交工具**: 获取 `combat-submit` 可执行文件
4. **算法代码准备就绪**: 您的算法代码已按照要求组织好

---

## 快速开始

### 最简单的提交方式

```bash
# 1. 设置您的 API Key
export COMBAT_API_KEY="your-api-key-here"

# 2. 提交您的算法
./combat-submit submit \
  --leaderboard-id 1 \
  --name "我的算法" \
  --dir /path/to/your/code
```

### 完整示例

```bash
# 设置环境变量
export COMBAT_API_KEY="sk_abcd1234..."
export UPLOAD_SERVER="http://your-server:10001"  # 可选，默认使用内置地址
export COMBATBENCH_API_URL="http://your-server:10000"  # 可选

# 提交命令
./combat-submit submit \
  --leaderboard-id 1 \
  --name "PPO算法v2.0" \
  --dir ./my_policy \
  --desc "使用PPO算法，训练100万步"
```

---

## 获取 API Key

### 通过 Web 界面获取

1. 访问 CombatBench Web 平台
2. 登录您的账号
3. 进入个人设置页面
4. 点击"生成 API Key"或"重新生成 API Key"
5. 复制生成的 API Key（格式如：`sk_xxxxx...`）

### API Key 安全须知

- **妥善保管**: API Key 相当于您的账号密码，请勿泄露给他人
- **定期更换**: 建议定期更换 API Key 以保证安全
- **最小权限**: API Key 仅用于提交代码，不要用于其他目的

---

## 准备算法代码

### 目录结构要求

您的算法代码目录应包含以下内容：

```
my_policy/
├── agent.py              # 必需：您的智能体实现
├── config.json           # 可选：算法配置参数
├── requirements.txt      # 可选：Python 依赖
├── README.md             # 可选：算法说明
└── models/               # 可选：预训练模型文件
    └── model.pkl
```

### 代码规范

1. **入口文件**: 必须包含可被评测系统调用的入口函数
2. **依赖声明**: 如果有第三方依赖，请在 `requirements.txt` 中声明
3. **文件大小**: 整个目录压缩后应小于 500MB
4. **编码格式**: 建议使用 UTF-8 编码

---

## 提交算法

### 命令格式

```bash
./combat-submit submit [选项]
```

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--leaderboard-id` | 排行榜 ID | `1`, `2`, `3` |
| `--name` | 提交名称 | `"我的PPO算法"` |
| `--dir` | 算法代码目录 | `./my_policy` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--desc` | 算法描述 | 无 |
| `--chunk-size` | 分片大小（MB） | 5 |

### 提交流程

提交工具会自动执行以下步骤：

```
┌─────────────────┐
│ 1. 验证 API Key  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 创建提交记录  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. 打包代码为ZIP │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. 分片上传文件  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. 更新提交状态  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   提交完成！     │
└─────────────────┘
```

### 上传进度显示

上传过程中会显示进度信息：

```
正在提交到排行榜 #1...
创建提交记录... ✓
打包代码... ✓ (12.5 MB)
正在上传: [████████████████████████████] 100% (12.5 MB / 12.5 MB)
提交成功！Submission ID: 123
```

---

## 查看排行榜

### 列出所有可用排行榜

```bash
./combat-submit leaderboards
```

### 输出示例

```
可用的排行榜：

ID  名称                          描述
1   1v1对战排行榜                单挑模式排行榜
2   3v3团战排行榜                三人团战模式
3   迷宫挑战排行榜               寻路挑战
```

---

## 提交状态查询

提交后，您可以通过以下方式查看提交状态：

### 通过命令行（如果支持）

```bash
./combat-submit status <submission-id>
```

### 通过 Web 界面

1. 访问 CombatBench Web 平台
2. 登录您的账号
3. 进入"我的提交"页面
4. 查看提交状态和评测结果

### 提交状态说明

| 状态 | 说明 |
|------|------|
| `pending` | 等待评测 |
| `running` | 正在评测中 |
| `completed` | 评测完成 |
| `failed` | 评测失败 |
| `error` | 系统错误 |

---

## 常见问题

### Q1: 提交失败，提示"Invalid API Key"

**原因**: API Key 无效或未设置

**解决方法**:
```bash
# 检查环境变量是否设置
echo $COMBAT_API_KEY

# 重新设置
export COMBAT_API_KEY="your-correct-api-key"
```

### Q2: 上传中断，怎么办？

**解决方法**: 直接重新运行相同的提交命令，工具会自动从断点继续上传：

```bash
# 重新运行，会自动续传
./combat-submit submit \
  --leaderboard-id 1 \
  --name "我的算法" \
  --dir ./my_policy
```

### Q3: 如何知道应该提交到哪个排行榜？

**解决方法**: 使用 `leaderboards` 命令查看所有可用的排行榜：

```bash
./combat-submit leaderboards
```

### Q4: 代码目录太大，上传很慢

**解决方法**:
- 检查是否包含了不必要的大文件（如数据集、日志文件）
- 可以使用更大的分片大小（但不超过 10MB）：
```bash
./combat-submit submit --leaderboard-id 1 --name "我的算法" --dir ./my_policy --chunk-size 10
```

### Q5: 评测失败，如何查看错误信息？

**解决方法**:
- 通过 Web 界面查看详细日志
- 检查代码是否符合评测系统的接口要求
- 确认所有依赖都在 `requirements.txt` 中声明

---

## 故障排除

### 连接服务器失败

**错误信息**: `connection refused` 或 `timeout`

**排查步骤**:

1. 检查网络连接
```bash
ping your-server
```

2. 检查服务器地址配置
```bash
echo $UPLOAD_SERVER
echo $COMBATBENCH_API_URL
```

3. 如需自定义服务器地址：
```bash
export UPLOAD_SERVER="http://your-server:10001"
export COMBATBENCH_API_URL="http://your-server:10000"
```

### 权限错误

**错误信息**: `401 Unauthorized` 或 `403 Forbidden`

**排查步骤**:

1. 确认 API Key 是否正确
2. 检查 API Key 是否已过期
3. 尝试重新生成 API Key

### 文件验证失败

**错误信息**: `400 Bad Request` 或文件验证相关错误

**排查步骤**:

1. 检查目录中是否包含不允许的文件类型
2. 确认压缩后文件大小不超过 500MB
3. 检查文件名是否包含特殊字符

---

## 环境变量参考

| 环境变量 | 说明 | 默认值 | 必需 |
|----------|------|--------|------|
| `COMBAT_API_KEY` | 您的 API Key | 无 | **是** |
| `UPLOAD_SERVER` | 上传服务器地址 | `http://localhost:10001` | 否 |
| `COMBATBENCH_API_URL` | WebAPI 地址 | `http://localhost:10000` | 否 |

---

## 技术支持

如遇到问题，请通过以下方式获取帮助：

1. **查阅文档**: 首先阅读本指南和相关技术文档
2. **GitHub Issues**: 在项目仓库提交问题
3. **联系管理员**: 通过官方渠道联系技术支持

---

## 附录

### A. 完整命令示例

```bash
#!/bin/bash
# 完整的提交流程示例

# 1. 设置环境变量
export COMBAT_API_KEY="sk_your_api_key_here"
export UPLOAD_SERVER="http://combatbench.example.com:10001"
export COMBATBENCH_API_URL="http://combatbench.example.com:10000"

# 2. 查看可用排行榜
./combat-submit leaderboards

# 3. 提交算法
./combat-submit submit \
  --leaderboard-id 1 \
  --name "PPO-v2.0" \
  --dir ./my_ppo_policy \
  --desc "使用PPO算法，训练100万步，LR=0.0003"

# 4. 等待评测完成，然后通过 Web 界面查看结果
```

### B. 目录打包规则

提交工具会自动将您的目录打包为 ZIP 文件，打包规则：

1. **包含所有文件**: 目录中的所有文件和子目录都会被包含
2. **保留目录结构**: ZIP 内保持原有的目录结构
3. **排除临时文件**: 建议在提交前清理 `__pycache__`, `.pyc`, `.log` 等临时文件

### C. 安全提示

- ⚠️ **不要在代码中硬编码 API Key**
- ⚠️ **不要将 API Key 提交到版本控制系统**
- ⚠️ **定期更换 API Key**
- ⚠️ **如怀疑 API Key 泄露，立即重新生成**

---

*最后更新时间: 2024年*
