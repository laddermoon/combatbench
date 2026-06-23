# 策略提交

## 1. 安装工具

```bash
curl -sSL http://www.combatbench.tech/install.sh | bash
```

## 2. 设置 API Key

网页注册账号，个人设置页生成 API Key，然后：

```bash
export COMBAT_API_KEY="sk_你的密钥"
```

## 3. 策略目录

提交一个目录，里面至少包含：

```
my_policy/
├── policy_blueprint.yaml   # 必需，入口配置
├── my_code.py              # 策略代码（文件名任意，由 blueprint 的 cls 指定）
├── model.pt                # 可选，模型权重
└── requirements.txt        # 可选，Python 依赖
```

`policy_blueprint.yaml` 用 `${DIR}` 引用同目录下的文件，打包提交后路径自动正确：

```yaml
version: 1
cls: "file:${DIR}/policy.py:MyPolicy"
config:
  stochastic: false
```

子目录里的文件同理用 `${DIR}` 引用，例如 `${DIR}/fallback/policy.py`。

## 4. 提交

```bash
combat-submit submit --dir ./my_policy --name "xxx" --leaderboard-id 1
```

`--leaderboard-id 1` 是 Humanoid21 环境，目前唯一支持的榜单。

上传支持断点续传，中断了重跑同一命令即可。

查看已提交的记录：

```bash
combat-submit list
```

## 5. 看结果

网页「我的提交」页面看状态和比赛视频。

---

**有第三方依赖**：目录里放 `requirements.txt`，平台自动安装。
