# Platform — combatbench.tech

> 类型：指南
> English: PLATFORM.md

在线评测平台：**人形机器人格斗策略的公开竞技平台**。
站点：[www.combatbench.tech](http://www.combatbench.tech)
（备用 IP：[180.76.152.227](http://180.76.152.227)）。

## 1. 这是什么

参赛者注册、提交策略目录，平台自动组织与其他提交的对战，
更新 Elo 榜单并公开比赛视频。当前唯一赛道是 **Humanoid21**
（`--leaderboard-id 1`）。

## 2. 价值

- **标准化能力证明**：所有提交用同一套规则、同一个环境、同一种
  评测——排名是可复现、可公开复核的凭证，而不是自报数字。
- **开发者优先**：榜单存在的意义是让策略作者展示能力
  （简历、论文、社区声望）。
- **结果可审计**：每场比赛产出可回放视频 + 逐物理步计分数据——
  排名背后是可看的对局，不是黑箱分数。

## 3. 为什么能做到（设计依据）

- **自包含提交包**：提交物是一个目录——`policy_blueprint.yaml` +
  你的代码（`policy.py`）+ 可选 `model.pt` + 可选 `requirements.txt`。
  任何实现只要满足策略契约（`act()` 返回 21 维动作）即可参赛——
  平台不绑定本仓框架。
- **`file:` 蓝图形态**：蓝图可引用包内代码
  （`file:${DIR}/policy.py:MyPolicy`），提交包保持可移植、自包含。
- **评测统一**：所有策略在同一规则（`RULE.md`）同一环境
  （`ENVIRONMENT.md`）下评测——这是 Elo 可比较的前提。
- **依赖自由**：包内 `requirements.txt` 会被自动安装——自带技术栈。

## 4. 完整闭环

```
本地训练（本仓：PPO 框架、rollout、runner）
        │
        ▼
打包自包含策略目录（SUBMISSION.md 契约）
        │
        ▼
combat-submit --dir ./my_policy --name "..." --leaderboard-id 1
        │
        ▼
平台自动组织对战 → Elo 更新 → 站内可查视频与排名
```

## 5. 用户接口面

| 界面 | 做什么 |
|---|---|
| 网站 | 注册、生成 API Key、看榜单、看比赛视频、查提交状态 |
| `combat-submit` CLI | `submit --dir ... --name ... --leaderboard-id 1`、`list`（见 `SUBMISSION.md`） |
| 提交包 | `policy_blueprint.yaml` + 代码 + `model.pt` + `requirements.txt` |

## 6. 边界

- 本文只覆盖**产品侧与用户接口侧**。平台服务端实现位于私有仓，
  不属于本文范围。
- 规则以 [`RULE.md`](RULE.md)（V1.0 纯血条制）为准——不在其中的
  规则均未上线。
- 场地规则见 [`ENVIRONMENT.md`](ENVIRONMENT.md)；提交包契约见
  [`SUBMISSION.md`](SUBMISSION.md)；本地已验证环境栈见
  [`RUNTIME.md`](RUNTIME.md)。
