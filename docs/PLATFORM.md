# Platform — combatbench.tech

> Type: Guide
> 中文版：PLATFORM_zh.md

The online evaluation platform: **a public competitive arena for humanoid
combat policies**. Website: [www.combatbench.tech](http://www.combatbench.tech)
(fallback IP: [180.76.152.227](http://180.76.152.227)).

## 1. What it is

Participants register, submit a policy directory, and the platform
automatically runs matches against other submissions, updates an Elo
leaderboard, and publishes match videos. The current (and only) arena is
**Humanoid21** (`--leaderboard-id 1`).

## 2. Value

- **Standardized proof**: identical rules, environment, and evaluation for
  every submission — a ranking is a reproducible, publicly checkable
  credential rather than a self-reported number.
- **Developer-first**: the leaderboard exists so policy authors can
  demonstrate ability (resumes, papers, community standing).
- **Auditable results**: every match produces a replayable video plus
  per-substep score data — rankings are backed by visible fights, not
  hidden scores.

## 3. Why it works this way (design rationale)

- **Self-contained submission package**: a submission is a directory —
  `policy_blueprint.yaml` + your code (`policy.py`) + optional `model.pt`
  + optional `requirements.txt`. Any implementation that satisfies the
  policy contract (`act()` returning a 21-dim action) can compete — the
  platform does not force this repo's framework on you.
- **`file:` blueprint form**: the blueprint may reference code inside the
  package (`file:${DIR}/policy.py:MyPolicy`), so submissions stay portable
  and self-contained.
- **Uniform evaluation**: every policy is evaluated under the same rules
  (`RULE.md`) and the same environment (`ENVIRONMENT.md`), which is what
  makes Elo comparisons meaningful.
- **Dependency freedom**: `requirements.txt` in the package is installed
  automatically — bring your own stack.

## 4. The loop

```
train locally (this repo: PPO framework, rollout, runners)
        │
        ▼
package a self-contained policy dir (SUBMISSION.md contract)
        │
        ▼
combat-submit --dir ./my_policy --name "..." --leaderboard-id 1
        │
        ▼
platform schedules matches automatically → Elo updates →
match videos & rankings visible on the site
```

## 5. User interface surface

| Surface | What you do there |
|---|---|
| Website | register, generate API key, browse leaderboards, watch match videos, check submission status |
| `combat-submit` CLI | `submit --dir ... --name ... --leaderboard-id 1`, `list` (see `SUBMISSION.md`) |
| Submission package | `policy_blueprint.yaml` + code + `model.pt` + `requirements.txt` |

## 6. Boundaries

- This document covers the **product and user-interface surface** only.
  The platform's server-side implementation lives in a separate private
  repository and is out of scope here.
- Rules are defined by [`RULE.md`](RULE.md) (V1.0, HP-focused) — anything
  not in it is not live.
- Rules of the arena: [`ENVIRONMENT.md`](ENVIRONMENT.md); packaging
  contract: [`SUBMISSION.md`](SUBMISSION.md); verified local stack:
  [`RUNTIME.md`](RUNTIME.md).
