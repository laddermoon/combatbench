# CombatBench Submission Tool

This directory contains pre-compiled binaries of the CombatBench submission tool.

## Download

You can also download the latest binaries from [GitHub Releases](https://github.com/laddermoon/combatbench/releases).

## Available Binaries

| Platform | Binary | Download |
|----------|--------|----------|
| **Linux (amd64)** | `combat-submit-linux-amd64` | [Download](https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-linux-amd64) |
| **macOS (Intel)** | `combat-submit-darwin-amd64` | [Download](https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-darwin-amd64) |
| **Windows (amd64)** | `combat-submit-windows-amd64.exe` | [Download](https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-windows-amd64.exe) |

## Quick Start

### 1. Download the appropriate binary for your platform

```bash
# Linux
wget https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-linux-amd64
chmod +x combat-submit-linux-amd64

# macOS (Intel)
wget https://github.com/laddermoon/combatbench/releases/latest/download/combat-submit-darwin-amd64
chmod +x combat-submit-darwin-amd64

# Windows
# Download from browser and use directly
```

### 2. Set your API Key

Get your API Key from the [CombatBench Web Platform](https://github.com/laddermoon/combatbench) after logging in.

```bash
export COMBAT_API_KEY="your-api-key-here"
```

### 3. Submit your policy

```bash
./combat-submit-linux-amd64 submit \
  --leaderboard-id 1 \
  --name "MyPolicy" \
  --dir ../my_policy \
  --desc "My awesome combat policy"
```

## Documentation

For complete usage instructions, see the [User Guide](../docs/SUBMISSION_GUIDE.md) or [中文指南](../docs/SUBMISSION_GUIDE_zh.md).

## Security

- Always verify the binary checksum before running
- Download only from official GitHub releases
- Report security issues to the maintainers

## License

See [LICENSE](../../LICENSE) for details.
