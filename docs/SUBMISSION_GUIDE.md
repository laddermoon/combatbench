# CombatBench Algorithm Submission Tool - User Guide

This guide is for users participating in the CombatBench leaderboard competitions, explaining how to use the `combat-submit` tool to submit your algorithm code.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Getting an API Key](#getting-an-api-key)
- [Preparing Your Algorithm Code](#preparing-your-algorithm-code)
- [Submitting Your Algorithm](#submitting-your-algorithm)
- [Viewing Leaderboards](#viewing-leaderboards)
- [Checking Submission Status](#checking-submission-status)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before using the submission tool, ensure you have:

1. **Registered Account**: You need to register an account on the CombatBench platform
2. **API Key**: You need an API Key for authentication when submitting code
3. **Submission Tool**: Obtain the `combat-submit` executable
4. **Algorithm Code Ready**: Your algorithm code should be organized according to requirements

---

## Quick Start

### Simplest Submission Method

```bash
# 1. Set your API Key
export COMBAT_API_KEY="your-api-key-here"

# 2. Submit your algorithm
./combat-submit submit \
  --leaderboard-id 1 \
  --name "My Algorithm" \
  --dir /path/to/your/code
```

### Complete Example

```bash
# Set environment variables
export COMBAT_API_KEY="sk_abcd1234..."
export UPLOAD_SERVER="http://your-server:10001"  # Optional, has default value
export COMBATBENCH_API_URL="http://your-server:10000"  # Optional

# Submit command
./combat-submit submit \
  --leaderboard-id 1 \
  --name "PPO Algorithm v2.0" \
  --dir ./my_policy \
  --desc "Using PPO algorithm, trained for 1M steps"
```

---

## Getting an API Key

### Through Web Interface

1. Visit the CombatBench Web platform
2. Log in to your account
3. Go to your profile/settings page
4. Click "Generate API Key" or "Regenerate API Key"
5. Copy the generated API Key (format: `sk_xxxxx...`)

### API Key Security Guidelines

- **Keep it Safe**: Your API Key is like your account password, do not share it with others
- **Rotate Regularly**: It's recommended to rotate your API Key periodically for security
- **Minimum Privilege**: The API Key is only for code submission, do not use it for other purposes

---

## Preparing Your Algorithm Code

### Directory Structure Requirements

Your algorithm code directory should contain the following:

```
my_policy/
├── agent.py              # Required: Your agent implementation
├── config.json           # Optional: Algorithm configuration parameters
├── requirements.txt      # Optional: Python dependencies
├── README.md             # Optional: Algorithm description
└── models/               # Optional: Pre-trained model files
    └── model.pkl
```

### Code Guidelines

1. **Entry File**: Must include an entry function that can be called by the evaluation system
2. **Dependency Declaration**: If there are third-party dependencies, declare them in `requirements.txt`
3. **File Size**: The compressed directory should be under 500MB
4. **Encoding**: UTF-8 encoding is recommended

---

## Submitting Your Algorithm

### Command Format

```bash
./combat-submit submit [OPTIONS]
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--leaderboard-id` | Leaderboard ID | `1`, `2`, `3` |
| `--name` | Submission name | `"My PPO Algorithm"` |
| `--dir` | Algorithm code directory | `./my_policy` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--desc` | Algorithm description | None |
| `--chunk-size` | Chunk size in MB | 5 |

### Submission Workflow

The submission tool automatically performs the following steps:

```
┌─────────────────┐
│ 1. Verify API Key │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Create Submission │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Package to ZIP │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Upload in Chunks │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Update Status │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Submission Complete! │
└─────────────────┘
```

### Upload Progress Display

Progress information is displayed during upload:

```
Submitting to leaderboard #1...
Creating submission record... ✓
Packaging code... ✓ (12.5 MB)
Uploading: [████████████████████████████] 100% (12.5 MB / 12.5 MB)
Submission successful! Submission ID: 123
```

---

## Viewing Leaderboards

### List All Available Leaderboards

```bash
./combat-submit leaderboards
```

### Output Example

```
Available Leaderboards:

ID  Name                          Description
1   1v1 Combat Leaderboard        Single combat mode leaderboard
2   3v3 Team Battle Leaderboard   Three-player team battle mode
3   Maze Challenge Leaderboard    Pathfinding challenge
```

---

## Checking Submission Status

After submission, you can check your submission status through:

### Via Command Line (if supported)

```bash
./combat-submit status <submission-id>
```

### Via Web Interface

1. Visit the CombatBench Web platform
2. Log in to your account
3. Go to "My Submissions" page
4. View submission status and evaluation results

### Submission Status Meanings

| Status | Description |
|--------|-------------|
| `pending` | Waiting for evaluation |
| `running` | Evaluation in progress |
| `completed` | Evaluation completed |
| `failed` | Evaluation failed |
| `error` | System error |

---

## FAQ

### Q1: Submission failed with "Invalid API Key"

**Cause**: API Key is invalid or not set

**Solution**:
```bash
# Check if environment variable is set
echo $COMBAT_API_KEY

# Reset it
export COMBAT_API_KEY="your-correct-api-key"
```

### Q2: Upload interrupted, what should I do?

**Solution**: Simply re-run the same submission command, the tool will automatically resume from where it left off:

```bash
# Re-run, will auto-resume
./combat-submit submit \
  --leaderboard-id 1 \
  --name "My Algorithm" \
  --dir ./my_policy
```

### Q3: How do I know which leaderboard to submit to?

**Solution**: Use the `leaderboards` command to view all available leaderboards:

```bash
./combat-submit leaderboards
```

### Q4: Code directory is too large, upload is very slow

**Solution**:
- Check if unnecessary large files are included (like datasets, log files)
- You can use a larger chunk size (but not exceeding 10MB):
```bash
./combat-submit submit --leaderboard-id 1 --name "My Algorithm" --dir ./my_policy --chunk-size 10
```

### Q5: Evaluation failed, how to view error messages?

**Solution**:
- View detailed logs through the Web interface
- Check if your code meets the evaluation system's interface requirements
- Ensure all dependencies are declared in `requirements.txt`

---

## Troubleshooting

### Failed to Connect to Server

**Error Message**: `connection refused` or `timeout`

**Troubleshooting Steps**:

1. Check network connection
```bash
ping your-server
```

2. Check server address configuration
```bash
echo $UPLOAD_SERVER
echo $COMBATBENCH_API_URL
```

3. If custom server address is needed:
```bash
export UPLOAD_SERVER="http://your-server:10001"
export COMBATBENCH_API_URL="http://your-server:10000"
```

### Permission Error

**Error Message**: `401 Unauthorized` or `403 Forbidden`

**Troubleshooting Steps**:

1. Confirm API Key is correct
2. Check if API Key has expired
3. Try regenerating the API Key

### File Validation Failed

**Error Message**: `400 Bad Request` or file validation related errors

**Troubleshooting Steps**:

1. Check if directory contains disallowed file types
2. Confirm compressed file size does not exceed 500MB
3. Check if filename contains special characters

---

## Environment Variables Reference

| Environment Variable | Description | Default | Required |
|---------------------|-------------|---------|----------|
| `COMBAT_API_KEY` | Your API Key | None | **Yes** |
| `UPLOAD_SERVER` | Upload server address | `http://localhost:10001` | No |
| `COMBATBENCH_API_URL` | WebAPI address | `http://localhost:10000` | No |

---

## Technical Support

If you encounter issues, get help through:

1. **Read Documentation**: First read this guide and related technical documentation
2. **GitHub Issues**: Submit issues in the project repository
3. **Contact Admin**: Contact technical support through official channels

---

## Appendix

### A. Complete Command Example

```bash
#!/bin/bash
# Complete submission workflow example

# 1. Set environment variables
export COMBAT_API_KEY="sk_your_api_key_here"
export UPLOAD_SERVER="http://combatbench.example.com:10001"
export COMBATBENCH_API_URL="http://combatbench.example.com:10000"

# 2. View available leaderboards
./combat-submit leaderboards

# 3. Submit algorithm
./combat-submit submit \
  --leaderboard-id 1 \
  --name "PPO-v2.0" \
  --dir ./my_ppo_policy \
  --desc "Using PPO algorithm, trained for 1M steps, LR=0.0003"

# 4. Wait for evaluation to complete, then check results via Web interface
```

### B. Directory Packaging Rules

The submission tool automatically packages your directory into a ZIP file, following these rules:

1. **Include All Files**: All files and subdirectories in the directory will be included
2. **Preserve Structure**: The original directory structure is maintained in the ZIP
3. **Exclude Temp Files**: It's recommended to clean up `__pycache__`, `.pyc`, `.log` and other temporary files before submission

### C. Security Tips

- ⚠️ **Do NOT hardcode API Key in your code**
- ⚠️ **Do NOT commit API Key to version control systems**
- ⚠️ **Rotate API Key regularly**
- ⚠️ **Regenerate immediately if you suspect API Key compromise**

---

*Last Updated: 2024*
