#!/usr/bin/env python3
"""
Humanoid 21 Training Log Monitor and Diagnostic Tool.

Parses machine-readable JSON logs from curriculum training and provides high-signal
diagnostic reports with clear conclusions, causes, and step-by-step evidence.

Usage:
    # 1. Analyze an existing log file
    python3 analyze_logs.py balance_recover8.log

    # 2. Watch a running log in real-time (like tail -f)
    python3 analyze_logs.py balance_recover8.log --watch
"""

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional


class LogAnalyzer:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.last_analysed_update = -1

    def feed_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Feed a log line. If it contains raw stats, parse and store it."""
        if "__RAW_STATS__" in line:
            try:
                # Find JSON payload
                json_str = line.split("__RAW_STATS__", 1)[1].strip()
                data = json.loads(json_str)
                self.history.append(data)
                return data
            except Exception as e:
                pass
        return None

    def run_diagnostics(self) -> List[Dict[str, Any]]:
        """Run diagnostic checks over the current history window."""
        if len(self.history) < 3:
            # Need at least 3 updates to establish a trend
            return []

        conclusions = []
        updates = [data["update"] for data in self.history]
        u_start, u_end = self.history[0]["update"], self.history[-1]["update"]

        # ----------------------------------------------------------------------
        # Check A: Exploration / Std Collapse (🚨 探索方差塌缩)
        # ----------------------------------------------------------------------
        std_mins = [d["stats"].get("std_min", 1.0) for d in self.history]
        std_means = [d["stats"].get("std_mean", 1.0) for d in self.history]
        avg_std_min = sum(std_mins) / len(std_mins)
        avg_std_mean = sum(std_means) / len(std_means)

        if avg_std_min <= 0.145:
            conclusions.append({
                "severity": "CRITICAL",
                "title": "关节探索能力濒临枯竭 (Exploration Collapse)",
                "conclusion": "至少有一个自由度关节（关节标准差已锁死在底线 0.13）失去了尝试新动作的能力。机器人正陷入刻板的肌肉硬记忆，极难在更高的扰动下泛化。",
                "evidence": f"在 Update {u_start} 到 {u_end} (10代窗口) 内：\n"
                            f"  - 最小单关节标准差 std_min 平均值为 {avg_std_min:.4f} (极限底线约为 0.13)\n"
                            f"  - 全关节平均标准差 std_mean 均值为 {avg_std_mean:.4f}\n"
                            f"  - 历史轨迹 std_min 走势: {[round(x, 3) for x in std_mins]}",
                "remedy": "1. 适当调大 policy 网络初始化或训练中的 log_std_min (例如在 config.py 中调至 -2.5 或 -2.0)。\n"
                          "2. 增加探索熵系数 entropy_coef 的值，或者混入部分低难度随机重放。"
            })

        # ----------------------------------------------------------------------
        # Check B: PPO Update Early Stop too fast (🚨 学习率过高)
        # ----------------------------------------------------------------------
        epochs_dones = [d["stats"].get("epochs_done", 0) for d in self.history]
        avg_epochs = sum(epochs_dones) / len(epochs_dones)
        # We assume the default maximum epochs is around 5
        if avg_epochs <= 1.8:
            conclusions.append({
                "severity": "WARNING",
                "title": "PPO 更新步子迈得太大 (Learning Rate Too High)",
                "conclusion": "策略更新在 1 到 2 个 Epoch 内就由于 KL 散度超标（Trust Region 破裂）而戛然而止。数据刚跑出来就被扔掉，利用率极低，容易卡死。",
                "evidence": f"在 Update {u_start} 到 {u_end} (10代窗口) 内：\n"
                            f"  - 实际完成 Epochs 平均值仅为 {avg_epochs:.1f} (最大 epoch 限制为 5)\n"
                            f"  - 各代实际完成 Epochs 分布: {epochs_dones}",
                "remedy": "1. 强烈建议将 Actor 的 Learning Rate 主动调小 30% 到 50% (例如从 5e-5 降低到 2e-5)，动作步幅变细腻后，能跑满 4-5 个 epoch。\n"
                          "2. 确认 Advantage 标准化是否被正常开启。"
            })

        # ----------------------------------------------------------------------
        # Check C: Critic Broken (🚨 价值评估失真 / 负解释方差)
        # ----------------------------------------------------------------------
        # Find active reward components
        first_data = self.history[-1]
        active_components = []
        for key in first_data["stats"].keys():
            if key.startswith("ev_"):
                # ev_{key}
                comp_name = key[3:]
                active_components.append(comp_name)

        for comp in active_components:
            evs = [d["stats"].get(f"ev_{comp}", 0.0) for d in self.history]
            avg_ev = sum(evs) / len(evs)
            
            # Check if this component has non-zero weight (is being actively trained)
            # Find weight index in experiment weights
            weights = first_data.get("weights", [])
            # If EV is consistently negative or zero
            if avg_ev <= 0.05:
                conclusions.append({
                    "severity": "CRITICAL",
                    "title": f"Critic 价值评估网络崩溃 ({comp} Critic Is Blind)",
                    "conclusion": f"负责评估奖励 '{comp}' 的 Critic 价值网络的解释方差 (Explained Variance) 逼近于 0 甚至为负数。这代表该 Critic 无法在当前状态下预测未来的回报，正在输出完全失真、黑白颠倒的 Advantage 信号，误导 Actor 学习！",
                    "evidence": f"在 Update {u_start} 到 {u_end} (10代窗口) 内：\n"
                                f"  - '{comp}' 对应的 Explained Variance 平均值为 {avg_ev:+.4f}\n"
                                f"  - 历史 EV 拟合走势: {[round(x, 3) for x in evs]}\n"
                                f"  - 该 Critic 的当前 value_loss 平均值: {sum([d['stats'].get(f'vloss_{comp}', 0) for d in self.history]) / len(self.history):.4f}",
                    "remedy": f"1. 开启非对称学习率：提高 {comp} Critic 的学习率，设为 Actor 的 3~4 倍 (例如 Actor 3e-5, Critic 1.2e-4)。\n"
                              f"2. 减小其折现因子 Gamma：如果是恢复平衡这种短长远任务，将其 gamma 降到 0.95 或 0.90，降低 Critic 预测难度。\n"
                              f"3. 扩充 Critic 网络的隐藏层容量 (Hidden Dimension) 或网络层数。"
                })

        # ----------------------------------------------------------------------
        # Check D: Rollout Survival Death Spiral (🚨 存活坠入黑洞)
        # ----------------------------------------------------------------------
        ep_means = [d["stats"].get("ep_len_mean", 1000.0) for d in self.history]
        avg_ep_mean = sum(ep_means) / len(ep_means)
        if avg_ep_mean <= 60.0:
            conclusions.append({
                "severity": "CRITICAL",
                "title": "机器人遭遇‘零成功率’探索黑洞",
                "conclusion": "机器人在本级扰动下出生即大批死亡，平均存活极短。Rollout 中 100% 充斥着失败倒地轨迹，没有任何能成功站立的数据。Actor 处于绝对黑暗中，没有梯度方向可以借鉴。",
                "evidence": f"在 Update {u_start} 到 {u_end} (10代窗口) 内：\n"
                            f"  - Rollout 平均 Episode 存活长度仅为 {avg_ep_mean:.1f} 步\n"
                            f"  - 历史平均存活步数走势: {[round(x, 1) for x in ep_means]}",
                "remedy": "1. 强烈建议引入【混合课程重放 (Mixed Batch)】：修改 exp_balance_recover.py 让 50% 的 episode 保持低一级难度扰动 (提供成功基准信号)，50% 尝试当前高难度扰动。\n"
                          "2. 减少扰动的增加步幅，或设计挣扎存活的部分奖励 (Shaping Reward)。"
                })

        return conclusions

    def analyze_progress_trend(self) -> str:
        """Analyze the progress trend in the current history window."""
        if len(self.history) < 5:
            return "\033[94m    [趋势] 正在收集数据以确立训练趋势 (至少需要 5 代更新)...\033[0m"

        # Calculate average of the first 2-3 updates vs last 2-3 updates in window
        half = min(3, len(self.history) // 2)
        first_group = list(self.history)[:half]
        last_group = list(self.history)[-half:]

        first_ep_mean = sum(d["stats"].get("ep_len_mean", 0.0) for d in first_group) / len(first_group)
        last_ep_mean = sum(d["stats"].get("ep_len_mean", 0.0) for d in last_group) / len(last_group)
        delta_ep = last_ep_mean - first_ep_mean

        first_surv = sum(d["sinfo"].get("survival_rate", 0.0) for d in first_group) / len(first_group)
        last_surv = sum(d["sinfo"].get("survival_rate", 0.0) for d in last_group) / len(last_group)
        # Fallback to eval survival in bsum if sinfo survival_rate is 0 (or fallback to survived)
        if first_surv == 0.0:
            first_surv = sum(d["bsum"].get("survived", 0.0) for d in first_group) / len(first_group)
        if last_surv == 0.0:
            last_surv = sum(d["bsum"].get("survived", 0.0) for d in last_group) / len(last_group)
        delta_surv = last_surv - first_surv

        current_level = self.history[-1]["sinfo"].get("level", 0)
        first_level = self.history[0]["sinfo"].get("level", 0)

        # Build output message
        status_lines = []
        if current_level > first_level:
            status_lines.append(f"    \033[92m🎉 史诗级进展：成功晋级！课程难度从 Level {first_level} 晋升到 Level {current_level}！\033[0m")
        
        # Decide progress state
        if delta_ep >= 8.0:
            status_lines.append(f"    \033[92m📈 训练正在大幅稳步前进！平均存活时间较 {len(self.history)} 代前提升了 +{delta_ep:.1f} 步！(存活率变动: {delta_surv:+.1%})\033[0m")
        elif delta_ep >= 3.0:
            status_lines.append(f"    \033[92m📈 训练正在微幅上升中。平均存活时间较 {len(self.history)} 代前提升了 +{delta_ep:.1f} 步。\033[0m")
        elif delta_ep <= -8.0:
            status_lines.append(f"    \033[91m📉 警告：检测到策略发生明显退化！存活步数下滑了 {delta_ep:.1f} 步，机器人基本站立基础可能在发生灾难性遗忘！\033[0m")
        elif delta_ep <= -3.0:
            status_lines.append(f"    \033[93m📉 提示：存活步数出现轻微波动下滑 ({delta_ep:.1f} 步)。\033[0m")
        else:
            # Flat (Stalled)
            # Check if survival is already perfect
            if last_ep_mean >= 195.0 or last_surv >= 0.98:
                status_lines.append(f"    \033[92m✨ 策略已臻化境：机器人在当前难度下已接近完美收敛 (平均存活 {last_ep_mean:.1f} 步，存活率 {last_surv:.1%})，等待晋级。\033[0m")
            else:
                status_lines.append(f"    \033[93m⚠️ 警告：训练进展陷入停滞瓶颈！过去 {len(self.history)} 代中存活步数原地踏步 (仅波动 {delta_ep:+.1f} 步，均值 {last_ep_mean:.1f})。机器人正困在当前阶段苦苦挣扎！\033[0m")

        return "\n".join(status_lines)


def tail_file(file_path):
    """Yield new lines from a file as they are written."""
    try:
        f = open(file_path, "r", encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Go to the end of file
    f.seek(0, os.SEEK_END)
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line


def format_report(conclusions: List[Dict[str, Any]]):
    """Print the diagnostic report nicely."""
    if not conclusions:
        print("\n\033[92m[HEALTHY] 诊断结果：该滑动窗口内未发现结构性异常，训练正在稳步进行中。\033[0m")
        return

    print(f"\n\033[93m============================================================\033[0m")
    print(f"\033[93m          强化学习训练异常诊断报告 (滑动窗口检测)          \033[0m")
    print(f"\033[93m============================================================\033[0m")

    for i, item in enumerate(conclusions, 1):
        color = "\033[91m" if item["severity"] == "CRITICAL" else "\033[93m"
        reset = "\033[0m"
        print(f"\n{color}[问题 {i}] {item['title']} ({item['severity']}){reset}")
        print(f"\n\033[1m【结论】\033[0m\n  {item['conclusion']}")
        print(f"\n\033[1m【依据 & 数据支撑】\033[0m\n{item['evidence']}")
        print(f"\n\033[1m【专家破局建议】\033[0m\n{item['remedy']}")
        print(f"------------------------------------------------------------")


def print_update_progress(data: Dict[str, Any], trend_msg: str):
    """Print a highly-readable big summary of the current update progress."""
    u = data["update"]
    sinfo = data.get("sinfo", {})
    stats = data.get("stats", {})
    bsum = data.get("bsum", {})

    level = sinfo.get("level", 0)
    scale = sinfo.get("perturb_scale", 0.0)
    consecutive = sinfo.get("consecutive_pass", 0)
    surv_rate = sinfo.get("survival_rate", 0.0)
    # Fallback to eval bsum
    if surv_rate == 0.0 and "survived" in bsum:
        surv_rate = bsum.get("survived", 0.0)

    ep_mean = stats.get("ep_len_mean", 0.0)
    ep_min = stats.get("ep_len_min", 0.0)
    ep_max = stats.get("ep_len_max", 0.0)

    std_mean = stats.get("std_mean", 0.0)
    std_min = stats.get("std_min", 0.0)
    std_max = stats.get("std_max", 0.0)
    entropy = stats.get("entropy", 0.0)
    epochs = stats.get("epochs_done", 0)

    print(f"\n\033[94m============================================================\033[0m")
    print(f"\033[1;94m>>> [Update {u:4d} 实时训练进度快报] <<<\033[0m")
    print(f"\033[94m============================================================\033[0m")
    print(f"  \033[1m【课程进度】\033[0m 难度级别: \033[1;92mLevel {level}\033[0m | 扰动系数: \033[92m{scale:.3f}\033[0m | 连过次数: {consecutive} | 评估存活率: \033[1;92m{surv_rate:.1%}\033[0m")
    print(f"  \033[1m【存活时长】\033[0m 平均站立: \033[1;92m{ep_mean:.1f} 步\033[0m (最惨: {ep_min:.1f} 步 | 最佳撑过: \033[92m{ep_max:.1f} 步\033[0m)")
    print(f"  \033[1m【探索状态】\033[0m 策略熵: {entropy:.2f} | 关节标准差: {std_mean:.3f} (min={std_min:.3f}, max={std_max:.3f}) | 更新Epochs: {epochs}")
    print(f"  \033[1m【核心趋势检测】\033[0m\n{trend_msg}")
    print(f"\033[94m============================================================\033[0m")


def main():
    parser = argparse.ArgumentParser(description="Humanoid 21 RL Training Log Diagnostics.")
    parser.add_argument("log_file", type=str, help="Path to the training log file.")
    parser.add_argument("--watch", action="store_true", help="Watch file and run real-time diagnostics.")
    parser.add_argument("--window", type=int, default=10, help="Diagnostic sliding window size.")
    args = parser.parse_args()

    analyzer = LogAnalyzer(window_size=args.window)

    print(f"Initializing log analysis on '{args.log_file}' with window_size={args.window}...")

    # Read existing contents
    try:
        with open(args.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{args.log_file}' not found.", file=sys.stderr)
        sys.exit(1)

    parsed_count = 0
    for line in lines:
        data = analyzer.feed_line(line)
        if data:
            parsed_count += 1

    print(f"Parsed {parsed_count} historical updates with machine-readable stats.")
    
    # Run historical analysis
    conclusions = analyzer.run_diagnostics()
    if parsed_count > 0:
        last_data = analyzer.history[-1]
        trend_msg = analyzer.analyze_progress_trend()
        print_update_progress(last_data, trend_msg)
        format_report(conclusions)
    else:
        print("\033[93m[WARN] 未在历史日志中发现包含 '__RAW_STATS__' 标签的机器可读数据。机器可读日志刚刚才被启用，如果是旧日志，需要等待新一轮训练产出该数据。\033[0m")

    if args.watch:
        print(f"\n\033[94m[WATCH] 正在实时监控日志文件 '{args.log_file}' 的写入... (按 Ctrl+C 退出)\033[0m")
        try:
            for line in tail_file(args.log_file):
                data = analyzer.feed_line(line)
                if data:
                    trend_msg = analyzer.analyze_progress_trend()
                    print_update_progress(data, trend_msg)
                    # Every time we get a new update, re-run diagnostics
                    conclusions = analyzer.run_diagnostics()
                    if conclusions:
                        format_report(conclusions)
                    else:
                        print("\n\033[92m[HEALTHY] 诊断结果：该滑动窗口内未发现结构性异常，训练正在稳步进行中。\033[0m")
                    print(f"\n\033[94m[WATCH] 正在实时监控日志中...\033[0m")
        except KeyboardInterrupt:
            print("\nExiting watch mode.")


if __name__ == "__main__":
    main()
