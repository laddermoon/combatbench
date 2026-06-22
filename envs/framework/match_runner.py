"""Multi-round combat match runner.

Runs a best-of-N match between two policies with HP carry-over between
rounds.  Each round is driven by :class:`RoundRunner`; this module
orchestrates the rounds, tracks cumulative HP, and determines the winner.

Match rules
-----------
* Both sides start with ``initial_health`` HP (default 100).
* Each round runs for the duration defined in the env blueprint.
* HP carries over: the health at the end of round *k* becomes the
  starting health for round *k+1*.
* **KO**: if either side's HP reaches 0 the match ends immediately.
* Otherwise, after all rounds, the side with more round wins takes
  the match.  Equal wins = draw.

Public surface
--------------
* :class:`MatchRunner` – orchestrator class
* :class:`MatchResult` – serialisable result dataclass

CLI
---
Run ``python -m envs.framework.match_runner --help`` for usage.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .blueprint import EnvBlueprint
from .common_plugins import VideoRecorderPlugin
from .parameterized_blueprint import ParameterizedEnvBlueprint
from .policy import PolicyBlueprint
from .round_runner import RoundRunner


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    """Outcome of a complete match."""

    total_rounds: int
    rounds_completed: int
    round_results: List[Dict[str, Any]] = field(default_factory=list)
    final_winner: str = "draw"           # 'robot_a' | 'robot_b' | 'draw'
    total_score: Dict[str, int] = field(
        default_factory=lambda: {"robot_a": 0, "robot_b": 0}
    )
    ko_winner: Optional[str] = None      # set when match ends by KO
    initial_health: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# MatchRunner
# ---------------------------------------------------------------------------
class MatchRunner:
    """Run a multi-round combat match between two policies.

    Parameters
    ----------
    env_blueprint:
        A *materialised* :class:`EnvBlueprint` (no ``${...}`` placeholders).
    policy_a_bp, policy_b_bp:
        :class:`PolicyBlueprint` instances for each robot.
    total_rounds:
        Number of rounds (default 6).
    initial_health:
        Starting HP for both sides (default 100).
    verbose:
        Print per-round and final summaries to stdout.
    """

    def __init__(
        self,
        env_blueprint: EnvBlueprint,
        policy_a_bp: PolicyBlueprint,
        policy_b_bp: PolicyBlueprint,
        total_rounds: int = 6,
        initial_health: float = 100.0,
        score_log_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.env_blueprint = env_blueprint
        self.policy_a_bp = policy_a_bp
        self.policy_b_bp = policy_b_bp
        self.total_rounds = total_rounds
        self.initial_health = initial_health
        self.score_log_dir = score_log_dir
        self.verbose = verbose

    def run(
        self,
        seed: Optional[int] = None,
        video_dir: Optional[str] = None,
    ) -> MatchResult:
        """Run all rounds and return a :class:`MatchResult`.

        Parameters
        ----------
        seed:
            Base seed; per-round seeds are derived via
            :class:`numpy.random.SeedSequence`.
        video_dir:
            If given, each round's video is saved as
            ``{video_dir}/round_{n}.mp4``.
        """
        policy_a = self.policy_a_bp.build()
        policy_b = self.policy_b_bp.build()

        # Derive per-round seeds from a single SeedSequence.
        if seed is not None:
            round_seeds: List[Optional[int]] = [
                int(s) for s in
                np.random.SeedSequence(int(seed)).generate_state(
                    self.total_rounds, dtype=np.uint32,
                )
            ]
        else:
            round_seeds = [None] * self.total_rounds

        round_results: List[Dict[str, Any]] = []
        total_score = {"robot_a": 0, "robot_b": 0}
        ko_winner: Optional[str] = None
        hp_a = self.initial_health
        hp_b = self.initial_health

        if self.verbose:
            self._print_header()

        for round_num in range(1, self.total_rounds + 1):
            if self.verbose:
                print(
                    f"\n>>> Round {round_num}/{self.total_rounds}  "
                    f"HP: A={hp_a:.1f}  B={hp_b:.1f}"
                )

            video_plugin = None
            if video_dir:
                Path(video_dir).mkdir(parents=True, exist_ok=True)
                video_plugin = VideoRecorderPlugin(
                    fps=30,
                    output_path=str(Path(video_dir) / f"round_{round_num}.mp4"),
                )

            with RoundRunner(
                blueprint=self.env_blueprint,
                policy_a=policy_a,
                policy_b=policy_b,
                video_plugin=video_plugin,
            ) as runner:
                # Per-round score log: {score_log_dir}/round_{n}.log
                round_score_log = None
                if self.score_log_dir:
                    Path(self.score_log_dir).mkdir(parents=True, exist_ok=True)
                    round_score_log = str(
                        Path(self.score_log_dir) / f"round_{round_num}.log"
                    )
                result = runner.run(
                    seed=round_seeds[round_num - 1],
                    initial_health_a=hp_a,
                    initial_health_b=hp_b,
                    score_log_file=round_score_log,
                )

            hp_a = result["health_a"]
            hp_b = result["health_b"]

            # Determine per-round winner.
            if hp_a <= 0 and hp_b <= 0:
                winner = "draw"
            elif hp_a <= 0:
                winner = "robot_b"
            elif hp_b <= 0:
                winner = "robot_a"
            elif hp_a > hp_b:
                winner = "robot_a"
            elif hp_b > hp_a:
                winner = "robot_b"
            else:
                winner = "draw"

            if winner == "robot_a":
                total_score["robot_a"] += 1
            elif winner == "robot_b":
                total_score["robot_b"] += 1

            round_entry = dict(result)
            round_entry["round_num"] = round_num
            round_entry["winner"] = winner
            round_results.append(round_entry)

            if self.verbose:
                print(
                    f"    Winner: {winner}  "
                    f"HP: A={hp_a:.1f}  B={hp_b:.1f}  "
                    f"Steps: {result['steps']}  "
                    f"Terms: {result['termination_reasons']}"
                )

            # KO — end match immediately.
            if hp_a <= 0 or hp_b <= 0:
                if hp_a <= 0 and hp_b > 0:
                    ko_winner = "robot_b"
                elif hp_b <= 0 and hp_a > 0:
                    ko_winner = "robot_a"
                if self.verbose and ko_winner:
                    print(f"\n!!! {ko_winner} wins by KO !!!")
                break

        # Final winner.
        if ko_winner:
            final_winner = ko_winner
        elif total_score["robot_a"] > total_score["robot_b"]:
            final_winner = "robot_a"
        elif total_score["robot_b"] > total_score["robot_a"]:
            final_winner = "robot_b"
        else:
            final_winner = "draw"

        match_result = MatchResult(
            total_rounds=self.total_rounds,
            rounds_completed=len(round_results),
            round_results=round_results,
            final_winner=final_winner,
            total_score=total_score,
            ko_winner=ko_winner,
            initial_health=self.initial_health,
        )

        if self.verbose:
            self._print_summary(match_result)

        return match_result

    # -- printing helpers -------------------------------------------------

    def _print_header(self) -> None:
        print("=" * 60)
        print("CombatBench Match")
        print(
            f"Rounds: {self.total_rounds}  "
            f"Initial HP: {self.initial_health}"
        )
        print("=" * 60)

    def _print_summary(self, mr: MatchResult) -> None:
        print("=" * 60)
        print(f"Rounds: {mr.rounds_completed}/{mr.total_rounds}")
        if mr.ko_winner:
            print(f"Winner: {mr.ko_winner} (by KO)")
        else:
            print(f"Winner: {mr.final_winner}")
        print(
            f"Score: A={mr.total_score['robot_a']}  "
            f"B={mr.total_score['robot_b']}"
        )
        print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_env_blueprint(path: str | Path) -> EnvBlueprint:
    """Load and materialise an env blueprint from file.

    Handles both plain :class:`EnvBlueprint` and
    :class:`ParameterizedEnvBlueprint` (using default parameter values).
    """
    pb = ParameterizedEnvBlueprint.load(path)
    return pb.materialize()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a multi-round combat match.",
    )
    parser.add_argument(
        "--env-blueprint", type=str, required=True,
        help="Path to the environment blueprint (YAML or JSON).",
    )
    parser.add_argument(
        "--policy-a-blueprint", type=str, required=True,
        help="Path to policy A blueprint.",
    )
    parser.add_argument(
        "--policy-b-blueprint", type=str, required=True,
        help="Path to policy B blueprint.",
    )
    parser.add_argument(
        "--total-rounds", type=int, default=6,
        help="Number of rounds (default: 6).",
    )
    parser.add_argument(
        "--initial-health", type=float, default=100.0,
        help="Starting HP for both robots (default: 100).",
    )
    parser.add_argument(
        "--score-log-dir", type=str, default=None,
        help="Directory to save per-round combat score audit logs.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base seed (default: random).",
    )
    parser.add_argument(
        "--video-dir", type=str, default=None,
        help="Directory to save per-round videos.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write match result JSON to this path.",
    )
    args = parser.parse_args()

    env_bp = load_env_blueprint(args.env_blueprint)
    policy_a_bp = PolicyBlueprint.load(args.policy_a_blueprint)
    policy_b_bp = PolicyBlueprint.load(args.policy_b_blueprint)

    runner = MatchRunner(
        env_blueprint=env_bp,
        policy_a_bp=policy_a_bp,
        policy_b_bp=policy_b_bp,
        total_rounds=args.total_rounds,
        initial_health=args.initial_health,
        score_log_dir=args.score_log_dir,
        verbose=True,
    )
    result = runner.run(seed=args.seed, video_dir=args.video_dir)

    if args.output:
        result.save(args.output)
        print(f"\nResult saved to {args.output}")
    else:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
