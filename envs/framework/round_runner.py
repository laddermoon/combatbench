from typing import Any, Dict, Optional
import numpy as np

from .common_plugins import VideoRecorderPlugin

class RoundRunner:
    """
    运行单局对战的封装类，支持 Policy 加载和视频录制。
    """
    def __init__(
        self,
        policy_a: Any,
        policy_b: Any,
        runtime: Any,
        verbose: bool = True
    ):
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.verbose = verbose
        
        self.runtime = runtime

    def _print_header(self):
        if not self.verbose: return
        print("=" * 60)
        print("CombatBench Round Started")
        print("=" * 60)

    def run(self, seed: Optional[int] = None, videosave_path: Optional[str] = None) -> Dict[str, Any]:
        if videosave_path is not None :
            VideoRecorderPlugin.set_videosave_path(videosave_path)
        result = self.runtime.reset(seed=seed)
        obs = result["obs"]
        info = result["info"]
        
        if hasattr(self.policy_a, "reset"): self.policy_a.reset()
        if hasattr(self.policy_b, "reset"): self.policy_b.reset()

        self._print_header()

        step_count = 0
        action_dim = self._resolve_action_dim()

        while True:
            try:
                act_a = self.policy_a.act(
                    self._extract_agent_obs(obs, "robot_a"),
                    self._build_policy_info(info, "robot_a"),
                ) if hasattr(self.policy_a, 'act') else np.zeros(action_dim)
            except Exception as e:
                if self.verbose: print(f"Warning: Policy A failed: {e}")
                act_a = np.zeros(action_dim)

            try:
                act_b = self.policy_b.act(
                    self._extract_agent_obs(obs, "robot_b"),
                    self._build_policy_info(info, "robot_b"),
                ) if hasattr(self.policy_b, 'act') else np.zeros(action_dim)
            except Exception as e:
                if self.verbose: print(f"Warning: Policy B failed: {e}")
                act_b = np.zeros(action_dim)

            result = self.runtime.step(
                np.asarray(act_a, dtype=np.float32),
                np.asarray(act_b, dtype=np.float32),
            )
            obs = result["obs"]
            info = result["info"]
            terminated = result["terminated"]
            truncated = result["truncated"]
            step_count += 1

            shared_info = self._extract_shared_info(info)
            if self.verbose:
                for event in shared_info.get('events', []):
                    if event['type'] == 'hit':
                        print(f"[Step {step_count}] {event['attacker']} hit {event['defender']} at {event['part']} for {event['damage']:.2f} damage!")

            if step_count % 100 == 0 and self.verbose:
                health = self._extract_health(shared_info, info)
                ha = health.get('robot_a', 100.0)
                hb = health.get('robot_b', 100.0)
                print(f"Step {step_count:03d} - HP: robot_a={ha:.1f}, robot_b={hb:.1f}")

            if terminated or truncated:
                break

        shared_info = self._extract_shared_info(info)
        final_health = self._extract_health(shared_info, info)
        result = {
            "steps": step_count,
            "winner": self._resolve_winner(info, shared_info, final_health),
            "final_health": final_health,
            "damage_taken": self._extract_damage_taken(shared_info, info),
            "termination_reasons": shared_info.get("termination_reasons", info.get("termination_reasons", []))
        }

        if self.verbose:
            print("-" * 60)
            print(f"Round ended. Total steps: {step_count}")
            print(f"Reason: {result['termination_reasons']}")
            print(f"Winner: {result['winner']}")
            print(f"Final HP: robot_a={result['final_health'].get('robot_a', 0):.1f}, robot_b={result['final_health'].get('robot_b', 0):.1f}")
            print("-" * 60)

        self.runtime.close()
        return result

    def _resolve_action_dim(self) -> int:
        action_space = getattr(self.runtime, "action_space", None)
        if action_space is not None and hasattr(action_space, "spaces") and "robot_a" in action_space.spaces:
            return int(action_space.spaces["robot_a"].shape[0])
        for policy in (self.policy_a, self.policy_b):
            if hasattr(policy, "ACTION_DIM"):
                return int(policy.ACTION_DIM)
        return 21

    def _extract_agent_obs(self, obs: Any, agent_id: str) -> Any:
        return obs[agent_id]

    def _build_policy_info(self, info: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        opponent_id = "robot_b" if agent_id == "robot_a" else "robot_a"
        shared_info = info["shared"]
        agent_info = info[agent_id]
        opponent_info = info[opponent_id]
        policy_info = dict(shared_info)
        policy_info["shared"] = shared_info
        policy_info["self"] = agent_info
        policy_info["opponent"] = opponent_info
        policy_info.update(agent_info)
        return policy_info

    def _extract_shared_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return info["shared"]

    def _extract_health(self, shared_info: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, float]:
        return dict(shared_info["health"])

    def _extract_damage_taken(self, shared_info: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, float]:
        return dict(shared_info["damage_taken"])

    def _resolve_winner(self, info: Dict[str, Any], shared_info: Dict[str, Any], final_health: Dict[str, float]) -> str:
        if isinstance(shared_info.get("winner"), str):
            return shared_info["winner"]
        health_a = float(final_health.get("robot_a", 0.0))
        health_b = float(final_health.get("robot_b", 0.0))
        if health_a <= 0.0 and health_b <= 0.0:
            return "draw"
        if health_a <= 0.0:
            return "robot_b"
        if health_b <= 0.0:
            return "robot_a"
        if health_a > health_b:
            return "robot_a"
        if health_b > health_a:
            return "robot_b"
        return "draw"


if __name__ == "__main__":
    from envs.humanoid21 import make_env
    
    runtime = make_env(
        plugins=[VideoRecorderPlugin(fps=30, output_path="match.mp4")],
        match_duration=30.0
    )
    
    class DummyPolicy:
        def act(self, obs, info):
            return [0.0] * 21
    
    policy_a = DummyPolicy()
    policy_b = DummyPolicy()
    
    runner = RoundRunner(policy_a, policy_b, runtime)
    result = runner.run(seed=42)
    
    print(f"Result: {result}")
