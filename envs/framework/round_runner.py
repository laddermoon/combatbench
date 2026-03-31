from typing import Any, Dict, Optional
import numpy as np

from .common_plugins import VideoRecorderPlugin
from .combat_gym import CombatGymEnv


class RoundRunner:
    """
    运行单局对战的封装类，支持 Policy 加载和视频录制。
    """
    def __init__(
        self,
        policy_a: Any,
        policy_b: Any,
        env: CombatGymEnv,
        verbose: bool = True
    ):
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.verbose = verbose
        
        self.env = env

    def _print_header(self):
        if not self.verbose: return
        print("=" * 60)
        print("CombatBench Round Started")
        print("=" * 60)

    def run(self, seed: Optional[int] = None, videosave_path: Optional[str] = None) -> Dict[str, Any]:
        if videosave_path is not None :
            VideoRecorderPlugin.set_videosave_path(videosave_path)
        obs, info = self.env.reset(seed=seed)
        
        if hasattr(self.policy_a, "reset"): self.policy_a.reset()
        if hasattr(self.policy_b, "reset"): self.policy_b.reset()

        self._print_header()

        step_count = 0
        action_dim = self.env.action_space.spaces["robot_a"].shape[0]

        while True:
            # 获取动作
            try:
                act_a = self.policy_a.act(obs["robot_a_obs"], info) if hasattr(self.policy_a, 'act') else np.zeros(action_dim)
            except Exception as e:
                if self.verbose: print(f"Warning: Policy A failed: {e}")
                act_a = np.zeros(action_dim)

            try:
                act_b = self.policy_b.act(obs["robot_b_obs"], info) if hasattr(self.policy_b, 'act') else np.zeros(action_dim)
            except Exception as e:
                if self.verbose: print(f"Warning: Policy B failed: {e}")
                act_b = np.zeros(action_dim)

            action = {
                "robot_a": np.asarray(act_a, dtype=np.float32),
                "robot_b": np.asarray(act_b, dtype=np.float32),
            }

            obs, reward, terminated, truncated, info = self.env.step(action)
            step_count += 1

            # 打印事件
            if self.verbose:
                for event in info.get('events', []):
                    if event['type'] == 'hit':
                        print(f"[Step {step_count}] {event['attacker']} hit {event['defender']} at {event['part']} for {event['damage']:.2f} damage!")

            if step_count % 100 == 0 and self.verbose:
                ha = info.get('health', {}).get('robot_a', 100)
                hb = info.get('health', {}).get('robot_b', 100)
                print(f"Step {step_count:03d} - HP: robot_a={ha:.1f}, robot_b={hb:.1f}")

            if terminated or truncated:
                break

        result = {
            "steps": step_count,
            "winner": info.get("winner", "draw"),
            "final_health": info.get("health", {}),
            "damage_taken": info.get("damage_taken", {}),
            "termination_reasons": info.get("termination_reasons", [])
        }

        if self.verbose:
            print("-" * 60)
            print(f"Round ended. Total steps: {step_count}")
            print(f"Reason: {result['termination_reasons']}")
            print(f"Winner: {result['winner']}")
            print(f"Final HP: robot_a={result['final_health'].get('robot_a', 0):.1f}, robot_b={result['final_health'].get('robot_b', 0):.1f}")
            print("-" * 60)

        self.env.close()
        return result


if __name__ == "__main__":
    # Example usage
    from envs.humanoid21 import make_env
    
    # Create environment with video recorder plugin
    env = make_env(
        plugins=[VideoRecorderPlugin(fps=30, output_path="match.mp4")],
        match_duration=30.0
    )
    
    # Create policies (dummy for example)
    class DummyPolicy:
        def act(self, obs, info):
            return [0.0] * 21  # 21DOF humanoid
    
    policy_a = DummyPolicy()
    policy_b = DummyPolicy()
    
    # Run round
    runner = RoundRunner(policy_a, policy_b, env)
    result = runner.run(seed=42)
    
    print(f"Result: {result}")
