from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backend import BaseSimulator
from .common_plugins import TimeoutPlugin
from .engine import SimEngine
from .plugin import BasePlugin
from .runtime_plugin import BaseObserver, BaseRewarder, RuntimeDriverPlugin
from .context import TerminationReason


class PolicyRuntime:
    AGENT_IDS = ("robot_a", "robot_b")

    def __init__(
        self,
        simulator: BaseSimulator,
        observers: Optional[Dict[str, BaseObserver]] = None,
        rewarders: Optional[Dict[str, BaseRewarder]] = None,
        plugins: Optional[List[BasePlugin]] = None,
        phy_steps_per_action: int = 1,
        max_steps: Optional[int] = None,
    ):
        self.engine = SimEngine(simulator, phy_steps_per_action)
        self.runtime_driver = RuntimeDriverPlugin()
        self.observers: Dict[str, Optional[BaseObserver]] = {
            agent_id: None for agent_id in self.AGENT_IDS
        }
        self.rewarders: Dict[str, Optional[BaseRewarder]] = {
            agent_id: None for agent_id in self.AGENT_IDS
        }
        self.latest_result: Optional[Dict[str, Any]] = None

        self.engine.attach_plugin(self.runtime_driver)

        if max_steps is not None:
            self.engine.attach_plugin(TimeoutPlugin(max_steps))

        for plugin in plugins or []:
            self.engine.attach_plugin(plugin)

        for agent_id, observer in (observers or {}).items():
            self.attach_observer(agent_id, observer)

        for agent_id, rewarder in (rewarders or {}).items():
            self.attach_rewarder(agent_id, rewarder)

    @property
    def is_episode_active(self) -> bool:
        return self.engine.is_episode_active

    def attach_plugin(self, plugin: BasePlugin) -> None:
        self.engine.attach_plugin(plugin)

    def detach_plugin(self, plugin: BasePlugin) -> None:
        if plugin is self.runtime_driver:
            raise ValueError("RuntimeDriverPlugin is managed internally by PolicyRuntime and cannot be detached.")
        self.engine.detach_plugin(plugin)

    def attach_observer(self, agent_id: str, observer: Optional[BaseObserver]) -> None:
        self._validate_agent_id(agent_id)
        current = self.observers[agent_id]
        if current is observer:
            return
        self.observers[agent_id] = observer
        if observer is None:
            self.runtime_driver.remove_observer(agent_id)
        else:
            self.runtime_driver.set_observer(agent_id, observer)
        if self.engine.is_episode_active:
            self.runtime_driver.refresh(self.engine.ctx, force=True)

    def detach_observer(self, agent_id: str) -> None:
        self.attach_observer(agent_id, None)

    def attach_rewarder(self, agent_id: str, rewarder: Optional[BaseRewarder]) -> None:
        self._validate_agent_id(agent_id)
        current = self.rewarders[agent_id]
        if current is rewarder:
            return
        self.rewarders[agent_id] = rewarder
        if rewarder is None:
            self.runtime_driver.remove_rewarder(agent_id)
        else:
            self.runtime_driver.set_rewarder(agent_id, rewarder)
        if self.engine.is_episode_active:
            self.runtime_driver.refresh(self.engine.ctx, force=True)

    def detach_rewarder(self, agent_id: str) -> None:
        self.attach_rewarder(agent_id, None)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.engine.reset(seed=seed, options=options)
        self.latest_result = self._collect_result()
        return self.latest_result

    def step(self, action_a: Any, action_b: Any) -> Dict[str, Any]:
        if not self.engine.is_episode_active:
            raise RuntimeError("PolicyRuntime.step() called before reset() or after episode termination.")
        self.engine.step(self._build_joint_action(action_a, action_b))
        self.latest_result = self._collect_result()
        return self.latest_result

    def render(self) -> Optional[np.ndarray]:
        return self.engine.simulator.get_broadcastview_image()

    def close(self) -> None:
        self.engine.close()

    def _collect_result(self) -> Dict[str, Any]:
        obs, observer_info = self._collect_observer_outputs()
        reward, reward_info = self._collect_rewarder_outputs()
        info = {
            "shared": self._build_shared_info(),
            "robot_a": {},
            "robot_b": {},
        }
        for agent_id in self.AGENT_IDS:
            info[agent_id].update(observer_info[agent_id])
            info[agent_id].update(reward_info[agent_id])
        terminated, truncated = self._resolve_termination_flags()
        return {
            "obs": obs,
            "reward": reward,
            "info": info,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _collect_observer_outputs(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        obs: Dict[str, Any] = {}
        info: Dict[str, Dict[str, Any]] = {agent_id: {} for agent_id in self.AGENT_IDS}
        for agent_id in self.AGENT_IDS:
            output = self.runtime_driver.get_observer_output(agent_id)
            obs[agent_id], info[agent_id] = self._normalize_observer_output(output)
        return obs, info

    def _collect_rewarder_outputs(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        reward: Dict[str, Any] = {}
        info: Dict[str, Dict[str, Any]] = {agent_id: {} for agent_id in self.AGENT_IDS}
        for agent_id in self.AGENT_IDS:
            output = self.runtime_driver.get_rewarder_output(agent_id)
            reward[agent_id], info[agent_id] = self._normalize_rewarder_output(output)
        return reward, info

    def _build_shared_info(self) -> Dict[str, Any]:
        ctx = self.engine.ctx
        return {
            "metrics": dict(ctx.metrics),
            "events": list(ctx.events),
            "termination_reasons": list(ctx.termination_proposals),
            "episode_step": ctx.episode_step,
            "physics_step": ctx.physics_step,
            "is_terminated": ctx.is_terminated,
        }

    def _resolve_termination_flags(self) -> Tuple[bool, bool]:
        proposals = self.engine.ctx.termination_proposals
        if not proposals:
            return False, False
        if TerminationReason.TIMEOUT in proposals:
            has_non_timeout_reason = any(reason != TerminationReason.TIMEOUT for reason in proposals)
            if has_non_timeout_reason:
                return True, False
            return False, True
        return True, False

    def _build_joint_action(self, action_a: Any, action_b: Any) -> Dict[str, Any]:
        return {
            "robot_a": action_a,
            "robot_b": action_b,
        }

    def _validate_agent_id(self, agent_id: str) -> None:
        if agent_id not in self.AGENT_IDS:
            raise ValueError(f"Unsupported agent_id: {agent_id}")

    @staticmethod
    def _normalize_observer_output(output: Any) -> Tuple[Any, Dict[str, Any]]:
        if output is None:
            return None, {}
        if isinstance(output, tuple) and len(output) == 2:
            payload, info = output
            if isinstance(info, dict):
                return payload, dict(info)
            return payload, {"observer_output": info}
        if isinstance(output, dict) and ("obs" in output or "observation" in output):
            obs = output.get("obs", output.get("observation"))
            info: Dict[str, Any] = {}
            raw_info = output.get("info")
            if isinstance(raw_info, dict):
                info.update(raw_info)
            elif raw_info is not None:
                info["observer_info"] = raw_info
            for key, value in output.items():
                if key not in {"obs", "observation", "info"}:
                    info[key] = value
            return obs, info
        return output, {}

    @staticmethod
    def _normalize_rewarder_output(output: Any) -> Tuple[Any, Dict[str, Any]]:
        if output is None:
            return None, {}
        if isinstance(output, tuple) and len(output) == 2:
            payload, info = output
            if isinstance(info, dict):
                return payload, dict(info)
            return payload, {"reward_output": info}
        if isinstance(output, dict) and ("reward" in output or "value" in output):
            reward = output.get("reward", output.get("value"))
            info = {key: value for key, value in output.items() if key not in {"reward", "value"}}
            return reward, info
        return output, {}
