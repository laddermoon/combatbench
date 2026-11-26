"""DeepMimic 单 clip 运动跟踪 PPO 实验。

环境：``envs/track_env.yaml``（DeepMimicTrackSimulator + RSI 合法相位
初始化 + DeepMimicGuardPlugin 终止 + DeepMimicRewarder 模仿奖励）。

- robot_a：训练中的跟踪策略；obs = 96 基础 + 参考扩展
  （ref_obs='full' → 150；'phase_only' → 97）。
- robot_b：StandingCombatPolicy 场景机（96 维 obs，不参与评分）。

奖励：单通道 r_track = 加权模仿奖励（pose/vel/keypoint/height/root）。

终止语义：track_fall / track_drift / track_clip_end → is_terminated；
timeout 视为未终止（max_steps 默认 = clip n_ctrl，不可达）。

eval 指标：survive / complete / fall / kp_err / q_rmse / dxy / dyaw；
best = survive − 0.5·fall_rate。

per-clip 默认参数由 ``clip_presets.suggest_env_params`` 推导
（max_steps=n_ctrl；地面类 clip fall_mode=contact）；class attribute
显式 --set 一律覆盖自动值。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint
from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.framework.rollout.job import Job, SamplingSpec

from baseline.experiments_ppo.base import CombatExperimentPPOBase
from .clip_presets import suggest_env_params
from .ref import CoreRefView

_PKG_DIR = Path(__file__).resolve().parent


class DeepMimicTrack(CombatExperimentPPOBase):

    name = "deepmimic"
    actor_blueprint = "init_policy_truncated_normal.yaml"

    obs_dim: int = 150
    action_dim: int = 21

    env_blueprint = "track_env.yaml"      # deepmimic/envs/ 下
    agent_used = "robot_a"
    clip_id: str = ""                     # 必须 --set clip_id=X
    clips_dir: str = "mocap/humanoid21/data/core"
    rsi: str = "random"
    ref_obs: str = "full"                 # 'full'|'phase_only'
    max_steps: int = 0                    # <=0 → clip_presets 自动 (n_ctrl)
    init_distance_min: float = 6.0
    init_distance_max: float = 6.0
    # 'auto' → clip_presets 按 label 决定；显式 height/contact/none 覆盖
    fall_mode: str = "auto"
    fall_height: float = 0.55
    drift_dxy: float | None = 0.8         # v6 一阶段；None=关闭
    drift_dyaw_deg: float | None = 75.0
    # 'simchar'=DeepMimic humanoid3d 对齐执行器（论文对齐）；
    # 'legacy'=原始固定 KP/KD
    actuation_profile: str = "simchar"

    # --- Reward ---
    _channel_name = "r_track"
    _gamma: float = 0.99
    _gae_lambda: float = 0.95

    # --- Exploration ---
    explore_factor: float = 0.0

    # --- PPO ---
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096

    episodes_per_update: int = 256
    max_updates: int = 2000
    eval_interval: int = 5
    eval_episodes: int = 16
    video_eval_interval: int = 10

    _OBS_KEY = "track_a"
    _best_score: float = -1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.clip_id:
            raise ValueError("deepmimic requires --set clip_id=<id>")
        self._ref = CoreRefView(self.clip_id, self.clips_dir)
        self.obs_dim = 96 + (54 if self.ref_obs == 'full' else 1)
        sug = suggest_env_params(self._ref)
        if self.max_steps <= 0:
            self.max_steps = int(sug['max_steps'])
        if self.fall_mode == 'auto':
            self.fall_mode = str(sug['fall_mode'])

    # ------------------------------------------------------------------
    def _env_pb(self) -> ParameterizedEnvBlueprint:
        return ParameterizedEnvBlueprint.load(
            _PKG_DIR / 'envs' / self.env_blueprint)

    def build_actor(self, device):
        blueprint_dir = _PKG_DIR.parent / 'humanoid21' / 'blueprints'
        bp = PolicyBlueprint.load(blueprint_dir / self.actor_blueprint)
        return bp.build(obs_dim=self.obs_dim).to(device)

    @staticmethod
    def _standing_bp() -> PolicyBlueprint:
        return PolicyBlueprint.load(
            _PKG_DIR.parents[1] / 'policy' / 'blueprints' / 'humanoid21'
            / 'standing.yaml')

    # ------------------------------------------------------------------
    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return (RewardChannel(
            name=self._channel_name, gamma=self._gamma,
            gae_lambda=self._gae_lambda),)

    def _materialize_kwargs(self) -> Dict[str, Any]:
        return dict(
            clip_id=self.clip_id, clips_dir=self.clips_dir, rsi=self.rsi,
            ref_obs=self.ref_obs, max_steps=self.max_steps,
            fall_mode=self.fall_mode, fall_height=self.fall_height,
            drift_dxy=self.drift_dxy, drift_dyaw_deg=self.drift_dyaw_deg,
            initial_distance=self.init_distance_min,
            actuation_profile=self.actuation_profile,
        )

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True) -> List[Job]:
        env_bp = self._env_pb().materialize(**self._materialize_kwargs())
        standing = self._standing_bp()
        sampling_a = SamplingSpec(explore_factor=self.explore_factor)
        return [Job(
            policy_a_bp=policy_bp,
            policy_b_bp=standing,
            env_bp=env_bp,
            seed=int(base_seed + i),
            episode_options={"initial_distance": self.init_distance_min},
            sampling_a=sampling_a,
            sampling_b=SamplingSpec(),
            stochastic=stochastic,
        ) for i in range(n_episodes)]

    # ------------------------------------------------------------------
    def build_trajectories(self, episodes) -> List[Trajectory]:
        out: List[Trajectory] = []
        for ep in episodes:
            T_full = ep.num_frames
            if T_full == 0:
                continue
            T = ep.agent_frame_boundary.get("robot_a", T_full)
            if T == 0:
                continue
            obs_all = ep.observations.get("robot_a")
            acts_all = ep.actions.get("robot_a")
            fin_obs = ep.final_observation.get("robot_a")
            if obs_all is None or acts_all is None or fin_obs is None:
                continue
            rew = extract_per_step_field(
                ep.observer_outputs, self._OBS_KEY, "reward", T_full)
            if rew is None:
                rew = np.zeros(T_full, dtype=np.float32)
            rew = np.asarray(rew[:T], dtype=np.float32)

            reason = ep.agent_termination_reason.get("robot_a", "")
            is_terminated = bool(reason) and reason != "timeout"

            out.append(Trajectory(
                obs=np.asarray(obs_all, dtype=np.float32)[:T],
                actions=np.asarray(acts_all, dtype=np.float32)[:T],
                last_obs=np.asarray(fin_obs, dtype=np.float32),
                channels={self._channel_name: ChannelData(
                    reward=rew, is_terminated=is_terminated,
                    actor_weight=np.ones(T, dtype=np.float32))},
                importance=1.0,
                sampling_ctx=self.extract_sampling_ctx(ep, "robot_a", T),
            ))
        return out

    # ------------------------------------------------------------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        survives, kp_errs, q_rmses, dxys, dyaws = [], [], [], [], []
        falls = completes = 0
        for ep in episodes:
            T = ep.agent_frame_boundary.get("robot_a", ep.num_frames)
            if T == 0:
                continue
            reason = ep.agent_termination_reason.get("robot_a", "")
            if reason == "track_fall":
                falls += 1
            if reason == "track_clip_end":
                completes += 1
            ph = extract_per_step_field(
                ep.observer_outputs, self._OBS_KEY, "phase", ep.num_frames)
            survives.append(
                float(np.max(ph)) if ph is not None and len(ph)
                else float(T) / self.max_steps)
            for key, acc in (("kp_err_m", kp_errs), ("q_rmse_deg", q_rmses),
                             ("root_dxy_m", dxys), ("root_dyaw_deg", dyaws)):
                v = extract_per_step_field(
                    ep.observer_outputs, self._OBS_KEY, key, ep.num_frames)
                if v is not None and len(v):
                    acc.append(float(np.mean(v[:T])))

        n_eps = max(len(episodes), 1)
        survive = float(np.mean(survives)) if survives else 0.0
        score = survive - 0.5 * (falls / n_eps)
        is_new_best = score > self._best_score
        if is_new_best:
            self._best_score = score

        def _m(xs):
            return round(float(np.mean(xs)), 3) if xs else 0.0

        return {
            "is_new_best": is_new_best,
            "info": {
                "survive": round(survive, 3),
                "complete": round(completes / n_eps, 3),
                "fall": round(falls / n_eps, 3),
                "kp_err": _m(kp_errs),
                "q_rmse": _m(q_rmses),
                "dxy": _m(dxys),
                "dyaw": _m(dyaws),
            },
        }

    def state(self) -> dict:
        return {**super().state(), "best_score": self._best_score}

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._best_score = float(state.get("best_score", -1.0))


EXPERIMENT_CLASS = DeepMimicTrack
