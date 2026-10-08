"""EpisodeRecorder — abandoned/closed episode regression tests.

``EnvRuntime.reset()`` mid-episode and ``EnvRuntime.close()`` with an
active episode issue ``request_termination`` *outside* a stepped frame —
those proposals arrive after the recorder's last ``on_post_action_step``
sweep. ``on_post_episode`` must do a final sweep itself so the produced
``Episode`` carries the real termination metadata instead of tripping the
all-agents-terminated validation.
"""
from __future__ import annotations

import unittest

from envs.framework.env_runtime import EnvRuntime
from envs.framework.tests.conftest import MockSimulator

from baseline.framework.rollout.episode_recorder import EpisodeRecorder


class TestAbandonedClosedEpisodes(unittest.TestCase):
    def _make_runtime(self) -> tuple[EnvRuntime, EpisodeRecorder]:
        recorder = EpisodeRecorder(blueprint_hash="testhash")
        runtime = EnvRuntime(
            simulator=MockSimulator(), recorders=[recorder]
        )
        return runtime, recorder

    def test_reset_abandon_records_abandoned_proposal(self):
        runtime, recorder = self._make_runtime()
        runtime.reset(seed=1, base_seed=1)
        runtime.step(None, None)

        runtime.reset(seed=2, base_seed=2)  # abandons episode 1

        episode = recorder.get_last_episode()
        records = episode.agent_termination_proposal_records
        for aid in ("robot_a", "robot_b"):
            reasons = [r for r, _step in records[aid]]
            self.assertIn("abandoned", reasons)

        runtime.close()

    def test_close_records_closed_proposal(self):
        runtime, recorder = self._make_runtime()
        runtime.reset(seed=1, base_seed=1)
        runtime.step(None, None)

        runtime.close()  # closes with an active episode

        episode = recorder.get_last_episode()
        records = episode.agent_termination_proposal_records
        for aid in ("robot_a", "robot_b"):
            reasons = [r for r, _step in records[aid]]
            self.assertIn("closed", reasons)


if __name__ == "__main__":
    unittest.main()
