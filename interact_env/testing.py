"""Contract fixture, not a fourth benchmark or a stand-in for ScreenSim/VH."""
from .protocol import EpisodeResult, Observation


class FixtureAdapter:
    def run(self, spec, session):
        for i in range(3):
            response = session.request(Observation("", f"visible turn {i}", profile="fixture"),
                                       tick=i * 0.25, time_unit="seconds", action_schema="fixture.free_text")
            if response != "arbitrary non-JSON speech":
                raise ValueError("the generic runtime must not parse cooking JSON")
        return EpisodeResult(session.episode_id, "fixture_finished", True, False, True,
                             1.0, "fixture_v1", {"turns": 3})
