"""Generic synchronous callback bridge. Native adapters own engine semantics."""
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
import time

from .protocol import Action, Decision, EpisodeSpec, PROTOCOL_VERSION


class Cancelled(BaseException):
    """Bypass a native runner's exception-to-silence handlers."""


class TransportFailure(BaseException):
    """Infrastructure failures must never become a native silent action/reward."""


class Session:
    def __init__(self, emit, episode_id, profiler=None):
        self.emit, self.episode_id, self.calls = emit, episode_id, 0
        self.profiler = profiler

    def request(self, observation, *, tick, time_unit, action_schema):
        try:
            return self._request(observation, tick=tick, time_unit=time_unit, action_schema=action_schema)
        except Exception as exc:
            raise TransportFailure(f"bridge request failed: {exc}") from exc

    def _request(self, observation, *, tick, time_unit, action_schema):
        decision = Decision(self.episode_id, self.calls, observation, tick, time_unit, action_schema)
        self.calls += 1
        if self.profiler:
            self.profiler.before_decision(decision.decision_id)
        self.emit("decision", asdict(decision))
        line = sys.stdin.readline()
        if not line:
            raise Cancelled("client closed")
        action = Action(**json.loads(line))
        if action.episode_id != self.episode_id or action.decision_id != decision.decision_id:
            raise Cancelled("response identity mismatch")
        audit_start = time.perf_counter()
        with open("decisions.jsonl", "a") as f:
            f.write(json.dumps({"decision": asdict(decision), "response": action.raw_response,
                                "prompt_hash": observation.content_hash}) + "\n")
        if self.profiler:
            self.profiler.after_reply(time.perf_counter()-audit_start)
        return action.raw_response


def main():
    wire = sys.stdout
    episode_id = sys.argv[2]

    def emit(kind, payload):
        wire.write(json.dumps(dict(protocol_version=PROTOCOL_VERSION, episode_id=episode_id,
                                   type=kind, payload=payload)) + "\n")
        wire.flush()

    try:
        with redirect_stdout(sys.stderr):
            profiler = None
            if os.environ.get('INTERACT_PROFILE') == '1':
                from .profiling import WorkerProfiler
                profiler = WorkerProfiler()
            spec = EpisodeSpec(**json.loads(Path(sys.argv[1]).read_text()))
            module, name = sys.argv[3].split(":", 1)
            adapter = getattr(importlib.import_module(module), name)()
            result = adapter.run(spec, Session(emit, episode_id, profiler))
            if result.episode_id != episode_id:
                raise RuntimeError("adapter returned another episode's result")
            emit("end", asdict(result))
    except Cancelled:
        return
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        wire.write(json.dumps(dict(protocol_version=PROTOCOL_VERSION, episode_id=episode_id,
                                   type="error", error=f"{type(exc).__name__}: {exc}")) + "\n")
        wire.flush()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
