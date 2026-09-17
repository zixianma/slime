"""Engine-neutral async transport; one owned native worker per episode."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import uuid

from .protocol import Action, Decision, EpisodeResult, EpisodeSpec, Observation, PROTOCOL_VERSION
from .registry import resolve


class Environment:
    def __init__(self, spec: EpisodeSpec, output_root: str | Path, *, timeout=180, python=None, launch_prefix=None):
        self.definition = resolve(spec.engine)
        self.spec = spec
        self.episode_id = uuid.uuid4().hex
        self.directory = Path(output_root).resolve() / self.episode_id
        self.timeout = timeout
        self.python = python or sys.executable
        self.launch_prefix = list(launch_prefix or [])
        self.process = None
        self.pending = None
        self._log = None
        self._closed = False

    async def reset(self):
        if self.process is not None or self._closed:
            raise RuntimeError("one episode per environment; construct a fresh client")
        self.directory.mkdir(parents=True)
        config = self.directory / "spec.json"
        config.write_text(json.dumps(asdict(self.spec), indent=2) + "\n")
        self._log = (self.directory / "worker.log").open("wb")
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.launch_prefix, self.python, "-m", "interact_env.worker", str(config), self.episode_id, self.definition.factory,
                cwd=self.directory, env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=self._log, limit=64 * 1024 * 1024, start_new_session=True,
            )
            return await self._receive()
        except BaseException:
            await self.close()
            raise

    async def _receive(self):
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), self.timeout)
        except TimeoutError as exc:
            raise TimeoutError(f'Native worker response timed out after {self.timeout}s; '
                               f'episode={self.episode_id}; log={self.directory / "worker.log"}') from exc
        if not line:
            code = await self.process.wait()
            raise RuntimeError(f"native worker exited ({code}); see {self.directory / 'worker.log'}")
        msg = json.loads(line)
        if msg.get("protocol_version") != PROTOCOL_VERSION or msg.get("episode_id") != self.episode_id:
            raise RuntimeError("worker protocol/episode mismatch")
        if msg["type"] == "error":
            raise RuntimeError(f"native worker failed: {msg['error']}; see {self.directory / 'worker.log'}")
        payload = msg["payload"]
        if msg["type"] == "decision":
            payload["observation"] = Observation(**payload["observation"])
            decision = Decision(**payload)
            self.pending = decision.decision_id
            return decision
        if msg["type"] == "end":
            self.pending = None
            return EpisodeResult(**payload)
        raise RuntimeError("unknown worker message")

    async def step(self, action: Action):
        if self._closed or self.pending is None or action.episode_id != self.episode_id or action.decision_id != self.pending:
            raise ValueError("response does not match the pending episode/decision")
        self.process.stdin.write((json.dumps(asdict(action)) + "\n").encode())
        self.pending = None  # a failed/uncertain delivery must not be replayed
        try:
            await self.process.stdin.drain()
            return await self._receive()
        except BaseException:
            await self.close()
            raise

    async def close(self):
        self._closed = True
        self.pending = None
        if self.process is not None and self.process.returncode is None:
            try:
                self.process.stdin.close()
                await asyncio.wait_for(self.process.wait(), 5)
            except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
                # Only our new process session, never a shared engine/service.
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(self.process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(self.process.wait(), 5)
                        break
                    except asyncio.TimeoutError:
                        continue
        if self._log is not None:
            self._log.close()
            self._log = None
