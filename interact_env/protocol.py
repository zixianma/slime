"""Versioned boundary shared by native assistant-supervision engines."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Protocol

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class EpisodeSpec:
    engine: str
    task_id: str
    seed: int = 0
    config: dict = field(default_factory=dict)  # private; never serialized into a policy prompt

    def __post_init__(self):
        if not self.engine or not self.task_id:
            raise ValueError("engine and task_id are required")
        json.dumps(asdict(self), allow_nan=False)

    @property
    def group_key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Observation:
    """Allowlist of policy-visible content; native image order is significant."""
    system: str
    user: str
    images: tuple[str, ...] = ()  # encoded image data URLs, no engine objects
    profile: str = "native"

    def __post_init__(self):
        if not isinstance(self.system, str) or not isinstance(self.user, str):
            raise TypeError("system/user must be strings")
        object.__setattr__(self, "images", tuple(self.images))
        if any(not s.startswith("data:image/") for s in self.images):
            raise ValueError("images must be encoded data URLs")

    @property
    def content_hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Decision:
    episode_id: str
    decision_id: int
    observation: Observation
    tick: float | None
    time_unit: str
    action_schema: str  # native taxonomy/schema version, not one universal action parser


@dataclass(frozen=True)
class Action:
    episode_id: str
    decision_id: int
    raw_response: str


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    outcome: str
    terminated: bool
    truncated: bool
    eligible: bool
    reward: float | None
    reward_version: str
    components: dict = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.terminated == self.truncated:
            raise ValueError("episode end must be exactly one of terminated/truncated")
        if self.eligible and (not self.terminated or self.reward is None or not math.isfinite(self.reward)):
            raise ValueError("eligible training episodes need a finite terminal reward")


class AssistantSession(Protocol):
    episode_id: str

    def request(self, observation: Observation, *, tick: float | None,
                time_unit: str, action_schema: str) -> str: ...


class NativeAdapter(Protocol):
    def run(self, spec: EpisodeSpec, session: AssistantSession) -> EpisodeResult: ...


class AsyncEnvironment(Protocol):
    async def reset(self) -> Decision | EpisodeResult: ...
    async def step(self, action: Action) -> Decision | EpisodeResult: ...
    async def close(self) -> None: ...
