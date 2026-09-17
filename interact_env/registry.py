"""Lazy adapter catalog: distinguish intended compatibility from implementation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterDefinition:
    factory: str | None
    resources: tuple[str, ...]
    native_boundary: str
    status: str


ADAPTERS = {
    "cooking": AdapterDefinition("interact_env.adapters.cooking:CookingAdapter", ("process", "optional-browser"),
                                 "e2e_stepin_rollout.run -> sup_client.sup_call", "implemented; CPU tested"),
    "screensim": AdapterDefinition("interact_env.adapters.screensim:ScreenSimAdapter", ("process", "browser"),
                                   "composite_run_v2 -> AssistantV2.client.text", "implemented; scripted-human native parity and GPU smoke tested"),
    "vh": AdapterDefinition(None, ("process", "unity-lease"),
                            "iv_e2e_stepin_rollout -> iv_protocol.assist_frames", "planned; not implemented"),
}


def resolve(engine):
    if engine not in ADAPTERS:
        raise ValueError(f"unknown engine {engine!r}; available catalog: {sorted(ADAPTERS)}")
    definition = ADAPTERS[engine]
    if definition.factory is None:
        raise NotImplementedError(f"{engine}: {definition.status}")
    return definition
