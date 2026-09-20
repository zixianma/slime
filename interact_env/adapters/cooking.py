"""CookSim-specific configuration, callback parsing, native runner and reward."""
from dataclasses import dataclass
import base64
from contextlib import ExitStack
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import re
import sys
import threading

from ..protocol import EpisodeResult, Observation
from ..worker import Cancelled
from .cooking_rewards import score

@dataclass(frozen=True)
class CookingSpec:
    case: str = "b3_easy_nops_medium_map_2"
    persona: str = "baseline"
    human: str = "scripted"  # scripted verification human or native Gemini
    human_model: str = "gemini-3.7-flash"
    observation: str = "text_state_map"
    latency_ticks: int = 0
    scripted_accept: bool = True
    seed: int = 0
    wall_seconds: int = 600
    renderer: str = "auto"  # Explicit experiment setting; never inherited implicitly.
    engine_root: str = str(Path(__file__).resolve().parents[3] / "cook-bench-engine")
    case_pool: str | None = None

    def validate(self):
        if self.renderer not in ("auto", "swiftshader", "vulkan", "egl"):
            raise ValueError("unsupported cooking renderer")
        if self.human not in ("scripted", "gemini"):
            raise ValueError("human must be scripted or gemini")
        if self.observation not in ("text_state_map", "frames", "agent+++"):
            raise ValueError("unsupported native observation profile")
        if self.latency_ticks < 0 or self.wall_seconds <= 0:
            raise ValueError("invalid time budget")
        pool = Path(self.case_pool) if self.case_pool else Path(self.engine_root) / "bench/v5/cases_composite_v4.json"
        if not pool.is_file():
            raise ValueError("cooking case pool does not exist")
        if self.human == "gemini" and self.human_model != "gemini-3.7-flash":
            raise ValueError("Gemini-user experiment requires pinned gemini-3.7-flash")


def parse_reply(raw: str) -> dict:
    """Permissive native-style JSON extraction; never trains rewritten tokens."""
    # Thinking is not spoken to the human and must not supply the action JSON.
    body = raw.rsplit("</think>", 1)[-1].strip()
    start = body.find("{")
    if start < 0:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(body[start:])
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict) or not isinstance(value.get("text", ""), str):
        return {}
    flag = value.get("flag")
    if flag is not None and not (isinstance(flag, dict) and isinstance(flag.get("error"), str)):
        return {}
    return value

def run(spec, assistant):
    spec.validate()
    root = Path(spec.engine_root).resolve()
    case_pool = Path(spec.case_pool).resolve() if spec.case_pool else root / "bench/v5/cases_composite_v4.json"
    # Fixed, explicit worker configuration overrides inherited benchmark switches.
    for key in list(os.environ):
        if key.startswith("COOKSIM_"):
            del os.environ[key]
    os.environ.update(COOKSIM_GRID=str(root / "bench/v5/grid.json"),
                      COOKSIM_CASES=str(case_pool),
                      COOKSIM_SCHED="v5", COOKSIM_PACE="tight",
                      COOKSIM_ASSIST_LATENCY=str(spec.latency_ticks),
                      COOKSIM_WALL_BUDGET=str(spec.wall_seconds))
    if spec.renderer == "swiftshader":
        os.environ["COOKSIM_FORCE_SWIFTSHADER"] = "1"
    if spec.human == "scripted":
        os.environ["COOKSIM_SCRIPTED_COOK"] = "template"
        os.environ["COOKSIM_SCRIPTED_ACCEPT"] = "1" if spec.scripted_accept else "0"
    if spec.observation == "text_state_map":
        os.environ["COOKSIM_NO_VIDEO"] = "1"
    sys.path[:0] = [str(root), str(root / "tools")]
    random.seed(spec.seed)
    import numpy as np
    np.random.seed(spec.seed)
    import e2e_stepin_rollout as native
    from cooksim.interact.persona_v2 import BY_NAME

    class NoModel:
        def sup_call(self, *args):
            raise Cancelled("unexpected model call in scripted-human mode")

    client = NoModel()
    if spec.human == "gemini":
        if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
            os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("Gemini user requires GEMINI_API_KEY or GOOGLE_API_KEY")
        os.environ["GEMINI_MODEL"] = spec.human_model
        from genai_client import make_client
        client = make_client()
    with ExitStack() as stack:
        page = None
        if spec.observation != "text_state_map":
            handler = partial(SimpleHTTPRequestHandler, directory=str(root / "web"))
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            from playwright.sync_api import sync_playwright
            pw = stack.enter_context(sync_playwright())
            flags = native.GPU
            if spec.renderer == "vulkan":
                flags = ["--no-sandbox", "--use-gl=angle", "--use-angle=vulkan",
                         "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu",
                         "--disable-software-rasterizer", "--autoplay-policy=no-user-gesture-required"]
            elif spec.renderer == 'egl':
                flags = ['--no-sandbox','--use-gl=angle','--use-angle=gl-egl',
                         '--ignore-gpu-blocklist','--enable-gpu','--disable-software-rasterizer',
                         '--autoplay-policy=no-user-gesture-required']
            browser = pw.chromium.launch(args=flags)
            stack.callback(browser.close)
            page = browser.new_page(viewport={"width": 800, "height": 580})
            def browser_message(message):
                with Path('browser-console.jsonl').open('a') as log:
                    log.write(json.dumps(dict(type=message.type, text=message.text))+'\n')
            page.on('console', browser_message)
            page.goto(f"http://127.0.0.1:{server.server_port}/headless_render.html", timeout=180000)
            page.wait_for_function("window.__ready===true", timeout=90000)
            page.wait_for_timeout(400)
        checked_assistant = assistant
        if spec.renderer in ('vulkan', 'egl') and page is not None:
            class CheckedAssistant:
                placement_checked = False
                def sup_call(self, *args, **kwargs):
                    backend = page.evaluate('''() => {
                        const gl=window.__rr?.renderer?.getContext();
                        if (!gl) return null;
                        const ext=gl.getExtension('WEBGL_debug_renderer_info');
                        return {renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),
                                context_lost:gl.isContextLost()};
                    }''')
                    Path('hardware_renderer.json').write_text(json.dumps(backend)+'\n')
                    name = str((backend or {}).get('renderer') or '').lower()
                    if (not backend or backend['context_lost'] or 'nvidia' not in name
                            or any(v in name for v in ('swiftshader', 'llvmpipe', 'software'))):
                        raise RuntimeError(f"Hardware cooking renderer failed: {backend}")
                    expected = os.environ.get('INTERACT_RENDER_GPU_UUID')
                    if expected and not self.placement_checked:
                        from examples.interact.cooking_gpu_handoff import graphics_processes
                        cdp = browser.new_browser_cdp_session()
                        processes = cdp.send('SystemInfo.getProcessInfo')['processInfo']
                        cdp.detach()
                        pids = {int(p['id']) for p in processes if p['type'].lower() == 'gpu'}
                        matches = [p for p in graphics_processes() if p['pid'] in pids]
                        if not matches or any(p['uuid'] != expected for p in matches):
                            raise RuntimeError(f'Renderer GPU placement mismatch: expected {expected}, got {matches}')
                        Path('renderer_placement.json').write_text(json.dumps(matches)+'\n')
                        self.placement_checked = True
                    return assistant.sup_call(*args, **kwargs)

                def stats(self):
                    return assistant.stats()
            checked_assistant = CheckedAssistant()
        return native.run(spec.case, f"pv2_{spec.persona}", "hardsoft", client, page,
                          persona_v2=BY_NAME[spec.persona], obs_mode=spec.observation,
                          sup_client=checked_assistant)


class CookingAdapter:
    def run(self, episode, session):
        config = dict(episode.config)
        reward_version = config.pop("reward_version", "native_outcome_v1")
        if "case" in config or "seed" in config:
            raise ValueError("case and seed belong in EpisodeSpec.task_id/seed")
        spec = CookingSpec(case=episode.task_id, seed=episode.seed, **config)

        class Assistant:
            def __init__(self):
                self.calls, self.invalid = 0, 0

            def sup_call(self, system, user, frames):
                self.calls += 1
                match = re.search(r"CURRENT TIMESTAMP: t=(\d+)", user)
                images = tuple("data:image/png;base64," + base64.b64encode(f).decode() for f in frames)
                raw = session.request(
                    Observation(system, user, images, spec.observation),
                    tick=int(match.group(1)) if match else None,
                    time_unit="cooksim_tick", action_schema="cooksim_native_v5",
                )
                parsed = parse_reply(raw)
                self.invalid += not bool(parsed)
                return parsed

            def stats(self):
                return {"model": "external-policy", "calls": self.calls, "invalid": self.invalid}

        policy = Assistant()
        report = run(spec, policy)
        path = Path("report.json").resolve()
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        reward, components = score(report, reward_version, assistant_turns=policy.calls)
        components.update(assistant_calls=policy.calls, invalid_responses=policy.invalid,
                          invalid_response_fraction=policy.invalid/max(policy.calls, 1),
                          native_ticks=report.get('ticks', 0))
        return EpisodeResult(session.episode_id, report["outcome"], True, False, True,
                             reward, reward_version, components, {"native_report": str(path)})
