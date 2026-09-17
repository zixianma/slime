"""Engine-neutral CLI. The provided raw policy response is a test control."""
import argparse
import asyncio
from dataclasses import asdict
import json

from .protocol import Action, Decision, EpisodeSpec
from .runtime import Environment


async def smoke(args):
    spec = EpisodeSpec(**json.loads(args.spec.read_text()))
    env = Environment(spec, args.output)
    count = 0
    try:
        msg = await env.reset()
        while isinstance(msg, Decision):
            count += 1
            if args.max_decisions and count >= args.max_decisions:
                print(json.dumps({"cancelled_after_decisions": count, "images": len(msg.observation.images),
                                  "directory": str(env.directory)}))
                return
            msg = await env.step(Action(env.episode_id, msg.decision_id, args.response))
        print(json.dumps({**asdict(msg), "decisions": count}, indent=2))
    finally:
        await env.close()


if __name__ == "__main__":
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--response", default='{"text":"","flag":null}')
    ap.add_argument("--max-decisions", type=int, default=0)
    ap.add_argument("--output", default="interact-runs/smoke")
    asyncio.run(smoke(ap.parse_args()))
