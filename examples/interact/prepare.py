"""Build an engine-neutral task file and provenance manifest; no training launch."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

# Also permit invocation as a script without an editable package installation.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from interact_env import EpisodeSpec
from interact_env.registry import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True, help="new output directory")
    args = ap.parse_args()
    specs = [EpisodeSpec(**json.loads(p.read_text())) for p in args.spec]
    for spec in specs:
        resolve(spec.engine)  # fail explicitly for unimplemented engines
    import slime.utils.types as types
    if not Path(types.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError("Python is importing another Slime checkout")
    packages = {}
    for name in ("torch", "transformers", "sglang", "sglang-router", "ray", "transformer-engine", "playwright",
                 "google-genai", "gymnasium", "pettingzoo"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    files = sorted((ROOT / "interact_env").rglob("*.py"))
    files += sorted(p for p in (ROOT / "examples/interact").rglob("*")
                    if p.suffix in (".py", ".json", ".jsonl", ".yaml", ".sh", ".sbatch", ".txt"))
    files.append(ROOT / "slime_plugins/models/qwen25_vl.py")
    manifest = dict(upstream="https://github.com/THUDM/slime.git",
                    revision=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                    python=sys.executable, packages=packages, validation_note="Preparation is not a GPU validation; see GPU_VALIDATION.md for tested profile.",
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    task_group_keys=[s.group_key for s in specs])
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [{"prompt": [{"role": "user", "content": "Native assistant episode"}],
             "metadata": {"episode": asdict(s)}} for s in specs]
    (args.output / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "tasks": len(specs), "revision": manifest["revision"]}))


if __name__ == "__main__":
    main()
