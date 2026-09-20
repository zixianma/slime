"""Load only W&B authentication from a user-approved dotenv file, then exec.

Never source the file as shell code or copy unrelated model-provider credentials.
The key remains in the child environment, not argv, manifests or printed output.
"""
import argparse
import os
from pathlib import Path

from dotenv import dotenv_values


def build_environment(path, base, provider_path=None):
    values = dotenv_values(path, interpolate=False)
    key = values.get("WANDB_API_KEY")
    if not key:
        raise ValueError("approved env file does not contain WANDB_API_KEY")
    endpoint = values.get("WANDB_BASE_URL") or "https://api.wandb.ai"
    if endpoint.rstrip("/") != "https://api.wandb.ai":
        raise ValueError("a non-default W&B endpoint needs explicit destination approval")
    env = dict(base)
    env["WANDB_API_KEY"] = key
    env["WANDB_BASE_URL"] = endpoint
    for name in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_SWEEP_ID"):
        env.pop(name, None)
    if provider_path is not None:
        provider = dotenv_values(provider_path, interpolate=False)
        google_key = provider.get("GOOGLE_API_KEY") or provider.get("GEMINI_API_KEY")
        if not google_key:
            raise ValueError("provider env file does not contain GOOGLE_API_KEY or GEMINI_API_KEY")
        env["GOOGLE_API_KEY"] = google_key
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--provider-env-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    env = build_environment(args.env_file, os.environ, args.provider_env_file)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
