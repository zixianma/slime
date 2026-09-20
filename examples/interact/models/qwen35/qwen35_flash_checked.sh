#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/qwen35_env.sh"
python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["INTERACT_JOB_DIR"])/"29-install-flash.status.json"; assert json.loads(p.read_text())["returncode"] == 0; import flash_attn; assert flash_attn.__version__ == "2.8.3"'
python examples/interact/models/qwen35/qwen35_provenance.py --output "$INTERACT_JOB_DIR/provenance-pilot09.json"
exec bash examples/interact/models/qwen35/run_qwen35_checked.sh "$@"
