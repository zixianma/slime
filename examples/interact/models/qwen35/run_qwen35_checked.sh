#!/usr/bin/env bash
set -euo pipefail
test -s "$INTERACT_JOB_DIR/gpu-preflight.json"
python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["INTERACT_JOB_DIR"])/os.environ.get("Q35_PREFLIGHT_STATUS", "10-kernels.status.json"); assert json.loads(p.read_text())["returncode"] == 0, "GPU preflight did not exit cleanly"'
python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["INTERACT_JOB_DIR"]); assert all(json.loads((p/f"{name}.status.json").read_text())["returncode"] == 0 for name in ("12-native-probe", "13-native-replay"))'
exec bash examples/interact/models/qwen35/run_qwen35_train.sh "$@"
