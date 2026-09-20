"""Release the owned allocation after a successful pilot and postflight."""
import json
import os
from pathlib import Path

job = Path(os.environ['INTERACT_JOB_DIR'])
for stage in (os.environ.get('Q35_PILOT_STAGE', '14-pilot'), os.environ.get('Q35_POSTFLIGHT_STAGE', '20-postflight')):
    if json.loads((job / f'{stage}.status.json').read_text())['returncode'] != 0:
        raise SystemExit(f'{stage} failed; controller remains available for a bounded diagnostic handoff')
(job / 'queue' / 'FINISH').touch()
print('PILOT_AND_POSTFLIGHT_FINISHED', flush=True)
