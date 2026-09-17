"""Read-only GPU process checks for the bounded cooking acceptance job."""
import subprocess
import xml.etree.ElementTree as ET


def canonical_gpu_uuid(value):
    value = str(value)
    return value if value.startswith('GPU-') else 'GPU-'+value


def graphics_processes():
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True))
    return [dict(uuid=gpu.findtext('uuid'), pid=int(p.findtext('pid')),
                 name=p.findtext('process_name'), type=p.findtext('type'))
            for gpu in root.findall('gpu') for p in gpu.findall('./processes/process_info')
            if 'G' in (p.findtext('type') or '')]


def assert_no_graphics():
    active = graphics_processes()
    if active:
        raise RuntimeError(f'GPU handoff blocked by live graphics processes: {active}')
    return dict(graphics_processes=[], handoff='browser contexts released before learner')
