"""Fail-closed Chromium hardware-WebGL preflight, without policy/API calls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROFILES = {
    'vulkan': ['--use-gl=angle', '--use-angle=vulkan', '--enable-features=Vulkan'],
    'egl': ['--use-gl=angle', '--use-angle=gl-egl'],
    'default': [],
    'software_control': ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
}


def probe(profile):
    from playwright.sync_api import sync_playwright
    flags = ['--no-sandbox', '--ignore-gpu-blocklist', '--enable-gpu'] + PROFILES[profile]
    if profile != 'software_control':
        flags.append('--disable-software-rasterizer')
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=flags, timeout=45000)
        try:
            page = browser.new_page()
            result = page.evaluate('''() => {
                const c = document.createElement('canvas'); c.width=64;c.height=64;
                const gl=c.getContext('webgl2', {preserveDrawingBuffer:true});
                if (!gl) return {context:false};
                const ext=gl.getExtension('WEBGL_debug_renderer_info');
                gl.clearColor(1,0,0,1);gl.clear(gl.COLOR_BUFFER_BIT);
                const pixel=new Uint8Array(4);gl.readPixels(0,0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,pixel);
                return {context:true, renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),
                    vendor:ext?gl.getParameter(ext.UNMASKED_VENDOR_WEBGL):gl.getParameter(gl.VENDOR),
                    context_lost:gl.isContextLost(),pixel:Array.from(pixel),error:gl.getError()};
            }''')
            result['gpu_info'] = browser.new_browser_cdp_session().send('SystemInfo.getInfo')['gpu']
            result['flags'] = flags
            renderer = result.get('renderer', '').lower()
            result['hardware_pass'] = (profile != 'software_control' and 'nvidia' in renderer
                and not any(x in renderer for x in ('swiftshader', 'llvmpipe', 'software'))
                and result.get('pixel') == [255, 0, 0, 255]
                and not result.get('context_lost') and result.get('error') == 0)
            return result
        finally:
            browser.close()


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'host':os.uname().nodename, 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'profiles':{}, 'scope':'WebGL capability gate only; cooking correctness still required'}
    for name, command in {
        'devices':['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv'],
        'libraries':['ldconfig','-p'],
    }.items():
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=20)
            lines = p.stdout.splitlines()
            if name == 'libraries':
                lines = [s for s in lines if any(v in s.lower() for v in ('vulkan','nvidia','libegl','libglx'))]
            report[name] = {'returncode':p.returncode, 'stdout':lines, 'stderr':p.stderr}
        except Exception as exc:
            report[name] = {'error':str(exc)}
    report['icds'] = {str(p):p.read_text() for folder in ('/usr/share/vulkan/icd.d','/etc/vulkan/icd.d')
                      for p in Path(folder).glob('*.json')}
    for profile in PROFILES:
        try:
            p = subprocess.run([sys.executable, __file__, '--profile', profile],
                               capture_output=True, text=True, timeout=90)
            (output/(profile+'.log')).write_text(p.stdout+'\n'+p.stderr)
            row = json.loads(p.stdout.split('PROBE_RESULT ')[-1]) if p.returncode == 0 else {'error':p.stderr[-4000:]}
        except Exception as exc:
            row = {'error':str(exc)}
        report['profiles'][profile] = row
        print(profile, json.dumps({k:v for k,v in row.items() if k != 'gpu_info'}), flush=True)
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    report['hardware_available'] = any(r.get('hardware_pass') for r in report['profiles'].values())
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('HARDWARE_AVAILABLE', report['hardware_available'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.profile:
        print('PROBE_RESULT '+json.dumps(probe(args.profile)), flush=True)
    else:
        main(args.output)
