"""Opt-in worker timing, never supplied to the policy or used for rewards."""
import json
import time


class WorkerProfiler:
    def __init__(self):
        self.start = time.perf_counter()
        self.browser = {}
        self.backend_recorded = False
        # Browser API timings include browser IPC and waiting; not pure GPU render.
        from playwright.sync_api import Page
        for name in ('evaluate', 'screenshot', 'set_content', 'goto', 'wait_for_function', 'wait_for_timeout'):
            original = getattr(Page, name)
            def wrapped(page, *args, _original=original, _name=name, **kwargs):
                begin = time.perf_counter()
                try:
                    result = _original(page, *args, **kwargs)
                finally:
                    self.browser[_name] = self.browser.get(_name, 0.) + time.perf_counter()-begin
                if _name == 'evaluate' and args and '__renderFrame' in str(args[0]) and not self.backend_recorded:
                    # Query the already-created context without changing its pixels.
                    info = _original(page, '''() => {
                        const gl = window.__rr?.renderer?.getContext();
                        if (!gl) return null;
                        const ext = gl.getExtension('WEBGL_debug_renderer_info');
                        return {renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
                                vendor: ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR)};
                    }''')
                    if info:
                        with open('browser_backend.json', 'w') as stream:
                            stream.write(json.dumps(info)+'\n')
                        self.backend_recorded = True
                return result
            setattr(Page, name, wrapped)

    def before_decision(self, decision_id):
        self.row = dict(decision_id=decision_id, native_segment_s=time.perf_counter()-self.start,
                        browser_api_s=dict(self.browser))
        self.browser.clear()

    def after_reply(self, audit_seconds):
        self.row['audit_write_s'] = audit_seconds
        with open('worker_timing.jsonl', 'a') as stream:
            stream.write(json.dumps(self.row)+'\n')
        self.start = time.perf_counter()
