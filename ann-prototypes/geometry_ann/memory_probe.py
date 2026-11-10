"""Process memory probes from /proc/self/status: total, anonymous (allocated) and file-backed RSS.

The kernel tracks only a total high-water mark (VmHWM), so peaks of the split are sampled."""
import threading
import time
from pathlib import Path


def read_memory() -> dict:
    usage = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "RssAnon:", "RssFile:", "VmHWM:")):
            key, kb, _ = line.split(); usage[key.rstrip(":")] = int(kb) * 1024
    return usage


class MemorySampler:
    """Peaks of total, anonymous and file-backed RSS; the kernel only tracks the total (VmHWM)."""
    PERIOD_S = 0.02

    def __init__(self) -> None:
        self.peak, self._running = {}, True
        self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()

    def _record(self) -> None:
        for key, value in read_memory().items():
            self.peak[key] = max(self.peak.get(key, 0), value)

    def _run(self) -> None:
        while self._running:
            self._record(); time.sleep(self.PERIOD_S)

    def stop(self) -> dict:
        self._running = False; self._thread.join(); self._record()
        return dict(build_peak_rss=self.peak.get("VmRSS", 0), build_peak_anon=self.peak.get("RssAnon", 0),
                    build_peak_file=self.peak.get("RssFile", 0), build_hwm=self.peak.get("VmHWM", 0))
