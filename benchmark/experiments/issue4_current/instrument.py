"""Add JSON-only build/RSS diagnostics to an archived harness; CSV contracts stay intact.

Run before measurement. The archived library and native adapter are untouched.
"""
from pathlib import Path
import difflib
import sys

w = Path(sys.argv[1]).resolve()
p = w/'source/benchmark/measure_memory.py'
old = p.read_text()
new = old.replace(
    'build_peak = memory_status()[1] - baseline',
    'build_peak_total = memory_status()[1]\n        build_peak = build_peak_total - baseline')
new = new.replace(
    'build_peak_rss_bytes=build_peak,',
    'build_peak_rss_bytes=build_peak, build_baseline_rss_bytes=baseline,\n'
    '                    build_peak_total_rss_bytes=build_peak_total,\n'
    '                    encode_s=index.encode_seconds, encode_vps=len(base)/index.encode_seconds,')
new = new.replace(
    'return dict(ram_accounted_bytes=accounted,',
    'return dict(serve_baseline_rss_bytes=baseline, ram_loaded_total_rss_bytes=loaded_rss,\n'
    '                    ram_peak_total_rss_bytes=baseline+max(load_peak, query_peak),\n'
    '                    ram_accounted_bytes=accounted,')
assert new != old and 'build_peak_total_rss_bytes' in new
p.write_text(new)
(w/'harness-instrumentation.patch').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True), fromfile='a/benchmark/measure_memory.py',
    tofile='b/benchmark/measure_memory.py')))

# storage() returns max(serialized bytes, component allocation) for native adapters. Retain
# that call's build/reload behavior, but label effective storage from the actual saved files.
p=w/'source/benchmark/run.py'
old=p.read_text()
needle='            total_bytes = index.storage(str(Path(temporary) / "index.bin"))'
replacement='''            saved = Path(temporary) / "index.bin"
            index.storage(str(saved))
            total_bytes = (sum(p.stat().st_size for p in saved.iterdir()) if saved.is_dir()
                           else saved.stat().st_size)'''
assert needle in old
new=old.replace(needle,replacement)
p.write_text(new)
with (w/'harness-instrumentation.patch').open('a') as f:
    f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
        fromfile='a/benchmark/run.py',tofile='b/benchmark/run.py')))
