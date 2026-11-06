"""Reuse the validated adapter instead of rebuilding it at each matching CLI invocation."""
import difflib
import hashlib
import json
from pathlib import Path
import sys

w=Path(sys.argv[1]).resolve()
b=w/'source/benchmark'
assert (w/'adapter-revalidation-exit-code').read_text().strip()=='0'
module=next((b/'build-native').glob('codec_native*.so'))
expected=hashlib.file_digest(module.open('rb'),'sha256').hexdigest()
original=json.loads((w/'source-hashes.json').read_text())[str(module.relative_to(w/'source'))]
assert expected==original,'Validated binary changed'
marker=dict(module=str(module),sha256=expected)
(b/'issue4-prebuilt.json').write_text(json.dumps(marker,indent=2))
p=b/'native_build.py'
old=p.read_text()
needle='def build_native(full_score=False):\n'
addition='''    marker = HERE/"issue4-prebuilt.json"
    if marker.exists() and not full_score:
        import hashlib
        saved = json.loads(marker.read_text())
        module = Path(saved["module"])
        if hashlib.file_digest(module.open("rb"), "sha256").hexdigest() != saved["sha256"]:
            raise RuntimeError("Frozen issue4 adapter hash changed")
        directory = HERE/"build-native"
        commands = audit_compile_commands(json.loads((directory/"compile_commands.json").read_text()))
        return NativeBuild(directory, HERE/".deps/rabitq-native", commands)
'''
assert needle in old and addition not in old
new=old.replace(needle,needle+addition)
p.write_text(new)
(w/'frozen-native.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
    fromfile='a/benchmark/native_build.py',tofile='b/benchmark/native_build.py')))
(w/'frozen-native.json').write_text(json.dumps(marker,indent=2))
print('Validated native adapter frozen at',expected,flush=True)
