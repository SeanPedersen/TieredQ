"""Wait for the owned primary process, then run the native timing control without overlap."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
HERE=Path(__file__).resolve().parent
pid=int(sys.argv[1])
try:
    descriptor=os.pidfd_open(pid)
except ProcessLookupError:
    descriptor=None
if descriptor is not None:
    select.select([descriptor],[],[])
    os.close(descriptor)
assert json.loads((HERE/'meta.json').read_text())['status']=='complete'
print('PRIMARY COMPLETE; starting native-pruned timing control',flush=True)
with (HERE/'native-job.log').open('w') as log:
    result=subprocess.run([sys.executable,'-u',str(HERE/'run_native.py')],stdout=log,stderr=subprocess.STDOUT)
(HERE/'native-exit-code').write_text(str(result.returncode)+'\n')
print('NATIVE EXIT',result.returncode,flush=True)
