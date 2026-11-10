"""Recover a verified sequential download prefix if the native downloader exits early."""
from pathlib import Path
import json
import os
import sys

big=Path(sys.argv[1])
old=big/'msmarco-qwen-1024-normalized.hdf5.partial'
resumed=big/'msmarco-qwen-http-resume.hdf5.partial'
if resumed.exists():
    print(resumed.name)
else:
    prefixes=list((big/'xet-download/.cache/huggingface/download').glob('*.incomplete'))
    prefix=max(prefixes,key=lambda p:p.stat().st_size,default=None)
    old_bytes=old.stat().st_blocks*512 if old.exists() else 0
    if prefix and prefix.stat().st_size > old_bytes+(128<<20):
        # The native downloader is configured for sequential writes. Discard
        # its trailing chunk so an interrupted append cannot introduce holes.
        safe=max(0,(prefix.stat().st_size//(64<<20)-1)*(64<<20))
        with prefix.open('r+b') as f:f.truncate(safe)
        os.replace(prefix,resumed)
        print(resumed.name)
    else: print(old.name)
