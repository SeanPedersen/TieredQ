import sys, re
from pathlib import Path
import codec_native
index = codec_native.Index.load(sys.argv[1], True)
text = Path("/proc/self/smaps_rollup").read_text()
print({k: v for k, v in re.findall(r"^(Rss|AnonHugePages):\s+(\d+ kB)", text, re.M)})
