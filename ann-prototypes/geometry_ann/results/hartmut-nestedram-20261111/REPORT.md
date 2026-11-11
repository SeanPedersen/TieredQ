nested-default: 0.2412 GiB, lists 2048, storage 0.1346 GiB, serialized 0.1392 GiB, build 35.9 s
  USS per step KiB {'start': 0.0, 'build': 182492.0, 'write_tier': 249980.0, 'attach_tier': 249980.0}  heap KiB {'arena': 127432, 'in_use': 4201867, 'free': 92752, 'mmapped': 4167188}
  grown: [('anon@7fd4f81f3000', 129076), ('[heap]', 21808), ('anon@7fd611c00000', 20700), ('/usr/lib/libopenblas.so.0.3', 4148), ('anon@7fd5e8000000', 3424), ('anon@7fd508000000', 2316)] ... 30 regions of 2.0-2.6 MiB = 67.9 MiB
    nprobe   16 cand 200: recall 0.9259  345 QPS  1432 QPS/GB
    nprobe   24 cand 200: recall 0.9546  334 QPS  1384 QPS/GB
    nprobe   32 cand 200: recall 0.9701  325 QPS  1348 QPS/GB
    nprobe   48 cand 200: recall 0.9849  312 QPS  1293 QPS/GB
    nprobe   64 cand 200: recall 0.9911  297 QPS  1232 QPS/GB
    nprobe   96 cand 200: recall 0.9968  274 QPS  1135 QPS/GB
    nprobe  128 cand 200: recall 0.9984  250 QPS  1036 QPS/GB
nested-default-s1: 0.2502 GiB, lists 2048, storage 0.1346 GiB, serialized 0.1392 GiB, build 36.4 s
  USS per step KiB {'start': 0.0, 'build': -207724.0, 'write_tier': -140248.0, 'attach_tier': -140244.0}  heap KiB {'arena': 126884, 'in_use': 4203122, 'free': 91333, 'mmapped': 4167572}
  grown: [('anon@7fb2941f9000', 129052), ('[heap]', 26768), ('anon@7fb3c7000000', 20700), ('/usr/lib/libopenblas.so.0.3', 4148), ('anon@7fb304000000', 3424), ('anon@7fb2a4000000', 2316)] ... 30 regions of 2.0-2.6 MiB = 67.9 MiB
    nprobe   16 cand 200: recall 0.9254  346 QPS  1384 QPS/GB
    nprobe   24 cand 200: recall 0.9547  336 QPS  1343 QPS/GB
    nprobe   32 cand 200: recall 0.9701  331 QPS  1322 QPS/GB
    nprobe   48 cand 200: recall 0.9853  314 QPS  1257 QPS/GB
    nprobe   64 cand 200: recall 0.9915  298 QPS  1191 QPS/GB
    nprobe   96 cand 200: recall 0.9967  277 QPS  1108 QPS/GB
    nprobe  128 cand 200: recall 0.9985  248 QPS  992 QPS/GB
nested-fix: 0.1778 GiB, lists 2048, storage 0.1346 GiB, serialized 0.1392 GiB, build 38.4 s
  USS per step KiB {'start': 0.0, 'build': 179280.0, 'write_tier': 183172.0, 'attach_tier': 183176.0}  heap KiB {'arena': 53416, 'in_use': 4201490, 'free': 19521, 'mmapped': 4167596}
  grown: [('anon@7fc7941f3000', 129076), ('[heap]', 27528), ('anon@7fc8aec00000', 20700), ('/usr/lib/libopenblas.so.0.3', 4148), ('/home/sean/experiments/TieredQ-issue43-heapretention-20261111/.venv-fix/lib/python3.12/site-packages/tieredq/_core.cpython-312-x86_64-linux-gnu.so', 1784), ('other', 3180)] ... 0 regions of 2.0-2.6 MiB = 0.0 MiB
    nprobe   16 cand 200: recall 0.9259  346 QPS  1947 QPS/GB
    nprobe   24 cand 200: recall 0.9546  332 QPS  1866 QPS/GB
    nprobe   32 cand 200: recall 0.9701  279 QPS  1567 QPS/GB
    nprobe   48 cand 200: recall 0.9849  293 QPS  1649 QPS/GB
    nprobe   64 cand 200: recall 0.9911  289 QPS  1623 QPS/GB
    nprobe   96 cand 200: recall 0.9968  265 QPS  1489 QPS/GB
    nprobe  128 cand 200: recall 0.9984  249 QPS  1400 QPS/GB
nested-fix-s1: 0.1832 GiB, lists 2048, storage 0.1346 GiB, serialized 0.1392 GiB, build 38.2 s
  USS per step KiB {'start': 0.0, 'build': -213100.0, 'write_tier': -210216.0, 'attach_tier': -210212.0}  heap KiB {'arena': 57124, 'in_use': 4203124, 'free': 21571, 'mmapped': 4167572}
  grown: [('anon@7f11bc1f9000', 129052), ('[heap]', 26536), ('anon@7f12e5c00000', 20700), ('/usr/lib/libopenblas.so.0.3', 4148), ('anon@7f1204000000', 2660), ('/home/sean/experiments/TieredQ-issue43-heapretention-20261111/.venv-fix/lib/python3.12/site-packages/tieredq/_core.cpython-312-x86_64-linux-gnu.so', 1784)] ... 0 regions of 2.0-2.6 MiB = 0.0 MiB
    nprobe   16 cand 200: recall 0.9254  348 QPS  1900 QPS/GB
    nprobe   24 cand 200: recall 0.9547  325 QPS  1776 QPS/GB
    nprobe   32 cand 200: recall 0.9701  279 QPS  1522 QPS/GB
    nprobe   48 cand 200: recall 0.9853  302 QPS  1648 QPS/GB
    nprobe   64 cand 200: recall 0.9915  292 QPS  1594 QPS/GB
    nprobe   96 cand 200: recall 0.9967  252 QPS  1377 QPS/GB
    nprobe  128 cand 200: recall 0.9985  245 QPS  1339 QPS/GB
