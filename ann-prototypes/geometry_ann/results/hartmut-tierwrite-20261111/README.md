# Tier-write time A/B: no regression since 9b158c6 (#52 follow-up)

Remote root: `/home/sean/experiments/TieredQ-issue52-tierwrite-20261111`. `../hartmut-quiethost-20261111` measured
`write_tier` at 117 s on 6298356, against 94 s at 9b158c6 (`../hartmut-pythondefaults-20261111`). This study asks
whether that came from our code or from the host.
- A is the 9b158c6 tree and package of that earlier run, unchanged since.
- B is 6298356, with the heap fixes and packed lists (the `../hartmut-packedlists-20261111` package).
- `study.py` runs `../hartmut-quiethost-20261111/build.py` in ABBA order. That is the 1-bit nested default through
  Python, defaults only, on Qwen/MS MARCO 8.84M from the SQ8 memmap, 32 threads, tier on NVMe.
- Each step records the host at its start and the build's user and system CPU seconds. The index and tier were
  removed after each step.

Host: the production containers were not pinned and the services ran. Load at each step's start was 1.6 / 22.7 /
21.5 / 24.7, mostly the previous step's 32 threads winding down.

## Results

| step | build | write_tier | save | user CPU | system CPU | peak RSS | index file |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 A | 214.5 s | 94.2 s | 8.3 s | 7,104 s | 9.7 s | 20.98 GB | 1.630 GB |
| 2 B | 208.1 s | 105.7 s | 7.8 s | 6,994 s | 44.5 s | 20.97 GB | 1.491 GB |
| 3 B | 207.9 s | 102.2 s | 7.7 s | 6,977 s | 45.3 s | 20.98 GB | 1.491 GB |
| 4 A | 210.9 s | 105.5 s | 7.5 s | 7,072 s | 7.6 s | 20.98 GB | 1.630 GB |

The tier file is 7.93 GB in both versions.

**No regression.** Build plus tier write averages 312.6 s for A and 312.0 s for B.
- B's `write_tier` averages 4% more than A's (104.0 s against 99.9 s), but A's own two runs differ by 12%.
- B's build is 2% faster (208.0 s against 212.7 s).
- B uses about 37 s more system CPU, from returning freed heap to the OS. Spread over 32 threads that is about 1 s of
  wall time, and it is what lowers the serving RAM.

The same A tree took 94 s at load 1.6 and 106 s at load 25. So the earlier 302 s against 336 s was host state, not
code. The heap fixes and packed lists cost no build time.
