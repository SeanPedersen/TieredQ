### RaBitQ 1 first stage (1,683 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| rabitq:5 | 980 | 32, 20 (0.952) | 64, 50 (0.977) | 128, 50 (0.984) | — | 0.984 (nprobe 128, k' 100) |
| rabitq:6 | 1,172 | 32, 20 (0.956) | 64, 20 (0.971) | 64, 50 (0.982) | — | 0.990 (nprobe 128, k' 100) |
| int8 | 1,536 | 32, 20 (0.960) | 32, 50 (0.971) | 64, 50 (0.986) | 128, 50 (0.994) | 0.995 (nprobe 128, k' 100) |

- R@10 ≥ 0.95: smallest tier rabitq:5 (980 B), nprobe 32, k' 20, recall 0.952
- R@10 ≥ 0.97: smallest tier rabitq:5 (980 B), nprobe 64, k' 50, recall 0.977
- R@10 ≥ 0.98: smallest tier rabitq:5 (980 B), nprobe 128, k' 50, recall 0.984
- R@10 ≥ 0.99: smallest tier int8 (1,536 B), nprobe 128, k' 50, recall 0.994

### TieredQ 512 first stage (627 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 517 | 32, 500 (0.950) | 128, 500 (0.970) | — | — | 0.970 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 702 | 32, 100 (0.954) | 64, 200 (0.970) | — | — | 0.979 (nprobe 128, k' 500) |
| int8 | 1,536 | 32, 50 (0.955) | 32, 500 (0.970) | 64, 100 (0.982) | 128, 100 (0.990) | 0.995 (nprobe 128, k' 500) |
| fp32 | 6,144 | 32, 50 (0.956) | 32, 200 (0.972) | 64, 100 (0.985) | 128, 100 (0.993) | 0.998 (nprobe 128, k' 500) |

- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (517 B), nprobe 32, k' 500, recall 0.950
- R@10 ≥ 0.97: smallest tier tieredq-refinement:3 (517 B), nprobe 128, k' 500, recall 0.970
- R@10 ≥ 0.98: smallest tier int8 (1,536 B), nprobe 64, k' 100, recall 0.982
- R@10 ≥ 0.99: smallest tier int8 (1,536 B), nprobe 128, k' 100, recall 0.990

### TieredQ 768 first stage (886 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 475 | 32, 500 (0.950) | — | — | — | 0.970 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 672 | 32, 50 (0.958) | 64, 50 (0.973) | 128, 50 (0.980) | — | 0.983 (nprobe 128, k' 500) |
| int8 | 1,536 | 32, 50 (0.967) | 32, 200 (0.970) | 64, 50 (0.983) | 128, 50 (0.991) | 0.995 (nprobe 128, k' 500) |
| fp32 | 6,144 | 32, 50 (0.969) | 32, 100 (0.972) | 64, 50 (0.985) | 128, 50 (0.994) | 0.998 (nprobe 128, k' 500) |

- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (475 B), nprobe 32, k' 500, recall 0.950
- R@10 ≥ 0.97: smallest tier tieredq-refinement:4 (672 B), nprobe 64, k' 50, recall 0.973
- R@10 ≥ 0.98: smallest tier tieredq-refinement:4 (672 B), nprobe 128, k' 50, recall 0.980
- R@10 ≥ 0.99: smallest tier int8 (1,536 B), nprobe 128, k' 50, recall 0.991

### TieredQ 1024 first stage (1,138 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 462 | 32, 50 (0.953) | 128, 50 (0.974) | — | — | 0.974 (nprobe 128, k' 100) |
| tieredq-refinement:4 | 647 | 32, 50 (0.960) | 64, 50 (0.976) | 128, 50 (0.983) | — | 0.984 (nprobe 128, k' 100) |
| int8 | 1,536 | 32, 20 (0.954) | 32, 100 (0.970) | 64, 50 (0.986) | 128, 50 (0.994) | 0.995 (nprobe 128, k' 100) |
| fp32 | 6,144 | 32, 20 (0.955) | 32, 50 (0.972) | 64, 50 (0.988) | 128, 50 (0.997) | 0.998 (nprobe 128, k' 100) |

- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (462 B), nprobe 32, k' 50, recall 0.953
- R@10 ≥ 0.97: smallest tier tieredq-refinement:3 (462 B), nprobe 128, k' 50, recall 0.974
- R@10 ≥ 0.98: smallest tier tieredq-refinement:4 (647 B), nprobe 128, k' 50, recall 0.983
- R@10 ≥ 0.99: smallest tier int8 (1,536 B), nprobe 128, k' 50, recall 0.994

