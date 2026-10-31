### RaBitQ 1 first stage (1,165 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| rabitq:5 | 660 | 32, 50 (0.919) | 128, 50 (0.966) | — | — | 0.970 (nprobe 128, k' 200) |
| rabitq:6 | 788 | 32, 50 (0.926) | 64, 50 (0.954) | 128, 50 (0.974) | — | 0.979 (nprobe 128, k' 200) |
| int8 | 1,024 | 32, 50 (0.928) | 64, 50 (0.957) | 128, 50 (0.978) | — | 0.984 (nprobe 128, k' 200) |
| fp32 | 4,096 | 32, 50 (0.932) | 64, 50 (0.961) | 128, 50 (0.983) | — | 0.989 (nprobe 128, k' 200) |

- R@10 ≥ 0.90: smallest tier rabitq:5 (660 B), nprobe 32, k' 50, recall 0.919
- R@10 ≥ 0.95: smallest tier rabitq:5 (660 B), nprobe 128, k' 50, recall 0.966
- R@10 ≥ 0.97: smallest tier rabitq:6 (788 B), nprobe 128, k' 50, recall 0.974

### TieredQ 256 first stage (357 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 349 | 64, 500 (0.920) | — | — | — | 0.935 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 477 | 64, 200 (0.905) | — | — | — | 0.944 (nprobe 128, k' 500) |
| int8 | 1,024 | 32, 500 (0.910) | 128, 500 (0.957) | — | — | 0.957 (nprobe 128, k' 500) |
| fp32 | 4,096 | 32, 500 (0.914) | 128, 500 (0.961) | — | — | 0.961 (nprobe 128, k' 500) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (349 B), nprobe 64, k' 500, recall 0.920
- R@10 ≥ 0.95: smallest tier int8 (1,024 B), nprobe 128, k' 500, recall 0.957

### TieredQ 384 first stage (478 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 339 | 32, 500 (0.906) | 128, 500 (0.952) | — | — | 0.952 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 470 | 32, 200 (0.904) | 128, 200 (0.951) | — | — | 0.964 (nprobe 128, k' 500) |
| int8 | 1,024 | 32, 200 (0.913) | 64, 500 (0.956) | 128, 500 (0.975) | — | 0.975 (nprobe 128, k' 500) |
| fp32 | 4,096 | 32, 100 (0.902) | 64, 500 (0.960) | 128, 500 (0.980) | — | 0.980 (nprobe 128, k' 500) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (339 B), nprobe 32, k' 500, recall 0.906
- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (339 B), nprobe 128, k' 500, recall 0.952
- R@10 ≥ 0.97: smallest tier int8 (1,024 B), nprobe 128, k' 500, recall 0.975

### TieredQ 512 first stage (608 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 327 | 32, 100 (0.901) | 128, 200 (0.956) | — | — | 0.960 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 455 | 32, 100 (0.911) | 64, 500 (0.952) | 128, 500 (0.972) | — | 0.972 (nprobe 128, k' 500) |
| int8 | 1,024 | 32, 50 (0.904) | 64, 200 (0.957) | 128, 200 (0.977) | — | 0.982 (nprobe 128, k' 500) |
| fp32 | 4,096 | 32, 50 (0.907) | 64, 100 (0.953) | 128, 100 (0.973) | — | 0.988 (nprobe 128, k' 500) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (327 B), nprobe 32, k' 100, recall 0.901
- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (327 B), nprobe 128, k' 200, recall 0.956
- R@10 ≥ 0.97: smallest tier tieredq-refinement:4 (455 B), nprobe 128, k' 500, recall 0.972

### TieredQ 768 first stage (867 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 281 | 32, 50 (0.908) | 128, 50 (0.955) | — | — | 0.960 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 409 | 32, 50 (0.918) | 64, 100 (0.952) | 128, 100 (0.971) | — | 0.973 (nprobe 128, k' 500) |
| int8 | 1,024 | 32, 50 (0.926) | 64, 50 (0.958) | 128, 50 (0.979) | — | 0.984 (nprobe 128, k' 500) |
| fp32 | 4,096 | 32, 50 (0.930) | 64, 50 (0.963) | 128, 50 (0.984) | 128, 500 (0.990) | 0.990 (nprobe 128, k' 500) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (281 B), nprobe 32, k' 50, recall 0.908
- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (281 B), nprobe 128, k' 50, recall 0.955
- R@10 ≥ 0.97: smallest tier tieredq-refinement:4 (409 B), nprobe 128, k' 100, recall 0.971
- R@10 ≥ 0.99: smallest tier fp32 (4,096 B), nprobe 128, k' 500, recall 0.990

### TieredQ 1024 first stage (1,133 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 265 | 32, 20 (0.909) | 128, 20 (0.955) | — | — | 0.964 (nprobe 128, k' 100) |
| tieredq-refinement:4 | 393 | 32, 20 (0.916) | 64, 50 (0.955) | 128, 50 (0.975) | — | 0.975 (nprobe 128, k' 100) |
| int8 | 1,024 | 32, 20 (0.921) | 64, 20 (0.952) | 128, 20 (0.973) | — | 0.984 (nprobe 128, k' 100) |
| fp32 | 4,096 | 32, 20 (0.925) | 64, 20 (0.956) | 128, 20 (0.977) | 128, 100 (0.990) | 0.990 (nprobe 128, k' 100) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (265 B), nprobe 32, k' 20, recall 0.909
- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (265 B), nprobe 128, k' 20, recall 0.955
- R@10 ≥ 0.97: smallest tier tieredq-refinement:4 (393 B), nprobe 128, k' 50, recall 0.975
- R@10 ≥ 0.99: smallest tier fp32 (4,096 B), nprobe 128, k' 100, recall 0.990

