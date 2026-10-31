### RaBitQ 1 first stage (1,165 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| rabitq:5 | 660 | 8, 50 (0.913) | 32, 50 (0.964) | 64, 50 (0.976) | — | 0.982 (nprobe 128, k' 200) |
| rabitq:6 | 788 | 8, 20 (0.903) | 16, 50 (0.954) | 32, 50 (0.971) | — | 0.989 (nprobe 128, k' 200) |
| int8 | 1,024 | 8, 20 (0.902) | 16, 50 (0.954) | 32, 50 (0.972) | 128, 50 (0.990) | 0.992 (nprobe 128, k' 200) |
| fp32 | 4,096 | 8, 20 (0.905) | 16, 50 (0.959) | 32, 50 (0.977) | 64, 50 (0.991) | 0.998 (nprobe 128, k' 200) |

- R@10 ≥ 0.90: smallest tier rabitq:5 (660 B), nprobe 8, k' 50, recall 0.913
- R@10 ≥ 0.95: smallest tier rabitq:5 (660 B), nprobe 32, k' 50, recall 0.964
- R@10 ≥ 0.97: smallest tier rabitq:5 (660 B), nprobe 64, k' 50, recall 0.976
- R@10 ≥ 0.99: smallest tier int8 (1,024 B), nprobe 128, k' 50, recall 0.990

### TieredQ 256 first stage (357 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 349 | 8, 500 (0.904) | 32, 500 (0.951) | — | — | 0.964 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 477 | 8, 200 (0.906) | 32, 200 (0.955) | 64, 500 (0.972) | — | 0.976 (nprobe 128, k' 500) |
| int8 | 1,024 | 8, 200 (0.912) | 16, 500 (0.953) | 64, 200 (0.973) | — | 0.986 (nprobe 128, k' 500) |
| fp32 | 4,096 | 8, 200 (0.915) | 16, 200 (0.950) | 32, 500 (0.975) | 128, 500 (0.992) | 0.992 (nprobe 128, k' 500) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (349 B), nprobe 8, k' 500, recall 0.904
- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (349 B), nprobe 32, k' 500, recall 0.951
- R@10 ≥ 0.97: smallest tier tieredq-refinement:4 (477 B), nprobe 64, k' 500, recall 0.972
- R@10 ≥ 0.99: smallest tier fp32 (4,096 B), nprobe 128, k' 500, recall 0.992

### TieredQ 384 first stage (478 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 339 | 8, 100 (0.901) | 32, 200 (0.954) | — | — | 0.970 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 470 | 8, 100 (0.906) | 32, 100 (0.957) | 64, 200 (0.972) | — | 0.980 (nprobe 128, k' 500) |
| int8 | 1,024 | 8, 100 (0.913) | 16, 200 (0.953) | 32, 200 (0.971) | 128, 500 (0.991) | 0.991 (nprobe 128, k' 500) |
| fp32 | 4,096 | 8, 50 (0.902) | 16, 100 (0.952) | 32, 200 (0.976) | 64, 500 (0.991) | 0.997 (nprobe 128, k' 500) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (339 B), nprobe 8, k' 100, recall 0.901
- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (339 B), nprobe 32, k' 200, recall 0.954
- R@10 ≥ 0.97: smallest tier tieredq-refinement:4 (470 B), nprobe 64, k' 200, recall 0.972
- R@10 ≥ 0.99: smallest tier int8 (1,024 B), nprobe 128, k' 500, recall 0.991

### TieredQ 512 first stage (608 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 327 | 8, 50 (0.902) | 32, 100 (0.956) | 128, 200 (0.972) | — | 0.972 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 455 | 8, 50 (0.908) | 32, 50 (0.958) | 64, 100 (0.975) | — | 0.982 (nprobe 128, k' 500) |
| int8 | 1,024 | 8, 50 (0.913) | 16, 100 (0.954) | 32, 100 (0.972) | 128, 200 (0.991) | 0.991 (nprobe 128, k' 500) |
| fp32 | 4,096 | 8, 50 (0.917) | 16, 50 (0.952) | 32, 100 (0.978) | 64, 200 (0.991) | 0.998 (nprobe 128, k' 500) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (327 B), nprobe 8, k' 50, recall 0.902
- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (327 B), nprobe 32, k' 100, recall 0.956
- R@10 ≥ 0.97: smallest tier tieredq-refinement:3 (327 B), nprobe 128, k' 200, recall 0.972
- R@10 ≥ 0.99: smallest tier int8 (1,024 B), nprobe 128, k' 200, recall 0.991

### TieredQ 768 first stage (867 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 281 | 8, 50 (0.907) | 32, 50 (0.956) | 128, 50 (0.970) | — | 0.971 (nprobe 128, k' 500) |
| tieredq-refinement:4 | 409 | 8, 20 (0.902) | 32, 20 (0.951) | 64, 50 (0.976) | — | 0.982 (nprobe 128, k' 500) |
| int8 | 1,024 | 8, 20 (0.906) | 16, 50 (0.956) | 32, 50 (0.974) | 128, 50 (0.990) | 0.992 (nprobe 128, k' 500) |
| fp32 | 4,096 | 8, 20 (0.910) | 16, 50 (0.960) | 32, 50 (0.979) | 64, 50 (0.991) | 0.998 (nprobe 128, k' 500) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (281 B), nprobe 8, k' 50, recall 0.907
- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (281 B), nprobe 32, k' 50, recall 0.956
- R@10 ≥ 0.97: smallest tier tieredq-refinement:3 (281 B), nprobe 128, k' 50, recall 0.970
- R@10 ≥ 0.99: smallest tier int8 (1,024 B), nprobe 128, k' 50, recall 0.990

### TieredQ 1024 first stage (1,133 index bits/vector), 256 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.90: nprobe, k' (recall) | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-refinement:3 | 265 | 8, 20 (0.905) | 32, 20 (0.955) | 128, 50 (0.972) | — | 0.972 (nprobe 128, k' 100) |
| tieredq-refinement:4 | 393 | 8, 20 (0.913) | 16, 50 (0.951) | 64, 20 (0.975) | — | 0.985 (nprobe 128, k' 100) |
| int8 | 1,024 | 8, 20 (0.917) | 16, 20 (0.952) | 32, 20 (0.970) | 128, 50 (0.992) | 0.992 (nprobe 128, k' 100) |
| fp32 | 4,096 | 8, 20 (0.920) | 16, 20 (0.957) | 32, 20 (0.975) | 64, 50 (0.992) | 0.998 (nprobe 128, k' 100) |

- R@10 ≥ 0.90: smallest tier tieredq-refinement:3 (265 B), nprobe 8, k' 20, recall 0.905
- R@10 ≥ 0.95: smallest tier tieredq-refinement:3 (265 B), nprobe 32, k' 20, recall 0.955
- R@10 ≥ 0.97: smallest tier tieredq-refinement:3 (265 B), nprobe 128, k' 50, recall 0.972
- R@10 ≥ 0.99: smallest tier int8 (1,024 B), nprobe 128, k' 50, recall 0.992

