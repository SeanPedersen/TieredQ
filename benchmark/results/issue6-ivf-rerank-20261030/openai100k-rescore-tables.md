### RaBitQ 1 first stage (1,711 index bits/vector), 64 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq:2 | 389 | 32, 50 (0.953) | — | — | — | 0.957 (nprobe 64, k' 100) |
| tieredq:3 | 581 | 16, 50 (0.957) | 32, 50 (0.972) | — | — | 0.977 (nprobe 64, k' 100) |
| rabitq:3 | 596 | — | — | — | — | 0.949 (nprobe 64, k' 100) |
| tieredq:4 | 773 | 16, 20 (0.951) | 32, 50 (0.981) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 100) |
| rabitq:4 | 788 | 16, 50 (0.951) | 64, 100 (0.970) | — | — | 0.970 (nprobe 64, k' 100) |
| tieredq:5 | 965 | 16, 20 (0.954) | 32, 50 (0.986) | 32, 50 (0.986) | 64, 50 (0.992) | 0.993 (nprobe 64, k' 100) |
| rabitq:5 | 980 | 16, 50 (0.962) | 32, 50 (0.979) | 64, 50 (0.985) | — | 0.985 (nprobe 64, k' 100) |
| rabitq:6 | 1,172 | 16, 20 (0.953) | 32, 50 (0.985) | 32, 50 (0.985) | 64, 50 (0.991) | 0.991 (nprobe 64, k' 100) |
| rabitq:7 | 1,364 | 16, 20 (0.956) | 16, 50 (0.972) | 32, 50 (0.989) | 64, 50 (0.996) | 0.996 (nprobe 64, k' 100) |
| int8 | 1,536 | 16, 20 (0.957) | 16, 50 (0.973) | 32, 50 (0.990) | 32, 50 (0.990) | 0.997 (nprobe 64, k' 100) |
| fp16 | 3,072 | 16, 20 (0.958) | 16, 50 (0.975) | 32, 50 (0.993) | 32, 50 (0.993) | 1.000 (nprobe 64, k' 100) |
| fp32 | 6,144 | 16, 20 (0.958) | 16, 50 (0.975) | 32, 50 (0.993) | 32, 50 (0.993) | 1.000 (nprobe 64, k' 100) |

- R@10 ≥ 0.95: smallest tier tieredq:2 (389 B), nprobe 32, k' 50, recall 0.953
- R@10 ≥ 0.97: smallest tier tieredq:3 (581 B), nprobe 32, k' 50, recall 0.972
- R@10 ≥ 0.98: smallest tier tieredq:4 (773 B), nprobe 32, k' 50, recall 0.981
- R@10 ≥ 0.99: smallest tier tieredq:5 (965 B), nprobe 64, k' 50, recall 0.992

### TieredQ 256 first stage (411 index bits/vector), 64 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-nested:3 | 399 | 16, 500 (0.953) | 64, 500 (0.971) | — | — | 0.973 (nprobe 64, k' 1000) |
| tieredq-nested:3 | 461 | 16, 500 (0.953) | 64, 500 (0.971) | — | — | 0.973 (nprobe 64, k' 1000) |
| tieredq-nested:3 | 474 | 16, 500 (0.953) | 64, 500 (0.971) | — | — | 0.973 (nprobe 64, k' 1000) |
| tieredq-nested:3 | 516 | 16, 500 (0.953) | 64, 500 (0.971) | — | — | 0.973 (nprobe 64, k' 1000) |
| tieredq-nested:3 | 549 | 16, 500 (0.953) | 64, 500 (0.971) | — | — | 0.973 (nprobe 64, k' 1000) |
| tieredq-nested:4 | 599 | 16, 200 (0.953) | 32, 500 (0.977) | 64, 500 (0.983) | — | 0.985 (nprobe 64, k' 1000) |
| tieredq-nested:4 | 658 | 16, 200 (0.953) | 32, 500 (0.977) | 64, 500 (0.983) | — | 0.985 (nprobe 64, k' 1000) |
| tieredq-nested:4 | 671 | 16, 200 (0.953) | 32, 500 (0.977) | 64, 500 (0.983) | — | 0.985 (nprobe 64, k' 1000) |
| tieredq-nested:4 | 711 | 16, 200 (0.953) | 32, 500 (0.977) | 64, 500 (0.983) | — | 0.985 (nprobe 64, k' 1000) |
| tieredq-nested:4 | 734 | 16, 200 (0.953) | 32, 500 (0.977) | 64, 500 (0.983) | — | 0.985 (nprobe 64, k' 1000) |
| fp32 | 6,144 | 16, 200 (0.964) | 16, 500 (0.974) | 32, 500 (0.991) | 32, 500 (0.991) | 1.000 (nprobe 64, k' 1000) |

- R@10 ≥ 0.95: smallest tier tieredq-nested:3 (549 B), nprobe 16, k' 500, recall 0.953
- R@10 ≥ 0.97: smallest tier tieredq-nested:3 (549 B), nprobe 64, k' 500, recall 0.971
- R@10 ≥ 0.98: smallest tier tieredq-nested:4 (734 B), nprobe 64, k' 500, recall 0.983
- R@10 ≥ 0.99: smallest tier fp32 (6,144 B), nprobe 32, k' 500, recall 0.991

### TieredQ 512 first stage (690 index bits/vector), 64 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-nested:3 | 399 | 16, 100 (0.951) | 64, 100 (0.971) | — | — | 0.973 (nprobe 64, k' 500) |
| tieredq-nested:3 | 461 | 16, 100 (0.951) | 64, 100 (0.971) | — | — | 0.973 (nprobe 64, k' 500) |
| tieredq-nested:3 | 474 | 16, 100 (0.951) | 64, 100 (0.971) | — | — | 0.973 (nprobe 64, k' 500) |
| tieredq-nested:3 | 516 | 16, 100 (0.951) | 64, 100 (0.971) | — | — | 0.973 (nprobe 64, k' 500) |
| tieredq-nested:3 | 549 | 16, 100 (0.951) | 64, 100 (0.971) | — | — | 0.973 (nprobe 64, k' 500) |
| tieredq-nested:4 | 599 | 16, 50 (0.953) | 32, 100 (0.976) | 64, 100 (0.982) | — | 0.985 (nprobe 64, k' 500) |
| tieredq-nested:4 | 658 | 16, 50 (0.953) | 32, 100 (0.976) | 64, 100 (0.982) | — | 0.985 (nprobe 64, k' 500) |
| tieredq-nested:4 | 671 | 16, 50 (0.953) | 32, 100 (0.976) | 64, 100 (0.982) | — | 0.985 (nprobe 64, k' 500) |
| tieredq-nested:4 | 711 | 16, 50 (0.953) | 32, 100 (0.976) | 64, 100 (0.982) | — | 0.985 (nprobe 64, k' 500) |
| tieredq-nested:4 | 734 | 16, 50 (0.953) | 32, 100 (0.976) | 64, 100 (0.982) | — | 0.985 (nprobe 64, k' 500) |
| fp32 | 6,144 | 16, 50 (0.964) | 16, 100 (0.973) | 32, 100 (0.990) | 32, 100 (0.990) | 1.000 (nprobe 64, k' 500) |

- R@10 ≥ 0.95: smallest tier tieredq-nested:3 (516 B), nprobe 16, k' 100, recall 0.951
- R@10 ≥ 0.97: smallest tier tieredq-nested:3 (516 B), nprobe 64, k' 100, recall 0.971
- R@10 ≥ 0.98: smallest tier tieredq-nested:4 (711 B), nprobe 64, k' 100, recall 0.982
- R@10 ≥ 0.99: smallest tier fp32 (6,144 B), nprobe 32, k' 100, recall 0.990

### TieredQ 768 first stage (976 index bits/vector), 64 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-nested:3 | 399 | 16, 50 (0.954) | 32, 200 (0.970) | — | — | 0.975 (nprobe 64, k' 200) |
| tieredq-nested:3 | 461 | 16, 50 (0.954) | 32, 200 (0.970) | — | — | 0.975 (nprobe 64, k' 200) |
| tieredq-nested:3 | 474 | 16, 50 (0.954) | 32, 200 (0.970) | — | — | 0.975 (nprobe 64, k' 200) |
| tieredq-nested:3 | 516 | 16, 50 (0.954) | 32, 200 (0.970) | — | — | 0.975 (nprobe 64, k' 200) |
| tieredq-nested:3 | 549 | 16, 50 (0.954) | 32, 200 (0.970) | — | — | 0.975 (nprobe 64, k' 200) |
| tieredq-nested:4 | 599 | 16, 50 (0.962) | 32, 50 (0.977) | 64, 50 (0.983) | — | 0.986 (nprobe 64, k' 200) |
| tieredq-nested:4 | 658 | 16, 50 (0.962) | 32, 50 (0.977) | 64, 50 (0.983) | — | 0.986 (nprobe 64, k' 200) |
| tieredq-nested:4 | 671 | 16, 50 (0.962) | 32, 50 (0.977) | 64, 50 (0.983) | — | 0.986 (nprobe 64, k' 200) |
| tieredq-nested:4 | 711 | 16, 50 (0.962) | 32, 50 (0.977) | 64, 50 (0.983) | — | 0.986 (nprobe 64, k' 200) |
| tieredq-nested:4 | 734 | 16, 50 (0.962) | 32, 50 (0.977) | 64, 50 (0.983) | — | 0.986 (nprobe 64, k' 200) |
| fp32 | 6,144 | 16, 50 (0.972) | 16, 50 (0.972) | 32, 50 (0.989) | 32, 100 (0.993) | 1.000 (nprobe 64, k' 200) |

- R@10 ≥ 0.95: smallest tier tieredq-nested:3 (474 B), nprobe 16, k' 50, recall 0.954
- R@10 ≥ 0.97: smallest tier tieredq-nested:3 (474 B), nprobe 32, k' 200, recall 0.970
- R@10 ≥ 0.98: smallest tier tieredq-nested:4 (671 B), nprobe 64, k' 50, recall 0.983
- R@10 ≥ 0.99: smallest tier fp32 (6,144 B), nprobe 32, k' 100, recall 0.993

### TieredQ 1024 first stage (1,262 index bits/vector), 64 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-nested:3 | 399 | 16, 50 (0.957) | 32, 50 (0.971) | — | — | 0.976 (nprobe 64, k' 50) |
| tieredq-nested:3 | 461 | 16, 50 (0.957) | 32, 50 (0.971) | — | — | 0.976 (nprobe 64, k' 50) |
| tieredq-nested:3 | 474 | 16, 50 (0.957) | 32, 50 (0.971) | — | — | 0.976 (nprobe 64, k' 50) |
| tieredq-nested:3 | 516 | 16, 50 (0.957) | 32, 50 (0.971) | — | — | 0.976 (nprobe 64, k' 50) |
| tieredq-nested:3 | 549 | 16, 50 (0.957) | 32, 50 (0.971) | — | — | 0.976 (nprobe 64, k' 50) |
| tieredq-nested:4 | 599 | 16, 20 (0.952) | 32, 50 (0.981) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-nested:4 | 658 | 16, 20 (0.952) | 32, 50 (0.981) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-nested:4 | 671 | 16, 20 (0.952) | 32, 50 (0.981) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-nested:4 | 711 | 16, 20 (0.952) | 32, 50 (0.981) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-nested:4 | 734 | 16, 20 (0.952) | 32, 50 (0.981) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| fp32 | 6,144 | 16, 20 (0.960) | 16, 50 (0.975) | 32, 50 (0.993) | 32, 50 (0.993) | 1.000 (nprobe 64, k' 100) |

- R@10 ≥ 0.95: smallest tier tieredq-nested:3 (461 B), nprobe 16, k' 50, recall 0.957
- R@10 ≥ 0.97: smallest tier tieredq-nested:3 (461 B), nprobe 32, k' 50, recall 0.971
- R@10 ≥ 0.98: smallest tier tieredq-nested:4 (658 B), nprobe 32, k' 50, recall 0.981
- R@10 ≥ 0.99: smallest tier fp32 (6,144 B), nprobe 32, k' 50, recall 0.993

### TieredQ 1416 first stage (1,706 index bits/vector), 64 lists

| Rerank tier | Bytes/vector | R@10 ≥ 0.95: nprobe, k' (recall) | R@10 ≥ 0.97: nprobe, k' (recall) | R@10 ≥ 0.98: nprobe, k' (recall) | R@10 ≥ 0.99: nprobe, k' (recall) | Best recall |
|---|---|---|---|---|---|---|
| tieredq-nested:2 | 203 | — | — | — | — | 0.949 (nprobe 64, k' 50) |
| tieredq:2 | 389 | 32, 20 (0.951) | — | — | — | 0.957 (nprobe 64, k' 50) |
| tieredq-nested:3 | 399 | 16, 20 (0.950) | 64, 20 (0.970) | — | — | 0.973 (nprobe 64, k' 50) |
| tieredq-residual:3 | 404 | 32, 50 (0.950) | — | — | — | 0.955 (nprobe 64, k' 50) |
| tieredq-nested:3 | 461 | 16, 20 (0.950) | 64, 20 (0.970) | — | — | 0.973 (nprobe 64, k' 50) |
| tieredq-nested:3 | 474 | 16, 20 (0.950) | 64, 20 (0.970) | — | — | 0.973 (nprobe 64, k' 50) |
| tieredq-nested:3 | 516 | 16, 20 (0.950) | 64, 20 (0.970) | — | — | 0.973 (nprobe 64, k' 50) |
| tieredq-nested:3 | 549 | 16, 20 (0.950) | 64, 20 (0.970) | — | — | 0.973 (nprobe 64, k' 50) |
| tieredq:3 | 581 | 16, 20 (0.954) | 32, 50 (0.972) | — | — | 0.977 (nprobe 64, k' 50) |
| tieredq-residual:4 | 595 | 16, 20 (0.953) | 64, 20 (0.972) | — | — | 0.974 (nprobe 64, k' 50) |
| rabitq:3 | 596 | — | — | — | — | 0.949 (nprobe 64, k' 50) |
| tieredq-nested:4 | 599 | 16, 20 (0.961) | 32, 20 (0.977) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-nested:4 | 658 | 16, 20 (0.961) | 32, 20 (0.977) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-nested:4 | 671 | 16, 20 (0.961) | 32, 20 (0.977) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-nested:4 | 711 | 16, 20 (0.961) | 32, 20 (0.977) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-nested:4 | 734 | 16, 20 (0.961) | 32, 20 (0.977) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-nested:5 | 769 | 16, 20 (0.965) | 32, 20 (0.981) | 32, 20 (0.981) | 64, 50 (0.992) | 0.992 (nprobe 64, k' 50) |
| tieredq:4 | 773 | 16, 20 (0.962) | 32, 20 (0.977) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| tieredq-residual:5 | 787 | 16, 20 (0.962) | 32, 20 (0.978) | 32, 50 (0.981) | — | 0.987 (nprobe 64, k' 50) |
| rabitq:4 | 788 | 16, 50 (0.951) | 64, 50 (0.970) | — | — | 0.970 (nprobe 64, k' 50) |
| tieredq:5 | 965 | 16, 20 (0.966) | 32, 20 (0.982) | 32, 20 (0.982) | 64, 50 (0.993) | 0.993 (nprobe 64, k' 50) |
| rabitq:5 | 980 | 16, 20 (0.959) | 32, 20 (0.976) | 64, 20 (0.982) | — | 0.985 (nprobe 64, k' 50) |
| rabitq:6 | 1,172 | 16, 20 (0.965) | 32, 20 (0.981) | 32, 20 (0.981) | 64, 50 (0.991) | 0.991 (nprobe 64, k' 50) |
| rabitq:7 | 1,364 | 16, 20 (0.968) | 16, 50 (0.972) | 32, 20 (0.985) | 64, 20 (0.991) | 0.996 (nprobe 64, k' 50) |
| int8 | 1,536 | 16, 20 (0.970) | 16, 50 (0.974) | 32, 20 (0.986) | 32, 50 (0.991) | 0.997 (nprobe 64, k' 50) |
| fp16 | 3,072 | 16, 20 (0.971) | 16, 20 (0.971) | 32, 20 (0.988) | 32, 50 (0.993) | 1.000 (nprobe 64, k' 50) |
| fp32 | 6,144 | 16, 20 (0.971) | 16, 20 (0.971) | 32, 20 (0.988) | 32, 50 (0.993) | 1.000 (nprobe 64, k' 50) |

- R@10 ≥ 0.95: smallest tier tieredq:2 (389 B), nprobe 32, k' 20, recall 0.951
- R@10 ≥ 0.97: smallest tier tieredq-nested:3 (399 B), nprobe 64, k' 20, recall 0.970
- R@10 ≥ 0.98: smallest tier tieredq-nested:4 (599 B), nprobe 32, k' 50, recall 0.981
- R@10 ≥ 0.99: smallest tier tieredq-nested:5 (769 B), nprobe 64, k' 50, recall 0.992

