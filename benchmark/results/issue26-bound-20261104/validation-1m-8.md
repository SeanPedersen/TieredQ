
## openai1m  (n=999000, D=1536, 8 bits/dim = 12288 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| seed | variant | fine prefix | add s (rows/s) | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | off | 0 | 95.2 (10,491) | 12288 | 1541 | 1541.8 | 0.9961 (39) | 13.375 | 22.138 | 0.9961 (388) | 13.547 | 25.355 |
| 0 | auto | 8 | 175.5 (5,693) | 12286 | 1541 | 1541.8 | 0.9992 (8) | 12.939 | 20.732 | 0.9987 (127) | 12.826 | 22.058 |

## msmarco1m  (n=1000000, D=1024, 8 bits/dim = 8192 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| seed | variant | fine prefix | add s (rows/s) | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | off | 0 | 60.5 (16,516) | 8192 | 1029 | 1029.5 | 0.9948 (52) | 8.713 | 14.359 | 0.9953 (474) | 8.741 | 15.642 |
| 0 | auto | 8 | 114.5 (8,732) | 8189 | 1029 | 1029.5 | 0.9972 (28) | 8.312 | 13.270 | 0.9987 (133) | 8.509 | 14.543 |
