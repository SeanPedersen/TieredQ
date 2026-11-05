
## gemini  (n=99000, D=768, budget 6144 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| cap | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 6144 | 773 | 776.5 | 0.9932 (68) | 0.678 | 1.301 | 0.9936 (641) | 0.658 | 1.371 |
| 10 | 6144 | 773 | 776.3 | 0.9985 (15) | 0.667 | 2.453 | 0.9987 (134) | 0.713 | 3.229 |
| 12 | 6136 | 773 | 776.3 | 0.9984 (16) | 0.662 | 2.320 | 0.9986 (138) | 0.712 | 3.233 |

## openai100k  (n=100000, D=1536, budget 12288 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| cap | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 12288 | 1541 | 1548.2 | 0.9962 (38) | 1.327 | 2.884 | 0.9961 (389) | 1.346 | 3.088 |
| 10 | 12285 | 1541 | 1548.0 | 0.9984 (16) | 1.362 | 5.086 | 0.9990 (98) | 1.447 | 7.148 |
| 12 | 12287 | 1541 | 1547.9 | 0.9990 (10) | 1.356 | 5.132 | 0.9991 (92) | 1.441 | 7.201 |

## glove  (n=100000, D=200, budget 1600 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| cap | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1600 | 205 | 206.0 | 0.9841 (159) | 0.184 | 0.244 | 0.9883 (1166) | 0.198 | 0.302 |
| 10 | 1593 | 205 | 206.0 | 0.9916 (84) | 0.196 | 0.498 | 0.9935 (653) | 0.219 | 0.780 |
| 12 | 1593 | 205 | 206.0 | 0.9916 (84) | 0.202 | 0.558 | 0.9935 (653) | 0.219 | 0.781 |
ALL_DONE
