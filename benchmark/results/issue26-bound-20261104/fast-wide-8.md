
## gemini  (n=99000, D=768, budget 6144 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| cap | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 6144 | 773 | 776.5 | 0.9932 (68) | 0.668 | 1.313 | 0.9936 (641) | 0.666 | 1.402 |
| 10 | 6144 | 773 | 776.3 | 0.9985 (15) | 0.680 | 1.278 | 0.9987 (134) | 0.775 | 1.632 |
| 12 | 6136 | 773 | 776.3 | 0.9984 (16) | 0.657 | 1.246 | 0.9986 (137) | 0.742 | 1.600 |

## openai100k  (n=100000, D=1536, budget 12288 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| cap | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 12288 | 1541 | 1548.2 | 0.9962 (38) | 1.355 | 2.731 | 0.9961 (389) | 1.336 | 2.969 |
| 10 | 12285 | 1541 | 1548.0 | 0.9984 (16) | 1.430 | 2.852 | 0.9990 (98) | 1.549 | 3.616 |
| 12 | 12287 | 1541 | 1547.9 | 0.9990 (10) | 1.419 | 2.970 | 0.9991 (92) | 1.544 | 3.670 |

## glove  (n=100000, D=200, budget 1600 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| cap | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1600 | 205 | 206.0 | 0.9841 (159) | 0.184 | 0.248 | 0.9883 (1166) | 0.191 | 0.331 |
| 10 | 1593 | 205 | 206.0 | 0.9916 (84) | 0.209 | 0.285 | 0.9935 (653) | 0.232 | 0.422 |
| 12 | 1593 | 205 | 206.0 | 0.9916 (84) | 0.199 | 0.315 | 0.9935 (653) | 0.241 | 0.427 |
ALL_DONE
