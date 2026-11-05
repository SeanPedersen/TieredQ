
## openai1m  (n=999000, D=1536, 6 bits/dim = 9216 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| seed | variant | fine prefix | add s (rows/s) | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | off | 0 | 53.1 (18,816) | 9216 | 1157 | 1157.7 | 0.9934 (66) | 6.044 | 16.769 | 0.9926 (743) | 6.094 | 16.779 |
| 0 | auto | 6 | 130.5 (7,652) | 9214 | 1157 | 1157.7 | 0.9976 (24) | 5.450 | 13.344 | 0.9968 (323) | 5.682 | 14.674 |

## msmarco1m  (n=1000000, D=1024, 6 bits/dim = 6144 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| seed | variant | fine prefix | add s (rows/s) | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | off | 0 | 33.0 (30,280) | 6144 | 773 | 773.4 | 0.9884 (116) | 3.970 | 10.917 | 0.9907 (930) | 4.028 | 11.096 |
| 0 | auto | 6 | 86.5 (11,562) | 6140 | 773 | 773.4 | 0.9970 (30) | 3.493 | 8.275 | 0.9967 (333) | 3.624 | 8.936 |
