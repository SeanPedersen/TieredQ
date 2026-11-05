
## openai1m  (n=999000, D=1536, 7 bits/dim = 10752 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| seed | variant | fine prefix | add s (rows/s) | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | off | 0 | 75.1 (13,311) | 10752 | 1349 | 1349.8 | 0.9951 (49) | 11.378 | 18.296 | 0.9953 (471) | 11.393 | 19.897 |
| 0 | auto | 7 | 152.6 (6,545) | 10743 | 1349 | 1349.7 | 0.9984 (16) | 10.666 | 17.165 | 0.9982 (178) | 10.829 | 18.817 |

## msmarco1m  (n=1000000, D=1024, 7 bits/dim = 7168 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| seed | variant | fine prefix | add s (rows/s) | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | off | 0 | 47.6 (20,997) | 7168 | 901 | 901.5 | 0.9927 (73) | 7.607 | 12.191 | 0.9937 (628) | 7.708 | 13.120 |
| 0 | auto | 7 | 102.6 (9,747) | 7166 | 901 | 901.4 | 0.9974 (26) | 6.820 | 11.197 | 0.9980 (197) | 6.972 | 12.372 |
