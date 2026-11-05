
## gemini  (n=99000, D=768, 8 bits/dim = 6144 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| variant | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| off | 6144 | 773 | 776.5 | 0.9932 (68) | 0.657 | 1.344 | 0.9936 (641) | 0.673 | 1.415 |
| auto | 6136 | 773 | 776.3 | 0.9984 (16) | 0.675 | 1.262 | 0.9986 (137) | 0.750 | 1.571 |
| on:10 | 6144 | 773 | 776.3 | 0.9985 (15) | 0.680 | 1.269 | 0.9987 (134) | 0.749 | 1.616 |
| on:12 | 6136 | 773 | 776.3 | 0.9984 (16) | 0.676 | 1.296 | 0.9986 (137) | 0.753 | 1.627 |

## openai100k  (n=100000, D=1536, 8 bits/dim = 12288 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| variant | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| off | 12288 | 1541 | 1548.2 | 0.9962 (38) | 1.335 | 2.839 | 0.9961 (389) | 1.340 | 3.058 |
| auto | 12287 | 1541 | 1547.9 | 0.9990 (10) | 1.431 | 2.948 | 0.9991 (92) | 1.547 | 3.483 |
| on:10 | 12285 | 1541 | 1548.0 | 0.9984 (16) | 1.424 | 2.952 | 0.9990 (98) | 1.551 | 3.520 |
| on:12 | 12287 | 1541 | 1547.9 | 0.9990 (10) | 1.426 | 2.876 | 0.9991 (92) | 1.561 | 3.537 |

## glove  (n=100000, D=200, 8 bits/dim = 1600 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| variant | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| off | 1600 | 205 | 206.0 | 0.9841 (159) | 0.184 | 0.257 | 0.9883 (1166) | 0.201 | 0.333 |
| auto | 1600 | 205 | 206.0 | 0.9841 (159) | 0.185 | 0.249 | 0.9883 (1166) | 0.202 | 0.327 |
| on:10 | 1593 | 205 | 206.0 | 0.9916 (84) | 0.200 | 0.285 | 0.9935 (653) | 0.238 | 0.424 |
| on:12 | 1593 | 205 | 206.0 | 0.9916 (84) | 0.210 | 0.287 | 0.9935 (653) | 0.250 | 0.424 |

Predicted fine-tier gain per training (gemini, openai100k, glove; variants auto, on:10, on:12):
```
fine-gain prefix=8 widest=11 gain_db=8.406
fine-gain prefix=8 widest=10 gain_db=8.268
fine-gain prefix=8 widest=11 gain_db=8.406
fine-gain prefix=8 widest=12 gain_db=9.167
fine-gain prefix=8 widest=10 gain_db=8.517
fine-gain prefix=8 widest=12 gain_db=9.167
fine-gain prefix=8 widest=9 gain_db=-0.192
fine-gain prefix=8 widest=9 gain_db=-0.192
fine-gain prefix=8 widest=9 gain_db=-0.192
```
