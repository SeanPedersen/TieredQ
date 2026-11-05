
## gemini  (n=99000, D=768, 7 bits/dim = 5376 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| variant | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| off | 5376 | 677 | 680.3 | 0.9907 (93) | 0.586 | 1.135 | 0.9925 (746) | 0.588 | 1.223 |
| auto | 5367 | 677 | 680.1 | 0.9965 (35) | 0.604 | 1.083 | 0.9975 (250) | 0.669 | 1.399 |
| on:10 | 5367 | 677 | 680.1 | 0.9965 (35) | 0.599 | 1.106 | 0.9975 (250) | 0.659 | 1.402 |
| on:12 | 5367 | 677 | 680.1 | 0.9965 (35) | 0.588 | 1.104 | 0.9975 (250) | 0.662 | 1.417 |

## openai100k  (n=100000, D=1536, 7 bits/dim = 10752 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| variant | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| off | 10752 | 1349 | 1355.8 | 0.9948 (52) | 1.180 | 2.597 | 0.9954 (463) | 1.194 | 2.630 |
| auto | 10745 | 1349 | 1355.5 | 0.9982 (18) | 1.291 | 2.526 | 0.9984 (163) | 1.397 | 3.205 |
| on:10 | 10747 | 1349 | 1355.5 | 0.9977 (23) | 1.260 | 2.544 | 0.9985 (154) | 1.393 | 3.115 |
| on:12 | 10745 | 1349 | 1355.5 | 0.9982 (18) | 1.259 | 2.545 | 0.9984 (163) | 1.394 | 3.102 |

## glove  (n=100000, D=200, 7 bits/dim = 1400 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| variant | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| off | 1400 | 180 | 180.9 | 0.9780 (220) | 0.165 | 0.215 | 0.9850 (1499) | 0.172 | 0.295 |
| auto | 1400 | 180 | 180.9 | 0.9780 (220) | 0.166 | 0.247 | 0.9850 (1499) | 0.175 | 0.292 |
| on:10 | 1393 | 180 | 180.9 | 0.9828 (172) | 0.184 | 0.264 | 0.9881 (1194) | 0.214 | 0.446 |
| on:12 | 1393 | 180 | 180.9 | 0.9828 (172) | 0.182 | 0.262 | 0.9881 (1194) | 0.243 | 0.401 |

Predicted fine-tier gain per training (gemini, openai100k, glove; variants auto, on:10, on:12):
```
fine-gain prefix=7 widest=10 gain_db=8.312
fine-gain prefix=7 widest=10 gain_db=8.312
fine-gain prefix=7 widest=10 gain_db=8.312
fine-gain prefix=7 widest=11 gain_db=9.081
fine-gain prefix=7 widest=10 gain_db=9.054
fine-gain prefix=7 widest=11 gain_db=9.081
fine-gain prefix=7 widest=8 gain_db=-0.192
fine-gain prefix=7 widest=8 gain_db=-0.192
fine-gain prefix=7 widest=8 gain_db=-0.192
```
