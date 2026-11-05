
## gemini  (n=99000, D=768, 6 bits/dim = 4608 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| variant | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| off | 4608 | 581 | 584.1 | 0.9874 (126) | 0.315 | 0.855 | 0.9890 (1102) | 0.311 | 0.950 |
| auto | 4604 | 581 | 583.0 | 0.9947 (53) | 0.343 | 0.811 | 0.9958 (425) | 0.413 | 1.083 |
| on:10 | 4604 | 581 | 583.0 | 0.9947 (53) | 0.331 | 0.830 | 0.9958 (425) | 0.408 | 1.115 |
| on:12 | 4604 | 581 | 583.0 | 0.9947 (53) | 0.347 | 0.818 | 0.9958 (425) | 0.412 | 1.071 |

## openai100k  (n=100000, D=1536, 6 bits/dim = 9216 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| variant | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| off | 9216 | 1157 | 1163.3 | 0.9924 (76) | 0.622 | 2.057 | 0.9929 (715) | 0.626 | 2.035 |
| auto | 9212 | 1157 | 1163.0 | 0.9971 (29) | 0.734 | 1.988 | 0.9972 (279) | 0.840 | 2.556 |
| on:10 | 9212 | 1157 | 1163.0 | 0.9971 (29) | 0.727 | 1.979 | 0.9972 (279) | 0.861 | 2.554 |
| on:12 | 9212 | 1157 | 1163.0 | 0.9971 (29) | 0.731 | 1.995 | 0.9972 (279) | 0.853 | 2.471 |

## glove  (n=100000, D=200, 6 bits/dim = 1200 code bits, 1000 queries; misses out of 1000 * k; ms/query median of 3)
| variant | code bits | bytes/vector | resident bytes/vector | recall@10 (misses) | batch ms/q@10 | latency ms@10 | recall@100 (misses) | batch ms/q@100 | latency ms@100 |
|---|---|---|---|---|---|---|---|---|---|
| off | 1200 | 155 | 155.8 | 0.9664 (336) | 0.091 | 0.159 | 0.9759 (2405) | 0.098 | 0.221 |
| auto | 1200 | 155 | 155.8 | 0.9664 (336) | 0.090 | 0.171 | 0.9759 (2405) | 0.098 | 0.221 |
| on:10 | 1193 | 155 | 155.8 | 0.9702 (298) | 0.109 | 0.202 | 0.9782 (2179) | 0.180 | 0.315 |
| on:12 | 1193 | 155 | 155.8 | 0.9702 (298) | 0.117 | 0.200 | 0.9782 (2179) | 0.141 | 0.347 |

Predicted fine-tier gain per training (gemini, openai100k, glove; variants auto, on:10, on:12):
```
fine-gain prefix=6 widest=9 gain_db=8.206
fine-gain prefix=6 widest=9 gain_db=8.206
fine-gain prefix=6 widest=9 gain_db=8.206
fine-gain prefix=6 widest=10 gain_db=8.978
fine-gain prefix=6 widest=10 gain_db=8.978
fine-gain prefix=6 widest=10 gain_db=8.978
fine-gain prefix=6 widest=7 gain_db=-0.191
fine-gain prefix=6 widest=7 gain_db=-0.191
fine-gain prefix=6 widest=7 gain_db=-0.191
```
