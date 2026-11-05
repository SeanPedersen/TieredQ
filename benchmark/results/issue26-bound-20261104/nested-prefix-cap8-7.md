
## glove: 100000 rows, D=200, 1000 queries, requested bits=1400, prefix cap=8, sample seed=None
| prefix bits/dim | cap | R | actual code bits | suffix row bytes | index+tier bytes/vector | recall@10 | misses | search s (median) |
|---|---|---|---|---|---|---|---|---|
| 6 | 8 | 64 | 1401 | 30 | 190.23 | 0.98280 | 172 | 0.1938 |
| 6 | 8 | 100 | 1401 | 30 | 190.23 | 0.98280 | 172 | 0.2092 |
| 6 | 8 | 200 | 1401 | 30 | 190.23 | 0.98280 | 172 | 0.2462 |
Payload mismatch: requested 1400, actual 1401; exclude from equal-budget claims.
| 6 | 10 | 64 | 1401 | 30 | 190.23 | 0.98280 | 172 | 0.2112 |
| 6 | 10 | 100 | 1401 | 30 | 190.23 | 0.98280 | 172 | 0.2250 |
| 6 | 10 | 200 | 1401 | 30 | 190.23 | 0.98280 | 172 | 0.2593 |
Payload mismatch: requested 1400, actual 1401; exclude from equal-budget claims.
| 6 | 12 | 64 | 1401 | 30 | 190.23 | 0.98280 | 172 | 0.2410 |
| 6 | 12 | 100 | 1401 | 30 | 190.23 | 0.98280 | 172 | 0.2608 |
| 6 | 12 | 200 | 1401 | 30 | 190.23 | 0.98280 | 172 | 0.3008 |
Payload mismatch: requested 1400, actual 1401; exclude from equal-budget claims.

## gemini: 99000 rows, D=768, 1000 queries, requested bits=5376, prefix cap=8, sample seed=None
| prefix bits/dim | cap | R | actual code bits | suffix row bytes | index+tier bytes/vector | recall@10 | misses | search s (median) |
|---|---|---|---|---|---|---|---|---|
| 6 | 8 | 64 | 5223 | 81 | 670.57 | 0.99590 | 41 | 0.8969 |
| 6 | 8 | 100 | 5223 | 81 | 670.57 | 0.99590 | 41 | 1.0050 |
| 6 | 8 | 200 | 5223 | 81 | 670.57 | 0.99590 | 41 | 1.2636 |
Payload mismatch: requested 5376, actual 5223; exclude from equal-budget claims.
| 6 | 10 | 64 | 5376 | 100 | 689.57 | 0.99680 | 32 | 0.9101 |
| 6 | 10 | 100 | 5376 | 100 | 689.57 | 0.99680 | 32 | 1.0210 |
| 6 | 10 | 200 | 5376 | 100 | 689.57 | 0.99680 | 32 | 1.2813 |
| 6 | 12 | 64 | 5376 | 100 | 689.57 | 0.99680 | 32 | 0.8959 |
| 6 | 12 | 100 | 5376 | 100 | 689.57 | 0.99680 | 32 | 0.9843 |
| 6 | 12 | 200 | 5376 | 100 | 689.57 | 0.99680 | 32 | 1.2603 |

## openai100k: 100000 rows, D=1536, 1000 queries, requested bits=10752, prefix cap=8, sample seed=None
| prefix bits/dim | cap | R | actual code bits | suffix row bytes | index+tier bytes/vector | recall@10 | misses | search s (median) |
|---|---|---|---|---|---|---|---|---|
| 6 | 8 | 64 | 10466 | 161 | 1331.28 | 0.99670 | 33 | 1.7942 |
| 6 | 8 | 100 | 10466 | 161 | 1331.28 | 0.99670 | 33 | 3.1793 |
| 6 | 8 | 200 | 10466 | 161 | 1331.28 | 0.99670 | 33 | 3.1351 |
Payload mismatch: requested 10752, actual 10466; exclude from equal-budget claims.
| 6 | 10 | 64 | 10748 | 196 | 1366.28 | 0.99800 | 20 | 1.8217 |
| 6 | 10 | 100 | 10748 | 196 | 1366.28 | 0.99800 | 20 | 2.0758 |
| 6 | 10 | 200 | 10748 | 196 | 1366.28 | 0.99800 | 20 | 2.6298 |
Payload mismatch: requested 10752, actual 10748; exclude from equal-budget claims.
| 6 | 12 | 64 | 10748 | 196 | 1366.28 | 0.99800 | 20 | 1.7514 |
| 6 | 12 | 100 | 10748 | 196 | 1366.28 | 0.99800 | 20 | 1.9816 |
| 6 | 12 | 200 | 10748 | 196 | 1366.28 | 0.99800 | 20 | 2.5570 |
Payload mismatch: requested 10752, actual 10748; exclude from equal-budget claims.
