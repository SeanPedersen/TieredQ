
## gemini  (n=99000, D=768, k=10, R=64)
| step | ms/query |
|---|---|
| flat fast cap 8, depth 10 | 0.657 |
| flat fast cap 8, depth 64 | 0.657 |
| IVF lists=1 fast cap 8, depth 10 | 0.880 |
| nested cap 12 stage 1 (candidates, depth 64) | 0.827 |
| nested cap 12 search (stage 1 + rescore 64) | 0.966 |

## openai100k  (n=100000, D=1536, k=10, R=64)
| step | ms/query |
|---|---|
| flat fast cap 8, depth 10 | 1.319 |
| flat fast cap 8, depth 64 | 1.340 |
| IVF lists=1 fast cap 8, depth 10 | 1.731 |
| nested cap 12 stage 1 (candidates, depth 64) | 1.586 |
| nested cap 12 search (stage 1 + rescore 64) | 2.036 |
ALL_DONE
