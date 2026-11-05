
## gemini  (n=99000, D=768, k=10, total code bits 6112; misses out of 10000; flat scan, ms/query median of 3)
| configuration | resident bytes/vector | misses | ms/query |
|---|---|---|---|
| fast cap 8 (today) | 773.0 | 68 | 0.692 |
| nested cap 10, prefix 5610 + suffix → 6072 bits, R=64 | 776.8 | 13 | 1.092 |
| nested cap 12, prefix 5585 + suffix → 6037 bits, R=64 | 772.8 | 14 | 1.015 |

## openai100k  (n=100000, D=1536, k=10, total code bits 12256; misses out of 10000; flat scan, ms/query median of 3)
| configuration | resident bytes/vector | misses | ms/query |
|---|---|---|---|
| fast cap 8 (today) | 1541.0 | 38 | 1.327 |
| nested cap 10, prefix 11419 + suffix → 12341 bits, R=64 | 1565.9 | 13 | 2.013 |
| nested cap 12, prefix 11356 + suffix → 12248 bits, R=64 | 1553.9 | 11 | 2.046 |

## gemini  (n=99000, D=768, k=100, total code bits 6112; misses out of 100000; flat scan, ms/query median of 3)
| configuration | resident bytes/vector | misses | ms/query |
|---|---|---|---|
| fast cap 8 (today) | 773.0 | 641 | 0.681 |
| nested cap 10, prefix 5610 + suffix → 6072 bits, R=150 | 776.8 | 193 | 1.249 |
| nested cap 12, prefix 5585 + suffix → 6037 bits, R=150 | 772.8 | 196 | 1.262 |

## openai100k  (n=100000, D=1536, k=100, total code bits 12256; misses out of 100000; flat scan, ms/query median of 3)
| configuration | resident bytes/vector | misses | ms/query |
|---|---|---|---|
| fast cap 8 (today) | 1541.0 | 389 | 1.339 |
| nested cap 10, prefix 11419 + suffix → 12341 bits, R=150 | 1565.9 | 131 | 2.606 |
| nested cap 12, prefix 11356 + suffix → 12248 bits, R=150 | 1553.9 | 139 | 2.600 |
ALL_DONE
