
## gemini  (n=99000, D=768, 8 bits/dim, 1000 queries; cells are missed top-100 hits out of 100000)
| configuration | code bits | bytes/vector | stage-1 alone | not in top 100 | not in top 150 | not in top 200 | not in top 300 | not in top 500 | not in top 1000 | rerank R=100 | rerank R=150 | rerank R=200 | rerank R=300 | rerank R=500 | rerank R=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fast cap 8 (scan 0.66 ms/q at depth 100, 0.74 at 1000) | 6144 | 773 | 641 | 641 | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — | — |
| fast cap 8 + float32 rerank | 6144 | 3845 | — | — | — | — | — | — | — | 641 | 0 | 0 | 0 | 0 | 0 |
| cap 8 two-stage (exact-only 309; stage-1 bits 6144; wide none) | 6144 | 773 | 309 | 309 | 0 | 0 | 0 | 0 | 0 | 309 | 309 | 309 | 309 | 309 | 309 |
| cap 10 two-stage (exact-only 213; stage-1 bits 5610; wide 10b×176, 9b×182) | 6144 | 773 | 492 | 492 | 0 | 0 | 0 | 0 | 0 | 492 | 213 | 213 | 213 | 213 | 213 |
| cap 12 two-stage (exact-only 194; stage-1 bits 5585; wide 11b×40, 10b×128, 9b×183) | 6144 | 773 | 561 | 561 | 0 | 0 | 0 | 0 | 0 | 561 | 194 | 194 | 194 | 194 | 194 |

## openai100k  (n=100000, D=1536, 8 bits/dim, 1000 queries; cells are missed top-100 hits out of 100000)
| configuration | code bits | bytes/vector | stage-1 alone | not in top 100 | not in top 150 | not in top 200 | not in top 300 | not in top 500 | not in top 1000 | rerank R=100 | rerank R=150 | rerank R=200 | rerank R=300 | rerank R=500 | rerank R=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fast cap 8 (scan 1.32 ms/q at depth 100, 1.39 at 1000) | 12288 | 1541 | 389 | 389 | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — | — |
| fast cap 8 + float32 rerank | 12288 | 7685 | — | — | — | — | — | — | — | 389 | 0 | 0 | 0 | 0 | 0 |
| cap 8 two-stage (exact-only 190; stage-1 bits 12288; wide none) | 12288 | 1541 | 190 | 190 | 0 | 0 | 0 | 0 | 0 | 190 | 190 | 190 | 190 | 190 | 190 |
| cap 10 two-stage (exact-only 112; stage-1 bits 11419; wide 10b×244, 9b×381) | 12288 | 1541 | 376 | 376 | 0 | 0 | 0 | 0 | 0 | 376 | 112 | 112 | 112 | 112 | 112 |
| cap 12 two-stage (exact-only 87; stage-1 bits 11356; wide 12b×16, 11b×72, 10b×142, 9b×368) | 12288 | 1541 | 425 | 425 | 0 | 0 | 0 | 0 | 0 | 425 | 87 | 87 | 87 | 87 | 87 |

## glove  (n=100000, D=200, 8 bits/dim, 1000 queries; cells are missed top-100 hits out of 100000)
| configuration | code bits | bytes/vector | stage-1 alone | not in top 100 | not in top 150 | not in top 200 | not in top 300 | not in top 500 | not in top 1000 | rerank R=100 | rerank R=150 | rerank R=200 | rerank R=300 | rerank R=500 | rerank R=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fast cap 8 (scan 0.20 ms/q at depth 100, 0.26 at 1000) | 1600 | 205 | 1166 | 1166 | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — | — |
| fast cap 8 + float32 rerank | 1600 | 1005 | — | — | — | — | — | — | — | 1166 | 0 | 0 | 0 | 0 | 0 |
| cap 8 two-stage (exact-only 608; stage-1 bits 1600; wide none) | 1600 | 205 | 608 | 608 | 0 | 0 | 0 | 0 | 0 | 608 | 608 | 608 | 608 | 608 | 608 |
| cap 10 two-stage (exact-only 614; stage-1 bits 1596; wide 9b×4) | 1600 | 205 | 929 | 929 | 0 | 0 | 0 | 0 | 0 | 929 | 614 | 614 | 614 | 614 | 614 |
| cap 12 two-stage (exact-only 614; stage-1 bits 1596; wide 9b×4) | 1600 | 205 | 929 | 929 | 0 | 0 | 0 | 0 | 0 | 929 | 614 | 614 | 614 | 614 | 614 |
