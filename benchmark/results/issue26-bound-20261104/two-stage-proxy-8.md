
## gemini  (n=99000, D=768, 8 bits/dim, 1000 queries; cells are missed top-10 hits out of 10000)
| configuration | code bits | bytes/vector | stage-1 alone | not in top 10 | not in top 64 | not in top 100 | not in top 200 | rerank R=10 | rerank R=64 | rerank R=100 | rerank R=200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fast cap 8 (nibble scan, 0.7s) | 6144 | 773 | 68 | 68 | 0 | 0 | 0 | — | — | — | — |
| fast cap 8 + float32 rerank | 6144 | 3845 | — | — | — | — | — | 68 | 0 | 0 | 0 |
| cap 8 two-stage (exact-only 30; stage-1 bits 6144; wide none) | 6144 | 773 | 30 | 30 | 0 | 0 | 0 | 30 | 30 | 30 | 30 |
| cap 10 two-stage (exact-only 21; stage-1 bits 5610; wide 10b×176, 9b×182) | 6144 | 773 | 54 | 54 | 0 | 0 | 0 | 54 | 21 | 21 | 21 |
| cap 12 two-stage (exact-only 20; stage-1 bits 5585; wide 11b×40, 10b×128, 9b×183) | 6144 | 773 | 71 | 71 | 0 | 0 | 0 | 71 | 20 | 20 | 20 |

## openai100k  (n=100000, D=1536, 8 bits/dim, 1000 queries; cells are missed top-10 hits out of 10000)
| configuration | code bits | bytes/vector | stage-1 alone | not in top 10 | not in top 64 | not in top 100 | not in top 200 | rerank R=10 | rerank R=64 | rerank R=100 | rerank R=200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fast cap 8 (nibble scan, 1.3s) | 12288 | 1541 | 38 | 38 | 0 | 0 | 0 | — | — | — | — |
| fast cap 8 + float32 rerank | 12288 | 7685 | — | — | — | — | — | 38 | 0 | 0 | 0 |
| cap 8 two-stage (exact-only 25; stage-1 bits 12288; wide none) | 12288 | 1541 | 25 | 25 | 0 | 0 | 0 | 25 | 25 | 25 | 25 |
| cap 10 two-stage (exact-only 8; stage-1 bits 11419; wide 10b×244, 9b×381) | 12288 | 1541 | 40 | 40 | 0 | 0 | 0 | 40 | 8 | 8 | 8 |
| cap 12 two-stage (exact-only 4; stage-1 bits 11356; wide 12b×16, 11b×72, 10b×142, 9b×368) | 12288 | 1541 | 46 | 46 | 0 | 0 | 0 | 46 | 4 | 4 | 4 |

## glove  (n=100000, D=200, 8 bits/dim, 1000 queries; cells are missed top-10 hits out of 10000)
| configuration | code bits | bytes/vector | stage-1 alone | not in top 10 | not in top 64 | not in top 100 | not in top 200 | rerank R=10 | rerank R=64 | rerank R=100 | rerank R=200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fast cap 8 (nibble scan, 0.2s) | 1600 | 205 | 159 | 159 | 0 | 0 | 0 | — | — | — | — |
| fast cap 8 + float32 rerank | 1600 | 1005 | — | — | — | — | — | 159 | 0 | 0 | 0 |
| cap 8 two-stage (exact-only 84; stage-1 bits 1600; wide none) | 1600 | 205 | 84 | 84 | 0 | 0 | 0 | 84 | 84 | 84 | 84 |
| cap 10 two-stage (exact-only 80; stage-1 bits 1596; wide 9b×4) | 1600 | 205 | 136 | 136 | 0 | 0 | 0 | 136 | 80 | 80 | 80 |
| cap 12 two-stage (exact-only 80; stage-1 bits 1596; wide 9b×4) | 1600 | 205 | 136 | 136 | 0 | 0 | 0 | 136 | 80 | 80 | 80 |
