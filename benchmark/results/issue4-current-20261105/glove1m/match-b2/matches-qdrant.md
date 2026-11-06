Bits/vector; recall@10 is mean ± std over seeds. Target = min of Qdrant TQ+'s metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | Qdrant TQ+ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | Qdrant TQ+ disk | Qdrant TQ+ RAM | Qdrant TQ+ peak | Qdrant TQ+ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| glove1m | disk | 2 | 464.0 | 416 | 458.4 | 472.4 | 485.4 | 0.6750 ± 0.0001 | 464.0 | 466.9 | 471.0 | 0.6228 ± 0.0000 | +0.0522 | yes | – |
| glove1m | ram | 2 | 466.3 | 408 | 450.4 | 462.7 | 476.2 | 0.6684 ± 0.0051 | 464.0 | 466.9 | 471.0 | 0.6228 ± 0.0000 | +0.0456 | yes | – |
| glove1m | peak | 2 | 470.4 | 400 | 442.4 | 456.1 | 468.9 | 0.6613 ± 0.0034 | 464.0 | 466.9 | 471.0 | 0.6228 ± 0.0000 | +0.0385 | yes | – |
