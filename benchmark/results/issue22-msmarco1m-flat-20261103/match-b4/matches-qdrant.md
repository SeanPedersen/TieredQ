Bits/vector; recall@10 is mean ± std over seeds. Target = min of Qdrant TQ+'s metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | Qdrant TQ+ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | Qdrant TQ+ disk | Qdrant TQ+ RAM | Qdrant TQ+ peak | Qdrant TQ+ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| msmarco1m | disk | 4 | 4,160.2 | 4,093 | 4,153.0 | 4,096.0 | 4,110.1 | 0.9824 ± 0.0006 | 4,160.2 | 4,162.9 | 4,167.5 | 0.9615 ± 0.0000 | +0.0209 | yes | +0.0190 |
| msmarco1m | ram | 4 | 4,162.3 | 4,159 | 4,217.2 | 4,160.1 | 4,174.0 | 0.9833 ± 0.0009 | 4,160.2 | 4,162.9 | 4,167.5 | 0.9615 ± 0.0000 | +0.0218 | yes | +0.0191 |
| msmarco1m | peak | 4 | 4,167.4 | 4,151 | 4,209.2 | 4,151.2 | 4,165.8 | 0.9824 ± 0.0012 | 4,160.2 | 4,162.9 | 4,167.5 | 0.9615 ± 0.0000 | +0.0209 | yes | +0.0178 |
