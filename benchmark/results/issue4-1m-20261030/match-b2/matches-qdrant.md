Bits/vector; recall@10 is mean ± std over seeds. Target = min of Qdrant TQ+'s metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | Qdrant TQ+ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | Qdrant TQ+ disk | Qdrant TQ+ RAM | Qdrant TQ+ peak | Qdrant TQ+ R@10 | Δ R@10 | Win |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| openai1m | disk | 2 | 3,136.3 | 3,038 | 3,121.0 | 3,129.1 | 3,162.0 | 0.9531 ± 0.0012 | 3,136.3 | 3,139.5 | 3,144.3 | 0.9104 ± 0.0000 | +0.0427 | yes |
| openai1m | ram | 2 | 3,138.9 | 3,038 | 3,121.0 | 3,129.1 | 3,162.0 | 0.9531 ± 0.0012 | 3,136.3 | 3,139.5 | 3,144.3 | 0.9104 ± 0.0000 | +0.0427 | yes |
| openai1m | peak | 2 | 3,143.7 | 3,005 | 3,088.8 | 3,096.9 | 3,129.3 | 0.9531 ± 0.0014 | 3,136.3 | 3,139.5 | 3,144.3 | 0.9104 ± 0.0000 | +0.0427 | yes |
