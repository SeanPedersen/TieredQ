Bits/vector; recall@10 is mean ± std over seeds. Target = min of Qdrant TQ+'s metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | Qdrant TQ+ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | Qdrant TQ+ disk | Qdrant TQ+ RAM | Qdrant TQ+ peak | Qdrant TQ+ R@10 | Δ R@10 | Win |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| openai1m | disk | 1 | 1,600.3 | 1,526 | 1,598.2 | 1,608.3 | 1,614.7 | 0.8935 ± 0.0006 | 1,600.3 | 1,603.5 | 1,608.4 | 0.8270 ± 0.0000 | +0.0665 | yes |
| openai1m | ram | 1 | 1,602.3 | 1,518 | 1,590.2 | 1,600.2 | 1,606.1 | 0.8925 ± 0.0018 | 1,600.3 | 1,603.5 | 1,608.4 | 0.8270 ± 0.0000 | +0.0655 | yes |
| openai1m | peak | 1 | 1,607.6 | 1,518 | 1,590.2 | 1,600.2 | 1,606.1 | 0.8925 ± 0.0018 | 1,600.3 | 1,603.5 | 1,608.4 | 0.8270 ± 0.0000 | +0.0655 | yes |
