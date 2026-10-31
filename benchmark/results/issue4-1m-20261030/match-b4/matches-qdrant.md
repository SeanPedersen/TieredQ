Bits/vector; recall@10 is mean ± std over seeds. Target = min of Qdrant TQ+'s metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | Qdrant TQ+ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | Qdrant TQ+ disk | Qdrant TQ+ RAM | Qdrant TQ+ peak | Qdrant TQ+ R@10 | Δ R@10 | Win |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| openai1m | disk | 4 | 6,208.3 | 6,112 | 6,202.4 | 6,211.9 | 6,251.2 | 0.9882 ± 0.0005 | 6,208.3 | 6,211.5 | 6,217.2 | 0.9723 ± 0.0000 | +0.0159 | yes |
| openai1m | ram | 4 | 6,210.3 | 6,080 | 6,170.4 | 6,179.8 | 6,222.6 | 0.9879 ± 0.0001 | 6,208.3 | 6,211.5 | 6,217.2 | 0.9723 ± 0.0000 | +0.0156 | yes |
| openai1m | peak | 4 | 6,215.6 | 6,048 | 6,138.4 | 6,147.7 | 6,187.1 | 0.9888 ± 0.0008 | 6,208.3 | 6,211.5 | 6,217.2 | 0.9723 ± 0.0000 | +0.0165 | yes |
