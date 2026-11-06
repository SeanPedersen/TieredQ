Bits/vector; recall@10 is mean ± std over seeds. Target = min of Qdrant TQ+'s metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | Qdrant TQ+ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | Qdrant TQ+ disk | Qdrant TQ+ RAM | Qdrant TQ+ peak | Qdrant TQ+ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| glove1m | disk | 1 | 264.0 | 207 | 258.4 | 272.4 | 285.1 | 0.4224 ± 0.0021 | 264.0 | 266.3 | 270.5 | 0.3556 ± 0.0000 | +0.0668 | yes | – |
| glove1m | ram | 1 | 266.3 | 199 | 250.3 | 263.5 | 278.6 | 0.4151 ± 0.0018 | 264.0 | 266.3 | 270.5 | 0.3556 ± 0.0000 | +0.0595 | yes | – |
| glove1m | peak | 1 | 270.4 | 191 | 234.3 | 247.3 | 261.2 | 0.3882 ± 0.0047 | 264.0 | 266.3 | 270.5 | 0.3556 ± 0.0000 | +0.0326 | yes | – |
