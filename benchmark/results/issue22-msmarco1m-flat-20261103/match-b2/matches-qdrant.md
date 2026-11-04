Bits/vector; recall@10 is mean ± std over seeds. Target = min of Qdrant TQ+'s metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | Qdrant TQ+ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | Qdrant TQ+ disk | Qdrant TQ+ RAM | Qdrant TQ+ peak | Qdrant TQ+ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| msmarco1m | disk | 2 | 2,112.2 | 2,055 | 2,107.5 | 2,055.3 | 2,067.1 | 0.9443 ± 0.0010 | 2,112.2 | 2,114.9 | 2,120.7 | 0.8721 ± 0.0000 | +0.0722 | yes | +0.0546 |
| msmarco1m | ram | 2 | 2,114.3 | 2,111 | 2,163.6 | 2,111.3 | 2,123.3 | 0.9467 ± 0.0015 | 2,112.2 | 2,114.9 | 2,120.7 | 0.8721 ± 0.0000 | +0.0746 | yes | +0.0577 |
| msmarco1m | peak | 2 | 2,119.4 | 2,100 | 2,155.6 | 2,103.7 | 2,115.4 | 0.9463 ± 0.0011 | 2,112.2 | 2,114.9 | 2,120.7 | 0.8721 ± 0.0000 | +0.0742 | yes | +0.0565 |
