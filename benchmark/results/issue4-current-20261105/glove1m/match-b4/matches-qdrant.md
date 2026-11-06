Bits/vector; recall@10 is mean ± std over seeds. Target = min of Qdrant TQ+'s metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | Qdrant TQ+ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | Qdrant TQ+ disk | Qdrant TQ+ RAM | Qdrant TQ+ peak | Qdrant TQ+ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| glove1m | disk | 4 | 864.0 | 821 | 858.5 | 873.0 | 885.7 | 0.9039 ± 0.0021 | 864.0 | 866.9 | 871.0 | 0.8818 ± 0.0000 | +0.0221 | yes | – |
| glove1m | ram | 4 | 866.3 | 813 | 850.5 | 865.0 | 877.7 | 0.9021 ± 0.0009 | 864.0 | 866.9 | 871.0 | 0.8818 ± 0.0000 | +0.0203 | yes | – |
| glove1m | peak | 4 | 870.4 | 801 | 842.5 | 857.0 | 869.8 | 0.8994 ± 0.0020 | 864.0 | 866.9 | 871.0 | 0.8818 ± 0.0000 | +0.0176 | yes | – |
