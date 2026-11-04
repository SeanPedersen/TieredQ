Bits/vector; recall@10 is mean ± std over seeds. Target = min of Qdrant TQ+'s metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | Qdrant TQ+ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | Qdrant TQ+ disk | Qdrant TQ+ RAM | Qdrant TQ+ peak | Qdrant TQ+ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| msmarco1m | disk | 1 | 1,088.2 | 1,018 | 1,080.6 | 1,031.1 | 1,042.2 | 0.8546 ± 0.0006 | 1,088.2 | 1,090.4 | 1,096.0 | 0.7554 ± 0.0000 | +0.0992 | yes | +0.0798 |
| msmarco1m | ram | 1 | 1,090.3 | 1,080 | 1,136.8 | 1,086.4 | 1,097.4 | 0.8637 ± 0.0003 | 1,088.2 | 1,090.4 | 1,096.0 | 0.7554 ± 0.0000 | +0.1083 | yes | +0.0868 |
| msmarco1m | peak | 1 | 1,095.4 | 1,069 | 1,128.8 | 1,078.5 | 1,089.7 | 0.8647 ± 0.0041 | 1,088.2 | 1,090.4 | 1,096.0 | 0.7554 ± 0.0000 | +0.1093 | yes | +0.0879 |
