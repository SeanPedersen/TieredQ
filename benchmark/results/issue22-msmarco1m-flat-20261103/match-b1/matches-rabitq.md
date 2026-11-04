Bits/vector; recall@10 is mean ± std over seeds. Target = min of RaBitQ's metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | RaBitQ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | RaBitQ disk | RaBitQ RAM | RaBitQ peak | RaBitQ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| msmarco1m | disk | 1 | 1,120.0 | 1,052 | 1,112.7 | 1,062.6 | 1,073.8 | 0.8627 ± 0.0020 | 1,120.0 | 1,121.6 | 1,126.2 | 0.7619 ± 0.0018 | +0.1008 | yes | +0.0821 |
| msmarco1m | ram | 1 | 1,121.6 | 1,108 | 1,169.0 | 1,119.5 | 1,130.3 | 0.8702 ± 0.0013 | 1,120.0 | 1,121.6 | 1,126.2 | 0.7619 ± 0.0018 | +0.1084 | yes | +0.0896 |
| msmarco1m | peak | 1 | 1,126.0 | 1,099 | 1,160.9 | 1,111.1 | 1,122.4 | 0.8692 ± 0.0028 | 1,120.0 | 1,121.6 | 1,126.2 | 0.7619 ± 0.0018 | +0.1074 | yes | +0.0879 |
