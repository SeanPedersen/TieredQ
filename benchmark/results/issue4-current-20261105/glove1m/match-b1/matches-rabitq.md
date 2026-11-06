Bits/vector; recall@10 is mean ± std over seeds. Target = min of RaBitQ's metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | RaBitQ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | RaBitQ disk | RaBitQ RAM | RaBitQ peak | RaBitQ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| glove1m | disk | 1 | 352.0 | 300 | 346.4 | 359.9 | 373.0 | 0.5413 ± 0.0008 | 352.0 | 353.6 | 356.7 | 0.4057 ± 0.0009 | +0.1357 | yes | – |
| glove1m | ram | 1 | 353.6 | 292 | 338.4 | 352.1 | 365.0 | 0.5349 ± 0.0058 | 352.0 | 353.6 | 356.7 | 0.4057 ± 0.0009 | +0.1292 | yes | – |
| glove1m | peak | 1 | 356.5 | 276 | 322.4 | 336.4 | 349.2 | 0.5089 ± 0.0016 | 352.0 | 353.6 | 356.7 | 0.4057 ± 0.0009 | +0.1032 | yes | – |
