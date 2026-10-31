Bits/vector; recall@10 is mean ± std over seeds. Target = min of RaBitQ's metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | RaBitQ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | RaBitQ disk | RaBitQ RAM | RaBitQ peak | RaBitQ R@10 | Δ R@10 | Win |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| openai1m | disk | 2 | 3,232.1 | 3,136 | 3,217.6 | 3,226.4 | 3,260.9 | 0.9575 ± 0.0015 | 3,232.1 | 3,233.3 | 3,238.4 | 0.9101 ± 0.0019 | +0.0473 | yes |
| openai1m | ram | 2 | 3,233.3 | 3,136 | 3,217.6 | 3,226.4 | 3,260.9 | 0.9575 ± 0.0015 | 3,232.1 | 3,233.3 | 3,238.4 | 0.9101 ± 0.0019 | +0.0473 | yes |
| openai1m | peak | 2 | 3,238.4 | 3,100 | 3,185.4 | 3,193.7 | 3,230.3 | 0.9565 ± 0.0007 | 3,232.1 | 3,233.3 | 3,238.4 | 0.9101 ± 0.0019 | +0.0464 | yes |
