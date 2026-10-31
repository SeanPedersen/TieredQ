Bits/vector; recall@10 is mean ± std over seeds. Target = min of RaBitQ's metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | RaBitQ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | RaBitQ disk | RaBitQ RAM | RaBitQ peak | RaBitQ R@10 | Δ R@10 | Win |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| openai1m | disk | 1 | 1,632.1 | 1,557 | 1,630.6 | 1,640.5 | 1,646.6 | 0.8955 ± 0.0022 | 1,632.1 | 1,633.3 | 1,638.6 | 0.8296 ± 0.0035 | +0.0660 | yes |
| openai1m | ram | 1 | 1,633.3 | 1,549 | 1,622.5 | 1,632.2 | 1,637.1 | 0.8942 ± 0.0015 | 1,632.1 | 1,633.3 | 1,638.6 | 0.8296 ± 0.0035 | +0.0646 | yes |
| openai1m | peak | 1 | 1,638.5 | 1,549 | 1,622.5 | 1,632.2 | 1,637.1 | 0.8942 ± 0.0015 | 1,632.1 | 1,633.3 | 1,638.6 | 0.8296 ± 0.0035 | +0.0646 | yes |
