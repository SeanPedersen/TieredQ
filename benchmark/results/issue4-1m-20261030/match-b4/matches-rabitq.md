Bits/vector; recall@10 is mean ± std over seeds. Target = min of RaBitQ's metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | RaBitQ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | RaBitQ disk | RaBitQ RAM | RaBitQ peak | RaBitQ R@10 | Δ R@10 | Win |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| openai1m | disk | 4 | 6,304.1 | 6,202 | 6,298.4 | 6,307.4 | 6,347.0 | 0.9888 ± 0.0008 | 6,304.1 | 6,305.3 | 6,310.5 | 0.9701 ± 0.0006 | +0.0187 | yes |
| openai1m | ram | 4 | 6,305.1 | 6,173 | 6,266.4 | 6,274.9 | 6,315.2 | 0.9887 ± 0.0002 | 6,304.1 | 6,305.3 | 6,310.5 | 0.9701 ± 0.0006 | +0.0186 | yes |
| openai1m | peak | 4 | 6,310.3 | 6,139 | 6,234.4 | 6,242.9 | 6,283.2 | 0.9893 ± 0.0007 | 6,304.1 | 6,305.3 | 6,310.5 | 0.9701 ± 0.0006 | +0.0192 | yes |
