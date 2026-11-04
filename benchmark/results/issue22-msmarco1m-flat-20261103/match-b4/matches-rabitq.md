Bits/vector; recall@10 is mean ± std over seeds. Target = min of RaBitQ's metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | RaBitQ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | RaBitQ disk | RaBitQ RAM | RaBitQ peak | RaBitQ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| msmarco1m | disk | 4 | 4,256.0 | 4,187 | 4,249.3 | 4,191.2 | 4,205.2 | 0.9829 ± 0.0001 | 4,256.0 | 4,257.6 | 4,262.2 | 0.9571 ± 0.0008 | +0.0258 | yes | +0.0214 |
| msmarco1m | ram | 4 | 4,257.6 | 4,252 | 4,313.4 | 4,255.2 | 4,269.3 | 0.9835 ± 0.0006 | 4,256.0 | 4,257.6 | 4,262.2 | 0.9571 ± 0.0008 | +0.0263 | yes | +0.0221 |
| msmarco1m | peak | 4 | 4,262.0 | 4,245 | 4,305.4 | 4,247.7 | 4,262.0 | 0.9839 ± 0.0014 | 4,256.0 | 4,257.6 | 4,262.2 | 0.9571 ± 0.0008 | +0.0268 | yes | +0.0223 |
