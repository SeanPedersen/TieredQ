Bits/vector; recall@10 is mean ± std over seeds. Target = min of RaBitQ's metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | RaBitQ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | RaBitQ disk | RaBitQ RAM | RaBitQ peak | RaBitQ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| msmarco1m | disk | 2 | 2,208.0 | 2,149 | 2,203.7 | 2,150.8 | 2,162.6 | 0.9483 ± 0.0016 | 2,208.0 | 2,209.6 | 2,214.1 | 0.8727 ± 0.0010 | +0.0756 | yes | +0.0598 |
| msmarco1m | ram | 2 | 2,209.6 | 2,204 | 2,259.8 | 2,207.4 | 2,219.2 | 0.9496 ± 0.0007 | 2,208.0 | 2,209.6 | 2,214.1 | 0.8727 ± 0.0010 | +0.0769 | yes | +0.0625 |
| msmarco1m | peak | 2 | 2,214.0 | 2,196 | 2,251.8 | 2,199.3 | 2,211.2 | 0.9509 ± 0.0002 | 2,208.0 | 2,209.6 | 2,214.1 | 0.8727 ± 0.0010 | +0.0782 | yes | +0.0634 |
