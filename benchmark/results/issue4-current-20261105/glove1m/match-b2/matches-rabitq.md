Bits/vector; recall@10 is mean ± std over seeds. Target = min of RaBitQ's metric over seeds; TieredQ memory is the max over seeds.

| Dataset | Matched on | RaBitQ bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM | TieredQ peak | TieredQ R@10 | RaBitQ disk | RaBitQ RAM | RaBitQ peak | RaBitQ R@10 | Δ R@10 | Win | Δ R@10 companion queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| glove1m | disk | 2 | 672.0 | 631 | 666.4 | 680.5 | 693.6 | 0.8262 ± 0.0013 | 672.0 | 673.6 | 677.0 | 0.6526 ± 0.0026 | +0.1736 | yes | – |
| glove1m | ram | 2 | 673.6 | 619 | 658.4 | 672.5 | 686.2 | 0.8213 ± 0.0031 | 672.0 | 673.6 | 677.0 | 0.6526 ± 0.0026 | +0.1687 | yes | – |
| glove1m | peak | 2 | 676.5 | 603 | 642.4 | 656.1 | 669.4 | 0.8143 ± 0.0010 | 672.0 | 673.6 | 677.0 | 0.6526 ± 0.0026 | +0.1616 | yes | – |
