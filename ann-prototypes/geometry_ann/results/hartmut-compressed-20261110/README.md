# TieredQ-compressed centroid routing (library default scan width)

Status: complete (exit 0); see `REPORT.md`. Remote root: `/home/sean/experiments/TieredQ-issue44-compressed-20261110`.
Superseded by `hartmut-compressed4-20261110`, which adds the 4-bit scan cap.

Setup: flat Lloyd 32K labels. Leaf centroids are coded by TieredQ (L2, fast mode, library default `max_scan_bits`) at 2 or
4 bits/dim. `search_fast` ranks 4,096 candidates, and the first R ∈ {512, 2048} are re-ranked with exact centroid distances.
Scanning uses fixed caps or patience. FP32 flat controls run on the same build.

Result: held-out recall and vectors equal exact FP32 routing (already at R=512). Routing costs 8.6 ms at 2 bits and 10.4 ms at 4 bits
(FP32 flat: 14.1 ms), with tables of 31 and 41.5 MiB including the codec. A laptop profile traced the cost to the single-query scan
kernel: at the default width (8 bits for L2), wide tiers on the top PCA ranks push every query onto the generic scan.
