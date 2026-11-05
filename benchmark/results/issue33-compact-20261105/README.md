# #33 compact storage: lossless headroom of the fast index (2026-11-05)

Gemini-768 (99K) and OpenAI-1536 (100K) on hartmut, ip, default public-index models
(`benchmark/experiments/issue33_compact`, `job.sh`). All values are bits per vector. The estimates are adaptive
KT code lengths, so model cost is included; they are lower bounds a real rANS/Elias-Fano coder gets within about 1%
of. `results.csv` holds every cell.

| | Gemini 1 / 2 / 4 / 8 b/dim | OpenAI 1 / 2 / 4 / 8 b/dim |
|---|---:|---:|
| Fast RAM = disk now | 820 / 1583 / 3128 / 6213 | 1592 / 3127 / 6217 / 12388 |
| Code bits → order-0 entropy-coded | −4.4 / −5.0 / −5.5 / −3.5% | −4.3 / −4.9 / −5.7 / −4.0% |
| Same, conditioned on centroid id | worse than order-0 (model cost > gain) | same |
| Metadata 40 → | 31 | 29–31 |
| Layout padding beyond code + metadata | 4 / 7 / 20 / 37 | 8 / 15 / 38 / 61 |
| **Total lossless saving** | **5.7 / 5.9 / 6.3 / 4.2%** | **5.3 / 5.6 / 6.4 / 4.5%** |
| IVF ids, uint32 → Elias-Fano (256 lists) | 32 → 10.3 | 32 → 10.3 |

The value of those bits, from the #4 precheck curves (`issue4-precheck-20261105/results.csv`, Gemini fast): about
+0.2 R@10 points per 1% more bits at about 1 bit/dim, +0.14 at about 2, and +0.01 at about 4. So ~5% buys about
+1 point at 1–2 bits/dim and nothing measurable at 4 or more.

Conclusions:
- A compact RAM mode is not worth it. It would buy about 5% for a decode in the scan loop, and the same RAM costs
  ≤1 R@10 point (1–2 b/dim) or about 0 (≥4 b/dim) by training with about 5% fewer `code_bits`.
- An entropy-coded disk format gains about 5% of disk, worth about +1 R@10 point at equal disk at 1–2 b/dim only,
  and adds a decode at load. It is low priority.
- Centroid-context coding does not help: at 100K rows, the cost of 256 times more models outweighs any per-centroid difference.
- The real free wins sit outside the codes, and none of them costs scan speed:
  - IVF ids: about 22 bits per vector, decoded only for the returned top-k. That is 2.8% at 1 b/dim on Gemini.
  - The centroid byte in IVF `lists==0`: 8 bits.
  - The 8-bit fine-tier layout padding: 37–61 bits per vector.

## Follow-up: where the compressible bits sit in the flat fast index (`results-fine.csv`)

These cells use the default fine tiers at 4/6/8 bits/dim, plus exact at 8. The scan reads the prefix, and the fine side
rows are read only when rescoring candidates.

| | Gemini fine 4 / 6 / 8, exact 8 | OpenAI fine 4 / 6 / 8, exact 8 |
|---|---:|---:|
| Fine bits, share of RAM | 2.4 / 12.1 / 9.0 / 9.4% | 2.4 / 10.0 / 7.5 / 7.8% |
| Fine bits saved by entropy coding (order-0, or conditioned on the prefix) | 0.0% | 0.0% |
| Scanned prefix saved by entropy coding, share of RAM | 5.4 / 5.0 / 3.6 / 4.5% | 5.7 / 5.4 / 4.0 / 4.8% |
| Row-stride padding (bits/vec) | 4 / 4 / 8 / 8 | 5 / 4 / 1 / 8 |
| Storage beyond rows × stride (bits/vec) | 16 / 19 / 29 / 3.5 | 33 / 53 / 60 / 5.5 |

- The fine bits are incompressible. They are the low bits inside one Lloyd-Max cell and are effectively uniform, even
  given the prefix. So compressing only the rescoring side would save nothing.
- All the lossless headroom (4–6%) is in the prefix, which the scan reads for every vector. In RAM it costs QPS.
- The storage beyond rows × stride is a fixed per-index cost, not per vector:
  - about 12 bits/vec (OpenAI, 100K) is the padding of the last 256-document block;
  - the rest is most likely the fine model's prefix-view state (inferred, not itemised), about 0.6 MB on OpenAI, which amortises to about 5 bits/vec at 1M.
