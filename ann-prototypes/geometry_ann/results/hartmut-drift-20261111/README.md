# Training on a shifted sample: the lists drift, the codec does not (#19 §2)

Remote root: `/home/sean/experiments/TieredQ-issue19-drift-20261111`. TieredQ 86f003d plus the heap fix and packed
lists (`../TieredQ-issue53-packedlists-20261111/.venv-packed`). Packed and padded lists return identical results
(`../hartmut-packedlists-20261111`). `drift.py` runs one model per process under the shared lock (32 threads). This is
recall only, so the host load during the run does not matter.

**Setup.** VIBE arxiv, 1,344,643 × 768, 1,000 queries; recall@10 with VIBE's 1e-3 tolerance. Flat IVF, 1.75 bits/dim,
2,048 lists. The base is cut into 100 clusters (a 100-list TieredQ IVF; `clusters.u32` on hartmut), and each seed holds
out 20 of them (19–22% of rows). Every model trains on 131,072 rows and then builds on all rows:
- `holdout`: codec and lists train on rows outside the held-out clusters;
- `random`: both train on a uniform sample;
- `codec-holdout` / `lists-holdout`: only one of them trains on the holdout sample. The first `train_lists()` trains
  the codec and the lists, and a second retrains only the lists.

Each seed's two samples are the same in every mode; `codec-holdout` reproduces `random`'s list sizes exactly. "Shifted"
queries are the 198–252 whose true nearest neighbour lies in the held-out clusters.

## Recall@10, three seeds (ranges)

| Training | nprobe 8 | 16 | 32 | 64 | largest list | rows in lists > 4× mean |
|---|---|---|---|---|---:|---:|
| random, shifted queries | 0.872–0.884 | 0.922–0.926 | 0.939–0.946 | 0.950–0.957 | 1,831–2,472 | 0 |
| codec on holdout, shifted | 0.865–0.883 | 0.916–0.920 | 0.933–0.941 | 0.946–0.953 | same as random | 0 |
| lists on holdout, shifted | 0.816–0.820 | 0.878–0.888 | 0.918–0.927 | 0.944–0.953 | 2,329–8,260 | 0–2.8% |
| holdout (both), shifted | 0.809–0.818 | 0.872–0.883 | 0.912–0.925 | 0.936–0.948 | 2,329–8,260 | 0–2.8% |
| holdout (both), all queries | 0.854–0.863 | 0.905–0.910 | 0.935–0.937 | 0.949–0.952 | | |
| random, all queries | 0.872–0.876 | 0.917–0.923 | 0.941–0.943 | 0.952–0.955 | | |

**Findings**
- **A shift in the training sample costs the shifted queries 0.05–0.07 recall at nprobe 8.** At nprobe 64 the loss
  is about 0.01, and over all queries it is 0.01–0.02 at low nprobe and under 0.005 at nprobe 64.
- **Almost all of it is the lists.** The held-out rows pile into the lists nearest them: the largest list grows up
  to 4× and up to 2.8% of rows sit in lists over 4× the mean. Nearby queries then probe too few good lists. A codec
  trained without those clusters costs 0.001–0.007 (about 0.005 at every nprobe, consistently), close to the seed
  spread.
- **Decision for inserts and drift:** keep the codec and retrain only the lists. `train_lists` on a fresh sample, or
  `set_lists` with centroids from elsewhere, then build. That is k-means and assignment, not codec training. List-size
  skew (largest list, share of rows in oversized lists) is a cheap drift signal, since the held-out runs show it and
  the random runs do not. One seed (`holdout-s0`) skewed only mildly (largest 2,329) and still lost 0.057 at nprobe 8,
  so a missing skew does not prove no drift.

Not measured: QPS (oversized lists make shifted queries scan more), retraining cost at 8.84M, and a natural shift
(new documents of a different source) instead of a held-out-cluster one.
