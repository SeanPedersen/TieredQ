# Issue #19.2: training sample size and distribution drift (2026-10-31)

How much a trained TieredQ codec loses when it is trained on little data or on data that later shifts, inside the
256-list IVF with rerank, against RaBitQ (no codec training) on the same partition. OpenAI-1M (DBpedia, 1,536-d,
document queries) and MS MARCO-1M (Cohere v3, 1,024-d, real dev questions and held-out passages).
Runs: [`job_drift.sh`](../../experiments/issue19_drift/job_drift.sh) with
[`ivf_eval.cpp`](../../experiments/issue6_ivf/ivf_eval.cpp) (`TRAIN_ROWS`, `TRAIN_EXCLUDE_ROWS`, `PARTITION_FROM`,
library `TieredQIVFIndex::train_lists`); shifted splits from
[`shift_data.py`](../../experiments/issue19_drift/shift_data.py). Untimed, Hartmut CPUs 8-31. TieredQ prefixes 768 /
1024 code bits with INT8 and `tieredq-refinement:4` rerank tiers (the refinement trains on the same rows);
RaBitQ 1 bit + INT8. Recall is at nprobe 128 unless noted; "candidates" = true top 10 among the k' candidates.

| Path | Contents |
|---|---|
| `sweep-<set>/` | codec trained on TRAIN_ROWS 1K-100K random rows, partition from all rows; `rabitq1` reference |
| `shift-<set>/manifest.json` | 64 k-means clusters, the 16 held out, row and query counts |
| `shift-<set>-{heldout,seen}/` | queries whose true top 10 lie mostly in held-out clusters, or not; configs below |

## Training sample size (same distribution)

| Training rows | OpenAI-1M 1024: candidates k'20 / refine:4 k'100 | MS MARCO real | MS MARCO doc |
|---|---|---|---|
| 1,000 (0.1%) | 0.919 / 0.864 | 0.927 / 0.915 | 0.959 / 0.933 |
| 2,000 | 0.956 / 0.958 | 0.954 / 0.961 | 0.978 / 0.977 |
| 4,000 | 0.969 / 0.978 | 0.967 / 0.973 | 0.988 / 0.984 |
| 8,000 | 0.976 / 0.984 | 0.971 / 0.975 | 0.989 / 0.983 |
| 16,000 | 0.979 / 0.984 | 0.973 / 0.973 | 0.991 / 0.986 |
| 32,768 (default) | 0.978 / 0.984 | 0.977 / 0.975 | 0.991 / 0.985 |
| 100,000 (10%) | 0.979 / 0.983 | 0.976 / 0.975 | 0.991 / 0.984 |

- Recall saturates at 8-16K training rows (about 32-64 rows per codec centroid); more data buys nothing.
- 4K rows cost 0.2-1 pt, 2K rows 2-3 pt; 1K rows (4 rows per centroid) collapse. Collections below ~8K vectors (cold
  start) should train fewer centroids. The 768 prefix follows the same curve (`sweep-*/`).

## Shifted distribution (held-out topics)
`all` = codec trained on all rows, `ex` = without the held-out clusters' rows (30% of OpenAI rows, 25% of MS MARCO);
`full` = partition from all rows, `sub` = partition (and RaBitQ's list means) from the training rows. Held-out
groups have 268 / 229 / 215 queries (OpenAI / real / doc), so differences below ~0.5 pt are noise.

Queries in held-out topics, nprobe 128 (candidates k'20 | INT8 k'100 | refine:4 k'100):

| Set | TieredQ 1024 all-full | ex-full (codec drift) | ex-sub (codec + partition) | RaBitQ all-full | RaBitQ ex-sub |
|---|---|---|---|---|---|
| OpenAI-1M | 0.973 / 0.994 / 0.982 | 0.966 / 0.994 / 0.978 | 0.965 / 0.993 / 0.978 | 0.977 / 0.994 | 0.964 / 0.993 |
| MS MARCO real | 0.986 / 0.991 / 0.983 | 0.981 / 0.991 / 0.974 | 0.972 / 0.982 / 0.966 | 0.966 / 0.990 | 0.944 / 0.982 |
| MS MARCO doc | 0.992 / 0.989 / 0.986 | 0.992 / 0.989 / 0.978 | 0.992 / 0.990 / 0.979 | 0.975 / 0.989 | 0.968 / 0.990 |

- **Codec drift is small**: a codec that never saw 25% of the topics loses 0-0.7 pt first-stage recall on them
  (768: 0-0.5 pt). An INT8 rerank absorbs it fully; the nested refinement keeps 0.4-0.9 pt of it.
- **Seen-topic queries are unaffected** (within +-0.3 pt; `shift-*-seen/`).
- **The partition drifts as much or more, for both codecs**: lists trained without the new topics cost TieredQ
  another 0-0.9 pt and RaBitQ 0.7-2.2 pt first-stage recall. RaBitQ's untrained codec does not avoid retraining an
  IVF; whatever retrains the lists can retrain TieredQ's codec on the same sample.
- With both drifted, TieredQ 1024 keeps its lead on MS MARCO (0.972 vs 0.944 real, 0.992 vs 0.968 doc) and ties
  RaBitQ on OpenAI-1M (0.965 vs 0.964), where RaBitQ led by 0.4 pt when both were trained on everything.

## Recommendation
- Train on a seeded sample of 16-32K rows (at least ~32 rows per codec centroid); the library's `train_sampled`
  reads it in place.
- Retrain codec and lists together when new data forms new topics; until then, an INT8 rerank tier makes the codec's
  share of the drift invisible after rerank. A new topic holding a quarter of the data costs the codec at most
  ~0.7 pt first-stage recall on that topic.
