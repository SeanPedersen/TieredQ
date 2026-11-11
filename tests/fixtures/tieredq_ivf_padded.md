# Padded-list IVF fixtures (#53)

Written by `ann-prototypes/geometry_ann/results/hartmut-packedlists-20261111/make_padded_fixture.py` with TieredQ
86f003d, the last build whose lists start on 32-document boundaries, padded with copies of their last row.

- `tieredq_ivf_padded.rows`: 1,200 rows of 32 little-endian float32 around 12 cluster centers (numpy seed 53).
- `tieredq_ivf_padded.queries`: 16 queries from the same centers.
- `tieredq_ivf_padded_ip2.tq`: inner product, 2 bits/dim, 40 coarse lists, a router (8 lists per super, rerank 16,
  at least 2 probed supers); payload version 5.
- `tieredq_ivf_padded_l2x6.tq`: L2 (input-space coarse lists), 6 bits/dim, 40 coarse lists; payload version 4.
- `tieredq_ivf_padded_<name>.ids` / `.scores`: the batch search results at k = 10, nprobe 6 (16 × 10 little-endian
  uint32 / float32).

The test loads each file and requires the same ids. The scores are kept for reference only, since the scan kernel and
so the last bits differ by CPU. It then rebuilds the index in place from the rows: the loaded
lists stay fixed, so the partition and codes are identical. The packed rebuild must return the same ids and scores,
store fewer documents, save as version 6 and round-trip.
