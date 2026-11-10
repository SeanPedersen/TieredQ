# Version 3 L2 IVF fixture

`tieredq_ivf_l2_v3.tq`: an L2 IVF index with 20 coarse lists over 600
16-dimensional rows (2 bits/dim, 8 codec centroids, no fine tiers), saved in
the version 3 payload, before L2 coarse lists moved to input space (version 4),
so its coarse centroids are in the codec's lifted model space.
`tieredq_ivf_l2_v3.queries`: 8 queries, 16 little-endian float32 each.
`tieredq_ivf_l2_v3.hits`: the ids that version returned for each query at
k = 10, nprobe = 3, 80 little-endian uint32. Tests load the file and require the
same hits, so indexes saved before version 4 keep their lists and probe order.
