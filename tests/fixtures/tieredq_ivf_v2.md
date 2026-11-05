# Version 2 IVF fixture

`tieredq_ivf_v2.tq`: a cosine IVF index with 20 coarse lists over 600
16-dimensional rows (2 bits/dim, 8 codec centroids, no fine tiers), saved
before #34 in the version 2 payload, which stores every padded document's id
raw. `tieredq_ivf_v2.documents`: its `id_documents()` at save time, 600
little-endian uint32. Tests load the file to keep old indexes readable now that
ids are Elias-Fano lists (version 3).
