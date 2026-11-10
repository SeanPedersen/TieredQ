# Restricted assignment, end to end

Remote root: `/home/sean/experiments/TieredQ-issue44-assignfast-20261110`. Status: queued. It labels 8.84M rows by
restricted assignment (256 supers, P=16 and P=8, from `fast_assign`) against the 32K Lloyd centroids, checks label
agreement and objective against exact assignment, and replays our index and Faiss IVF-SQ8 on the same partitions.
On a 500K-row slice: P=16 gives 99.0% exact labels at +0.1% objective and is 8.5× faster. Results go to #48.
