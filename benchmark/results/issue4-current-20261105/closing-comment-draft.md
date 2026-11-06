Refreshed benchmark on library c74b4e354b92d848a004e0f50199c3f7db7e0ca4, default flat fast TieredQ with fine tiers enabled.

Six primary datasets, including GloVe-1M and MS MARCO-1M real queries. All 108 disk/steady-RAM/peak-RAM matched comparisons have higher three-seed mean recall for TieredQ. Equal-recall 8-thread batch-32 throughput is 0.76–3.36x RaBitQ and 3.32–13.36x Qdrant TQ+ against each reference’s fastest evaluated budget reaching the target; the report retains losses and build tradeoffs.

Completed 42 equal-recall/0.95 checks and 168 timed configurations across four schedules. Final audit passed all cells and 48 fresh reference checks. Total expanded benchmark wall time was 11.98 hours; the under-one-hour acceptance is not met. Timing interruption and resumption are recorded.

Report: docs/benchmark-report.md
Results/provenance: benchmark/results/issue4-current-20261105/
GloVe-100K scaling evidence remains in the appendix; investigation #38 remains separate.

Draft only: do not post or close #4 without approval.
