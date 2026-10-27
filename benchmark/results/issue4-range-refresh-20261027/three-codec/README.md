# Interrupted three-codec extension

Stopped at the user's request on 27 October 2026 to investigate SAQ instead. This is an incomplete experiment, not an extension of the primary report. The original report and audited main/supplemental artifacts are unchanged.

Partial native captures and storage-selection attempts are retained. `initial-capture.log` contains the first completed Gemini TurboQuant+ capture; `job.log` records continuation and the interruption. `interrupted-selection/` retains an abandoned storage-selection attempt. No four-dataset or completed three-codec claims can be drawn from this directory. Workspaces and models remain on Hartmut beneath the original refresh workspace's `three-codec/` directory.

The new native adapter selects production cosine/L2 and orders scores appropriately; its overlay seeds only production calibration and observes allocations/timing. The seed-0 Gemini recall matches the issue-15 evidence. The extra L2 flag changes adapter shell size. All accepted captures use the new adapter binary; its hash is in metadata. Original issue-15 source/model/binary artifacts were preserved.

`report-before.md` preserves the unchanged report snapshot, verified before the interruption (110 numerical rows, 38 relative links, 120 reused RaBitQ rows). Draft report/plot scripts here have not been applied to it.
