# Scan policies: boundary spill and adaptive stopping (offline)

Status: complete (exit 0); see `REPORT.md`. Spill gives no gain (±0.2 pt for 12–85% extra vectors). Patience stopping reaches equal coverage with about 1.25–2× fewer mean vectors (E-means partition up to about 2×, Lloyd about 1.25–1.5×). This is a calibration-query simulation and must be confirmed in real search.
Remote root: `/home/sean/experiments/TieredQ-issue44-policies-20261110`.

Goal: fewer scanned clusters/vectors at the same recall, using geometry and the leaf neighbourhood where it
can actually help (convex pruning bounds are ruled out by `../hartmut-cells-20261110/`).

1. **Boundary spill (hyperplane margin + neighbour graph).** A row x in leaf a with alternative leaf b
   (nearest among a's 32 nearest centroids) has relative margin (||x-c_b||² - ||x-c_a||²)/||x-c_a||². This is
   proportional to its distance to the a/b bisector hyperplane. Queries near x on b's side rank b first, so
   duplicating low-margin rows into b targets the "probed the neighbour, missed the row" failure. Caps count the
   duplicated copies, so coverage gains are net of their scan cost. Reported with the extra-vector (RAM) fraction.
2. **Patience stopping.** Scan in centroid order; stop after W consecutive leaves do not change the top-10.
   Compare mean vectors/leaves at equal coverage against fixed caps.

Simulated on the best flat 32K partitions from `../hartmut-emeans-20261110/` (E-means 4 epochs and Lloyd, loaded with
`--assignments`), using SQ8 member distances for 250 calibration queries (`cell_ceiling --policies 1`).
Patience windows 4–256 leaves (about 270 vectors per leaf). Policies that look good here go to real search.
