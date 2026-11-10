# Cell-ceiling results

Certified = leaves whose lower bound <= reference (max SQ8 distance of the 10 supplied true IDs):
the leaves an exact search using that bound could not rule out. Means over queries; p95 in parentheses.
Ranked columns scan whole leaves in bound order (ties by centroid distance) until 10/10 truth IDs are covered.
Hull = convex hull of decoded members (ceiling for any hyperplane/convex metadata); undecided hulls
hit the iteration budget. Bound ns are per leaf, measured inside a multi-threaded diagnostic.

## qwen-cells

Leaves 8192, build 951 s, slab build 19.6 s (72.4 MB for 32 directions), 50 queries. Oracle vectors for 10/10: 8640 (p95 22531); truth clusters 4.28.

| Score | Certified leaves | Certified vectors | Undecided leaves | Pure-order vectors | Hybrid vectors | Truth rank | Bound/ref | ns/leaf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| centroid | 8192.0 (8192.0) | 8840823 (8840823) | 0.0 (0.0) | 130127 (682907) | 130127 (682907) | 18.27 | 0.000 | 0 |
| ball | 8140.9 (8155.0) | 8840662 (8840718) | 0.0 (0.0) | 365119 (2364492) | 132348 (676040) | 35.80 | 0.000 | 46 |
| knn8-single | 8139.8 (8155.0) | 8840657 (8840718) | 0.0 (0.0) | 911338 (4112680) | 127003 (641055) | 100.27 | 0.001 | 482 |
| knn8-joint4 | 8139.3 (8155.0) | 8840653 (8840718) | 0.0 (0.0) | 909850 (4080260) | 126859 (641055) | 99.98 | 0.001 | 910 |
| knn8-joint | 8138.8 (8155.0) | 8840652 (8840718) | 0.0 (0.0) | 909793 (4080260) | 126859 (641055) | 99.97 | 0.001 | 1853 |
| knn32-single | 8137.0 (8155.0) | 8840642 (8840718) | 0.0 (0.0) | 1247241 (4659539) | 143145 (669108) | 130.03 | 0.002 | 1366 |
| knn32-joint4 | 8135.3 (8155.0) | 8840627 (8840718) | 0.0 (0.0) | 1241179 (4471677) | 142896 (667623) | 128.00 | 0.002 | 3384 |
| knn32-joint | 8131.5 (8155.0) | 8840589 (8840718) | 0.0 (0.0) | 1239486 (4455866) | 142816 (667623) | 127.73 | 0.002 | 12205 |
| hull | 888.3 (4648.0) | 1357938 (6575721) | 16.2 (62.0) | — | — | 0.00 | 0.002 | 1972044 |
| nearest-member | 4.3 (8.0) | 8640 (22531) | 0.0 (0.0) | 8640 (22531) | 13640 (36045) | 2.28 | 0.718 | 0 |

## sift-cells

Leaves 8192, build 24 s, slab build 0.5 s (72.4 MB for 32 directions), 50 queries. Oracle vectors for 10/10: 1475 (p95 2173); truth clusters 6.94.

| Score | Certified leaves | Certified vectors | Undecided leaves | Pure-order vectors | Hybrid vectors | Truth rank | Bound/ref | ns/leaf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| centroid | 8192.0 (8192.0) | 1000000 (1000000) | 0.0 (0.0) | 14582 (40901) | 14582 (40901) | 22.36 | 0.000 | 0 |
| ball | 4819.2 (7122.0) | 569427 (812588) | 0.0 (0.0) | 34451 (112036) | 14651 (41366) | 56.09 | 0.001 | 21 |
| knn8-single | 4700.6 (7113.0) | 558900 (812588) | 0.0 (0.0) | 52804 (137231) | 14846 (44814) | 78.60 | 0.010 | 369 |
| knn8-joint4 | 4652.8 (7103.0) | 554760 (811190) | 0.0 (0.0) | 50564 (135916) | 14935 (45055) | 75.61 | 0.010 | 813 |
| knn8-joint | 4651.5 (7102.0) | 554694 (811190) | 0.0 (0.0) | 50568 (135916) | 14934 (45055) | 75.61 | 0.010 | 1624 |
| knn32-single | 4489.2 (7078.0) | 538808 (811483) | 0.0 (0.0) | 31239 (80700) | 14180 (36588) | 45.46 | 0.020 | 1074 |
| knn32-joint4 | 4301.8 (7012.0) | 520591 (807942) | 0.0 (0.0) | 27030 (66396) | 14025 (37160) | 40.28 | 0.022 | 3147 |
| knn32-joint | 4283.2 (6984.0) | 519270 (807933) | 0.0 (0.0) | 27004 (66404) | 14026 (37160) | 40.25 | 0.022 | 12641 |
| hull | 163.3 (349.0) | 28706 (58003) | 1.9 (4.0) | — | — | 0.00 | 0.022 | 7798 |
| nearest-member | 7.0 (9.0) | 1493 (2186) | 0.0 (0.0) | 1484 (2186) | 4339 (10003) | 3.63 | 0.887 | 0 |

