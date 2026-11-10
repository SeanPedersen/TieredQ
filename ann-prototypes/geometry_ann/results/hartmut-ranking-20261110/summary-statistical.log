# Cell-ceiling results

Certified = leaves whose lower bound <= reference (max SQ8 distance of the 10 supplied true IDs):
the leaves an exact search using that bound could not rule out. Means over queries; p95 in parentheses.
Ranked columns scan whole leaves in bound order (ties by centroid distance) until 10/10 truth IDs are covered.
Hull = convex hull of decoded members (ceiling for any hyperplane/convex metadata); undecided hulls
hit the iteration budget. Bound ns are per leaf, measured inside a multi-threaded diagnostic.

## qwen-statistical

Leaves 8192, build 951 s, slab build 0.0 s (0.0 MB for 32 directions), 250 queries. Oracle vectors for 10/10: 8391 (p95 19869); truth clusters 4.15.

| Score | Certified leaves | Certified vectors | Undecided leaves | Pure-order vectors | Hybrid vectors | Truth rank | Bound/ref | ns/leaf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| centroid | 8192.0 (8192.0) | 8840823 (8840823) | 0.0 (0.0) | 147196 (879005) | 147196 (879005) | 20.28 | 0.000 | 0 |
| ball | 8133.0 (8155.0) | 8839771 (8840721) | 0.0 (0.0) | 311296 (1613952) | 149676 (870906) | 31.51 | 0.000 | 34 |
| evt-iso-b0 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 240540 (1073866) | 148117 (699439) | 41.64 | 0.000 | 10 |
| evt-iso-b0.5 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 220554 (999729) | 144744 (678224) | 33.03 | 0.000 | 10 |
| evt-iso-b1 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 204201 (914621) | 142554 (678177) | 27.32 | 0.000 | 9 |
| evt-iso-b1.5 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 191317 (870803) | 140634 (691863) | 23.40 | 0.000 | 9 |
| evt-iso-b2 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 186318 (867367) | 139872 (709565) | 20.95 | 0.000 | 9 |
| evt-pca8-b0.5 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 221853 (1078542) | 143462 (713744) | 32.50 | 0.000 | 38 |
| evt-pca8-b1 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 218393 (1035218) | 141412 (694521) | 27.97 | 0.000 | 38 |
| evt-pca8-b1.5 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 251689 (1223163) | 144215 (798953) | 28.46 | 0.000 | 37 |
| evt-pca8-b2 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 363625 (2131897) | 152477 (737741) | 38.24 | 0.000 | 37 |
| nearest-member | 4.2 (8.0) | 8404 (19869) | 0.0 (0.0) | 8391 (19869) | 15091 (55527) | 2.23 | 0.731 | 0 |

Mean truth coverage (recall proxy) when scanning whole leaves in score order under a vector cap:

| Score | 35000 | 88408 | 176816 | 442041 |
|---|---:|---:|---:|---:|
| centroid | 0.8832 | 0.9352 | 0.9632 | 0.9864 |
| ball | 0.8760 | 0.9280 | 0.9468 | 0.9680 |
| evt-iso-b0 | 0.8516 | 0.9160 | 0.9456 | 0.9780 |
| evt-iso-b0.5 | 0.8568 | 0.9184 | 0.9556 | 0.9840 |
| evt-iso-b1 | 0.8592 | 0.9188 | 0.9608 | 0.9848 |
| evt-iso-b1.5 | 0.8600 | 0.9208 | 0.9620 | 0.9864 |
| evt-iso-b2 | 0.8600 | 0.9204 | 0.9620 | 0.9868 |
| evt-pca8-b0.5 | 0.8532 | 0.9160 | 0.9544 | 0.9808 |
| evt-pca8-b1 | 0.8580 | 0.9200 | 0.9560 | 0.9780 |
| evt-pca8-b1.5 | 0.8544 | 0.9116 | 0.9448 | 0.9756 |
| evt-pca8-b2 | 0.8188 | 0.8820 | 0.9184 | 0.9624 |
| nearest-member | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## sift-statistical

Leaves 8192, build 23 s, slab build 0.0 s (0.0 MB for 32 directions), 250 queries. Oracle vectors for 10/10: 1408 (p95 2400); truth clusters 6.70.

| Score | Certified leaves | Certified vectors | Undecided leaves | Pure-order vectors | Hybrid vectors | Truth rank | Bound/ref | ns/leaf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| centroid | 8192.0 (8192.0) | 1000000 (1000000) | 0.0 (0.0) | 12314 (43555) | 12314 (43555) | 19.93 | 0.000 | 0 |
| ball | 4495.0 (7050.0) | 541046 (805946) | 0.0 (0.0) | 27872 (109322) | 12420 (43294) | 44.16 | 0.001 | 19 |
| evt-iso-b0 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 16437 (49654) | 12763 (46768) | 22.29 | 0.000 | 10 |
| evt-iso-b0.5 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 15947 (47751) | 12529 (44522) | 20.72 | 0.000 | 10 |
| evt-iso-b1 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 15680 (47352) | 12353 (42209) | 19.73 | 0.000 | 10 |
| evt-iso-b1.5 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 16044 (50668) | 12369 (40976) | 19.51 | 0.000 | 10 |
| evt-iso-b2 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 17334 (59013) | 12491 (43011) | 20.50 | 0.000 | 10 |
| evt-pca8-b0.5 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 15919 (50303) | 12376 (45377) | 20.82 | 0.000 | 37 |
| evt-pca8-b1 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 16533 (52999) | 12241 (43174) | 21.07 | 0.000 | 37 |
| evt-pca8-b1.5 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 20279 (75775) | 12422 (43741) | 25.32 | 0.000 | 37 |
| evt-pca8-b2 | 0.0 (0.0) | 0 (0) | 0.0 (0.0) | 33297 (124188) | 13213 (46112) | 45.25 | 0.000 | 37 |
| nearest-member | 6.8 (10.0) | 1421 (2521) | 0.0 (0.0) | 1414 (2407) | 3771 (10185) | 3.54 | 0.869 | 0 |

Mean truth coverage (recall proxy) when scanning whole leaves in score order under a vector cap:

| Score | 5000 | 10000 | 20000 | 35000 |
|---|---:|---:|---:|---:|
| centroid | 0.8388 | 0.9256 | 0.9776 | 0.9916 |
| ball | 0.8096 | 0.8836 | 0.9288 | 0.9532 |
| evt-iso-b0 | 0.8004 | 0.9004 | 0.9528 | 0.9864 |
| evt-iso-b0.5 | 0.8124 | 0.9032 | 0.9572 | 0.9872 |
| evt-iso-b1 | 0.8152 | 0.9012 | 0.9592 | 0.9868 |
| evt-iso-b1.5 | 0.8072 | 0.8984 | 0.9600 | 0.9856 |
| evt-iso-b2 | 0.7976 | 0.8896 | 0.9588 | 0.9824 |
| evt-pca8-b0.5 | 0.8024 | 0.9008 | 0.9576 | 0.9876 |
| evt-pca8-b1 | 0.7952 | 0.8912 | 0.9584 | 0.9832 |
| evt-pca8-b1.5 | 0.7516 | 0.8640 | 0.9464 | 0.9772 |
| evt-pca8-b2 | 0.5984 | 0.7484 | 0.8728 | 0.9444 |
| nearest-member | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

