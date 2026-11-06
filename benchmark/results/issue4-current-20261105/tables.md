# Complete current-run tables

## Memory targets

| Dataset | Reference | Reference nominal bits/dim | Disk TQ max / ref min (bits/vector; TQ code bits) | Steady RAM TQ max / ref min | Peak RAM TQ max / ref min |
| --- | --- | --- | --- | --- | --- |
| SIFT1M | RaBitQ | 1 | 201.5 / 224.0 (160 code bits) | 223.8 / 225.6 (160 code bits) | 212.4 / 228.4 (134 code bits) |
| SIFT1M | RaBitQ | 2 | 393.6 / 416.0 (352 code bits) | 416.0 / 417.6 (352 code bits) | 406.4 / 420.6 (315 code bits) |
| SIFT1M | RaBitQ | 4 | 649.6 / 672.0 (608 code bits) | 672.1 / 673.4 (608 code bits) | 663.7 / 676.3 (576 code bits) |
| SIFT1M | Qdrant TQ+ | 1 | 201.5 / 224.0 (160 code bits) | 223.8 / 227.5 (160 code bits) | 212.4 / 231.5 (134 code bits) |
| SIFT1M | Qdrant TQ+ | 2 | 329.5 / 352.0 (285 code bits) | 352.0 / 355.6 (285 code bits) | 342.2 / 359.5 (253 code bits) |
| SIFT1M | Qdrant TQ+ | 4 | 585.6 / 608.0 (538 code bits) | 606.5 / 611.5 (538 code bits) | 599.3 / 615.5 (509 code bits) |
| GloVe-200 1M | RaBitQ | 1 | 346.4 / 352.0 (300 code bits) | 352.1 / 353.6 (292 code bits) | 349.2 / 356.5 (276 code bits) |
| GloVe-200 1M | RaBitQ | 2 | 666.4 / 672.0 (631 code bits) | 672.5 / 673.6 (619 code bits) | 669.4 / 676.5 (603 code bits) |
| GloVe-200 1M | RaBitQ | 4 | 1178.5 / 1184.0 (1139 code bits) | 1185.3 / 1185.6 (1131 code bits) | 1182.4 / 1188.8 (1114 code bits) |
| GloVe-200 1M | Qdrant TQ+ | 1 | 258.4 / 264.0 (207 code bits) | 263.5 / 266.3 (199 code bits) | 261.2 / 270.4 (191 code bits) |
| GloVe-200 1M | Qdrant TQ+ | 2 | 458.4 / 464.0 (416 code bits) | 462.7 / 466.3 (408 code bits) | 468.9 / 470.4 (400 code bits) |
| GloVe-200 1M | Qdrant TQ+ | 4 | 858.5 / 864.0 (821 code bits) | 865.0 / 866.3 (813 code bits) | 869.8 / 870.4 (801 code bits) |
| OpenAI-1536 100K | RaBitQ | 1 | 1631.2 / 1632.6 (1304 code bits) | 1626.3 / 1643.0 (1205 code bits) | 1685.9 / 1696.1 (1125 code bits) |
| OpenAI-1536 100K | RaBitQ | 2 | 3228.6 / 3232.6 (2797 code bits) | 3238.8 / 3243.0 (2709 code bits) | 3287.3 / 3296.5 (2568 code bits) |
| OpenAI-1536 100K | RaBitQ | 4 | 6302.2 / 6304.6 (5752 code bits) | 6308.8 / 6315.0 (5620 code bits) | 6345.2 / 6368.1 (5267 code bits) |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | 1597.2 / 1602.6 (1268 code bits) | 1626.3 / 1640.7 (1205 code bits) | 1685.9 / 1701.0 (1125 code bits) |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | 3134.6 / 3138.6 (2709 code bits) | 3164.4 / 3176.5 (2639 code bits) | 3235.8 / 3236.5 (2535 code bits) |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | 6205.6 / 6210.6 (5653 code bits) | 6243.9 / 6248.5 (5570 code bits) | 6304.9 / 6308.8 (5221 code bits) |
| Gemini-768 99K | RaBitQ | 1 | 858.6 / 864.4 (706 code bits) | 859.6 / 869.5 (630 code bits) | 901.3 / 902.9 (523 code bits) |
| Gemini-768 99K | RaBitQ | 2 | 1692.0 / 1696.4 (1513 code bits) | 1698.0 / 1701.6 (1421 code bits) | 1728.1 / 1734.4 (1302 code bits) |
| Gemini-768 99K | RaBitQ | 4 | 3226.0 / 3232.4 (3029 code bits) | 3234.4 / 3237.7 (2917 code bits) | 3242.0 / 3270.8 (2648 code bits) |
| Gemini-768 99K | Qdrant TQ+ | 1 | 825.9 / 833.3 (673 code bits) | 859.6 / 864.2 (630 code bits) | 901.3 / 909.9 (523 code bits) |
| Gemini-768 99K | Qdrant TQ+ | 2 | 1601.1 / 1601.3 (1429 code bits) | 1626.5 / 1632.4 (1352 code bits) | 1669.8 / 1675.8 (1250 code bits) |
| Gemini-768 99K | Qdrant TQ+ | 4 | 3136.8 / 3137.3 (2936 code bits) | 3161.0 / 3167.6 (2862 code bits) | 3184.1 / 3211.3 (2600 code bits) |
| OpenAI-1536 1M | RaBitQ | 1 | 1630.6 / 1632.1 (1557 code bits) | 1626.9 / 1633.3 (1540 code bits) | 1631.2 / 1638.5 (1532 code bits) |
| OpenAI-1536 1M | RaBitQ | 2 | 3226.4 / 3232.1 (3137 code bits) | 3227.0 / 3233.3 (3129 code bits) | 3231.7 / 3238.4 (3108 code bits) |
| OpenAI-1536 1M | RaBitQ | 4 | 6299.6 / 6304.1 (6208 code bits) | 6301.5 / 6305.1 (6197 code bits) | 6306.0 / 6310.3 (6153 code bits) |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | 1598.3 / 1600.3 (1524 code bits) | 1595.1 / 1602.3 (1507 code bits) | 1600.2 / 1607.6 (1499 code bits) |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | 3129.6 / 3136.3 (3048 code bits) | 3134.5 / 3138.9 (3040 code bits) | 3137.9 / 3143.7 (3024 code bits) |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | 6203.6 / 6208.3 (6116 code bits) | 6203.8 / 6210.3 (6091 code bits) | 6205.5 / 6215.6 (6059 code bits) |
| MS MARCO-1M real | RaBitQ | 1 | 1112.7 / 1120.0 (1052 code bits) | 1120.9 / 1121.6 (1108 code bits) | 1123.9 / 1126.0 (1101 code bits) |
| MS MARCO-1M real | RaBitQ | 2 | 2203.9 / 2208.0 (2137 code bits) | 2153.2 / 2209.6 (2145 code bits) | 2212.8 / 2214.0 (2189 code bits) |
| MS MARCO-1M real | RaBitQ | 4 | 4249.1 / 4256.0 (4180 code bits) | 4245.3 / 4257.6 (4239 code bits) | 4261.2 / 4262.0 (4232 code bits) |
| MS MARCO-1M real | Qdrant TQ+ | 1 | 1080.6 / 1088.2 (1018 code bits) | 1087.4 / 1090.3 (1079 code bits) | 1093.1 / 1095.4 (1069 code bits) |
| MS MARCO-1M real | Qdrant TQ+ | 2 | 2107.5 / 2112.2 (2054 code bits) | 2112.0 / 2114.3 (2106 code bits) | 2117.3 / 2119.4 (2099 code bits) |
| MS MARCO-1M real | Qdrant TQ+ | 4 | 4152.8 / 4160.2 (4087 code bits) | 4156.7 / 4162.3 (4150 code bits) | 4158.4 / 4167.4 (4133 code bits) |

## Speed at every equal-memory target

| Dataset | Reference | bits/dim | Match metric | Code bits | Threads | Batch | TieredQ QPS | Reference QPS | QPS ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT1M | RaBitQ | 1 | disk | 160 | 1 | 1 | 124.4 | 264.1 | 0.47 |
| SIFT1M | RaBitQ | 1 | disk | 160 | 1 | 32 | 478.5 | 266.8 | 1.79 |
| SIFT1M | RaBitQ | 1 | disk | 160 | 8 | 1 | 765.1 | 267.4 | 2.86 |
| SIFT1M | RaBitQ | 1 | disk | 160 | 8 | 32 | 2898.5 | 1904.7 | 1.52 |
| SIFT1M | RaBitQ | 1 | ram | 160 | 1 | 1 | 124.4 | 264.1 | 0.47 |
| SIFT1M | RaBitQ | 1 | ram | 160 | 1 | 32 | 478.5 | 266.8 | 1.79 |
| SIFT1M | RaBitQ | 1 | ram | 160 | 8 | 1 | 765.1 | 267.4 | 2.86 |
| SIFT1M | RaBitQ | 1 | ram | 160 | 8 | 32 | 2898.5 | 1904.7 | 1.52 |
| SIFT1M | RaBitQ | 1 | peak | 134 | 1 | 1 | 339.4 | 264.1 | 1.29 |
| SIFT1M | RaBitQ | 1 | peak | 134 | 1 | 32 | 449.0 | 266.8 | 1.68 |
| SIFT1M | RaBitQ | 1 | peak | 134 | 8 | 1 | 2391.5 | 267.4 | 8.94 |
| SIFT1M | RaBitQ | 1 | peak | 134 | 8 | 32 | 3257.8 | 1904.7 | 1.71 |
| SIFT1M | RaBitQ | 2 | disk | 352 | 1 | 1 | 69.6 | 260.5 | 0.27 |
| SIFT1M | RaBitQ | 2 | disk | 352 | 1 | 32 | 311.1 | 259.2 | 1.20 |
| SIFT1M | RaBitQ | 2 | disk | 352 | 8 | 1 | 448.4 | 257.3 | 1.74 |
| SIFT1M | RaBitQ | 2 | disk | 352 | 8 | 32 | 2089.5 | 2060.0 | 1.01 |
| SIFT1M | RaBitQ | 2 | ram | 352 | 1 | 1 | 69.6 | 260.5 | 0.27 |
| SIFT1M | RaBitQ | 2 | ram | 352 | 1 | 32 | 311.1 | 259.2 | 1.20 |
| SIFT1M | RaBitQ | 2 | ram | 352 | 8 | 1 | 448.4 | 257.3 | 1.74 |
| SIFT1M | RaBitQ | 2 | ram | 352 | 8 | 32 | 2089.5 | 2060.0 | 1.01 |
| SIFT1M | RaBitQ | 2 | peak | 315 | 1 | 1 | 78.9 | 260.5 | 0.30 |
| SIFT1M | RaBitQ | 2 | peak | 315 | 1 | 32 | 325.1 | 259.2 | 1.25 |
| SIFT1M | RaBitQ | 2 | peak | 315 | 8 | 1 | 507.3 | 257.3 | 1.97 |
| SIFT1M | RaBitQ | 2 | peak | 315 | 8 | 32 | 2154.3 | 2060.0 | 1.05 |
| SIFT1M | RaBitQ | 4 | disk | 608 | 1 | 1 | 45.5 | 239.5 | 0.19 |
| SIFT1M | RaBitQ | 4 | disk | 608 | 1 | 32 | 255.4 | 239.1 | 1.07 |
| SIFT1M | RaBitQ | 4 | disk | 608 | 8 | 1 | 278.8 | 237.1 | 1.18 |
| SIFT1M | RaBitQ | 4 | disk | 608 | 8 | 32 | 1703.4 | 1778.0 | 0.96 |
| SIFT1M | RaBitQ | 4 | ram | 608 | 1 | 1 | 45.5 | 239.5 | 0.19 |
| SIFT1M | RaBitQ | 4 | ram | 608 | 1 | 32 | 255.4 | 239.1 | 1.07 |
| SIFT1M | RaBitQ | 4 | ram | 608 | 8 | 1 | 278.8 | 237.1 | 1.18 |
| SIFT1M | RaBitQ | 4 | ram | 608 | 8 | 32 | 1703.4 | 1778.0 | 0.96 |
| SIFT1M | RaBitQ | 4 | peak | 576 | 1 | 1 | 52.0 | 239.5 | 0.22 |
| SIFT1M | RaBitQ | 4 | peak | 576 | 1 | 32 | 262.0 | 239.1 | 1.10 |
| SIFT1M | RaBitQ | 4 | peak | 576 | 8 | 1 | 301.4 | 237.1 | 1.27 |
| SIFT1M | RaBitQ | 4 | peak | 576 | 8 | 32 | 1700.5 | 1778.0 | 0.96 |
| SIFT1M | Qdrant TQ+ | 1 | disk | 160 | 1 | 1 | 124.4 | 36.0 | 3.46 |
| SIFT1M | Qdrant TQ+ | 1 | disk | 160 | 1 | 32 | 478.5 | 35.9 | 13.32 |
| SIFT1M | Qdrant TQ+ | 1 | disk | 160 | 8 | 1 | 765.1 | 36.3 | 21.06 |
| SIFT1M | Qdrant TQ+ | 1 | disk | 160 | 8 | 32 | 2898.5 | 269.6 | 10.75 |
| SIFT1M | Qdrant TQ+ | 1 | ram | 160 | 1 | 1 | 124.4 | 36.0 | 3.46 |
| SIFT1M | Qdrant TQ+ | 1 | ram | 160 | 1 | 32 | 478.5 | 35.9 | 13.32 |
| SIFT1M | Qdrant TQ+ | 1 | ram | 160 | 8 | 1 | 765.1 | 36.3 | 21.06 |
| SIFT1M | Qdrant TQ+ | 1 | ram | 160 | 8 | 32 | 2898.5 | 269.6 | 10.75 |
| SIFT1M | Qdrant TQ+ | 1 | peak | 134 | 1 | 1 | 339.4 | 36.0 | 9.44 |
| SIFT1M | Qdrant TQ+ | 1 | peak | 134 | 1 | 32 | 449.0 | 35.9 | 12.49 |
| SIFT1M | Qdrant TQ+ | 1 | peak | 134 | 8 | 1 | 2391.5 | 36.3 | 65.84 |
| SIFT1M | Qdrant TQ+ | 1 | peak | 134 | 8 | 32 | 3257.8 | 269.6 | 12.08 |
| SIFT1M | Qdrant TQ+ | 2 | disk | 285 | 1 | 1 | 89.7 | 57.1 | 1.57 |
| SIFT1M | Qdrant TQ+ | 2 | disk | 285 | 1 | 32 | 340.8 | 57.4 | 5.93 |
| SIFT1M | Qdrant TQ+ | 2 | disk | 285 | 8 | 1 | 551.2 | 56.7 | 9.72 |
| SIFT1M | Qdrant TQ+ | 2 | disk | 285 | 8 | 32 | 2246.9 | 441.8 | 5.09 |
| SIFT1M | Qdrant TQ+ | 2 | ram | 285 | 1 | 1 | 89.7 | 57.1 | 1.57 |
| SIFT1M | Qdrant TQ+ | 2 | ram | 285 | 1 | 32 | 340.8 | 57.4 | 5.93 |
| SIFT1M | Qdrant TQ+ | 2 | ram | 285 | 8 | 1 | 551.2 | 56.7 | 9.72 |
| SIFT1M | Qdrant TQ+ | 2 | ram | 285 | 8 | 32 | 2246.9 | 441.8 | 5.09 |
| SIFT1M | Qdrant TQ+ | 2 | peak | 253 | 1 | 1 | 91.3 | 57.1 | 1.60 |
| SIFT1M | Qdrant TQ+ | 2 | peak | 253 | 1 | 32 | 354.7 | 57.4 | 6.18 |
| SIFT1M | Qdrant TQ+ | 2 | peak | 253 | 8 | 1 | 555.4 | 56.7 | 9.79 |
| SIFT1M | Qdrant TQ+ | 2 | peak | 253 | 8 | 32 | 2252.7 | 441.8 | 5.10 |
| SIFT1M | Qdrant TQ+ | 4 | disk | 538 | 1 | 1 | 55.0 | 54.2 | 1.01 |
| SIFT1M | Qdrant TQ+ | 4 | disk | 538 | 1 | 32 | 261.9 | 54.3 | 4.83 |
| SIFT1M | Qdrant TQ+ | 4 | disk | 538 | 8 | 1 | 311.0 | 54.2 | 5.74 |
| SIFT1M | Qdrant TQ+ | 4 | disk | 538 | 8 | 32 | 1746.5 | 428.7 | 4.07 |
| SIFT1M | Qdrant TQ+ | 4 | ram | 538 | 1 | 1 | 55.0 | 54.2 | 1.01 |
| SIFT1M | Qdrant TQ+ | 4 | ram | 538 | 1 | 32 | 261.9 | 54.3 | 4.83 |
| SIFT1M | Qdrant TQ+ | 4 | ram | 538 | 8 | 1 | 311.0 | 54.2 | 5.74 |
| SIFT1M | Qdrant TQ+ | 4 | ram | 538 | 8 | 32 | 1746.5 | 428.7 | 4.07 |
| SIFT1M | Qdrant TQ+ | 4 | peak | 509 | 1 | 1 | 56.2 | 54.2 | 1.04 |
| SIFT1M | Qdrant TQ+ | 4 | peak | 509 | 1 | 32 | 261.9 | 54.3 | 4.83 |
| SIFT1M | Qdrant TQ+ | 4 | peak | 509 | 8 | 1 | 316.1 | 54.2 | 5.83 |
| SIFT1M | Qdrant TQ+ | 4 | peak | 509 | 8 | 32 | 1769.2 | 428.7 | 4.13 |
| GloVe-200 1M | RaBitQ | 1 | disk | 300 | 1 | 1 | 204.5 | 175.3 | 1.17 |
| GloVe-200 1M | RaBitQ | 1 | disk | 300 | 1 | 32 | 334.6 | 175.4 | 1.91 |
| GloVe-200 1M | RaBitQ | 1 | disk | 300 | 8 | 1 | 1331.0 | 174.9 | 7.61 |
| GloVe-200 1M | RaBitQ | 1 | disk | 300 | 8 | 32 | 2460.9 | 1249.2 | 1.97 |
| GloVe-200 1M | RaBitQ | 1 | ram | 292 | 1 | 1 | 213.5 | 175.3 | 1.22 |
| GloVe-200 1M | RaBitQ | 1 | ram | 292 | 1 | 32 | 340.6 | 175.4 | 1.94 |
| GloVe-200 1M | RaBitQ | 1 | ram | 292 | 8 | 1 | 1375.1 | 174.9 | 7.86 |
| GloVe-200 1M | RaBitQ | 1 | ram | 292 | 8 | 32 | 2477.9 | 1249.2 | 1.98 |
| GloVe-200 1M | RaBitQ | 1 | peak | 276 | 1 | 1 | 232.1 | 175.3 | 1.32 |
| GloVe-200 1M | RaBitQ | 1 | peak | 276 | 1 | 32 | 350.2 | 175.4 | 2.00 |
| GloVe-200 1M | RaBitQ | 1 | peak | 276 | 8 | 1 | 1440.2 | 174.9 | 8.23 |
| GloVe-200 1M | RaBitQ | 1 | peak | 276 | 8 | 32 | 2591.7 | 1249.2 | 2.07 |
| GloVe-200 1M | RaBitQ | 2 | disk | 631 | 1 | 1 | 125.4 | 177.3 | 0.71 |
| GloVe-200 1M | RaBitQ | 2 | disk | 631 | 1 | 32 | 229.5 | 176.8 | 1.30 |
| GloVe-200 1M | RaBitQ | 2 | disk | 631 | 8 | 1 | 828.9 | 175.9 | 4.71 |
| GloVe-200 1M | RaBitQ | 2 | disk | 631 | 8 | 32 | 1717.4 | 1382.1 | 1.24 |
| GloVe-200 1M | RaBitQ | 2 | ram | 619 | 1 | 1 | 125.5 | 177.3 | 0.71 |
| GloVe-200 1M | RaBitQ | 2 | ram | 619 | 1 | 32 | 230.0 | 176.8 | 1.30 |
| GloVe-200 1M | RaBitQ | 2 | ram | 619 | 8 | 1 | 795.0 | 175.9 | 4.52 |
| GloVe-200 1M | RaBitQ | 2 | ram | 619 | 8 | 32 | 1724.8 | 1382.1 | 1.25 |
| GloVe-200 1M | RaBitQ | 2 | peak | 603 | 1 | 1 | 129.5 | 177.3 | 0.73 |
| GloVe-200 1M | RaBitQ | 2 | peak | 603 | 1 | 32 | 229.9 | 176.8 | 1.30 |
| GloVe-200 1M | RaBitQ | 2 | peak | 603 | 8 | 1 | 812.5 | 175.9 | 4.62 |
| GloVe-200 1M | RaBitQ | 2 | peak | 603 | 8 | 32 | 1704.5 | 1382.1 | 1.23 |
| GloVe-200 1M | RaBitQ | 4 | disk | 1139 | 1 | 1 | 75.5 | 171.3 | 0.44 |
| GloVe-200 1M | RaBitQ | 4 | disk | 1139 | 1 | 32 | 143.9 | 171.7 | 0.84 |
| GloVe-200 1M | RaBitQ | 4 | disk | 1139 | 8 | 1 | 482.6 | 171.1 | 2.82 |
| GloVe-200 1M | RaBitQ | 4 | disk | 1139 | 8 | 32 | 1105.7 | 1353.8 | 0.82 |
| GloVe-200 1M | RaBitQ | 4 | ram | 1131 | 1 | 1 | 76.3 | 171.3 | 0.45 |
| GloVe-200 1M | RaBitQ | 4 | ram | 1131 | 1 | 32 | 145.2 | 171.7 | 0.85 |
| GloVe-200 1M | RaBitQ | 4 | ram | 1131 | 8 | 1 | 496.8 | 171.1 | 2.90 |
| GloVe-200 1M | RaBitQ | 4 | ram | 1131 | 8 | 32 | 1113.5 | 1353.8 | 0.82 |
| GloVe-200 1M | RaBitQ | 4 | peak | 1114 | 1 | 1 | 77.7 | 171.3 | 0.45 |
| GloVe-200 1M | RaBitQ | 4 | peak | 1114 | 1 | 32 | 147.0 | 171.7 | 0.86 |
| GloVe-200 1M | RaBitQ | 4 | peak | 1114 | 8 | 1 | 502.4 | 171.1 | 2.94 |
| GloVe-200 1M | RaBitQ | 4 | peak | 1114 | 8 | 32 | 1125.9 | 1353.8 | 0.83 |
| GloVe-200 1M | Qdrant TQ+ | 1 | disk | 207 | 1 | 1 | 256.4 | 35.8 | 7.17 |
| GloVe-200 1M | Qdrant TQ+ | 1 | disk | 207 | 1 | 32 | 389.3 | 36.0 | 10.82 |
| GloVe-200 1M | Qdrant TQ+ | 1 | disk | 207 | 8 | 1 | 1755.1 | 35.8 | 49.01 |
| GloVe-200 1M | Qdrant TQ+ | 1 | disk | 207 | 8 | 32 | 2880.9 | 266.4 | 10.82 |
| GloVe-200 1M | Qdrant TQ+ | 1 | ram | 199 | 1 | 1 | 259.2 | 35.8 | 7.25 |
| GloVe-200 1M | Qdrant TQ+ | 1 | ram | 199 | 1 | 32 | 394.8 | 36.0 | 10.97 |
| GloVe-200 1M | Qdrant TQ+ | 1 | ram | 199 | 8 | 1 | 1728.6 | 35.8 | 48.27 |
| GloVe-200 1M | Qdrant TQ+ | 1 | ram | 199 | 8 | 32 | 2851.8 | 266.4 | 10.71 |
| GloVe-200 1M | Qdrant TQ+ | 1 | peak | 191 | 1 | 1 | 266.0 | 35.8 | 7.44 |
| GloVe-200 1M | Qdrant TQ+ | 1 | peak | 191 | 1 | 32 | 404.7 | 36.0 | 11.25 |
| GloVe-200 1M | Qdrant TQ+ | 1 | peak | 191 | 8 | 1 | 1852.4 | 35.8 | 51.72 |
| GloVe-200 1M | Qdrant TQ+ | 1 | peak | 191 | 8 | 32 | 2968.0 | 266.4 | 11.14 |
| GloVe-200 1M | Qdrant TQ+ | 2 | disk | 416 | 1 | 1 | 182.4 | 32.8 | 5.56 |
| GloVe-200 1M | Qdrant TQ+ | 2 | disk | 416 | 1 | 32 | 287.7 | 32.9 | 8.74 |
| GloVe-200 1M | Qdrant TQ+ | 2 | disk | 416 | 8 | 1 | 1147.5 | 32.8 | 34.99 |
| GloVe-200 1M | Qdrant TQ+ | 2 | disk | 416 | 8 | 32 | 2156.2 | 255.9 | 8.43 |
| GloVe-200 1M | Qdrant TQ+ | 2 | ram | 408 | 1 | 1 | 187.2 | 32.8 | 5.70 |
| GloVe-200 1M | Qdrant TQ+ | 2 | ram | 408 | 1 | 32 | 290.6 | 32.9 | 8.83 |
| GloVe-200 1M | Qdrant TQ+ | 2 | ram | 408 | 8 | 1 | 1153.8 | 32.8 | 35.18 |
| GloVe-200 1M | Qdrant TQ+ | 2 | ram | 408 | 8 | 32 | 2161.8 | 255.9 | 8.45 |
| GloVe-200 1M | Qdrant TQ+ | 2 | peak | 400 | 1 | 1 | 186.9 | 32.8 | 5.70 |
| GloVe-200 1M | Qdrant TQ+ | 2 | peak | 400 | 1 | 32 | 291.4 | 32.9 | 8.86 |
| GloVe-200 1M | Qdrant TQ+ | 2 | peak | 400 | 8 | 1 | 1148.9 | 32.8 | 35.03 |
| GloVe-200 1M | Qdrant TQ+ | 2 | peak | 400 | 8 | 32 | 2131.6 | 255.9 | 8.33 |
| GloVe-200 1M | Qdrant TQ+ | 4 | disk | 821 | 1 | 1 | 99.9 | 32.0 | 3.13 |
| GloVe-200 1M | Qdrant TQ+ | 4 | disk | 821 | 1 | 32 | 186.1 | 32.1 | 5.80 |
| GloVe-200 1M | Qdrant TQ+ | 4 | disk | 821 | 8 | 1 | 651.4 | 32.1 | 20.31 |
| GloVe-200 1M | Qdrant TQ+ | 4 | disk | 821 | 8 | 32 | 1411.3 | 248.8 | 5.67 |
| GloVe-200 1M | Qdrant TQ+ | 4 | ram | 813 | 1 | 1 | 101.1 | 32.0 | 3.16 |
| GloVe-200 1M | Qdrant TQ+ | 4 | ram | 813 | 1 | 32 | 187.1 | 32.1 | 5.83 |
| GloVe-200 1M | Qdrant TQ+ | 4 | ram | 813 | 8 | 1 | 651.1 | 32.1 | 20.30 |
| GloVe-200 1M | Qdrant TQ+ | 4 | ram | 813 | 8 | 32 | 1427.4 | 248.8 | 5.74 |
| GloVe-200 1M | Qdrant TQ+ | 4 | peak | 801 | 1 | 1 | 103.6 | 32.0 | 3.24 |
| GloVe-200 1M | Qdrant TQ+ | 4 | peak | 801 | 1 | 32 | 189.4 | 32.1 | 5.91 |
| GloVe-200 1M | Qdrant TQ+ | 4 | peak | 801 | 8 | 1 | 648.0 | 32.1 | 20.21 |
| GloVe-200 1M | Qdrant TQ+ | 4 | peak | 801 | 8 | 32 | 1425.5 | 248.8 | 5.73 |
| OpenAI-1536 100K | RaBitQ | 1 | disk | 1304 | 1 | 1 | 883.4 | 594.0 | 1.49 |
| OpenAI-1536 100K | RaBitQ | 1 | disk | 1304 | 1 | 32 | 1176.2 | 599.2 | 1.96 |
| OpenAI-1536 100K | RaBitQ | 1 | disk | 1304 | 8 | 1 | 3169.0 | 593.9 | 5.34 |
| OpenAI-1536 100K | RaBitQ | 1 | disk | 1304 | 8 | 32 | 7901.9 | 4159.1 | 1.90 |
| OpenAI-1536 100K | RaBitQ | 1 | ram | 1205 | 1 | 1 | 875.3 | 594.0 | 1.47 |
| OpenAI-1536 100K | RaBitQ | 1 | ram | 1205 | 1 | 32 | 1252.4 | 599.2 | 2.09 |
| OpenAI-1536 100K | RaBitQ | 1 | ram | 1205 | 8 | 1 | 3202.0 | 593.9 | 5.39 |
| OpenAI-1536 100K | RaBitQ | 1 | ram | 1205 | 8 | 32 | 9002.7 | 4159.1 | 2.16 |
| OpenAI-1536 100K | RaBitQ | 1 | peak | 1125 | 1 | 1 | 987.8 | 594.0 | 1.66 |
| OpenAI-1536 100K | RaBitQ | 1 | peak | 1125 | 1 | 32 | 1311.0 | 599.2 | 2.19 |
| OpenAI-1536 100K | RaBitQ | 1 | peak | 1125 | 8 | 1 | 3294.2 | 593.9 | 5.55 |
| OpenAI-1536 100K | RaBitQ | 1 | peak | 1125 | 8 | 32 | 9469.2 | 4159.1 | 2.28 |
| OpenAI-1536 100K | RaBitQ | 2 | disk | 2797 | 1 | 1 | 385.7 | 635.0 | 0.61 |
| OpenAI-1536 100K | RaBitQ | 2 | disk | 2797 | 1 | 32 | 641.5 | 639.6 | 1.00 |
| OpenAI-1536 100K | RaBitQ | 2 | disk | 2797 | 8 | 1 | 1376.0 | 636.9 | 2.16 |
| OpenAI-1536 100K | RaBitQ | 2 | disk | 2797 | 8 | 32 | 4719.5 | 4560.2 | 1.03 |
| OpenAI-1536 100K | RaBitQ | 2 | ram | 2709 | 1 | 1 | 404.0 | 635.0 | 0.64 |
| OpenAI-1536 100K | RaBitQ | 2 | ram | 2709 | 1 | 32 | 655.0 | 639.6 | 1.02 |
| OpenAI-1536 100K | RaBitQ | 2 | ram | 2709 | 8 | 1 | 1400.3 | 636.9 | 2.20 |
| OpenAI-1536 100K | RaBitQ | 2 | ram | 2709 | 8 | 32 | 4701.3 | 4560.2 | 1.03 |
| OpenAI-1536 100K | RaBitQ | 2 | peak | 2568 | 1 | 1 | 414.8 | 635.0 | 0.65 |
| OpenAI-1536 100K | RaBitQ | 2 | peak | 2568 | 1 | 32 | 680.3 | 639.6 | 1.06 |
| OpenAI-1536 100K | RaBitQ | 2 | peak | 2568 | 8 | 1 | 1430.6 | 636.9 | 2.25 |
| OpenAI-1536 100K | RaBitQ | 2 | peak | 2568 | 8 | 32 | 4896.2 | 4560.2 | 1.07 |
| OpenAI-1536 100K | RaBitQ | 4 | disk | 5752 | 1 | 1 | 168.4 | 530.4 | 0.32 |
| OpenAI-1536 100K | RaBitQ | 4 | disk | 5752 | 1 | 32 | 299.6 | 539.7 | 0.56 |
| OpenAI-1536 100K | RaBitQ | 4 | disk | 5752 | 8 | 1 | 548.9 | 537.0 | 1.02 |
| OpenAI-1536 100K | RaBitQ | 4 | disk | 5752 | 8 | 32 | 1985.6 | 3716.7 | 0.53 |
| OpenAI-1536 100K | RaBitQ | 4 | ram | 5620 | 1 | 1 | 178.4 | 530.4 | 0.34 |
| OpenAI-1536 100K | RaBitQ | 4 | ram | 5620 | 1 | 32 | 317.8 | 539.7 | 0.59 |
| OpenAI-1536 100K | RaBitQ | 4 | ram | 5620 | 8 | 1 | 566.5 | 537.0 | 1.05 |
| OpenAI-1536 100K | RaBitQ | 4 | ram | 5620 | 8 | 32 | 2044.0 | 3716.7 | 0.55 |
| OpenAI-1536 100K | RaBitQ | 4 | peak | 5267 | 1 | 1 | 190.0 | 530.4 | 0.36 |
| OpenAI-1536 100K | RaBitQ | 4 | peak | 5267 | 1 | 32 | 333.9 | 539.7 | 0.62 |
| OpenAI-1536 100K | RaBitQ | 4 | peak | 5267 | 8 | 1 | 592.3 | 537.0 | 1.10 |
| OpenAI-1536 100K | RaBitQ | 4 | peak | 5267 | 8 | 32 | 2113.2 | 3716.7 | 0.57 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | disk | 1268 | 1 | 1 | 945.6 | 134.3 | 7.04 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | disk | 1268 | 1 | 32 | 1249.0 | 128.4 | 9.73 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | disk | 1268 | 8 | 1 | 3101.7 | 128.0 | 24.23 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | disk | 1268 | 8 | 32 | 8923.2 | 1049.1 | 8.51 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | ram | 1205 | 1 | 1 | 875.3 | 134.3 | 6.52 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | ram | 1205 | 1 | 32 | 1252.4 | 128.4 | 9.75 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | ram | 1205 | 8 | 1 | 3202.0 | 128.0 | 25.01 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | ram | 1205 | 8 | 32 | 9002.7 | 1049.1 | 8.58 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | peak | 1125 | 1 | 1 | 987.8 | 134.3 | 7.36 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | peak | 1125 | 1 | 32 | 1311.0 | 128.4 | 10.21 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | peak | 1125 | 8 | 1 | 3294.2 | 128.0 | 25.73 |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | peak | 1125 | 8 | 32 | 9469.2 | 1049.1 | 9.03 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | disk | 2709 | 1 | 1 | 404.0 | 118.8 | 3.40 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | disk | 2709 | 1 | 32 | 655.0 | 117.7 | 5.57 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | disk | 2709 | 8 | 1 | 1400.3 | 117.5 | 11.92 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | disk | 2709 | 8 | 32 | 4701.3 | 952.5 | 4.94 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | ram | 2639 | 1 | 1 | 411.4 | 118.8 | 3.46 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | ram | 2639 | 1 | 32 | 673.5 | 117.7 | 5.72 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | ram | 2639 | 8 | 1 | 1408.5 | 117.5 | 11.99 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | ram | 2639 | 8 | 32 | 4907.8 | 952.5 | 5.15 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | peak | 2535 | 1 | 1 | 428.1 | 118.8 | 3.60 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | peak | 2535 | 1 | 32 | 696.6 | 117.7 | 5.92 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | peak | 2535 | 8 | 1 | 1403.7 | 117.5 | 11.94 |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | peak | 2535 | 8 | 32 | 5078.9 | 952.5 | 5.33 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | disk | 5653 | 1 | 1 | 179.5 | 101.8 | 1.76 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | disk | 5653 | 1 | 32 | 315.9 | 101.2 | 3.12 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | disk | 5653 | 8 | 1 | 571.8 | 101.7 | 5.62 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | disk | 5653 | 8 | 32 | 2034.0 | 814.8 | 2.50 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | ram | 5570 | 1 | 1 | 182.1 | 101.8 | 1.79 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | ram | 5570 | 1 | 32 | 323.0 | 101.2 | 3.19 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | ram | 5570 | 8 | 1 | 574.9 | 101.7 | 5.65 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | ram | 5570 | 8 | 32 | 2076.0 | 814.8 | 2.55 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | peak | 5221 | 1 | 1 | 192.1 | 101.8 | 1.89 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | peak | 5221 | 1 | 32 | 339.9 | 101.2 | 3.36 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | peak | 5221 | 8 | 1 | 598.6 | 101.7 | 5.89 |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | peak | 5221 | 8 | 32 | 2193.8 | 814.8 | 2.69 |
| Gemini-768 99K | RaBitQ | 1 | disk | 706 | 1 | 1 | 1596.3 | 1154.7 | 1.38 |
| Gemini-768 99K | RaBitQ | 1 | disk | 706 | 1 | 32 | 2024.9 | 1163.5 | 1.74 |
| Gemini-768 99K | RaBitQ | 1 | disk | 706 | 8 | 1 | 6226.7 | 1141.0 | 5.46 |
| Gemini-768 99K | RaBitQ | 1 | disk | 706 | 8 | 32 | 14173.1 | 8248.5 | 1.72 |
| Gemini-768 99K | RaBitQ | 1 | ram | 630 | 1 | 1 | 1710.6 | 1154.7 | 1.48 |
| Gemini-768 99K | RaBitQ | 1 | ram | 630 | 1 | 32 | 2241.6 | 1163.5 | 1.93 |
| Gemini-768 99K | RaBitQ | 1 | ram | 630 | 8 | 1 | 6400.6 | 1141.0 | 5.61 |
| Gemini-768 99K | RaBitQ | 1 | ram | 630 | 8 | 32 | 15683.3 | 8248.5 | 1.90 |
| Gemini-768 99K | RaBitQ | 1 | peak | 523 | 1 | 1 | 1889.8 | 1154.7 | 1.64 |
| Gemini-768 99K | RaBitQ | 1 | peak | 523 | 1 | 32 | 2460.3 | 1163.5 | 2.11 |
| Gemini-768 99K | RaBitQ | 1 | peak | 523 | 8 | 1 | 7109.4 | 1141.0 | 6.23 |
| Gemini-768 99K | RaBitQ | 1 | peak | 523 | 8 | 32 | 16959.7 | 8248.5 | 2.06 |
| Gemini-768 99K | RaBitQ | 2 | disk | 1513 | 1 | 1 | 866.9 | 1203.0 | 0.72 |
| Gemini-768 99K | RaBitQ | 2 | disk | 1513 | 1 | 32 | 1139.5 | 1219.2 | 0.93 |
| Gemini-768 99K | RaBitQ | 2 | disk | 1513 | 8 | 1 | 3856.2 | 1216.2 | 3.17 |
| Gemini-768 99K | RaBitQ | 2 | disk | 1513 | 8 | 32 | 7978.9 | 8843.1 | 0.90 |
| Gemini-768 99K | RaBitQ | 2 | ram | 1421 | 1 | 1 | 911.4 | 1203.0 | 0.76 |
| Gemini-768 99K | RaBitQ | 2 | ram | 1421 | 1 | 32 | 1182.2 | 1219.2 | 0.97 |
| Gemini-768 99K | RaBitQ | 2 | ram | 1421 | 8 | 1 | 4058.5 | 1216.2 | 3.34 |
| Gemini-768 99K | RaBitQ | 2 | ram | 1421 | 8 | 32 | 8357.6 | 8843.1 | 0.95 |
| Gemini-768 99K | RaBitQ | 2 | peak | 1302 | 1 | 1 | 931.9 | 1203.0 | 0.77 |
| Gemini-768 99K | RaBitQ | 2 | peak | 1302 | 1 | 32 | 1275.3 | 1219.2 | 1.05 |
| Gemini-768 99K | RaBitQ | 2 | peak | 1302 | 8 | 1 | 4222.7 | 1216.2 | 3.47 |
| Gemini-768 99K | RaBitQ | 2 | peak | 1302 | 8 | 32 | 9122.9 | 8843.1 | 1.03 |
| Gemini-768 99K | RaBitQ | 4 | disk | 3029 | 1 | 1 | 358.7 | 1171.9 | 0.31 |
| Gemini-768 99K | RaBitQ | 4 | disk | 3029 | 1 | 32 | 593.3 | 1181.3 | 0.50 |
| Gemini-768 99K | RaBitQ | 4 | disk | 3029 | 8 | 1 | 1331.0 | 1164.9 | 1.14 |
| Gemini-768 99K | RaBitQ | 4 | disk | 3029 | 8 | 32 | 3697.4 | 8199.4 | 0.45 |
| Gemini-768 99K | RaBitQ | 4 | ram | 2917 | 1 | 1 | 369.4 | 1171.9 | 0.32 |
| Gemini-768 99K | RaBitQ | 4 | ram | 2917 | 1 | 32 | 603.1 | 1181.3 | 0.51 |
| Gemini-768 99K | RaBitQ | 4 | ram | 2917 | 8 | 1 | 1389.2 | 1164.9 | 1.19 |
| Gemini-768 99K | RaBitQ | 4 | ram | 2917 | 8 | 32 | 3793.1 | 8199.4 | 0.46 |
| Gemini-768 99K | RaBitQ | 4 | peak | 2648 | 1 | 1 | 399.8 | 1171.9 | 0.34 |
| Gemini-768 99K | RaBitQ | 4 | peak | 2648 | 1 | 32 | 645.3 | 1181.3 | 0.55 |
| Gemini-768 99K | RaBitQ | 4 | peak | 2648 | 8 | 1 | 1493.6 | 1164.9 | 1.28 |
| Gemini-768 99K | RaBitQ | 4 | peak | 2648 | 8 | 32 | 4078.9 | 8199.4 | 0.50 |
| Gemini-768 99K | Qdrant TQ+ | 1 | disk | 673 | 1 | 1 | 1616.4 | 232.4 | 6.96 |
| Gemini-768 99K | Qdrant TQ+ | 1 | disk | 673 | 1 | 32 | 2082.3 | 243.4 | 8.56 |
| Gemini-768 99K | Qdrant TQ+ | 1 | disk | 673 | 8 | 1 | 6275.7 | 241.3 | 26.01 |
| Gemini-768 99K | Qdrant TQ+ | 1 | disk | 673 | 8 | 32 | 14303.6 | 1794.8 | 7.97 |
| Gemini-768 99K | Qdrant TQ+ | 1 | ram | 630 | 1 | 1 | 1710.6 | 232.4 | 7.36 |
| Gemini-768 99K | Qdrant TQ+ | 1 | ram | 630 | 1 | 32 | 2241.6 | 243.4 | 9.21 |
| Gemini-768 99K | Qdrant TQ+ | 1 | ram | 630 | 8 | 1 | 6400.6 | 241.3 | 26.53 |
| Gemini-768 99K | Qdrant TQ+ | 1 | ram | 630 | 8 | 32 | 15683.3 | 1794.8 | 8.74 |
| Gemini-768 99K | Qdrant TQ+ | 1 | peak | 523 | 1 | 1 | 1889.8 | 232.4 | 8.13 |
| Gemini-768 99K | Qdrant TQ+ | 1 | peak | 523 | 1 | 32 | 2460.3 | 243.4 | 10.11 |
| Gemini-768 99K | Qdrant TQ+ | 1 | peak | 523 | 8 | 1 | 7109.4 | 241.3 | 29.47 |
| Gemini-768 99K | Qdrant TQ+ | 1 | peak | 523 | 8 | 32 | 16959.7 | 1794.8 | 9.45 |
| Gemini-768 99K | Qdrant TQ+ | 2 | disk | 1429 | 1 | 1 | 910.1 | 219.1 | 4.15 |
| Gemini-768 99K | Qdrant TQ+ | 2 | disk | 1429 | 1 | 32 | 1204.2 | 227.8 | 5.29 |
| Gemini-768 99K | Qdrant TQ+ | 2 | disk | 1429 | 8 | 1 | 4061.5 | 227.0 | 17.89 |
| Gemini-768 99K | Qdrant TQ+ | 2 | disk | 1429 | 8 | 32 | 8073.3 | 1759.1 | 4.59 |
| Gemini-768 99K | Qdrant TQ+ | 2 | ram | 1352 | 1 | 1 | 924.5 | 219.1 | 4.22 |
| Gemini-768 99K | Qdrant TQ+ | 2 | ram | 1352 | 1 | 32 | 1255.5 | 227.8 | 5.51 |
| Gemini-768 99K | Qdrant TQ+ | 2 | ram | 1352 | 8 | 1 | 4142.1 | 227.0 | 18.25 |
| Gemini-768 99K | Qdrant TQ+ | 2 | ram | 1352 | 8 | 32 | 8616.6 | 1759.1 | 4.90 |
| Gemini-768 99K | Qdrant TQ+ | 2 | peak | 1250 | 1 | 1 | 844.6 | 219.1 | 3.86 |
| Gemini-768 99K | Qdrant TQ+ | 2 | peak | 1250 | 1 | 32 | 1299.8 | 227.8 | 5.71 |
| Gemini-768 99K | Qdrant TQ+ | 2 | peak | 1250 | 8 | 1 | 4243.1 | 227.0 | 18.69 |
| Gemini-768 99K | Qdrant TQ+ | 2 | peak | 1250 | 8 | 32 | 8899.2 | 1759.1 | 5.06 |
| Gemini-768 99K | Qdrant TQ+ | 4 | disk | 2936 | 1 | 1 | 369.3 | 188.0 | 1.96 |
| Gemini-768 99K | Qdrant TQ+ | 4 | disk | 2936 | 1 | 32 | 594.3 | 191.5 | 3.10 |
| Gemini-768 99K | Qdrant TQ+ | 4 | disk | 2936 | 8 | 1 | 1338.4 | 189.8 | 7.05 |
| Gemini-768 99K | Qdrant TQ+ | 4 | disk | 2936 | 8 | 32 | 3753.4 | 1477.6 | 2.54 |
| Gemini-768 99K | Qdrant TQ+ | 4 | ram | 2862 | 1 | 1 | 375.1 | 188.0 | 1.99 |
| Gemini-768 99K | Qdrant TQ+ | 4 | ram | 2862 | 1 | 32 | 610.3 | 191.5 | 3.19 |
| Gemini-768 99K | Qdrant TQ+ | 4 | ram | 2862 | 8 | 1 | 1360.9 | 189.8 | 7.17 |
| Gemini-768 99K | Qdrant TQ+ | 4 | ram | 2862 | 8 | 32 | 3964.7 | 1477.6 | 2.68 |
| Gemini-768 99K | Qdrant TQ+ | 4 | peak | 2600 | 1 | 1 | 407.3 | 188.0 | 2.17 |
| Gemini-768 99K | Qdrant TQ+ | 4 | peak | 2600 | 1 | 32 | 658.1 | 191.5 | 3.44 |
| Gemini-768 99K | Qdrant TQ+ | 4 | peak | 2600 | 8 | 1 | 1447.8 | 189.8 | 7.63 |
| Gemini-768 99K | Qdrant TQ+ | 4 | peak | 2600 | 8 | 32 | 4162.1 | 1477.6 | 2.82 |
| OpenAI-1536 1M | RaBitQ | 1 | disk | 1557 | 1 | 1 | 61.9 | 50.0 | 1.24 |
| OpenAI-1536 1M | RaBitQ | 1 | disk | 1557 | 1 | 32 | 109.1 | 50.0 | 2.18 |
| OpenAI-1536 1M | RaBitQ | 1 | disk | 1557 | 8 | 1 | 359.7 | 49.8 | 7.23 |
| OpenAI-1536 1M | RaBitQ | 1 | disk | 1557 | 8 | 32 | 848.9 | 373.0 | 2.28 |
| OpenAI-1536 1M | RaBitQ | 1 | ram | 1540 | 1 | 1 | 61.7 | 50.0 | 1.24 |
| OpenAI-1536 1M | RaBitQ | 1 | ram | 1540 | 1 | 32 | 111.4 | 50.0 | 2.23 |
| OpenAI-1536 1M | RaBitQ | 1 | ram | 1540 | 8 | 1 | 361.4 | 49.8 | 7.26 |
| OpenAI-1536 1M | RaBitQ | 1 | ram | 1540 | 8 | 32 | 844.8 | 373.0 | 2.26 |
| OpenAI-1536 1M | RaBitQ | 1 | peak | 1532 | 1 | 1 | 62.7 | 50.0 | 1.26 |
| OpenAI-1536 1M | RaBitQ | 1 | peak | 1532 | 1 | 32 | 110.6 | 50.0 | 2.21 |
| OpenAI-1536 1M | RaBitQ | 1 | peak | 1532 | 8 | 1 | 361.2 | 49.8 | 7.26 |
| OpenAI-1536 1M | RaBitQ | 1 | peak | 1532 | 8 | 32 | 842.7 | 373.0 | 2.26 |
| OpenAI-1536 1M | RaBitQ | 2 | disk | 3137 | 1 | 1 | 36.2 | 52.0 | 0.70 |
| OpenAI-1536 1M | RaBitQ | 2 | disk | 3137 | 1 | 32 | 60.4 | 51.9 | 1.16 |
| OpenAI-1536 1M | RaBitQ | 2 | disk | 3137 | 8 | 1 | 173.9 | 51.9 | 3.35 |
| OpenAI-1536 1M | RaBitQ | 2 | disk | 3137 | 8 | 32 | 455.2 | 386.0 | 1.18 |
| OpenAI-1536 1M | RaBitQ | 2 | ram | 3129 | 1 | 1 | 36.9 | 52.0 | 0.71 |
| OpenAI-1536 1M | RaBitQ | 2 | ram | 3129 | 1 | 32 | 60.9 | 51.9 | 1.17 |
| OpenAI-1536 1M | RaBitQ | 2 | ram | 3129 | 8 | 1 | 179.6 | 51.9 | 3.46 |
| OpenAI-1536 1M | RaBitQ | 2 | ram | 3129 | 8 | 32 | 456.2 | 386.0 | 1.18 |
| OpenAI-1536 1M | RaBitQ | 2 | peak | 3108 | 1 | 1 | 38.6 | 52.0 | 0.74 |
| OpenAI-1536 1M | RaBitQ | 2 | peak | 3108 | 1 | 32 | 61.9 | 51.9 | 1.19 |
| OpenAI-1536 1M | RaBitQ | 2 | peak | 3108 | 8 | 1 | 201.3 | 51.9 | 3.88 |
| OpenAI-1536 1M | RaBitQ | 2 | peak | 3108 | 8 | 32 | 473.1 | 386.0 | 1.23 |
| OpenAI-1536 1M | RaBitQ | 4 | disk | 6208 | 1 | 1 | 17.8 | 51.9 | 0.34 |
| OpenAI-1536 1M | RaBitQ | 4 | disk | 6208 | 1 | 32 | 33.1 | 52.1 | 0.64 |
| OpenAI-1536 1M | RaBitQ | 4 | disk | 6208 | 8 | 1 | 98.4 | 51.9 | 1.90 |
| OpenAI-1536 1M | RaBitQ | 4 | disk | 6208 | 8 | 32 | 249.5 | 390.6 | 0.64 |
| OpenAI-1536 1M | RaBitQ | 4 | ram | 6197 | 1 | 1 | 17.7 | 51.9 | 0.34 |
| OpenAI-1536 1M | RaBitQ | 4 | ram | 6197 | 1 | 32 | 33.3 | 52.1 | 0.64 |
| OpenAI-1536 1M | RaBitQ | 4 | ram | 6197 | 8 | 1 | 98.5 | 51.9 | 1.90 |
| OpenAI-1536 1M | RaBitQ | 4 | ram | 6197 | 8 | 32 | 249.6 | 390.6 | 0.64 |
| OpenAI-1536 1M | RaBitQ | 4 | peak | 6153 | 1 | 1 | 18.1 | 51.9 | 0.35 |
| OpenAI-1536 1M | RaBitQ | 4 | peak | 6153 | 1 | 32 | 33.5 | 52.1 | 0.64 |
| OpenAI-1536 1M | RaBitQ | 4 | peak | 6153 | 8 | 1 | 99.3 | 51.9 | 1.91 |
| OpenAI-1536 1M | RaBitQ | 4 | peak | 6153 | 8 | 32 | 252.4 | 390.6 | 0.65 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | disk | 1524 | 1 | 1 | 66.2 | 13.3 | 4.96 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | disk | 1524 | 1 | 32 | 113.6 | 12.8 | 8.89 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | disk | 1524 | 8 | 1 | 381.2 | 13.4 | 28.45 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | disk | 1524 | 8 | 32 | 870.4 | 99.2 | 8.78 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | ram | 1507 | 1 | 1 | 65.6 | 13.3 | 4.92 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | ram | 1507 | 1 | 32 | 115.3 | 12.8 | 9.02 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | ram | 1507 | 8 | 1 | 382.1 | 13.4 | 28.52 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | ram | 1507 | 8 | 32 | 876.7 | 99.2 | 8.84 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | peak | 1499 | 1 | 1 | 67.4 | 13.3 | 5.05 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | peak | 1499 | 1 | 32 | 115.4 | 12.8 | 9.03 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | peak | 1499 | 8 | 1 | 384.9 | 13.4 | 28.73 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | peak | 1499 | 8 | 32 | 873.5 | 99.2 | 8.81 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | disk | 3048 | 1 | 1 | 40.1 | 11.6 | 3.45 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | disk | 3048 | 1 | 32 | 62.7 | 11.9 | 5.25 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | disk | 3048 | 8 | 1 | 203.7 | 11.8 | 17.30 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | disk | 3048 | 8 | 32 | 484.1 | 94.0 | 5.15 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | ram | 3040 | 1 | 1 | 40.2 | 11.6 | 3.46 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | ram | 3040 | 1 | 32 | 63.2 | 11.9 | 5.29 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | ram | 3040 | 8 | 1 | 203.6 | 11.8 | 17.29 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | ram | 3040 | 8 | 32 | 481.5 | 94.0 | 5.12 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | peak | 3024 | 1 | 1 | 40.1 | 11.6 | 3.45 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | peak | 3024 | 1 | 32 | 63.3 | 11.9 | 5.30 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | peak | 3024 | 8 | 1 | 204.4 | 11.8 | 17.35 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | peak | 3024 | 8 | 32 | 484.1 | 94.0 | 5.15 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | disk | 6116 | 1 | 1 | 18.2 | 10.2 | 1.79 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | disk | 6116 | 1 | 32 | 33.9 | 10.2 | 3.33 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | disk | 6116 | 8 | 1 | 100.9 | 10.1 | 9.95 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | disk | 6116 | 8 | 32 | 253.5 | 82.4 | 3.08 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | ram | 6091 | 1 | 1 | 18.3 | 10.2 | 1.80 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | ram | 6091 | 1 | 32 | 34.2 | 10.2 | 3.36 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | ram | 6091 | 8 | 1 | 100.7 | 10.1 | 9.92 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | ram | 6091 | 8 | 32 | 254.6 | 82.4 | 3.09 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | peak | 6059 | 1 | 1 | 18.2 | 10.2 | 1.78 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | peak | 6059 | 1 | 32 | 34.0 | 10.2 | 3.34 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | peak | 6059 | 8 | 1 | 100.1 | 10.1 | 9.87 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | peak | 6059 | 8 | 32 | 256.5 | 82.4 | 3.11 |
| MS MARCO-1M real | RaBitQ | 1 | disk | 1052 | 1 | 1 | 83.0 | 71.3 | 1.16 |
| MS MARCO-1M real | RaBitQ | 1 | disk | 1052 | 1 | 32 | 151.5 | 71.8 | 2.11 |
| MS MARCO-1M real | RaBitQ | 1 | disk | 1052 | 8 | 1 | 502.8 | 71.6 | 7.02 |
| MS MARCO-1M real | RaBitQ | 1 | disk | 1052 | 8 | 32 | 1144.6 | 532.0 | 2.15 |
| MS MARCO-1M real | RaBitQ | 1 | ram | 1108 | 1 | 1 | 80.8 | 71.3 | 1.13 |
| MS MARCO-1M real | RaBitQ | 1 | ram | 1108 | 1 | 32 | 146.6 | 71.8 | 2.04 |
| MS MARCO-1M real | RaBitQ | 1 | ram | 1108 | 8 | 1 | 493.6 | 71.6 | 6.89 |
| MS MARCO-1M real | RaBitQ | 1 | ram | 1108 | 8 | 32 | 1099.9 | 532.0 | 2.07 |
| MS MARCO-1M real | RaBitQ | 1 | peak | 1101 | 1 | 1 | 81.2 | 71.3 | 1.14 |
| MS MARCO-1M real | RaBitQ | 1 | peak | 1101 | 1 | 32 | 147.5 | 71.8 | 2.06 |
| MS MARCO-1M real | RaBitQ | 1 | peak | 1101 | 8 | 1 | 489.6 | 71.6 | 6.84 |
| MS MARCO-1M real | RaBitQ | 1 | peak | 1101 | 8 | 32 | 1061.2 | 532.0 | 1.99 |
| MS MARCO-1M real | RaBitQ | 2 | disk | 2137 | 1 | 1 | 46.7 | 77.1 | 0.61 |
| MS MARCO-1M real | RaBitQ | 2 | disk | 2137 | 1 | 32 | 85.8 | 77.2 | 1.11 |
| MS MARCO-1M real | RaBitQ | 2 | disk | 2137 | 8 | 1 | 260.3 | 76.9 | 3.39 |
| MS MARCO-1M real | RaBitQ | 2 | disk | 2137 | 8 | 32 | 636.4 | 573.7 | 1.11 |
| MS MARCO-1M real | RaBitQ | 2 | ram | 2145 | 1 | 1 | 46.5 | 77.1 | 0.60 |
| MS MARCO-1M real | RaBitQ | 2 | ram | 2145 | 1 | 32 | 84.6 | 77.2 | 1.10 |
| MS MARCO-1M real | RaBitQ | 2 | ram | 2145 | 8 | 1 | 266.0 | 76.9 | 3.46 |
| MS MARCO-1M real | RaBitQ | 2 | ram | 2145 | 8 | 32 | 634.1 | 573.7 | 1.11 |
| MS MARCO-1M real | RaBitQ | 2 | peak | 2189 | 1 | 1 | 46.4 | 77.1 | 0.60 |
| MS MARCO-1M real | RaBitQ | 2 | peak | 2189 | 1 | 32 | 83.7 | 77.2 | 1.08 |
| MS MARCO-1M real | RaBitQ | 2 | peak | 2189 | 8 | 1 | 262.8 | 76.9 | 3.42 |
| MS MARCO-1M real | RaBitQ | 2 | peak | 2189 | 8 | 32 | 625.7 | 573.7 | 1.09 |
| MS MARCO-1M real | RaBitQ | 4 | disk | 4180 | 1 | 1 | 26.4 | 76.3 | 0.35 |
| MS MARCO-1M real | RaBitQ | 4 | disk | 4180 | 1 | 32 | 49.1 | 76.4 | 0.64 |
| MS MARCO-1M real | RaBitQ | 4 | disk | 4180 | 8 | 1 | 150.5 | 76.1 | 1.98 |
| MS MARCO-1M real | RaBitQ | 4 | disk | 4180 | 8 | 32 | 368.3 | 573.5 | 0.64 |
| MS MARCO-1M real | RaBitQ | 4 | ram | 4239 | 1 | 1 | 26.2 | 76.3 | 0.34 |
| MS MARCO-1M real | RaBitQ | 4 | ram | 4239 | 1 | 32 | 48.5 | 76.4 | 0.64 |
| MS MARCO-1M real | RaBitQ | 4 | ram | 4239 | 8 | 1 | 148.4 | 76.1 | 1.95 |
| MS MARCO-1M real | RaBitQ | 4 | ram | 4239 | 8 | 32 | 364.3 | 573.5 | 0.64 |
| MS MARCO-1M real | RaBitQ | 4 | peak | 4232 | 1 | 1 | 26.3 | 76.3 | 0.34 |
| MS MARCO-1M real | RaBitQ | 4 | peak | 4232 | 1 | 32 | 48.6 | 76.4 | 0.64 |
| MS MARCO-1M real | RaBitQ | 4 | peak | 4232 | 8 | 1 | 148.2 | 76.1 | 1.95 |
| MS MARCO-1M real | RaBitQ | 4 | peak | 4232 | 8 | 32 | 361.0 | 573.5 | 0.63 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | disk | 1018 | 1 | 1 | 88.8 | 18.6 | 4.77 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | disk | 1018 | 1 | 32 | 154.4 | 17.8 | 8.68 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | disk | 1018 | 8 | 1 | 512.3 | 17.7 | 28.96 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | disk | 1018 | 8 | 32 | 1168.9 | 148.0 | 7.90 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | ram | 1079 | 1 | 1 | 81.5 | 18.6 | 4.38 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | ram | 1079 | 1 | 32 | 146.5 | 17.8 | 8.24 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | ram | 1079 | 8 | 1 | 496.2 | 17.7 | 28.05 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | ram | 1079 | 8 | 32 | 1123.5 | 148.0 | 7.59 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | peak | 1069 | 1 | 1 | 81.7 | 18.6 | 4.39 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | peak | 1069 | 1 | 32 | 149.8 | 17.8 | 8.42 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | peak | 1069 | 8 | 1 | 481.2 | 17.7 | 27.20 |
| MS MARCO-1M real | Qdrant TQ+ | 1 | peak | 1069 | 8 | 32 | 1139.9 | 148.0 | 7.70 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | disk | 2054 | 1 | 1 | 46.9 | 17.0 | 2.76 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | disk | 2054 | 1 | 32 | 88.6 | 17.3 | 5.12 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | disk | 2054 | 8 | 1 | 278.5 | 17.2 | 16.15 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | disk | 2054 | 8 | 32 | 673.5 | 135.3 | 4.98 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | ram | 2106 | 1 | 1 | 47.1 | 17.0 | 2.77 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | ram | 2106 | 1 | 32 | 87.8 | 17.3 | 5.07 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | ram | 2106 | 8 | 1 | 285.3 | 17.2 | 16.54 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | ram | 2106 | 8 | 32 | 665.3 | 135.3 | 4.92 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | peak | 2099 | 1 | 1 | 47.0 | 17.0 | 2.76 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | peak | 2099 | 1 | 32 | 87.5 | 17.3 | 5.05 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | peak | 2099 | 8 | 1 | 280.0 | 17.2 | 16.23 |
| MS MARCO-1M real | Qdrant TQ+ | 2 | peak | 2099 | 8 | 32 | 669.1 | 135.3 | 4.95 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | disk | 4087 | 1 | 1 | 27.4 | 14.7 | 1.86 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | disk | 4087 | 1 | 32 | 49.7 | 14.9 | 3.33 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | disk | 4087 | 8 | 1 | 153.2 | 14.7 | 10.45 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | disk | 4087 | 8 | 32 | 374.0 | 116.1 | 3.22 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | ram | 4150 | 1 | 1 | 26.2 | 14.7 | 1.78 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | ram | 4150 | 1 | 32 | 48.9 | 14.9 | 3.28 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | ram | 4150 | 8 | 1 | 148.3 | 14.7 | 10.11 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | ram | 4150 | 8 | 32 | 367.5 | 116.1 | 3.16 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | peak | 4133 | 1 | 1 | 25.8 | 14.7 | 1.76 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | peak | 4133 | 1 | 32 | 48.9 | 14.9 | 3.28 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | peak | 4133 | 8 | 1 | 144.6 | 14.7 | 9.85 |
| MS MARCO-1M real | Qdrant TQ+ | 4 | peak | 4133 | 8 | 32 | 366.7 | 116.1 | 3.16 |

## Best measured configurations at equal recall

| Dataset | Reference | Target bits/dim | Best TieredQ QPS (schedule) | Fastest reference QPS reaching target | Ratio |
| --- | --- | --- | --- | --- | --- |
| SIFT1M | RaBitQ | 1 | 3667.1 (8t·b32) | 2060.0 (2b, 8t·b32) | 1.78× |
| SIFT1M | RaBitQ | 2 | 3060.5 (8t·b32) | 2060.0 (2b, 8t·b32) | 1.49× |
| SIFT1M | RaBitQ | 4 | 2092.5 (8t·b32) | 1778.0 (4b, 8t·b32) | 1.18× |
| SIFT1M | Qdrant TQ+ | 1 | 3728.2 (8t·b32) | 441.8 (2b, 8t·b32) | 8.44× |
| SIFT1M | Qdrant TQ+ | 2 | 3021.1 (8t·b32) | 441.8 (2b, 8t·b32) | 6.84× |
| SIFT1M | Qdrant TQ+ | 4 | 2089.5 (8t·b32) | 428.7 (4b, 8t·b32) | 4.87× |
| GloVe-200 1M | RaBitQ | 1 | 2936.1 (8t·b32) | 1382.1 (2b, 8t·b32) | 2.12× |
| GloVe-200 1M | RaBitQ | 2 | 2145.2 (8t·b32) | 1382.1 (2b, 8t·b32) | 1.55× |
| GloVe-200 1M | RaBitQ | 4 | 1490.8 (8t·b32) | 1353.8 (4b, 8t·b32) | 1.10× |
| GloVe-200 1M | Qdrant TQ+ | 1 | 3105.6 (8t·b32) | 266.4 (1b, 8t·b32) | 11.66× |
| GloVe-200 1M | Qdrant TQ+ | 2 | 2304.9 (8t·b32) | 255.9 (2b, 8t·b32) | 9.01× |
| GloVe-200 1M | Qdrant TQ+ | 4 | 1508.1 (8t·b32) | 248.8 (4b, 8t·b32) | 6.06× |
| OpenAI-1536 100K | RaBitQ | 1 | 11341.0 (8t·b32) | 4560.2 (2b, 8t·b32) | 2.49× |
| OpenAI-1536 100K | RaBitQ | 2 | 7362.7 (8t·b32) | 4560.2 (2b, 8t·b32) | 1.61× |
| OpenAI-1536 100K | RaBitQ | 4 | 2808.1 (8t·b32) | 3716.7 (4b, 8t·b32) | 0.76× |
| OpenAI-1536 100K | Qdrant TQ+ | 1 | 11341.0 (8t·b32) | 1049.1 (1b, 8t·b32) | 10.81× |
| OpenAI-1536 100K | Qdrant TQ+ | 2 | 7068.1 (8t·b32) | 952.5 (2b, 8t·b32) | 7.42× |
| OpenAI-1536 100K | Qdrant TQ+ | 4 | 2705.2 (8t·b32) | 814.8 (4b, 8t·b32) | 3.32× |
| Gemini-768 99K | RaBitQ | 1 | 16695.2 (8t·b32) | 8843.1 (2b, 8t·b32) | 1.89× |
| Gemini-768 99K | RaBitQ | 2 | 12418.3 (8t·b32) | 8843.1 (2b, 8t·b32) | 1.40× |
| Gemini-768 99K | RaBitQ | 4 | 6816.3 (8t·b32) | 8199.4 (4b, 8t·b32) | 0.83× |
| Gemini-768 99K | Qdrant TQ+ | 1 | 18582.1 (8t·b32) | 1794.8 (1b, 8t·b32) | 10.35× |
| Gemini-768 99K | Qdrant TQ+ | 2 | 13196.9 (8t·b32) | 1759.1 (2b, 8t·b32) | 7.50× |
| Gemini-768 99K | Qdrant TQ+ | 4 | 6513.6 (8t·b32) | 1477.6 (4b, 8t·b32) | 4.41× |
| OpenAI-1536 1M | RaBitQ | 1 | 1311.2 (8t·b32) | 390.6 (4b, 8t·b32) | 3.36× |
| OpenAI-1536 1M | RaBitQ | 2 | 743.7 (8t·b32) | 390.6 (4b, 8t·b32) | 1.90× |
| OpenAI-1536 1M | RaBitQ | 4 | 369.7 (8t·b32) | 390.6 (4b, 8t·b32) | 0.95× |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | 1324.8 (8t·b32) | 99.2 (1b, 8t·b32) | 13.36× |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | 743.7 (8t·b32) | 94.0 (2b, 8t·b32) | 7.91× |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | 355.8 (8t·b32) | 82.4 (4b, 8t·b32) | 4.32× |
| MS MARCO-1M real | RaBitQ | 1 | 1556.4 (8t·b32) | 573.7 (2b, 8t·b32) | 2.71× |
| MS MARCO-1M real | RaBitQ | 2 | 1095.9 (8t·b32) | 573.7 (2b, 8t·b32) | 1.91× |
| MS MARCO-1M real | RaBitQ | 4 | 592.0 (8t·b32) | 573.5 (4b, 8t·b32) | 1.03× |
| MS MARCO-1M real | Qdrant TQ+ | 1 | 1708.0 (8t·b32) | 148.0 (1b, 8t·b32) | 11.54× |
| MS MARCO-1M real | Qdrant TQ+ | 2 | 1095.9 (8t·b32) | 135.3 (2b, 8t·b32) | 8.10× |
| MS MARCO-1M real | Qdrant TQ+ | 4 | 565.9 (8t·b32) | 116.1 (4b, 8t·b32) | 4.87× |

Raw single-query p50/p95 and all schedule latencies: `all-timing.csv`. Raw total bytes, bits/vector, per-seed build time and memory: `all-memory.csv`. Absolute process build peaks and encoding rates: `all-build-diagnostics.csv`.


[Codec-first configuration tables](configurations.md) combine accuracy, storage, serving RAM, build cost and all four timing schedules for each evaluated configuration.
