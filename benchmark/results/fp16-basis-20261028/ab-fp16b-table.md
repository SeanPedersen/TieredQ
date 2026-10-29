| Index | Schedule | Tree | Rounds | QPS (median) | QPS ratio | Paired ratio range | p50 ratio | p95 ratio | Reference spread | IDs vs reference |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| gemini-2 | t1-b1 | adaptive-tmpl | 6 | 217 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.022 | identical |
| gemini-2 | t1-b1 | fp16 | 6 | 216 | 0.998 | 0.971–1.018 | 1.003 | 1.004 | 1.022 | differs |
| gemini-2 | t1-b32 | adaptive-tmpl | 6 | 554 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.019 | identical |
| gemini-2 | t1-b32 | fp16 | 6 | 547 | 0.986 | 0.978–1.000 | 0.999 | 1.003 | 1.019 | differs |
| gemini-2 | t8-b32 | adaptive-tmpl | 6 | 3,138 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.091 | identical |
| gemini-2 | t8-b32 | fp16 | 6 | 3,210 | 1.023 | 0.840–1.095 | 0.975 | 1.174 | 1.091 | differs |
| openai-2 | t1-b1 | adaptive-tmpl | 6 | 100 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.024 | identical |
| openai-2 | t1-b1 | fp16 | 6 | 101 | 1.011 | 0.984–1.023 | 0.987 | 0.990 | 1.024 | differs |
| openai-2 | t1-b32 | adaptive-tmpl | 6 | 239 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.015 | identical |
| openai-2 | t1-b32 | fp16 | 6 | 244 | 1.021 | 1.018–1.023 | 0.979 | 0.982 | 1.015 | differs |
| openai-2 | t8-b32 | adaptive-tmpl | 6 | 1,402 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.051 | identical |
| openai-2 | t8-b32 | fp16 | 6 | 1,462 | 1.042 | 0.977–1.081 | 0.980 | 0.810 | 1.051 | differs |
| openai-4 | t1-b1 | adaptive-tmpl | 6 | 87 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.020 | identical |
| openai-4 | t1-b1 | fp16 | 6 | 89 | 1.017 | 1.000–1.027 | 0.983 | 0.988 | 1.020 | differs |
| openai-4 | t1-b32 | adaptive-tmpl | 6 | 224 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.017 | identical |
| openai-4 | t1-b32 | fp16 | 6 | 225 | 1.004 | 0.989–1.019 | 0.982 | 0.996 | 1.017 | differs |
| openai-4 | t8-b32 | adaptive-tmpl | 6 | 1,261 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.037 | identical |
| openai-4 | t8-b32 | fp16 | 6 | 1,303 | 1.033 | 1.019–1.056 | 0.993 | 0.689 | 1.037 | differs |
