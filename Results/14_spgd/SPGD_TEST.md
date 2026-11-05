# SPGD test (Step 14)

| scenario | target | frames | correlation start -> best | fidelity start -> SPGD | at final state | physics optimum | verdict |
|---|---|---|---|---|---|---|---|
| static | gaussian | 349 | 0.76 -> 0.93 | 0.40 -> 0.34 | 0.34 | 0.68 | FAIL |
| static | LG01 | 365 | 0.75 -> 0.88 | 0.11 -> 0.17 | 0.17 | 0.52 | FAIL |
| temp_noise | gaussian | 347 | 0.76 -> 0.91 | 0.40 -> 0.34 | 0.34 | 0.68 | FAIL |
| temp_noise | LG01 | 361 | 0.75 -> 0.88 | 0.11 -> 0.17 | 0.17 | 0.52 | FAIL |
| temp_drift | gaussian | 349 | 0.76 -> 0.94 | 0.40 -> 0.51 | 0.50 | 0.68 | FAIL |
| temp_drift | LG01 | 362 | 0.75 -> 0.88 | 0.11 -> 0.17 | 0.18 | 0.52 | FAIL |
| jitter | gaussian | 345 | 0.76 -> 0.88 | 0.40 -> 0.43 | 0.43 | 0.68 | FAIL |
| jitter | LG01 | 363 | 0.74 -> 0.85 | 0.11 -> 0.21 | 0.21 | 0.52 | FAIL |
| all | gaussian | 351 | 0.75 -> 0.88 | 0.40 -> 0.29 | 0.29 | 0.68 | FAIL |
| all | LG01 | 362 | 0.77 -> 0.87 | 0.11 -> 0.23 | 0.24 | 0.52 | FAIL |
