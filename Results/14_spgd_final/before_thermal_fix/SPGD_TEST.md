# SPGD test (Step 14)

| scenario | target | frames | correlation start -> best | fidelity start -> SPGD | at final state | physics optimum | verdict |
|---|---|---|---|---|---|---|---|
| static | gaussian | 390 | 0.91 -> 0.95 | 0.46 -> 0.56 | 0.56 | 0.92 | FAIL |
| static | LG01 | 380 | 0.64 -> 0.81 | 0.28 -> 0.10 | 0.10 | 0.66 | FAIL |
| temp_noise | gaussian | 382 | 0.91 -> 0.95 | 0.46 -> 0.33 | 0.33 | 0.92 | FAIL |
| temp_noise | LG01 | 366 | 0.64 -> 0.81 | 0.28 -> 0.05 | 0.05 | 0.66 | FAIL |
| temp_drift | gaussian | 379 | 0.91 -> 0.95 | 0.46 -> 0.32 | 0.33 | 0.92 | FAIL |
| temp_drift | LG01 | 379 | 0.65 -> 0.81 | 0.28 -> 0.07 | 0.08 | 0.66 | FAIL |
| jitter | gaussian | 387 | 0.88 -> 0.90 | 0.46 -> 0.69 | 0.69 | 0.92 | FAIL |
| jitter | LG01 | 378 | 0.62 -> 0.78 | 0.28 -> 0.14 | 0.14 | 0.66 | FAIL |
| all | gaussian | 381 | 0.88 -> 0.90 | 0.46 -> 0.41 | 0.42 | 0.92 | FAIL |
| all | LG01 | 371 | 0.63 -> 0.74 | 0.28 -> 0.24 | 0.26 | 0.66 | FAIL |
