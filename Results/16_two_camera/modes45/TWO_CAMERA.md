# Two-camera iterative correction (Step 16)

| scenario | target | method | frames | fidelity start -> end | best | physics optimum |
|---|---|---|---|---|---|---|
| static | gaussian | model-based (design model) | 32 | 0.46 -> 0.51 | 0.51 | 0.92 |
| static | gaussian | response matrix | 374 | 0.46 -> 0.73 | 0.73 | 0.92 |
| static | gaussian | response matrix, oracle field | 374 | 0.46 -> 0.74 | 0.74 | 0.92 |
| static | LG01 | model-based (design model) | 32 | 0.28 -> 0.34 | 0.34 | 0.66 |
| static | LG01 | response matrix | 374 | 0.28 -> 0.53 | 0.53 | 0.66 |
| static | LG01 | response matrix, oracle field | 374 | 0.28 -> 0.54 | 0.54 | 0.66 |
