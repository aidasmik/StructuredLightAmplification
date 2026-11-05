# Two-camera iterative correction (Step 16)

| scenario | target | method | frames | fidelity start -> end | best | physics optimum |
|---|---|---|---|---|---|---|
| static | gaussian | model-based (design model) | 32 | 0.55 -> 0.60 | 0.60 | 0.94 |
| static | gaussian | response matrix | 240 | 0.55 -> 0.80 | 0.81 | 0.94 |
| static | gaussian | response matrix, oracle field | 238 | 0.55 -> 0.79 | 0.79 | 0.94 |
| static | LG01 | model-based (design model) | 32 | 0.41 -> 0.47 | 0.47 | 0.77 |
| static | LG01 | response matrix | 238 | 0.41 -> 0.59 | 0.61 | 0.77 |
| static | LG01 | response matrix, oracle field | 238 | 0.41 -> 0.63 | 0.63 | 0.77 |
