# Two-camera iterative correction (Step 16)

| scenario | target | method | frames | fidelity start -> end | best | physics optimum |
|---|---|---|---|---|---|---|
| static | gaussian | model-based (design model) | 32 | 0.46 -> 0.51 | 0.51 | 0.92 |
| static | gaussian | response matrix | 238 | 0.46 -> 0.68 | 0.75 | 0.92 |
| static | gaussian | response matrix, oracle field | 238 | 0.46 -> 0.77 | 0.77 | 0.92 |
| static | LG01 | model-based (design model) | 32 | 0.28 -> 0.34 | 0.34 | 0.66 |
| static | LG01 | response matrix | 238 | 0.28 -> 0.42 | 0.47 | 0.66 |
| static | LG01 | response matrix, oracle field | 238 | 0.28 -> 0.53 | 0.53 | 0.66 |
| temp_noise | gaussian | model-based (design model) | 32 | 0.46 -> 0.51 | 0.53 | 0.92 |
| temp_noise | gaussian | response matrix | 238 | 0.46 -> 0.72 | 0.76 | 0.92 |
| temp_noise | gaussian | response matrix, oracle field | 238 | 0.45 -> 0.75 | 0.75 | 0.92 |
| temp_noise | LG01 | model-based (design model) | 32 | 0.28 -> 0.35 | 0.36 | 0.66 |
| temp_noise | LG01 | response matrix | 238 | 0.29 -> 0.43 | 0.45 | 0.66 |
| temp_noise | LG01 | response matrix, oracle field | 238 | 0.28 -> 0.51 | 0.51 | 0.66 |
| all | gaussian | model-based (design model) | 32 | 0.46 -> 0.52 | 0.53 | 0.92 |
| all | gaussian | response matrix | 238 | 0.44 -> 0.62 | 0.76 | 0.92 |
| all | gaussian | response matrix, oracle field | 238 | 0.46 -> 0.76 | 0.76 | 0.92 |
| all | LG01 | model-based (design model) | 32 | 0.28 -> 0.34 | 0.36 | 0.66 |
| all | LG01 | response matrix | 238 | 0.27 -> 0.45 | 0.46 | 0.66 |
| all | LG01 | response matrix, oracle field | 238 | 0.27 -> 0.52 | 0.52 | 0.66 |
