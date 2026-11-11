# How fine must the correction be, and is it a property of the disk state alone? (Step 23 diagnostic)

6 held-out disk states x 8 test shapes, 150 Adam steps with the disk known, field fidelity, mean.
Ceiling r x r: the correction is an r x r map (bilinear to the 256 x 256 SLM crop); the network outputs 128 x 128.

| family | n | uncorrected | network | ceiling, 16 x 16 | ceiling, 32 x 32 | ceiling, 64 x 64 | ceiling, 128 x 128 | ceiling, 256 x 256 | universal, its fit shapes | universal, other shapes | Gaussian's correction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| blob | 4 | 0.13 | 0.23 | 0.72 | 0.87 | 0.89 | 0.89 | 0.89 | 0.71 | 0.03 | 0.02 |
| flattop | 5 | 0.52 | 0.73 | 0.89 | 0.92 | 0.94 | 0.94 | 0.94 | 0.67 | 0.32 | 0.30 |
| line | 8 | 0.21 | 0.43 | 0.74 | 0.83 | 0.86 | 0.87 | 0.87 | 0.48 | 0.09 | 0.12 |
| polygon | 2 | 0.45 | 0.62 | 0.82 | 0.90 | 0.93 | 0.93 | 0.93 | 0.79 | 0.28 | 0.16 |
| ring | 8 | 0.28 | 0.37 | 0.71 | 0.82 | 0.85 | 0.85 | 0.85 | 0.57 | 0.10 | 0.09 |
| spiral | 5 | 0.14 | 0.33 | 0.74 | 0.83 | 0.85 | 0.86 | 0.86 | 0.61 | 0.11 | 0.05 |
| spot | 7 | 0.68 | 0.89 | 0.90 | 0.94 | 0.95 | 0.96 | 0.96 | 0.55 | 0.50 | 0.47 |
| spots | 4 | 0.02 | 0.12 | 0.57 | 0.68 | 0.71 | 0.71 | 0.72 | 0.39 | 0.01 | 0.01 |
| strokes | 5 | 0.12 | 0.21 | 0.68 | 0.78 | 0.81 | 0.82 | 0.83 | 0.45 | 0.04 | 0.02 |
| all | 48 | 0.29 | 0.45 | 0.76 | 0.84 | 0.87 | 0.87 | 0.87 | 0.56 | 0.17 | 0.15 |
