# Is the correction one phase map per disk state, at the disk plane? (Step 23 diagnostic 2)

8 held-out disk states x 8 test shapes, disk known, 250 Adam steps on the disk-plane phase (32 x 32 map + tilt, defocus, astigmatism), field fidelity, mean.

| family | n | uncorrected | per shape | universal, its fit shapes | universal, other shapes | from the Gaussian |
|---|---|---|---|---|---|---|
| blob | 9 | 0.10 | 0.76 | 0.66 | 0.39 | 0.09 |
| flattop | 9 | 0.46 | 0.88 | 0.74 | 0.72 | 0.40 |
| line | 9 | 0.25 | 0.81 | 0.67 | 0.56 | 0.22 |
| polygon | 4 | 0.09 | 0.65 | 0.50 | 0.23 | 0.11 |
| ring | 2 | 0.09 | 0.64 | 0.61 | 0.24 | 0.08 |
| spiral | 7 | 0.18 | 0.73 | 0.70 | 0.46 | 0.16 |
| spot | 7 | 0.63 | 0.91 | 0.68 | 0.79 | 0.71 |
| spots | 10 | 0.18 | 0.80 | 0.77 | 0.42 | 0.10 |
| strokes | 7 | 0.09 | 0.68 | 0.62 | 0.35 | 0.13 |
| all | 64 | 0.25 | 0.78 | 0.68 | 0.50 | 0.23 |
