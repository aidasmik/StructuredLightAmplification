# Networks against SPGD, like for like (Step 24, layout: SLM imaged onto the disk)

Held-out disk states (240-360 W), 84 pairs of disk state and shape, means. SPGD: Xie et al. 2015, far-field camera, 14 Zernike terms, signal 'pearson', delta 0.01, eta 10.0 (best of a sweep).

## One camera: intensity similarity x power kept

| target | n | uncorrected | SPGD, 200 frames | SPGD, 600 frames | network relay | ceiling (disk known) |
|---|---|---|---|---|---|---|
| Gaussian | 16 | 0.64 | 0.87 | 0.89 | 0.95 | 0.98 |
| LG01 vortex | 16 | 0.68 | 0.96 | 0.96 | 0.97 | 0.99 |
| shapes | 36 | 0.68 | 0.86 | 0.86 | 0.90 | 0.97 |
| shapes never seen | 16 | 0.80 | 0.94 | 0.95 | 0.93 | 0.99 |

## Two cameras: field fidelity x power kept

| target | n | uncorrected | SPGD, 200 frames | SPGD, 600 frames | network relay | ceiling (disk known) |
|---|---|---|---|---|---|---|
| Gaussian | 16 | 0.42 | 0.74 | 0.79 | 0.89 | 0.97 |
| LG01 vortex | 16 | 0.46 | 0.90 | 0.90 | 0.94 | 0.99 |
| shapes | 36 | 0.42 | 0.70 | 0.71 | 0.83 | 0.91 |
| shapes never seen | 16 | 0.54 | 0.76 | 0.79 | 0.88 | 0.93 |
