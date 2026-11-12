# Networks against SPGD, like for like (Step 24, layout: SLM 2.9 m before the disk)

Held-out disk states (240-360 W), 84 pairs of disk state and shape, means. SPGD: Xie et al. 2015, far-field camera, 14 Zernike terms, signal 'pearson', delta 0.01, eta 3.0 (best of a sweep).

## One camera: intensity similarity x power kept

| target | n | uncorrected | SPGD, 200 frames | SPGD, 600 frames | network onecam | ceiling (disk known) |
|---|---|---|---|---|---|---|
| Gaussian | 16 | 0.64 | 0.74 | 0.75 | 0.73 | 0.79 |
| LG01 vortex | 16 | 0.64 | 0.73 | 0.71 | 0.79 | 0.84 |
| shapes | 36 | 0.67 | 0.75 | 0.75 | 0.81 | 0.90 |
| shapes never seen | 16 | 0.73 | 0.81 | 0.80 | 0.85 | 0.91 |

## Two cameras: field fidelity x power kept

| target | n | uncorrected | SPGD, 200 frames | SPGD, 600 frames | network free | ceiling (disk known) |
|---|---|---|---|---|---|---|
| Gaussian | 16 | 0.42 | 0.53 | 0.54 | 0.58 | 0.61 |
| LG01 vortex | 16 | 0.25 | 0.31 | 0.31 | 0.37 | 0.50 |
| shapes | 36 | 0.27 | 0.29 | 0.29 | 0.44 | 0.56 |
| shapes never seen | 16 | 0.28 | 0.27 | 0.26 | 0.45 | 0.55 |
