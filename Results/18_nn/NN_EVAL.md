# Neural-network pre-correction (Step 18)

Fresh random disk states (100), true field fidelity through the real LCOS model, mean (10th percentile). Gaussian / LG01.

| method | static | temp_noise | temp_drift | cam_noise |
|---|---|---|---|---|
| uncorrected | 0.64 (0.19) / 0.36 (0.01) | 0.64 (0.19) / 0.36 (0.01) | 0.64 (0.17) / 0.36 (0.01) | 0.64 (0.19) / 0.36 (0.01) |
| reference correction only (no network) | 0.68 (0.21) / 0.46 (0.01) | 0.68 (0.21) / 0.46 (0.01) | 0.67 (0.20) / 0.45 (0.01) | 0.68 (0.21) / 0.46 (0.01) |
| network 'resnet_all' (resnet, residual, all; 1 shot) | 0.92 (0.83) / 0.80 (0.64) | 0.92 (0.83) / 0.80 (0.63) | 0.91 (0.81) / 0.79 (0.62) | 0.92 (0.82) / 0.80 (0.64) |
| network 'resnet_cams' (resnet, residual, cams; 1 shot) | 0.92 (0.84) / 0.80 (0.63) | 0.92 (0.84) / 0.80 (0.61) | 0.91 (0.83) / 0.79 (0.60) | 0.91 (0.80) / 0.80 (0.61) |
| physics optimum (projections + gradient ascent, phase-only, LCOS) | 0.94 (0.86) / 0.86 (0.73) | 0.94 (0.86) / 0.86 (0.73) | 0.94 (0.84) / 0.86 (0.71) | 0.94 (0.86) / 0.86 (0.73) |
| network 'resnet_all' (resnet, residual, all; 1 shot), scored on the 12.5 um grid | 0.92 (0.83) / 0.80 (0.64) | | | |
| network 'resnet_cams' (resnet, residual, cams; 1 shot), scored on the 12.5 um grid | 0.92 (0.84) / 0.80 (0.62) | | | |
| physics optimum (projections + gradient ascent, phase-only, LCOS), scored on the 12.5 um grid | 0.94 (0.84) / 0.86 (0.70) | | | |
| response matrix (Step 16, ~240 frames) | 0.80 / 0.59 | | | |
| SPGD (slides loop, ~390 frames) | 0.41 / 0.15 | | | |
