# Neural-network pre-correction (Step 18)

Fresh random disk states (100), true field fidelity through the real LCOS model, mean (10th percentile). Gaussian / LG01.

| method | static | temp_noise | temp_drift | cam_noise |
|---|---|---|---|---|
| uncorrected | 0.36 (0.10) / 0.17 (0.01) | 0.36 (0.09) / 0.17 (0.01) | 0.35 (0.09) / 0.17 (0.01) | 0.36 (0.10) / 0.17 (0.01) |
| reference correction only (no network) | 0.38 (0.11) / 0.19 (0.01) | 0.38 (0.11) / 0.19 (0.01) | 0.37 (0.11) / 0.19 (0.01) | 0.38 (0.11) / 0.19 (0.01) |
| network 'resnet_all' (resnet, residual, all; 1 shot) | 0.59 (0.20) / 0.40 (0.04) | 0.59 (0.20) / 0.40 (0.04) | 0.58 (0.18) / 0.39 (0.03) | 0.53 (0.18) / 0.37 (0.03) |
| network 'resnet_ft' (resnet, residual, all; 1 shot) | 0.57 (0.18) / 0.40 (0.03) | 0.57 (0.18) / 0.40 (0.03) | 0.56 (0.17) / 0.39 (0.02) | 0.53 (0.17) / 0.37 (0.02) |
| physics optimum (projections + gradient ascent, phase-only, LCOS) | 0.84 (0.60) / 0.73 (0.50) | 0.83 (0.61) / 0.73 (0.50) | 0.82 (0.56) / 0.72 (0.48) | 0.84 (0.60) / 0.73 (0.50) |
| network 'resnet_all' (resnet, residual, all; 1 shot), scored on the 12.5 um grid | 0.59 (0.20) / 0.40 (0.03) | | | |
| network 'resnet_ft' (resnet, residual, all; 1 shot), scored on the 12.5 um grid | 0.57 (0.19) / 0.40 (0.03) | | | |
| physics optimum (projections + gradient ascent, phase-only, LCOS), scored on the 12.5 um grid | 0.83 (0.59) / 0.72 (0.47) | | | |
| response matrix (Step 16, ~240 frames) | 0.80 / 0.59 | | | |
| SPGD (slides loop, ~390 frames) | 0.41 / 0.15 | | | |
