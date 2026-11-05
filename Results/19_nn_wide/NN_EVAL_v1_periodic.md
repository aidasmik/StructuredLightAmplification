# Neural-network pre-correction (Step 18)

Fresh random disk states (100), true field fidelity through the real LCOS model, mean (10th percentile). Gaussian / LG01.

| method | static | temp_noise | temp_drift | cam_noise |
|---|---|---|---|---|
| uncorrected | 0.43 (0.12) / 0.20 (0.01) | 0.43 (0.12) / 0.20 (0.01) | 0.42 (0.12) / 0.20 (0.01) | 0.43 (0.12) / 0.20 (0.01) |
| reference correction only (no network) | 0.46 (0.14) / 0.25 (0.01) | 0.46 (0.15) / 0.25 (0.01) | 0.45 (0.14) / 0.24 (0.01) | 0.46 (0.14) / 0.25 (0.01) |
| network 'resnet_all' (resnet, residual, all; 1 shot) | 0.86 (0.67) / 0.70 (0.39) | 0.86 (0.67) / 0.69 (0.37) | 0.85 (0.65) / 0.68 (0.36) | 0.85 (0.64) / 0.68 (0.37) |
| network 'resnet_cams' (resnet, residual, cams; 1 shot) | 0.85 (0.66) / 0.70 (0.41) | 0.85 (0.66) / 0.70 (0.40) | 0.84 (0.64) / 0.69 (0.39) | 0.83 (0.61) / 0.68 (0.36) |
| physics optimum (projections + gradient ascent, phase-only, LCOS) | 0.87 (0.69) / 0.76 (0.53) | 0.87 (0.69) / 0.76 (0.53) | 0.86 (0.67) / 0.75 (0.49) | 0.87 (0.69) / 0.76 (0.53) |
| response matrix (Step 16, ~240 frames) | 0.80 / 0.59 | | | |
| SPGD (slides loop, ~390 frames) | 0.41 / 0.15 | | | |
