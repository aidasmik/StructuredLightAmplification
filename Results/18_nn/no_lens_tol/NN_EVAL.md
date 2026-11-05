# Neural-network pre-correction (Step 18)

Fresh random disk states (100), true field fidelity through the real LCOS model, mean (10th percentile). Gaussian / LG01.

| method | static | temp_noise | temp_drift | cam_noise |
|---|---|---|---|---|
| uncorrected | 0.46 (0.16) / 0.23 (0.02) | 0.46 (0.15) / 0.23 (0.03) | 0.46 (0.17) / 0.23 (0.02) | 0.46 (0.16) / 0.23 (0.02) |
| reference correction only (no network) | 0.50 (0.19) / 0.30 (0.03) | 0.49 (0.18) / 0.30 (0.03) | 0.49 (0.20) / 0.30 (0.04) | 0.50 (0.19) / 0.30 (0.03) |
| network 'resnet_all' (resnet, residual, all; 1 shot) | 0.79 (0.49) / 0.63 (0.10) | 0.79 (0.49) / 0.63 (0.10) | 0.78 (0.49) / 0.63 (0.08) | 0.79 (0.51) / 0.62 (0.09) |
| physics optimum (phase-only, LCOS) | 0.67 (0.33) / 0.46 (0.20) | 0.67 (0.33) / 0.46 (0.19) | 0.66 (0.31) / 0.45 (0.18) | 0.67 (0.33) / 0.46 (0.20) |
| response matrix (Step 16, ~240 frames) | 0.80 / 0.59 | | | |
| SPGD (slides loop, ~390 frames) | 0.41 / 0.15 | | | |
