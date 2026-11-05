# Neural-network pre-correction (Step 18)

Fresh random disk states (100), true field fidelity through the real LCOS model, mean (10th percentile). Gaussian / LG01.

| method | static | temp_noise | temp_drift | cam_noise |
|---|---|---|---|---|
| uncorrected | 0.65 (0.38) / 0.34 (0.07) | 0.65 (0.38) / 0.34 (0.07) | 0.65 (0.36) / 0.34 (0.07) | 0.65 (0.38) / 0.34 (0.07) |
| reference correction only (no network) | 0.70 (0.40) / 0.44 (0.09) | 0.70 (0.41) / 0.44 (0.09) | 0.70 (0.39) / 0.44 (0.08) | 0.70 (0.40) / 0.44 (0.09) |
| network 'resnet_all' (resnet, residual, all; 1 shot) | 0.93 (0.89) / 0.81 (0.70) | 0.93 (0.89) / 0.81 (0.69) | 0.92 (0.88) / 0.81 (0.70) | 0.93 (0.88) / 0.81 (0.70) |
| network 'resnet_cams' (resnet, residual, cams; 1 shot) | 0.93 (0.89) / 0.82 (0.70) | 0.93 (0.89) / 0.81 (0.68) | 0.92 (0.88) / 0.81 (0.68) | 0.92 (0.88) / 0.81 (0.68) |
| physics optimum (projections + gradient ascent, phase-only, LCOS) | 0.97 (0.94) / 0.89 (0.81) | 0.97 (0.93) / 0.89 (0.80) | 0.96 (0.93) / 0.88 (0.80) | 0.97 (0.94) / 0.89 (0.81) |
| response matrix (Step 16, ~240 frames) | 0.80 / 0.59 | | | |
| SPGD (slides loop, ~390 frames) | 0.41 / 0.15 | | | |
