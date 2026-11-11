# Neural-network pre-correction (Step 18)

Fresh random disk states (100), true field fidelity through the real LCOS model, mean (10th percentile). Gaussian / LG01.

| method | static | temp_noise | temp_drift | cam_noise |
|---|---|---|---|---|
| uncorrected | 0.36 (0.10) / 0.17 (0.01) | 0.36 (0.10) / 0.17 (0.01) | 0.35 (0.09) / 0.16 (0.00) | 0.36 (0.10) / 0.17 (0.01) |
| reference correction only (no network) | 0.43 (0.11) / 0.24 (0.01) | 0.43 (0.11) / 0.24 (0.01) | 0.42 (0.11) / 0.23 (0.01) | 0.43 (0.11) / 0.24 (0.01) |
| network 'resnet_ft' (resnet, residual, all; 1 shot) | 0.70 (0.40) / 0.54 (0.14) | 0.70 (0.40) / 0.54 (0.14) | 0.68 (0.37) / 0.53 (0.13) | 0.69 (0.38) / 0.53 (0.14) |
| network 'resnet_v2' (resnet, residual, all; 1 shot) | 0.77 (0.46) / 0.61 (0.26) | 0.76 (0.46) / 0.61 (0.26) | 0.75 (0.43) / 0.60 (0.23) | 0.76 (0.45) / 0.60 (0.21) |
| physics optimum (projections + gradient ascent, phase-only, LCOS) | 0.84 (0.60) / 0.73 (0.50) | 0.84 (0.60) / 0.73 (0.50) | 0.82 (0.57) / 0.72 (0.47) | 0.84 (0.60) / 0.73 (0.50) |
| network 'resnet_ft' (resnet, residual, all; 1 shot), scored on the 12.5 um grid | 0.70 (0.40) / 0.54 (0.14) | | | |
| network 'resnet_v2' (resnet, residual, all; 1 shot), scored on the 12.5 um grid | 0.76 (0.44) / 0.60 (0.24) | | | |
| physics optimum (projections + gradient ascent, phase-only, LCOS), scored on the 12.5 um grid | 0.83 (0.60) / 0.72 (0.47) | | | |
| response matrix (Step 16, ~240 frames) | 0.80 / 0.59 | | | |
| SPGD (slides loop, ~390 frames) | 0.41 / 0.15 | | | |
