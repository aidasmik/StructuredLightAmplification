# One network for any shape (Step 23)

Network 'shapes_onecam', layout as modelled (SLM 2.9 m before the disk). Held-out disk states of the wide range (240-360 W, no lab nuisances), ideal SLM. One camera (far field): the phase is not rated.
Target = what the nominal system makes from the shape's base hologram. Each cell: intensity similarity x power kept (field fidelity alone / output power relative to the target's), mean.

| shapes | family | n | uncorrected | network, one shot | ceiling (disk known) |
|---|---|---|---|---|---|
| test | blob | 26 | 0.69 (0.22 / 1.03) | 0.81 (0.20 / 1.06) | 0.89 (0.23 / 1.02) |
| test | flattop | 26 | 0.65 (0.33 / 1.00) | 0.83 (0.29 / 1.01) | 0.89 (0.29 / 1.00) |
| test | line | 25 | 0.66 (0.28 / 1.00) | 0.80 (0.35 / 1.03) | 0.88 (0.40 / 1.00) |
| test | polygon | 26 | 0.69 (0.29 / 1.01) | 0.86 (0.26 / 1.02) | 0.91 (0.24 / 1.00) |
| test | ring | 26 | 0.66 (0.14 / 1.02) | 0.77 (0.16 / 1.10) | 0.88 (0.21 / 1.02) |
| test | spiral | 26 | 0.71 (0.26 / 1.08) | 0.79 (0.24 / 1.29) | 0.92 (0.28 / 1.10) |
| test | spot | 26 | 0.70 (0.55 / 1.00) | 0.76 (0.59 / 1.00) | 0.80 (0.56 / 0.99) |
| test | spots | 25 | 0.70 (0.27 / 1.04) | 0.77 (0.29 / 1.14) | 0.90 (0.33 / 1.02) |
| test | strokes | 26 | 0.64 (0.26 / 1.03) | 0.78 (0.24 / 1.06) | 0.88 (0.26 / 1.01) |
| test | all | 232 | 0.68 (0.29 / 1.02) | 0.80 (0.29 / 1.08) | 0.88 (0.31 / 1.02) |
| unseen | letter | 24 | 0.67 (0.20 / 1.02) | 0.80 (0.24 / 1.04) | 0.88 (0.27 / 1.01) |
| unseen | square | 24 | 0.69 (0.35 / 1.00) | 0.86 (0.37 / 1.02) | 0.91 (0.37 / 1.00) |
| unseen | all | 48 | 0.68 (0.27 / 1.01) | 0.83 (0.30 / 1.03) | 0.90 (0.32 / 1.00) |
| named | gaussian | 16 | 0.64 (0.43 / 1.00) | 0.76 (0.58 / 1.00) | 0.82 (0.61 / 1.00) |
| named | vortex | 16 | 0.66 (0.24 / 1.01) | 0.79 (0.12 / 1.01) | 0.89 (0.15 / 1.00) |
| named | all | 32 | 0.65 (0.33 / 1.00) | 0.78 (0.35 / 1.00) | 0.86 (0.38 / 1.00) |
