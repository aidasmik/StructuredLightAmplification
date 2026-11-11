# One network for any shape (Step 23)

Network 'shapes_relay', SLM imaged onto the disk. Held-out disk states of the wide range (240-360 W, no lab nuisances), ideal SLM.
Target = what the nominal system makes from the shape's base hologram. Each cell: fidelity x power kept (field fidelity alone / output power relative to the target's), mean.

| shapes | family | n | uncorrected | network, one shot | ceiling (disk known) |
|---|---|---|---|---|---|
| test | blob | 26 | 0.48 (0.48 / 1.00) | 0.84 (0.88 / 0.95) | 0.91 (0.93 / 0.97) |
| test | flattop | 26 | 0.44 (0.44 / 1.00) | 0.90 (0.91 / 0.98) | 0.95 (0.97 / 0.98) |
| test | line | 25 | 0.44 (0.44 / 1.00) | 0.87 (0.89 / 0.98) | 0.94 (0.96 / 0.97) |
| test | polygon | 26 | 0.50 (0.50 / 1.00) | 0.91 (0.93 / 0.98) | 0.96 (0.97 / 0.99) |
| test | ring | 26 | 0.45 (0.45 / 1.00) | 0.86 (0.88 / 0.96) | 0.93 (0.94 / 0.99) |
| test | spiral | 26 | 0.46 (0.46 / 1.00) | 0.85 (0.88 / 0.96) | 0.92 (0.94 / 0.98) |
| test | spot | 26 | 0.42 (0.42 / 1.00) | 0.88 (0.90 / 0.97) | 0.93 (0.95 / 0.97) |
| test | spots | 25 | 0.49 (0.49 / 1.00) | 0.84 (0.88 / 0.95) | 0.92 (0.94 / 0.98) |
| test | strokes | 26 | 0.44 (0.44 / 1.00) | 0.81 (0.84 / 0.95) | 0.91 (0.93 / 0.97) |
| test | all | 232 | 0.46 (0.46 / 1.00) | 0.86 (0.89 / 0.96) | 0.93 (0.95 / 0.98) |
| unseen | letter | 24 | 0.41 (0.41 / 1.00) | 0.83 (0.85 / 0.96) | 0.93 (0.95 / 0.98) |
| unseen | square | 24 | 0.51 (0.51 / 1.00) | 0.91 (0.93 / 0.98) | 0.96 (0.97 / 0.99) |
| unseen | all | 48 | 0.46 (0.46 / 1.00) | 0.87 (0.89 / 0.97) | 0.94 (0.96 / 0.98) |
| named | gaussian | 16 | 0.43 (0.43 / 1.00) | 0.89 (0.90 / 1.00) | 0.98 (0.99 / 0.99) |
| named | vortex | 16 | 0.46 (0.46 / 1.00) | 0.94 (0.94 / 1.00) | 0.99 (0.99 / 1.00) |
| named | all | 32 | 0.44 (0.44 / 1.00) | 0.92 (0.92 / 1.00) | 0.99 (0.99 / 1.00) |
