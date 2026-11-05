# Numerical and physical checks

One-off diagnostics used while building Steps 18-21. Run them from `Simulation_Code/`, e.g.
`NN_RANGE=wide python checks/fine_grid.py resnet_all`; most need a trained network and a dataset.

| script | question |
|---|---|
| `realism_consistency.py` | Do mirror tilts and gain maps agree between the numpy and torch amplifier, single and batched? |
| `nuisance_cost.py` | How much does each nuisance term (mirror tilt, seed error, SLM stroke) cost on the nominal disk? |
| `numerics.py` | Do the corrections exploit the periodic FFT window (periodic vs absorbing window vs real LCOS)? |
| `fine_grid.py` | Do the corrections hold on the 12.5 µm grid? (Showed that the pixel-level ceiling was a grid artefact.) |
| `fine_grid_samples.py` | The same as figures: patterns propagated on the 25 µm and 12.5 µm grids. |
| `wraparound.py` | Where do the raster lines come from? (Light wrapping around the FFT window; fixed by `propagation.absorber`.) |
| `nuisance_ablation.py` | Which nuisance effect limits the Step 20 network? |
| `drift_ablation.py` | How much does the shot-to-shot drift cost the Step 21 network? |

`numerics.py` and `wraparound.py` were written before the absorbing window became part of every
propagation step; their "periodic" case is now absorbing as well.
