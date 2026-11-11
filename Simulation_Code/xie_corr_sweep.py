"""
xie_corr_sweep.py - Step 22b: the published SPGD driven by the image correlation, tuned for that signal.

The slides show the loop's correlation against the number of SPGD iterations (vortex: 0.68 -> 0.95 in
about 60 iterations). In xie_spgd.py the gain and perturbation were tuned for the paper's signal and
only reused for the correlation, which then rises slowly. Here both are swept for the correlation
signal itself, 200 iterations each, on the same disk state (300 W, hot spots, shape error).

Usage:  NN_RANGE=narrow python xie_corr_sweep.py
Output: Results/22_xie_spgd/corr_sweep.npz  (key <target>_<delta>_<eta>: columns frames, correlation, true fidelity)
"""
import os
import time
import numpy as np

os.environ.setdefault("NN_RANGE", "narrow")
import spgd_test as st
import xie_spgd as xs
from spgd import zernike_basis

DELTAS, ETAS, ITER = (0.05, 0.15, 0.3), (10.0, 30.0, 100.0), 200
t0 = time.time()
plant = st.Plant("static", 10000, seed=0)
ff = xs.FarFieldPlant(plant)
basis = zernike_basis(xs.G, radius=3.0e-3, n_modes=xs.N_TERMS)
curves = {}
for name, tgt in plant.targets.items():
    base = st.base_pattern(name)
    ff.set_exposure(base)
    ideal = ff.ideal_image(tgt)
    for delta in DELTAS:
        for eta in ETAS:
            plant.frame = 0
            log = xs.xie_loop(ff, base, basis, ideal, tgt, delta, eta, ITER, metric="pearson")
            curves[f"{name}_{delta}_{eta}"] = log
            print(f"{name} delta {delta} eta {eta:.0f}: correlation {log[1, 1]:.3f} -> {np.nanmean(log[-5:, 1]):.3f} (at 60: {log[60, 1]:.3f}), "
                  f"fidelity {log[0, 2]:.3f} -> {log[-5:, 2].mean():.3f}  ({time.time() - t0:.0f} s)", flush=True)
np.savez(os.path.join(xs.OUT, "corr_sweep.npz"), **curves)
print("saved", xs.OUT)
