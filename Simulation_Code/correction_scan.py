"""Why does the complex modulation give ~0 efficiency? Scan the scale s of
the back-projected (adjoint) field at the SLM: overlap with the wanted field
vs. power efficiency, and compare the wanted field's size with the seed."""
import os, sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import thin_disk_system as tds, correction as co
from correction_benchmark import DISTORTIONS, setup
from seed_beam import Grid, results_dir
from optics import _flip_x

g = Grid(**tds.SIM_GRID)
dev = tds.SLM_DEVICE
fig, axes = plt.subplots(1, len(DISTORTIONS), figsize=(4 * len(DISTORTIONS), 3.8), sharey=True)
rows = []
for ax, dist in zip(axes, DISTORTIONS):
    amp, fin, ref = setup(g, dist)
    env = np.sqrt(dev.reflectivity) * dev.fill_factor * np.abs(fin)
    w_seed = np.sqrt(4 * np.sum(np.abs(fin)**2 * g.R**2) / np.sum(np.abs(fin)**2))   # 2*sigma radius
    for name, tgt in co.target_modes(g, ref).items():
        want = _flip_x(amp.adjoint(tgt))
        P = np.abs(want)**2
        w_want = np.sqrt(4 * np.sum(P * g.R**2) / P.sum())
        inside = P[g.R < w_seed].sum() / P.sum()
        ss = np.geomspace(0.01, 100, 80) * env.max() / np.abs(want).max()
        ov, eff = [], []
        for s in ss:
            amp_s = np.minimum(s * np.abs(want), env)
            f = amp_s * np.exp(1j * np.angle(want))
            ov.append(co.fidelity(f, want)); eff.append(np.vdot(f, f).real / np.vdot(env, env).real)
        ov, eff = np.array(ov), np.array(eff)
        ax.semilogx(eff, ov, label=name)
        rows.append((dist, name, w_seed, w_want, inside, ov.max(), eff[ov.argmax()]))
        print(f"{dist:15s} {name:8s}: seed 2sigma {w_seed*1e3:.2f} mm, wanted {w_want*1e3:.2f} mm, "
              f"power inside seed radius {inside:.2f} | best SLM-plane overlap {ov.max():.3f} at eff {eff[ov.argmax()]:.3f}", flush=True)
    ax.set_title(dist); ax.set_xlabel("power efficiency after SLM"); ax.grid(alpha=0.3)
axes[0].set_ylabel("overlap with wanted field (SLM plane)"); axes[0].legend(fontsize=8)
fig.suptitle("Complex-amplitude pre-correction: how much of the required input field a 1.5 mm Gaussian seed can supply")
fig.tight_layout()
fig.savefig(os.path.join(results_dir("10_correction"), "fidelity_vs_efficiency.png"), dpi=110)
print("done")
