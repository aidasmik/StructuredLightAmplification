"""
comparison_figures.py - figures of the README section "Networks against iterative loops".

Reads the saved curves and tables (no simulation, runs anywhere with numpy + matplotlib):
    Results/readme/iterations_curves.npz      iteration_curves.py   (slides SPGD, response matrix, one disk state)
    Results/22_xie_spgd/xie_curves.npz        xie_spgd.py           (published SPGD, small steps)
    Results/22_xie_spgd/corr_sweep.npz        xie_corr_sweep.py     (published SPGD tuned for the correlation)
    Results/23_nn_shapes/iter_<layout>.npz/csv  nn_shapes_iter.py   (like-for-like comparison)
Output: Results/readme/it_*.png, cmp_*.png
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
OUT = os.path.join(R, "readme")
it = np.load(os.path.join(OUT, "iterations_curves.npz"))
xie = np.load(os.path.join(R, "22_xie_spgd", "xie_curves.npz"))
sweep = np.load(os.path.join(R, "22_xie_spgd", "corr_sweep.npz"))
NAMES = {"gaussian": "Gaussian", "LG01": "LG01 vortex"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
save = lambda fig, name: (fig.tight_layout(), fig.savefig(os.path.join(OUT, name), dpi=170), plt.close(fig))


def panels(title, ylabel="fidelity / loop signal"):
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1), sharey=True)
    for a, t in zip(ax, NAMES):
        a.set_title(NAMES[t]); a.set_xlabel("camera frames"); a.set_ylim(0, 1); a.grid(alpha=.25)
    ax[0].set_ylabel(ylabel); fig.suptitle(title, fontsize=10.5)
    return fig, ax


def curve(a, c, color, label):                           # columns of c: frames, the loop's own signal, true fidelity
    a.plot(c[:, 0], c[:, 2], color=color, lw=2, label=label + ": true field fidelity")
    a.plot(c[:, 0], c[:, 1], color=color, lw=1.3, ls=":", label=label + ": the loop's own signal")


# ---- one disk state: the three loops ----------------------------------------------------------------
fig, ax = panels("One near-field camera: SPGD as read from the slides")
for a, t in zip(ax, NAMES):
    curve(a, it[t + "_spgd"], "#d62728", "loop")
ax[1].legend(fontsize=8, loc="center right", bbox_to_anchor=(1, .58)); save(fig, "it_slides.png")

fig, ax = panels("One far-field camera: published SPGD (Xie et al. 2015), small steps")
for a, t in zip(ax, NAMES):
    curve(a, xie[t + "_paper"], "#1f77b4", "paper's signal")
    curve(a, xie[t + "_pearson"], "#2ca02c", "correlation signal")
ax[1].legend(fontsize=7.5, loc="lower right"); save(fig, "it_published.png")

fig, ax = panels("Two cameras: response-matrix loop (calibration frames included)", "field fidelity")
for a, t in zip(ax, NAMES):
    a.plot(it[t + "_rm"][:, 0], it[t + "_rm"][:, 1], "o-", color="#ff7f0e", lw=2, ms=4, label="true field fidelity")
ax[0].legend(fontsize=8, loc="lower right"); save(fig, "it_matrix.png")

# ---- correlation against SPGD iterations, as plotted in the slides (delta 0.05 rad, gain 10) --------
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
for a, (t, name) in zip(ax, (("LG01", "vortex beam"), ("gaussian", "Gaussian beam"))):
    c = sweep[f"{t}_0.05_10.0"]; k = np.arange(len(c))
    a.plot(k[1:], c[1:, 1], color="#1f77b4", lw=1.8, label="correlation (what the loop sees)")
    a.plot(k, c[:, 2], color="#555555", lw=1.5, ls="--", label="true field fidelity (simulation only)")
    a.set_title("Target: " + name); a.set_xlabel("# of SPGD iterations"); a.set_xlim(0, 200); a.set_ylim(0.3, 1); a.grid(alpha=.25)
ax[0].set_ylabel("correlation / fidelity"); ax[1].legend(fontsize=8, loc="lower right"); save(fig, "it_corr.png")

# ---- like for like: mean score against camera frames, the one-shot network as a line ----------------
GROUPS = ["Gaussian", "LG01 vortex", "shapes", "shapes never seen"]


def compare(layout, k, net, ylabel, name, with_rm):
    z = np.load(os.path.join(R, "23_nn_shapes", f"iter_{layout}.npz"))
    rows = list(csv.DictReader(open(os.path.join(R, "23_nn_shapes", f"iter_{layout}.csv"))))
    mean = lambda g, c: np.mean([float(r[c]) for r in rows if r["group"] == g])
    col = {"I": 1, "D": 2}[k]                            # columns of the curves: frames, I, D
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 2.9), sharey=True)
    for a, g in zip(ax, GROUPS):
        s = z[f"{g}|spgd"]; a.plot(s[:, 0], s[:, col], color="#1f77b4", lw=1.8, label="SPGD (1 camera)")
        if with_rm:
            r = z[f"{g}|rm"][:-1]; a.plot(r[:, 0], r[:, col], "o-", ms=3, color="#ff7f0e", lw=1.6, label="response matrix (2 cameras)")
        a.axhline(mean(g, f"{k}_{net}"), color="#d62728", lw=1.8, label="network, one shot")
        a.axhline(mean(g, f"{k}_ceiling_{k}"), color="k", lw=1, ls="--", label="best possible (disk known)")
        a.set_title(g); a.set_xlabel("camera frames"); a.set_ylim(0, 1); a.set_xlim(0, 600); a.grid(alpha=.25)
    ax[0].set_ylabel(ylabel); ax[0].legend(fontsize=7, loc="lower right"); save(fig, name)


compare("free", "I", "shapes_onecam", "intensity similarity x power kept", "cmp_one.png", False)
compare("free", "D", "shapes_free", "fidelity x power kept", "cmp_two.png", True)
compare("relay", "D", "shapes_relay", "fidelity x power kept", "cmp_relay.png", True)
print("saved", OUT)
