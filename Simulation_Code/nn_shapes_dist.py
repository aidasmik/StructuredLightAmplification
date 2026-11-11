"""
nn_shapes_dist.py - how does the SLM-to-disk distance limit the pre-correction? (Step 23 diagnostic 3)

The slide of the setup shows SLM -> BS -> M0 -> mirror array -> disk with flat mirrors only (no
telescope) and gives no distances; our 2.91 m is a placement we chose. This scan moves the SLM along
the same seed beam to a distance d before the disk (d = 0: SLM imaged onto the disk) and measures,
on held-out disk states with the disk known:

    uncorrected    the base hologram alone
    ceiling        best phase-only correction (Adam on fidelity x power kept, as nn_shapes.ceiling)

for the Gaussian, the vortex and 6 shapes whose holograms are redesigned at every distance.
Numerically: the seed is propagated by (2.91 m - d) to the new SLM plane, the hologram is applied
there and the field is propagated back to the old SLM plane, where the amplifier model starts.

Usage:  NN_RANGE=wide python nn_shapes_dist.py [n_states]      (GPU PC)
Output: Results/23_nn_shapes/DIAG_distance.md, diag_distance.csv, diag_distance.png
"""
import os
import sys
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thin_disk_system as tds
import nn_shapes as ns

DEV, C = ns.DEV, ns.C
N_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
Z0 = tds._path_length("SLM", "BS", "M0", "A1", "Disk")  # distance of the model, 2.91 m
DIST = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, Z0)            # SLM-to-disk distance [m]

opt = ns.Optics()
fin0, H0 = opt.fin, opt.H_disk                          # seed at the model's SLM plane; free space over Z0
states = ns.load_states()
val_s = np.random.default_rng(5).choice(np.where(states["index"] % 20 == 0)[0], N_STATES, replace=False)
screens = opt.phys.screens(torch.as_tensor(states["opd"][val_s]).to(DEV))
S_bank = dict(np.load(os.path.join(ns.OUT, "bank_relay.npz")))
pick = [int(np.where(S_bank["test_family"] == f)[0][0]) for f in ns.FAMILIES]
S = torch.as_tensor(np.concatenate([S_bank["test_S"][pick], S_bank["unseen_S"][:1]]), device=DEV)   # 5 families + a letter
names = ["gaussian", "vortex"] + list(ns.FAMILIES) + ["letter"]

rows = []
for d in DIST:
    # SLM at distance d before the disk: Optics' relay branch with the transfer function of (Z0 - d)
    Hd = torch.as_tensor(ns._transfer_function(ns.N, float(ns.G.dx), float(2 * np.pi / tds.LAM * ns.refractive_index("air", tds.LAM)), Z0 - d), device=DEV)
    opt.H_disk, opt.relay = Hd, True
    opt.fin = torch.fft.ifft2(torch.fft.fft2(fin0) * Hd)
    phi_S, sim, kept = ns.design(opt, S, steps=60)
    named = torch.as_tensor(S_bank["named_phi"].astype(np.float32), device=DEV)            # Gaussian, charge-1 vortex
    phis = torch.cat([named, phi_S])
    print(f"d = {d:.2f} m: hologram design similarity {sim.mean():.2f}, power kept {kept.mean():.2f}", flush=True)
    for j, name in enumerate(names):
        phi = phis[j:j + 1].expand(N_STATES, -1, -1)
        with torch.no_grad():
            T = opt.output(phi, opt.screen0)
            D0 = ns.delivered(opt.output(phi, screens), T)
        Dc = ns.delivered(ns.ceiling(opt, phi, screens, T, torch.zeros_like(phi)), T)
        for k in range(N_STATES):
            rows.append(dict(d=d, target=name, state=int(val_s[k]), D0=D0[k].item(), D_ceiling=Dc[k].item()))
    m = lambda key, sel: np.mean([r[key] for r in rows if r["d"] == d and sel(r["target"])])
    print("   uncorrected -> ceiling: " + ", ".join(f"{n} {m('D0', lambda t: t == n):.2f} -> {m('D_ceiling', lambda t: t == n):.2f}" for n in names), flush=True)

with open(os.path.join(ns.OUT, "diag_distance.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
groups = {"Gaussian": ["gaussian"], "vortex": ["vortex"], "6 shapes": names[2:]}
mean = lambda key, d, g: np.mean([r[key] for r in rows if r["d"] == d and r["target"] in groups[g]])
lines = ["# Pre-correction against the SLM-to-disk distance (Step 23 diagnostic 3)", "",
         f"{N_STATES} held-out disk states (240-360 W), disk known, phase-only SLM. Cell: fidelity x power kept, uncorrected -> ceiling.",
         "d = 0: SLM imaged onto the disk; 2.91 m: the model's placement.", "",
         "| SLM-to-disk distance [m] | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
for d in DIST:
    lines.append(f"| {d:.2f} | " + " | ".join(f"{mean('D0', d, g):.2f} -> {mean('D_ceiling', d, g):.2f}" for g in groups) + " |")
open(os.path.join(ns.OUT, "DIAG_distance.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)

fig, ax = plt.subplots(figsize=(6.5, 4))
for i, g in enumerate(groups):
    ax.plot(DIST, [mean("D_ceiling", d, g) for d in DIST], "o-", color=f"C{i}", label=f"{g}: best phase-only correction")
    ax.plot(DIST, [mean("D0", d, g) for d in DIST], "o:", color=f"C{i}", label=f"{g}: uncorrected")
ax.set_xlabel("SLM-to-disk distance [m]"); ax.set_ylabel("fidelity x power kept"); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=7)
ax.set_title("Phase-only pre-correction limit against the SLM position")
fig.tight_layout(); fig.savefig(os.path.join(ns.OUT, "diag_distance.png"), dpi=120)
