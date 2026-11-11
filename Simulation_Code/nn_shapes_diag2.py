"""
nn_shapes_diag2.py - is the any-shape correction ONE phase map per disk state, at the DISK plane?

nn_shapes_diag.py showed that a map added at the SLM cannot serve all shapes. Reason (layout): the SLM
is 2.91 m of free space before the disk, so a shaped beam arrives on the disk already spread into its
pattern; the disk error acts on a different light distribution for every shape. The -I relay images
the disk onto itself, so all passes together act as one phase screen psi at the disk plane. Its exact
pre-correction is  P(-z) [ exp(-i psi) P(z) [ seed exp(i phi_S) ] ]  (P: free space over z = 2.91 m);
the SLM shows the phase of that field (ns.to_slm). Tested here with the disk known (Adam on psi):

    per shape          psi fitted for every (state, shape) pair: the ceiling of this parametrisation
                       (psi = 32 x 32 map + tilt, defocus, astigmatism)
    universal          ONE psi per disk state, fitted on 8 training shapes, scored on 8 other shapes
    from the Gaussian  psi fitted on the plain Gaussian only, applied to the 8 other shapes

Usage:  NN_RANGE=wide python nn_shapes_diag2.py [n_states]      (GPU PC)
Output: Results/23_nn_shapes/DIAG_disk_plane.md, diag_disk_plane.csv
"""
import os
import sys
import csv

import numpy as np
import torch

import nn_shapes as ns

DEV, C = ns.DEV, ns.C
N_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
STEPS, K, RES = 250, 8, 32

opt = ns.Optics()
x = (torch.arange(C, device=DEV) - C / 2 + 0.5) * float(ns.G.dx) * 1e3   # [mm]
Y, X = torch.meshgrid(x, x, indexing="ij")
LOW = torch.stack([X, Y, X ** 2 + Y ** 2, X ** 2 - Y ** 2, 2 * X * Y])   # tilt, defocus, astigmatism [rad at 1 mm]
full = lambda p: ns.upsample(p[0]) + torch.einsum("bk,kxy->bxy", p[1], LOW)
fid = lambda phi, p, screens, T: ns.fidelity(opt.output(opt.to_slm(phi, full(p)), screens), T)


def fit(phi, screens, T, shared):
    """Adam on the disk-plane phase psi = RES x RES map + tilt / defocus / astigmatism (the edge of the
    disk needs tens of radians, too far for the map alone). One per example, or one shared by the batch."""
    n = 1 if shared else len(phi)
    p = [torch.zeros(n, RES, RES, device=DEV, requires_grad=True), torch.zeros(n, len(LOW), device=DEV, requires_grad=True)]
    adam = torch.optim.Adam(p, lr=0.05)
    for _ in range(STEPS):
        loss = (1 - fid(phi, p, screens, T)).sum()
        adam.zero_grad(); loss.backward(); adam.step()
    return [v.detach() for v in p]


states, bank = ns.load_states(), dict(np.load(os.path.join(ns.OUT, "bank_v2.npz")))
bt = ns.Batches(opt, states, bank)
val_s = np.where(states["index"] % 20 == 0)[0]
rng = np.random.default_rng(5)
zero = torch.zeros(1, C, C, device=DEV)
with torch.no_grad():
    T_gauss = opt.output(zero, opt.screen0)
    print("layer check, psi = 0 returns the hologram: fidelity", fid(zero, [torch.zeros(1, RES, RES, device=DEV), torch.zeros(1, len(LOW), device=DEV)], opt.screen0, T_gauss).item(), flush=True)

rows = []
for n, s in enumerate(rng.choice(val_s, N_STATES, replace=False)):
    si = np.full(K, s)
    ti_fit, ti = rng.integers(len(bank["train_phi"]), size=K), rng.choice(len(bank["test_phi"]), K, replace=False)
    _, phi_fit, screens, T_fit = bt.make(si, "train", ti_fit, noise=0)
    U = fit(phi_fit, screens, T_fit, shared=True)
    Gp = fit(zero, screens[:1], T_gauss, shared=True)
    _, phi, screens, T = bt.make(si, "test", ti, noise=0)
    own = fit(phi, screens, T, shared=False)
    with torch.no_grad():
        F = dict(F0=ns.fidelity(opt.output(phi, screens), T), F_own=fid(phi, own, screens, T), F_universal_fit=fid(phi_fit, U, screens, T_fit),
                 F_universal=fid(phi, U, screens, T), F_gauss=fid(phi, Gp, screens, T))
        Fg0, Fg = ns.fidelity(opt.output(zero, screens[:1]), T_gauss).item(), fid(zero, Gp, screens[:1], T_gauss).item()
    for k in range(K):
        rows.append(dict(state=int(s), family=bank["test_family"][ti[k]], **{key: v[k].item() for key, v in F.items()}))
    m = lambda key: np.mean([r[key] for r in rows[-K:]])
    print(f"state {n}: uncorrected {m('F0'):.2f}, per shape {m('F_own'):.2f}; universal: fit shapes {m('F_universal_fit'):.2f} -> other shapes "
          f"{m('F_universal'):.2f}; from the Gaussian {m('F_gauss'):.2f} (Gaussian itself {Fg0:.2f} -> {Fg:.2f})", flush=True)

with open(os.path.join(ns.OUT, "diag_disk_plane.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cols = ("F0", "F_own", "F_universal_fit", "F_universal", "F_gauss")
lines = ["# Is the correction one phase map per disk state, at the disk plane? (Step 23 diagnostic 2)", "",
         f"{N_STATES} held-out disk states x {K} test shapes, disk known, {STEPS} Adam steps on the disk-plane phase ({RES} x {RES} map + tilt, defocus, astigmatism), field fidelity, mean.", "",
         "| family | n | uncorrected | per shape | universal, its fit shapes | universal, other shapes | from the Gaussian |", "|---|---|---|---|---|---|---|"]
for f in sorted(set(r["family"] for r in rows)) + ["all"]:
    rr = [r for r in rows if f in (r["family"], "all")]
    lines.append(f"| {f} | {len(rr)} | " + " | ".join(f"{np.mean([r[c] for r in rr]):.2f}" for c in cols) + " |")
open(os.path.join(ns.OUT, "DIAG_disk_plane.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
