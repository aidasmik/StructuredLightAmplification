"""
nn_shapes_diag.py - why does the any-shape network (Step 23) stall far below the ceiling?

On held-out disk states, with the true disk known (Adam on the correction, as nn_shapes.ceiling):

    per-shape ceiling at resolution r    the correction is an r x r map, bilinear to the 256 x 256 SLM crop.
                                         r = 128 is what the network can output, r = 256 the full ceiling.
                                         Tells how fine the correction has to be.
    universal (r = 32)                   ONE smooth correction per disk state, fitted on 8 training shapes,
                                         scored on those shapes ("fit") and on 8 other shapes.
                                         Tells whether the correction is a property of the disk state alone.
    Gaussian transfer (r = 32)           the smooth correction of the plain Gaussian applied to the 8 other shapes
    network                              shapes_v2, one shot from the shape's own camera frames

Usage:  NN_RANGE=wide python nn_shapes_diag.py [n_states]      (GPU PC)
Output: Results/23_nn_shapes/DIAG_resolution.md, diag_resolution.csv
"""
import os
import sys
import csv

import numpy as np
import torch
import torch.nn.functional as Fn

import nn_shapes as ns
import nn_train as nt

DEV, C = ns.DEV, ns.C
N_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STEPS, K = 150, 8
RES = (16, 32, 64, 128, 256)

opt = ns.Optics()
up = lambda c: c if c.shape[-1] == C else Fn.interpolate(c[:, None], size=(C, C), mode="bilinear", align_corners=False)[:, 0]
fid = lambda phi, corr, screens, T: ns.fidelity(opt.output(phi + corr, screens), T)


def optimise(phi, screens, T, res, shared=False):
    """Adam on a res x res correction map (one per example, or one shared by the batch). Returns the 256 x 256 map."""
    c = torch.zeros(1 if shared else len(phi), res, res, device=DEV, requires_grad=True)
    adam = torch.optim.Adam([c], lr=0.05)
    for _ in range(STEPS):
        loss = (1 - fid(phi, up(c), screens, T)).sum()
        adam.zero_grad(); loss.backward(); adam.step()
    return up(c.detach())


states, bank = ns.load_states(), dict(np.load(os.path.join(ns.OUT, "bank_v2.npz")))
bt = ns.Batches(opt, states, bank)
ck = torch.load(os.path.join(ns.OUT, "shapes_v2", "model.pt"), map_location=DEV)
net = nt.build("resnet", 13, ck["args"]["width"]); net.load_state_dict(ck["state"]); net.eval()
val_s = np.where(states["index"] % 20 == 0)[0]
rng = np.random.default_rng(5)
zero = torch.zeros(1, C, C, device=DEV)
with torch.no_grad():
    T_gauss = opt.output(zero, opt.screen0)

rows = []
for n, s in enumerate(rng.choice(val_s, N_STATES, replace=False)):
    si = np.full(K, s)
    ti_fit, ti = rng.integers(len(bank["train_phi"]), size=K), rng.choice(len(bank["test_phi"]), K, replace=False)
    _, phi_fit, screens, T_fit = bt.make(si, "train", ti_fit, noise=0)
    U = optimise(phi_fit, screens, T_fit, 32, shared=True)
    Gc = optimise(zero, screens[:1], T_gauss, 32, shared=True)
    x, phi, screens, T = bt.make(si, "test", ti)
    own = {r: optimise(phi, screens, T, r) for r in RES}
    with torch.no_grad():
        F = dict(F0=fid(phi, 0, screens, T), F_net=fid(phi, net(x), screens, T), F_universal=fid(phi, U, screens, T),
                 F_universal_fit=fid(phi_fit, U, screens, T_fit), F_gauss=fid(phi, Gc, screens, T),
                 **{f"F_own_{r}": fid(phi, own[r], screens, T) for r in RES})
        Fg0, Fg = fid(zero, 0, screens[:1], T_gauss).item(), fid(zero, Gc, screens[:1], T_gauss).item()
    for k in range(K):
        rows.append(dict(state=int(s), family=bank["test_family"][ti[k]], **{key: v[k].item() for key, v in F.items()}))
    m = lambda key: np.mean([r[key] for r in rows[-K:]])
    print(f"state {n}: uncorrected {m('F0'):.2f}, network {m('F_net'):.2f}, per-shape ceiling at r = "
          + ", ".join(f"{r}: {m(f'F_own_{r}'):.2f}" for r in RES)
          + f"; universal fit {m('F_universal_fit'):.2f} -> other shapes {m('F_universal'):.2f}; Gaussian transfer {m('F_gauss'):.2f}"
          f" (Gaussian itself {Fg0:.2f} -> {Fg:.2f})", flush=True)

with open(os.path.join(ns.OUT, "diag_resolution.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cols = ["F0", "F_net"] + [f"F_own_{r}" for r in RES] + ["F_universal_fit", "F_universal", "F_gauss"]
head = ["uncorrected", "network"] + [f"ceiling, {r} x {r}" for r in RES] + ["universal, its fit shapes", "universal, other shapes", "Gaussian's correction"]
lines = ["# How fine must the correction be, and is it a property of the disk state alone? (Step 23 diagnostic)", "",
         f"{N_STATES} held-out disk states x {K} test shapes, {STEPS} Adam steps with the disk known, field fidelity, mean.",
         "Ceiling r x r: the correction is an r x r map (bilinear to the 256 x 256 SLM crop); the network outputs 128 x 128.", "",
         "| family | n | " + " | ".join(head) + " |", "|" + "---|" * (len(head) + 2)]
for f in sorted(set(r["family"] for r in rows)) + ["all"]:
    rr = [r for r in rows if f in (r["family"], "all")]
    lines.append(f"| {f} | {len(rr)} | " + " | ".join(f"{np.mean([r[c] for r in rr]):.2f}" for c in cols) + " |")
open(os.path.join(ns.OUT, "DIAG_resolution.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
