"""Visual samples for the numerical-artefact check: the same patterns propagated on the 25 um
working grid and on the 12.5 um design grid. Pixel-level optimum (old ceiling) vs the 50 um
parametrised one (new ceiling) vs the network."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the simulation modules
import os, sys, numpy as np, torch
from scipy.ndimage import zoom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.environ.setdefault("NN_RANGE", "wide")
import nn_dataset as nd, nn_train as nt, nn_eval as ne, correction as co

G = nd.G; pad = (G.n - nd.CROP_LABEL) // 2
sysm = nd.System(); phys = nt.Physics(np.load(os.path.join(nt.DATA, "meta.npz"))); ref = nt.load_reference()
net, a = ne.load_model("resnet_all")
fine = ne.FineGrid()

def refine_pixels(pattern, rec, t_id, base, steps=100, lr=0.05):
    """The old ceiling: every 25 um pixel free."""
    st = nt.state_tensors({k: np.asarray(v)[None] for k, v in rec.items()}, [0])
    c = torch.as_tensor(nd.crop(np.angle(np.exp(1j * (pattern - base))), nd.CROP_LABEL), dtype=torch.float32, device=nt.DEV)[None].requires_grad_()
    opt = torch.optim.Adam([c], lr=lr); tgt = torch.as_tensor([t_id], device=nt.DEV)
    for _ in range(steps):
        F = phys.fidelity(c, st, tgt).sum(); opt.zero_grad(); (-F).backward(); opt.step()
    return base + np.pad(c.detach()[0].cpu().numpy(), pad)

def out_coarse(pattern, st, name):
    sysm.amp.set_thermal_opd(st["opd"])
    out = sysm.amp.forward(co.ideal_slm(sysm.fin, pattern))
    return out, co.fidelity(out, sysm.targets[name])

def out_fine(pattern, st, p, name):
    F = fine.fidelity(pattern, st, p, name)                           # arms the fine amplifier
    z = np.exp(1j * pattern); phi = np.angle(zoom(z.real, 2, order=1) + 1j * zoom(z.imag, 2, order=1))
    return fine.amp.forward(co.ideal_slm(fine.fin, phi)), F

rows = []
for i in range(2):
    rng = np.random.default_rng([i, 99]); p, st = sysm.draw_state(rng)
    T_pts = sysm.temperature_points(st["T_mean"], rng); P_meas = p["P"] * (1 + 0.01 * rng.standard_normal()); dop = sysm.doping_input(p)
    rec = sysm.record(st, p)
    for t_id, name in enumerate(nd.TARGETS):
        near, far = sysm.frames(sysm.output(sysm.base[name], st, p), p)
        x = nt.make_input(near[None], far[None], T_pts[None], [P_meas], dop[None], [t_id], ref, a["inputs"], a["formulation"])
        with torch.no_grad():
            corr = nt.full_correction(net(x), torch.as_tensor([t_id], device=nt.DEV), ref, a["formulation"])[0].cpu().numpy()
        pat_net = sysm.base[name] + np.pad(corr, pad)
        pat_gs, _ = sysm.optimum(name, st, p)
        pat_px = refine_pixels(pat_gs, rec, t_id, sysm.base[name])
        pat_50 = ne.refine_optimum(phys, pat_gs, rec, t_id, sysm.base[name], 100)
        rows.append((f"state {i}, {name}, {p['P']:.0f} W", name, st, p, [("network", pat_net), ("pixel-level optimum (old ceiling)", pat_px), ("50 um optimum (new ceiling)", pat_50)]))
        print(f"state {i} {name}: " + ", ".join(f"{k}: {out_coarse(pt, st, name)[1]:.3f} / {out_fine(pt, st, p, name)[1]:.3f}" for k, pt in rows[-1][4]), flush=True)

fig, ax = plt.subplots(len(rows), 9, figsize=(27, 3.2 * len(rows)))
for r, (title, name, st, p, pats) in enumerate(rows):
    for j, (k, pat) in enumerate(pats):
        oc, Fc = out_coarse(pat, st, name); of, Ff = out_fine(pat, st, p, name)
        c = G.n // 2; w = 60; c2 = fine.G2.n // 2; w2 = 120
        ax[r, 3 * j].imshow(np.angle(np.exp(1j * (pat - sysm.base[name])))[c - 128:c + 128, c - 128:c + 128], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[r, 3 * j].set_title(f"{k}\n{title}" if j == 0 else k, fontsize=9)
        ax[r, 3 * j + 1].imshow(np.abs(oc[c - w:c + w, c - w:c + w]) ** 2, cmap="inferno"); ax[r, 3 * j + 1].set_title(f"output, 25 um grid: F = {Fc:.3f}", fontsize=9)
        ax[r, 3 * j + 2].imshow(np.abs(of[c2 - w2:c2 + w2, c2 - w2:c2 + w2]) ** 2, cmap="inferno"); ax[r, 3 * j + 2].set_title(f"output, 12.5 um grid: F = {Ff:.3f}", fontsize=9)
    for a_ in ax[r]: a_.set_xticks([]); a_.set_yticks([])
fig.suptitle("Same SLM patterns propagated on the working grid (25 um) and the design grid (12.5 um); corrections shown without the base pattern")
fig.tight_layout(); out = os.path.join(ne.RESULTS, "numeric_samples.png"); fig.savefig(out, dpi=90); print("saved", out)
