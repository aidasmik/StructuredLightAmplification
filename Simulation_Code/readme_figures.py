"""
readme_figures.py - figures for the README: the modelled system as it is used in Steps 17-21.

    layout.png         the beam path (BS - SLM - 10 disk passes via the mirror array - output),
                       relay tuned to the hot 300 W disk
    disk_state.png     one random disk state: temperature, thermal OPD, gain per reflection
    beam_path.png      Gaussian and LG01 seed through the amplifier (SLM, disk passes, output, target),
                       nominal 300 W disk and the random state
    correction.png     the same state, one-shot network correction (Step 21 network, all lab
                       nuisances of setup 0): target, uncorrected, corrected
    profiles.png       x and y intensity profiles of that output: target, uncorrected, corrected

Usage:  NN_RANGE=setup python readme_figures.py [network name]      (needs the GPU and Step 21 results)
Output: Results/readme/
"""
import os
import sys
from dataclasses import replace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("NN_RANGE", "setup")
import nn_dataset as nd
import nn_train as nt
import nn_eval as ne
import thin_disk_system as tds
import correction as co
import slm
from seed_beam import results_dir

OUT = results_dir("readme")
G = nd.G
MM = 1e3
sysm = nd.System()                                   # relay retuned to the hot nominal disk
pad = (G.n - nd.CROP_LABEL) // 2
ext = [-G.size / 2 * MM, G.size / 2 * MM] * 2


def crop_ext(a, half=4e-3):
    """Central +-half region of a grid map and its extent in mm."""
    k = int(half / G.dx); c = G.n // 2
    return a[c - k:c + k, c - k:c + k], [-half * MM, half * MM, -half * MM, half * MM]


def show(ax, I, title, cmap="inferno", half=4e-3, profiles=False, ref=None, **kw):
    """Map of I (+-half). profiles: overlay the x and y intensity profiles through the beam
    centroid, as a beam profiler shows them (x profile along the bottom, y profile along the
    left edge, each normalised to its peak); ref: a second intensity (the target) drawn dashed
    along the same lines."""
    img, e = crop_ext(I, half)
    im = ax.imshow(img, extent=e, origin="lower", cmap=cmap, **kw)
    if profiles:
        n = img.shape[0]
        x = np.linspace(e[0], e[1], n)
        w = img / img.sum()
        cy = int(round((w.sum(axis=1) * np.arange(n)).sum())); cx = int(round((w.sum(axis=0) * np.arange(n)).sum()))
        h = 0.35 * (e[1] - e[0])                                       # profile height: 35 % of the panel
        for a, style, alpha in ((img, "-", 1.0),) + (((crop_ext(ref, half)[0], "--", 0.8),) if ref is not None else ()):
            px, py = a[cy, :], a[:, cx]
            ax.plot(x, e[2] + h * px / (px.max() + 1e-30), style, color="cyan", lw=1.0, alpha=alpha)
            ax.plot(e[0] + h * py / (py.max() + 1e-30), x, style, color="lime", lw=1.0, alpha=alpha)
        ax.axhline(x[cy], color="cyan", lw=0.4, alpha=0.5); ax.axvline(x[cx], color="lime", lw=0.4, alpha=0.5)
        ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3])
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    return im


# ---- one random disk state (wide range, gain on), no lab nuisances for the physics figures ----
rng = np.random.default_rng([3, 99])
p, st = sysm.draw_state(rng)
for k in ("seed", "tilts", "stroke", "cam", "setup"):
    p.pop(k, None)
p.update({k: v for k, v in sysm.nominal_params().items() if k in ("seed", "tilts", "stroke", "cam")})
p0 = sysm.nominal_params()
st0 = sysm.state(p0)
print(f"state: P = {p['P']:.0f} W, hot spots {p['hot_rms']*100:.1f} %, decentre ({p['dx']*1e3:.2f}, {p['dy']*1e3:.2f}) mm, "
      f"shape x{p['shape_scale']:.2f}, doping {p['dop_amp']*2:.1f} %, lens change {p['lens_change']*100:+.0f} %", flush=True)

# ---- layout ---------------------------------------------------------------------------------
tds.plot_layout(OUT)
os.replace(os.path.join(OUT, "system_layout.png"), os.path.join(OUT, "layout.png"))

# ---- disk state -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
R = G.R < tds.DISK.diameter / 2
im = show(ax[0], np.where(R, st0["T_mean"], np.nan), "temperature, nominal 300 W disk [K]", cmap="hot", half=3e-3); fig.colorbar(im, ax=ax[0])
im = show(ax[1], np.where(R, st["T_mean"], np.nan), f"temperature, random state ({p['P']:.0f} W) [K]", cmap="hot", half=3e-3); fig.colorbar(im, ax=ax[1])
opd = st["opd"] - st0["opd"]
im = show(ax[2], np.where(R, opd * 1e9, np.nan), "OPD change vs nominal disk [nm]", cmap="RdBu_r", half=3e-3); fig.colorbar(im, ax=ax[2])
im = show(ax[3], np.where(R, st["gain"], np.nan), "power gain per reflection", cmap="viridis", half=3e-3); fig.colorbar(im, ax=ax[3])
fig.suptitle("Thin disk (Yb:YAG, 0.5 mm, 10 %, 5 mm): hot spots, decentred pump, shape error and doping clusters of one random state")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "disk_state.png"), dpi=100); plt.close(fig)

# ---- beam through the amplifier -------------------------------------------------------------
def trace_snapshots(pattern, state):
    tds.DISK.thermal_opd, tds.DISK.gain = state["opd"], state["gain"]
    snaps = {}
    out, _ = tds.trace(pattern, G, None, snaps)
    tds.DISK.thermal_opd = tds.DISK.gain = None
    disk = [snaps[k] for k in sorted(snaps) if "Disk" in k]
    slm_b = [snaps[k] for k in sorted(snaps) if "SLM" in k][0]
    return slm_b, disk, out

passes = (0, 2, 5, 9)
rows = [("Gaussian", slm.flat(), "gaussian"), ("LG01 vortex", slm.vortex(1), "LG01")]
fig, ax = plt.subplots(4, 7, figsize=(19, 12))
r = 0
for label, pat, tname in rows:
    for sname, state in (("nominal 300 W disk", st0), (f"random state, {p['P']:.0f} W", st)):
        b_slm, disk, out = trace_snapshots(pat, state)
        F = co.fidelity(out.field, sysm.targets[tname])
        show(ax[r, 0], np.abs(b_slm.field) ** 2, f"{label}, {sname}\nafter the SLM", profiles=True)
        for j, k in enumerate(passes):
            show(ax[r, 1 + j], np.abs(disk[k].field) ** 2, f"on the disk, pass {k + 1}", profiles=True)
        show(ax[r, 5], np.abs(out.field) ** 2, f"output, F = {F:.2f} (target dashed)", half=5e-3, profiles=True,
             ref=np.abs(sysm.targets[tname]) ** 2)
        show(ax[r, 6], np.abs(sysm.targets[tname]) ** 2, "target", half=5e-3, profiles=True)
        r += 1
fig.suptitle("Beam through the 10-pass amplifier (uncorrected: flat / vortex pattern on the SLM); intensity, +-4 mm (output +-5 mm)\n"
             "profiles through the beam centroid: x (cyan, bottom), y (green, left), each normalised to its peak; dashed: target")
fig.tight_layout(h_pad=2.2); fig.savefig(os.path.join(OUT, "beam_path.png"), dpi=90); plt.close(fig)

# ---- one-shot network correction (full lab model: setup 0 nuisances, gain, LCOS, cameras) ---
name = sys.argv[1] if len(sys.argv) > 1 else "resnet_ft"
net, a = ne.load_model(name)
ref = nt.load_reference()
rng = np.random.default_rng([3, 99])
ps, sts = sysm.draw_state(rng, setup=0)                      # the same disk state, with the nuisances of setup 0
T_pts = sysm.temperature_points(sts["T_mean"], rng); P_meas = ps["P"] * (1 + 0.01 * rng.standard_normal()); dop = sysm.doping_input(ps)
fig, ax = plt.subplots(2, 5, figsize=(16, 7))
outs = {}
for t_id, tname in enumerate(nd.TARGETS):
    out0 = sysm.output(sysm.base[tname], sts, ps)
    near, far = sysm.frames(out0, ps)
    x = nt.make_input(near[None], far[None], T_pts[None], [P_meas], dop[None], [t_id], ref, a["inputs"], a["formulation"], [0], a.get("ref_images", False))
    with torch.no_grad():
        corr = nt.full_correction(net(x), torch.as_tensor([t_id], device=nt.DEV), ref, a["formulation"], [0])[0].cpu().numpy()
    out1 = sysm.output(sysm.base[tname] + np.pad(corr, pad), sts, ps)
    tgt = sysm.targets[tname]
    outs[tname] = (out0, out1, tgt)
    ax[t_id, 0].imshow(near, cmap="hot"); ax[t_id, 0].set_title(f"{tname}: near-field camera (input)", fontsize=9)
    ax[t_id, 1].imshow(far, cmap="hot"); ax[t_id, 1].set_title("far-field camera (input)", fontsize=9)
    ax[t_id, 2].imshow(corr, cmap="twilight", vmin=-np.pi, vmax=np.pi); ax[t_id, 2].set_title("network correction on the SLM", fontsize=9)
    show(ax[t_id, 3], np.abs(out0) ** 2, f"output, uncorrected: F = {co.fidelity(out0, tgt):.2f}", half=4e-3, profiles=True, ref=np.abs(tgt) ** 2)
    show(ax[t_id, 4], np.abs(out1) ** 2, f"output, one-shot corrected: F = {co.fidelity(out1, tgt):.2f}", half=4e-3, profiles=True, ref=np.abs(tgt) ** 2)
    for aa in ax[t_id, :3]:
        aa.set_xticks([]); aa.set_yticks([])
fig.suptitle(f"One-shot correction by the network ('{name}', Step 21) with gain, seed / mirror / SLM / camera errors of the setup\n"
             "output profiles through the beam centroid: x (cyan), y (green), solid = output, dashed = target")
fig.tight_layout(h_pad=2.5); fig.savefig(os.path.join(OUT, "correction.png"), dpi=90); plt.close(fig)
# ---- line profiles of the same outputs -------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
x = (np.arange(G.n) - G.n / 2) * G.dx * MM
c = G.n // 2                                                 # cuts through the axis (the target's centre)
for r, tname in enumerate(nd.TARGETS):
    out0, out1, tgt = outs[tname]
    norm = lambda f: np.abs(f) ** 2 / (np.abs(f) ** 2).sum()           # same power -> comparable heights
    for j, (axis, cut) in enumerate((("x", lambda I: I[c, :]), ("y", lambda I: I[:, c]))):
        a = ax[r, j]
        a.fill_between(x, cut(norm(tgt)), color="0.85", label="target")
        a.plot(x, cut(norm(out0)), color="C3", lw=1.4, label=f"uncorrected, F = {co.fidelity(out0, tgt):.2f}")
        a.plot(x, cut(norm(out1)), color="C0", lw=1.8, label=f"one-shot corrected, F = {co.fidelity(out1, tgt):.2f}")
        a.set_xlim(-4, 4); a.set_title(f"{'Gaussian' if tname == 'gaussian' else 'LG01 vortex'}: {axis} profile through the axis", fontsize=10)
        a.set_ylabel("intensity (equal power)"); a.set_yticks([]); a.grid(alpha=0.3)
        if j == 0:
            a.legend(fontsize=8, loc="upper right")
for a in ax[1]:
    a.set_xlabel("position at the output [mm]")
fig.suptitle("Output beam profiles: target, uncorrected and corrected by the network in one shot (same disk state and setup)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "profiles.png"), dpi=110); plt.close(fig)
print("saved", OUT)
