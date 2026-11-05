"""
gain_study.py - Step 15: amplification of the structured seed with the pump
over the whole 5 mm disk (300 W, flat top), uniform doping vs the 3-D clustered doping
(+-2 %), including the doping's thermal / index OPD from Step 11.

For the Gaussian and the vortex seed:
    * amplification = output power with gain / output power without gain
      (same optics, so losses and clipping cancel out),
    * output intensity and phase for uniform and clustered doping, and the
      residual (clustered - uniform), with the fidelity between the two.
Also: gain maps (uniform, clustered, relative difference) and a table of the
10-pass small-signal gain vs pump spot radius at 300 W.

Output: Results/15_gain/
"""
import os
import csv
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

import limits_benchmark as lb
import thin_disk_system as tds
import slm
import correction as co
from seed_beam import Grid, results_dir
from doping import DopedDiskThermal, doping_map_3d, N_SITES
from gain import gain_map, transparency_intensity
from thermal import DiskThermal

warnings.filterwarnings("ignore")
RESULTS = results_dir("15_gain")
G = lb.G
P = 300.0
R_PUMP = lb.PUMP_RADIUS
ORDER = lb.PUMP_ORDER
AMPL = 0.02                                    # doping cluster amplitude

# --- disk states on a 512 grid (3-D doping), interpolated to the trace grid -------------
g5 = Grid(n=512, size=G.size)
th = DopedDiskThermal(pump_power=P, doping=0.10, n_pump_passes=24, pump_radius=R_PUMP, pump_order=ORDER)
h = th.thickness
c3, delta3, z = doping_map_3d(g5, h, nz=50, nominal=0.10, amplitude=AMPL, cluster_size=0.3e-3, seed=1)
clustered = th.solve_3d(g5, c3, z)
uniform = DopedDiskThermal(pump_power=P, doping=0.10, doping_field=np.full((g5.n, g5.n), 0.10),
                           n_pump_passes=24, pump_radius=R_PUMP, pump_order=ORDER).solve(g5)
up = lambda a: zoom(a, G.n / g5.n, order=1)

states = {}
for name, st, n_col in (("uniform", uniform, np.full((g5.n, g5.n), N_SITES * 0.10 * h)),
                        ("clustered", clustered, N_SITES * c3.sum(axis=0) * (z[1] - z[0]))):
    Gmap, beta = gain_map(st["Q"] / th.eta_heat, n_col, st["T_mean"])
    states[name] = dict(gain=up(Gmap), beta=up(beta), opd=up(st["opd"]), T=up(st["T_mean"]),
                        c_mean=up(c3.mean(axis=0)) if name == "clustered" else np.full((G.n, G.n), 0.10))
c = G.n // 2
I_pump = P / (np.pi * R_PUMP ** 2)
print(f"pump {P:.0f} W over r = {R_PUMP*1e3:.1f} mm: {I_pump/1e7:.2f} kW/cm^2, transparency needs "
      f"{transparency_intensity(states['uniform']['T'][c, c])/1e7:.2f} kW/cm^2 at {states['uniform']['T'][c, c]:.0f} K")
for name, s in states.items():
    print(f"{name:9s}: beta centre {s['beta'][c, c]:.4f}, gain per reflection {s['gain'][c, c]:.4f}, "
          f"10 passes {s['gain'][c, c]**10:.3f}; gain map rms variation inside 2 mm {s['gain'][G.R < 2e-3].std() / s['gain'][G.R < 2e-3].mean():.2%}")

# --- traces: Gaussian and vortex seed, with and without gain ----------------------------
f_th, _ = lb.per_pass_lens(states["uniform"]["opd"])
tds.MIRROR_ROC.update(lb.retuned_mirrors(f_th))
patterns = {"gaussian": slm.flat(), "vortex": slm.vortex(1)}
out = {}
for dop, s in states.items():
    tds.DISK.thermal_opd = s["opd"]
    for seed_name, pat in patterns.items():
        tds.DISK.gain = None
        ref, _ = tds.trace(pat, G)
        tds.DISK.gain = s["gain"]
        amp, _ = tds.trace(pat, G)
        out[(dop, seed_name)] = (ref, amp)
        print(f"{dop:9s} {seed_name:8s}: amplification {amp.power / ref.power:.3f}  (output {amp.power*1e3:.1f} mW from {tds.SEED_POWER*1e3:.0f} mW seed incl. BS and losses)", flush=True)
tds.DISK.gain = None

# --- gain vs pump radius table (uniform disk) -------------------------------------------
rows = []
gq = Grid(n=256, size=G.size)
for r in (2.0e-3, 2.5e-3, 3.0e-3, 3.5e-3, 4.0e-3):
    t = DiskThermal(pump_power=P, doping=0.10, pump_radius=r, pump_order=ORDER)
    st = t.solve(gq)
    Gm, b = gain_map(st["Q"] / t.eta_heat, N_SITES * 0.10 * h, st["T_mean"])
    k = gq.n // 2
    rows.append(dict(pump_radius_mm=r * 1e3, pump_intensity_kW_cm2=P / (np.pi * r ** 2) / 1e7, T_centre_K=st["T_mean"][k, k],
                     beta=b[k, k], gain_per_reflection=Gm[k, k], gain_10_passes=Gm[k, k] ** 10))
with open(os.path.join(RESULTS, "gain_vs_pump_radius.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\n10-pass small-signal gain at the centre vs pump radius (300 W, uniform 10 % disk):")
for r in rows:
    print(f"  r = {r['pump_radius_mm']:.1f} mm: {r['pump_intensity_kW_cm2']:.2f} kW/cm^2, {r['T_centre_K']:.0f} K, beta {r['beta']:.3f}, "
          f"G = {r['gain_per_reflection']:.3f} per reflection, x{r['gain_10_passes']:.2f} after 10 passes")

# --- figures ------------------------------------------------------------------------------
ext = np.array([-1, 1, -1, 1]) * G.size / 2 * 1e3
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
im = ax[0].imshow(states["clustered"]["c_mean"] * 100, extent=ext, origin="lower", cmap="viridis"); ax[0].set_title("doping, thickness mean [at.%]")
plt.colorbar(im, ax=ax[0], fraction=0.046)
for a, name in zip(ax[1:3], ("uniform", "clustered")):
    im = a.imshow(states[name]["gain"], extent=ext, origin="lower", cmap="RdBu_r",
                  vmin=2 - states["uniform"]["gain"].max(), vmax=states["uniform"]["gain"].max())
    a.set_title(f"gain per reflection, {name} doping"); plt.colorbar(im, ax=a, fraction=0.046)
d = (states["clustered"]["gain"] / states["uniform"]["gain"] - 1) * 100
im = ax[3].imshow(d, extent=ext, origin="lower", cmap="RdBu_r", vmin=-np.abs(d).max(), vmax=np.abs(d).max())
ax[3].set_title("gain difference clustered / uniform - 1 [%]"); plt.colorbar(im, ax=ax[3], fraction=0.046)
for a in ax:
    a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
fig.suptitle(f"Small-signal gain of the disk: {P:.0f} W pump over r = {R_PUMP*1e3:.0f} mm "
             f"({I_pump/1e7:.2f} kW/cm^2), 0.5 mm / 10 % Yb:YAG, doping clusters +-{AMPL*100:.0f} %")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "gain_maps.png"), dpi=110)

for seed_name in patterns:
    u, cl = out[("uniform", seed_name)][1], out[("clustered", seed_name)][1]
    ref_u = out[("uniform", seed_name)][0]
    fid = co.fidelity(cl.field, u.field)
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.5))
    Imax = max(u.intensity.max(), cl.intensity.max())
    for j, (b, title) in enumerate(((u, "uniform doping"), (cl, "clustered doping"))):
        ax[0, j].imshow(b.intensity, extent=ext, origin="lower", cmap="inferno", vmin=0, vmax=Imax)
        ax[0, j].set_title(f"{seed_name} output intensity, {title}\n(amplification x{b.power / ref_u.power:.3f})")
        m = b.intensity > 1e-2 * b.intensity.max()
        ph = np.where(m, np.angle(b.field * np.conj(b.field[c, c + 1])), np.nan)
        ax[1, j].imshow(ph, extent=ext, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[1, j].set_title(f"phase, {title}")
    dI = cl.intensity - u.intensity
    im = ax[0, 2].imshow(dI / Imax * 100, extent=ext, origin="lower", cmap="RdBu_r", vmin=-np.abs(dI).max() / Imax * 100, vmax=np.abs(dI).max() / Imax * 100)
    ax[0, 2].set_title(f"residual intensity, clustered - uniform [% of peak]\nfidelity between the two outputs {fid:.4f}")
    plt.colorbar(im, ax=ax[0, 2], fraction=0.046)
    m = (u.intensity > 1e-2 * u.intensity.max())
    dphi = np.where(m, np.angle(cl.field * np.conj(u.field)), np.nan)
    dphi = dphi - np.nanmean(dphi)
    lim = np.nanmax(np.abs(dphi))
    im = ax[1, 2].imshow(dphi, extent=ext, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1, 2].set_title(f"residual phase [rad], rms {np.nanstd(dphi):.3f}"); plt.colorbar(im, ax=ax[1, 2], fraction=0.046)
    for a in ax.ravel():
        a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
    fig.suptitle(f"{seed_name} seed after 10 passes with gain: uniform vs clustered doping (+-{AMPL*100:.0f} %), {P:.0f} W over the whole disk")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f"amplified_{seed_name}.png"), dpi=110)
print("saved", os.path.abspath(RESULTS))
