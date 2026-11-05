"""
spgd_demo.py - The feedback loop of the slides run on the simulated amplifier:
SPGD on the correlation between the output CCD image and the target image,
SLM phase parametrised by Zernike polynomials, real LCOS model.

Operating point (as agreed): pump over the whole disk, 300 W, relay re-tuned
to the hot disk, plus a static shape error and 10 % hot spots as the
distortion the loop has to find. Targets: Gaussian (the cold output) and the
vortex LG01. One SPGD iteration costs 1-2 camera frames = 1-2 traces through
the amplifier (about 7 s each on the CPU at 1024 x 1024).

Outputs (Results/14_spgd/): correlation vs iteration, SLM phase before/after,
CCD before/after vs target, and the true field fidelity for comparison with
the physics-optimal patterns of the benchmark.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("PUMP_RADIUS", "6e-3")               # whole-disk pump (must be set before the import)
import limits_benchmark as lb
import thin_disk_system as tds
import slm
import correction as co
from correction_benchmark import hot_spot_field
from seed_beam import Grid, results_dir
from spgd import SPGD, Camera, correlation, field_overlap, zernike_basis

warnings.filterwarnings("ignore")

ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RESULTS = results_dir("14_spgd")
G = Grid(**tds.SIM_GRID)

# --- the plant: distorted amplifier at the operating point -----------------
P = 300
th = lb.thermal(P)
opd = th.solve(G, Q=th.heat_flux(G) * (1 + hot_spot_field(G, rms=0.1)))["opd"] + lb.shape_error(1, G)
f_th, _ = lb.per_pass_lens(lb.thermal(P).opd(G), G)
tds.DISK.thermal_opd = None
tds.MIRROR_ROC.update(lb.COLD_ROC)
ref, _ = tds.trace(slm.flat(), G)                           # cold Gaussian output = Gaussian target
targets = {k: v for k, v in co.target_modes(G, ref).items() if k in ("gaussian", "LG01")}
tds.DISK.thermal_opd = opd
tds.MIRROR_ROC.update(lb.retuned_mirrors(f_th))
amp = co.Amplifier(G)
fin_beam = co.field_at_slm(G)

cam = Camera(binning=int(round(50e-6 / G.dx)), blur=0.5, noise=0.01, bits=12)       # 50 um camera pixels
basis = zernike_basis(G, radius=3.0e-3, n_modes=20)         # tip .. secondary astigmatism, over the beam at the SLM


def measure(phase_map):
    """SLM phase map -> real LCOS -> amplifier -> CCD image at the output."""
    field = tds.SLM_DEVICE.modulate(fin_beam, slm.map_pattern(phase_map, G), bias=tds.SLM_BIAS).field
    return cam.image(amp.forward(field))


def output_field(phase_map):
    field = tds.SLM_DEVICE.modulate(fin_beam, slm.map_pattern(phase_map, G), bias=tds.SLM_BIAS).field
    return amp.forward(field)


ext = np.array([-1, 1, -1, 1]) * G.size / 2 * 1e3
summary = []
for name, tgt in targets.items():
    cam._scale = None
    target_img = Camera(binning=int(round(50e-6 / G.dx)), blur=0.5, noise=0.0, bits=16).image(tgt)
    phi_base = np.zeros((G.n, G.n)) if name == "gaussian" else slm.vortex(1)(G.X, G.Y)   # structured light: vortex on the SLM
    opt = SPGD(measure, lambda img: correlation(img, target_img), basis, sigma=0.2, gain=5.0,
               rule="slides", phi_base=phi_base, seed=1)
    before = output_field(opt.phase())
    print(f"--- target {name}: start correlation {correlation(cam.image(before), target_img):.3f}, "
          f"field fidelity {field_overlap(before, tgt):.3f}", flush=True)
    hist = opt.run(ITERATIONS, verbose=10)
    after = output_field(opt.phase(opt.best[1]))
    F_after = field_overlap(after, tgt)
    print(f"--- target {name}: correlation {hist[0]:.3f} -> {opt.best[0]:.3f} after {ITERATIONS} iterations "
          f"({opt.frames} frames); field fidelity {field_overlap(before, tgt):.3f} -> {F_after:.3f}", flush=True)
    summary.append((name, hist[0], opt.best[0], field_overlap(before, tgt), F_after, opt.frames))
    np.save(os.path.join(RESULTS, f"spgd_coefficients_{name}.npy"), opt.best[1])

    fig, ax = plt.subplots(2, 3, figsize=(14, 8.5))
    ax[0, 0].imshow(np.angle(np.exp(1j * (opt.phase(np.zeros(len(basis))) + tds.SLM_BIAS))), extent=ext, origin="lower", cmap="gray")
    ax[0, 0].set_title("SLM phase, uncompensated")
    ax[0, 1].imshow(np.angle(np.exp(1j * (opt.phase(opt.best[1]) + tds.SLM_BIAS))), extent=ext, origin="lower", cmap="gray")
    ax[0, 1].set_title("SLM phase, compensated (SPGD)")
    ax[0, 2].plot(hist, lw=1); ax[0, 2].set_xlabel("# of SPGD iterations"); ax[0, 2].set_ylabel("correlation")
    ax[0, 2].set_title(f"target: {name}"); ax[0, 2].grid(alpha=0.3); ax[0, 2].set_ylim(0, 1.02)
    for a, img, title in ((ax[1, 0], cam.image(before), f"CCD, uncompensated (C = {hist[0]:.2f})"),
                          (ax[1, 1], cam.image(after), f"CCD, compensated (C = {opt.best[0]:.2f}, F = {F_after:.2f})"),
                          (ax[1, 2], target_img, "target")):
        a.imshow(img, extent=ext, origin="lower", cmap="hot", vmin=0, vmax=max(img.max(), 1e-9))
        a.set_title(title)
    for a in ax.ravel()[[0, 1, 3, 4, 5]]:
        a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"SPGD feedback on the simulated amplifier (300 W, whole-disk pump, shape error + hot spots), target {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"spgd_{name}.png"), dpi=100)

with open(os.path.join(RESULTS, "spgd_summary.csv"), "w") as fh:
    fh.write("target,correlation_start,correlation_best,fidelity_start,fidelity_end,camera_frames\n")
    for row in summary:
        fh.write(",".join(str(x) for x in row) + "\n")
print("saved", os.path.abspath(RESULTS))
