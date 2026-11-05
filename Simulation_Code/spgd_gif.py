"""
spgd_gif.py - Animation of the SPGD feedback loop of the slides running on
the simulated amplifier (static scenario of spgd_test.py): per iteration the
SLM phase, the CCD image next to the target and the growing correlation
curve, for the Gaussian and the vortex target.

Output: Results/14_spgd/spgd_feedback.gif  (+ the last frame as PNG)
"""
import io
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import thin_disk_system as tds
from seed_beam import results_dir
from spgd import SPGD, Camera, correlation, field_overlap, zernike_basis
import spgd_test as st

warnings.filterwarnings("ignore")
ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
EVERY = 2                                    # record every n-th iteration
RESULTS = results_dir("14_spgd")
G = st.G

plant = st.Plant("static", int(ITERATIONS * 1.7), seed=0)
basis = zernike_basis(G, radius=3.0e-3, n_modes=20)
target_cam = Camera(binning=int(round(50e-6 / G.dx)), blur=0.5, noise=0.0, bits=16)
ext = np.array([-1, 1, -1, 1]) * G.size / 2 * 1e3

records = {}
for name, tgt in plant.targets.items():
    plant.cam._scale = None
    plant.frame = 0
    target_img = target_cam.image(tgt)
    opt = SPGD(plant.measure, lambda img: correlation(img, target_img), basis, sigma=0.2, gain=5.0,
               rule="slides", phi_base=st.base_pattern(name), seed=1)
    frames = []

    def record(i, o):
        if i % EVERY == 0 or i == ITERATIONS - 1:
            F = field_overlap(plant.output_nominal(o.phase()), tgt)      # true fidelity of the current pattern
            frames.append((i, np.angle(np.exp(1j * (o.phase() + tds.SLM_BIAS))), plant.last_image.copy(),
                           np.array(o.history), F))

    opt.run(ITERATIONS, verbose=25, callback=record)
    records[name] = (frames, target_img)
    print(f"{name}: correlation {opt.history[0]:.3f} -> {opt.best[0]:.3f}, true fidelity at the end {frames[-1][4]:.3f}", flush=True)

# --- compose the GIF: one row per target --------------------------------------
names = list(records)
n_frames = min(len(records[n][0]) for n in names)
images = []
for k in range(n_frames):
    fig, ax = plt.subplots(len(names), 4, figsize=(15, 3.6 * len(names)))
    for r, name in enumerate(names):
        i, phase, ccd, hist, F = records[name][0][k]
        target_img = records[name][1]
        ax[r, 0].imshow(phase, extent=ext, origin="lower", cmap="gray", vmin=-np.pi, vmax=np.pi)
        ax[r, 0].set_title(f"{name}: SLM phase, iteration {i}")
        ax[r, 1].imshow(ccd, extent=ext, origin="lower", cmap="hot", vmin=0, vmax=max(ccd.max(), 1e-9))
        ax[r, 1].set_title(f"CCD (correlation {hist[-1]:.2f}, field fidelity {F:.2f})")
        ax[r, 2].imshow(target_img, extent=ext, origin="lower", cmap="hot", vmin=0, vmax=target_img.max())
        ax[r, 2].set_title("target")
        ax[r, 3].plot(hist, lw=1.2, color="C0")
        ax[r, 3].set_xlim(0, ITERATIONS); ax[r, 3].set_ylim(0, 1.02); ax[r, 3].grid(alpha=0.3)
        ax[r, 3].set_xlabel("# of SPGD iterations"); ax[r, 3].set_ylabel("correlation")
        for a in ax[r, :3]:
            a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("SPGD feedback (slides' method) on the simulated amplifier: 300 W, whole-disk pump, re-tuned relay, shape error + hot spots")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=70); plt.close(fig)
    buf.seek(0); images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
images[-1].save(os.path.join(RESULTS, "spgd_feedback_last.png"))
images[0].save(os.path.join(RESULTS, "spgd_feedback.gif"), save_all=True, append_images=images[1:] + [images[-1]] * 12,
               duration=120, loop=0)
print("saved", os.path.join(RESULTS, "spgd_feedback.gif"), f"({len(images)} frames)")
