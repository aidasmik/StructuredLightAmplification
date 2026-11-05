"""
iteration_curves.py - fidelity against camera frames for the iterative methods and the network,
on one and the same disk state (the static state of Steps 14/16: 300 W, hot spots, nominal shape
error; relay tuned to the hot nominal disk). Everything with the current model (absorbing boundaries).

    SPGD (slides)        one camera, correlation of the near-field image with the target image,
                         20 Zernike modes, slides update rule (Step 14). Its own metric is logged
                         too: it rises while the true fidelity does not.
    response matrix      two cameras, Gerchberg-Saxton field retrieval, measured complex Jacobian
                         of 28 Zernike modes, ridge least squares, one recalibration (Step 16)
    network              Step 18 network (narrow range), one shot from two camera frames
    ceiling              best phase-only pattern with the true disk state (projections + 100
                         gradient steps at 50 um)

Usage:  NN_RANGE=narrow python iteration_curves.py
Output: Results/readme/iterations.png, iterations.csv
"""
import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

os.environ.setdefault("NN_RANGE", "narrow")
import spgd_test as st
import two_camera_correction as tc
from spgd import SPGD, Camera, correlation, field_overlap, zernike_basis
import limits_benchmark as lb
from correction_benchmark import hot_spot_field
import nn_dataset as nd
import nn_train as nt
import nn_eval as ne
from seed_beam import results_dir

OUT = results_dir("readme")
G = st.G
SPGD_ITER = 200

plant = st.Plant("static", 1000, seed=0)                      # the disk state of Steps 14/16
sysm = nd.System()                                            # network-side model of the same system
phys = nt.Physics(np.load(os.path.join(nt.DATA, "meta.npz")))
ref = nt.load_reference()
net, a = ne.load_model("resnet_all")
th = lb.thermal(st.P0)
T_mean = th.solve(G, Q=th.heat_flux(G) * (1 + hot_spot_field(G, rms=0.1)))["T_mean"]
state = dict(opd=plant.opd(st.P0), T_mean=T_mean)
p = sysm.nominal_params()
pad = (G.n - nd.CROP_LABEL) // 2

curves, rows = {}, []
for t_id, (name, tgt) in enumerate(plant.targets.items()):
    print(f"--- {name}", flush=True)
    # SPGD as in the slides, true fidelity of the accepted pattern after every iteration
    plant.cam._scale = None; plant.frame = 0
    target_img = Camera(binning=int(round(50e-6 / G.dx)), blur=0.5, noise=0.0, bits=16).image(tgt)
    opt = SPGD(plant.measure, lambda img: correlation(img, target_img), zernike_basis(G, radius=3.0e-3, n_modes=20),
               sigma=0.2, gain=5.0, rule="slides", phi_base=st.base_pattern(name), seed=1)
    log = [(0, np.nan, field_overlap(plant.output_nominal(opt.phase()), tgt))]
    opt.run(SPGD_ITER, callback=lambda i, o: log.append((o.frames, o.history[-1], field_overlap(plant.output_nominal(o.phase()), tgt))))
    curves[(name, "spgd")] = np.array(log)
    print(f"SPGD: fidelity {log[0][2]:.3f} -> {log[-1][2]:.3f}, correlation {log[1][1]:.3f} -> {log[-1][1]:.3f} ({log[-1][0]} frames)", flush=True)
    # response matrix with two cameras
    cams = (Camera(binning=1, blur=0.5, noise=tc.CAM_NOISE, bits=12, rng=np.random.default_rng(3)),
            Camera(binning=1, blur=0.5, noise=tc.CAM_NOISE, bits=12, rng=np.random.default_rng(4)))
    rm = tc.ResponseMatrixCorrector(plant, tgt, st.base_pattern(name), *cams, zernike_basis(G, radius=3.0e-3, n_modes=28),
                                    phase_init_model=tc.model_amplifier(plant))
    h = rm.run(6, recalibrate_at=3, verbose=False)
    F = np.array([x["fidelity_shown"] for x in h])
    fr = rm.frames_calibration + 2 * np.arange(len(F)) + 4 * 28 * (np.arange(len(F)) > 3)
    curves[(name, "rm")] = np.stack([np.r_[0, fr], np.r_[log[0][2], F]], 1)
    print(f"response matrix: {log[0][2]:.3f} -> {F[-1]:.3f} ({rm.frames} frames)", flush=True)
    # network, one shot, on the same disk state
    out0 = sysm.output(sysm.base[name], state, p)
    near, far = sysm.frames(out0, p)
    x = nt.make_input(near[None], far[None], sysm.temperature_points(T_mean)[None], [st.P0], np.zeros((1, 128, 128)), [t_id], ref, a["inputs"], a["formulation"])
    with torch.no_grad():
        corr = nt.full_correction(net(x), torch.as_tensor([t_id], device=nt.DEV), ref, a["formulation"])[0].cpu().numpy()
    F_net = field_overlap(plant.output_nominal(sysm.base[name] + np.pad(corr, pad)), tgt)
    # ceiling
    pat, _ = sysm.optimum(name, state, p)
    pat = ne.refine_optimum(phys, pat, sysm.record(state, p), t_id, sysm.base[name], 100)
    F_opt = field_overlap(plant.output_nominal(pat), tgt)
    print(f"network {F_net:.3f} (2 frames), ceiling {F_opt:.3f}", flush=True)
    rows.append(dict(target=name, F_uncorrected=log[0][2], F_spgd=log[-1][2], frames_spgd=log[-1][0], F_rm=F[-1], F_rm_best=F.max(),
                     frames_rm=rm.frames, F_network=F_net, frames_network=2, F_ceiling=F_opt))
    curves[(name, "net")], curves[(name, "opt")] = F_net, F_opt

with open(os.path.join(OUT, "iterations.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for j, name in enumerate(plant.targets):
    A = ax[j]
    s, r = curves[(name, "spgd")], curves[(name, "rm")]
    A.plot(np.maximum(s[:, 0], 1), s[:, 2], color="C3", lw=1.6, label="SPGD as in the slides (1 camera): true fidelity")
    A.plot(np.maximum(s[1:, 0], 1), s[1:, 1], color="C3", lw=1, ls=":", label="SPGD: its own metric (image correlation)")
    A.plot(np.maximum(r[:, 0], 1), r[:, 1], "s-", color="C1", ms=4, lw=1.6, label="response matrix (2 cameras, calibration incl.)")
    A.plot([2], [curves[(name, "net")]], "*", color="C0", ms=17, label="neural network, one shot (2 frames)")
    A.axhline(curves[(name, "opt")], color="k", ls="--", lw=1, label="ceiling (true disk state known)")
    A.set_xscale("log"); A.set_xlim(1, 600); A.set_ylim(0, 1.0)
    A.set_xlabel("camera frames"); A.set_title("Gaussian" if name == "gaussian" else "LG01 vortex"); A.grid(alpha=0.3, which="both")
ax[0].set_ylabel("field fidelity with the target")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.suptitle("Iterative correction vs one-shot network on the same disk state (300 W, hot spots, shape error)")
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(OUT, "iterations.png"), dpi=110)
print("saved", OUT)
