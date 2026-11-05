"""
spgd_tuning.py - Why does SPGD on the CCD correlation stop far below the
physics optimum, and what fixes it? Same plant as spgd_test.py (static
scenario), Gaussian and vortex targets, 200 iterations each:

    baseline      slides rule, sigma 0.2, gain 5, near-field correlation (as in spgd_test)
    small_step    slides rule, sigma 0.1, gain 5
    classic       two-sided rule, sigma 0.1, gain 2
    two_plane     slides rule, sigma 0.1; metric = mean correlation of the
                  near-field CCD and a second camera in the focal plane of a
                  f = 1 m lens (the far field fixes the wavefront)
    oracle        slides rule, sigma 0.1; metric = true field overlap
                  (what a complete field measurement would give: the limit of
                  the algorithm itself, independent of the metric)

Records the CCD metric and the true field fidelity every 5 iterations.
Output: Results/14_spgd/spgd_tuning.png, spgd_tuning.csv
"""
import os
import csv
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seed_beam import results_dir
from propagation import propagate
from optics import ideal_lens
from dataclasses import replace
from spgd import SPGD, Camera, correlation, field_overlap, zernike_basis
import spgd_test as st
import limits_benchmark as lb
import correction as co

warnings.filterwarnings("ignore")
ITER = 200
RESULTS = results_dir("14_spgd")
G = st.G
plant = st.Plant("static", int(ITER * 2.5), seed=0)
basis = zernike_basis(G, radius=3.0e-3, n_modes=20)
def clean_image(field):
    """Noise-free camera image, exposure scaled to this image alone (a fresh camera each time)."""
    return Camera(binning=int(round(50e-6 / G.dx)), blur=0.5, noise=0.0, bits=16).image(field)
F_LENS = 1.0


def far_field(field):
    """Output field -> intensity in the focal plane of a f = 1 m lens."""
    b = replace(plant.ref, field=field)
    return propagate(ideal_lens(b, F_LENS), F_LENS, "air").field


CONFIGS = {
    # one camera only:
    "far_only":   dict(rule="slides", sigma=0.1, gain=5.0, metric="far"),         # the one camera sits in the focal plane
    "diversity":  dict(rule="slides", sigma=0.1, gain=5.0, metric="diversity"),   # one near-field camera, two frames per
                                                                                  # evaluation: pattern, and pattern + known defocus on the SLM
    "shape_l1":   dict(rule="slides", sigma=0.1, gain=5.0, metric="shape_l1"),    # one near-field camera, shape metric:
                                                                                  # 1 - 0.5 * sum |I/sum I - T/sum T| (penalises missing parts)
    "baseline":   dict(rule="slides", sigma=0.2, gain=5.0, metric="near"),
    "small_step": dict(rule="slides", sigma=0.1, gain=5.0, metric="near"),
    "classic":    dict(rule="classic", sigma=0.1, gain=2.0, metric="near"),
    "two_plane":  dict(rule="slides", sigma=0.1, gain=5.0, metric="two_plane"),
    "oracle":     dict(rule="slides", sigma=0.1, gain=5.0, metric="field"),
}

import sys
if len(sys.argv) > 1:                                       # optional: run only the named configs
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in sys.argv[1:]}
# model of the system for the diversity target: the nominal thermal lens only (what the design knows),
# the pattern that produces the target in that model, and its image with the known defocus added
DIVERSITY = 1.5 * basis[2]                                   # defocus (Noll 4), 1.5 rad rms over the control radius
model_opd = plant.opd_thermal * 0 + (lambda f: -G.R ** 2 / (2 * f))(lb.per_pass_lens(lb.thermal(st.P0).opd(G), G)[0])
plant.amp.set_thermal_opd(model_opd)
model_patterns = {n: co.optimise_pattern(plant.amp, plant.fin, t, 12)[0] for n, t in plant.targets.items()}
div_targets = {}
for n, t in plant.targets.items():
    out_div = plant.amp.forward(co.ideal_slm(plant.fin, model_patterns[n] + DIVERSITY))
    div_targets[n] = clean_image(out_div)
plant.amp.set_thermal_opd(plant.opd(st.P0))

rows, curves = [], {}
for name, tgt in plant.targets.items():
    tgt_near = clean_image(tgt)
    tgt_div = div_targets[name]
    tgt_far = Camera(binning=1, blur=0.5, noise=0.0, bits=16).image(far_field(tgt))
    far_cam = Camera(binning=1, blur=0.5, noise=0.01, bits=12, rng=np.random.default_rng(7))   # 12.5 um pixels: the focal spot is ~0.2 mm
    for cfg_name, cfg in CONFIGS.items():
        plant.cam._scale = None; far_cam._scale = None
        plant.frame = 0

        def measure(phase_map, cfg=cfg):
            """One frame: what the chosen metric can see."""
            plant.set_state(plant.pump_now())
            out = plant.amp.forward(plant.slm_field(phase_map))
            plant.frame += 1
            if cfg["metric"] in ("near", "shape_l1"):
                return plant.cam.image(out)
            if cfg["metric"] == "far":
                return far_cam.image(far_field(out))
            if cfg["metric"] == "two_plane":
                return plant.cam.image(out), far_cam.image(far_field(out))
            if cfg["metric"] == "diversity":                             # second frame: same camera, defocus added on the SLM
                plant.set_state(plant.pump_now())
                out2 = plant.amp.forward(plant.slm_field(phase_map + DIVERSITY))
                plant.frame += 1
                return plant.cam.image(out), plant.cam.image(out2)
            return out                                                   # oracle: the field itself

        def metric(m, cfg=cfg, tgt=tgt, tgt_near=tgt_near, tgt_far=tgt_far, tgt_div=tgt_div):
            if cfg["metric"] == "near":
                return correlation(m, tgt_near)
            if cfg["metric"] == "shape_l1":
                a, b = np.clip(m, 0, None), tgt_near
                return 1 - 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()
            if cfg["metric"] == "far":
                return correlation(m, tgt_far)
            if cfg["metric"] == "two_plane":
                return 0.5 * (correlation(m[0], tgt_near) + correlation(m[1], tgt_far))
            if cfg["metric"] == "diversity":
                return 0.5 * (correlation(m[0], tgt_near) + correlation(m[1], tgt_div))
            return field_overlap(m, tgt)

        opt = SPGD(measure, metric, basis, sigma=cfg["sigma"], gain=cfg["gain"], rule=cfg["rule"],
                   phi_base=st.base_pattern(name), seed=1)
        fid = []

        def record(i, o, fid=fid, tgt=tgt):
            if i % 5 == 0 or i == ITER - 1:
                fid.append((i, field_overlap(plant.output_nominal(o.phase()), tgt),
                            field_overlap(plant.output_nominal(o.phase(o.best[1])), tgt)))

        hist = opt.run(ITER, callback=record)
        fid = np.array(fid)
        out_best = plant.output_nominal(opt.phase(opt.best[1]))
        F_best = field_overlap(out_best, tgt)
        C_near = correlation(clean_image(out_best), tgt_near)          # shape at the camera, same for all variants
        rows.append(dict(target=name, config=cfg_name, frames=opt.frames, metric_start=hist[0], metric_best=opt.best[0],
                         F_start=fid[0, 1], F_best_pattern=F_best, F_max_seen=fid[:, 1].max(), near_corr_best=C_near))
        curves[(name, cfg_name)] = (hist, fid)
        print(f"{name:8s} {cfg_name:10s}: metric {hist[0]:.3f} -> {opt.best[0]:.3f} ({opt.frames} frames); "
              f"true fidelity of the best pattern {F_best:.3f} (near-field shape correlation {C_near:.3f}), highest fidelity on the way {fid[:, 1].max():.3f}", flush=True)

with open(os.path.join(RESULTS, "spgd_tuning" + ("_" + "_".join(CONFIGS) if len(sys.argv) > 1 else "") + ".csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for r, name in enumerate(plant.targets):
    for i, cfg_name in enumerate(CONFIGS):
        hist, fid = curves[(name, cfg_name)]
        ax[r, 0].plot(hist, lw=1, color=f"C{i}", alpha=0.8, label=cfg_name)
        ax[r, 1].plot(fid[:, 0], fid[:, 2], "-", color=f"C{i}", lw=1.5, label=cfg_name)
    F_ref = float(open(os.path.join(RESULTS, f"reference_{name}.txt")).read())
    ax[r, 1].axhline(F_ref, color="k", ls="--", lw=1, label="physics optimum (phase-only)")
    ax[r, 0].set_title(f"{name}: metric seen by the loop"); ax[r, 1].set_title(f"{name}: true field fidelity of the best pattern so far")
    for a in ax[r]:
        a.set_xlabel("# of SPGD iterations"); a.set_ylim(0, 1.02); a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=8); ax[0, 1].legend(fontsize=8)
fig.suptitle("SPGD variants on the simulated amplifier (static scenario): step size, update rule and metric")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "spgd_tuning" + ("_" + "_".join(CONFIGS) if len(sys.argv) > 1 else "") + ".png"), dpi=110)
print("saved", os.path.join(RESULTS, "spgd_tuning.png"))
