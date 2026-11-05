"""
xie_spgd.py - Step 22: the SPGD loop exactly as in Xie et al., Opt. Lett. 40, 1197 (2015)
("Phase correction for a distorted OAM beam using a Zernike-polynomials-based SPGD algorithm"),
run on the disk state of Steps 14/16 (300 W, hot spots, nominal shape error) in the current model.

The paper's loop (their Eqs. 2-4, Fig. 2), per iteration k:
    C1  = C(phi_k)                        one far-field camera frame
    s_n = random +-1 for each of N Zernike terms (N = 14 in the paper)
    C2  = C(phi_k + delta * sum_n s_n Z_n)  second frame          (delta ~ 0.01 in the paper)
    a_n <- a_n + eta * s_n * (C2 - C1)     one-sided update, no accept/reject (eta = 200 in the paper)
with the feedback signal C = integral of I(r) * I_ideal(r) over the FAR-FIELD intensity (their Eq. 2):
the camera image times the ideal image of the undistorted mode. Here it is normalised by the
ideal image's own integral so that C = 1 for a perfect frame at the same exposure; the Pearson
correlation of the slides (Step 14) is run too for comparison. 50-100 iterations sufficed in the
paper (turbulence, 14 terms); their C is the loop's own measure of success, we also log the true
field fidelity of the displayed pattern after every iteration.

Differences to Step 14 (spgd.py, "slides" rule): far-field camera instead of near-field, the paper's
intensity-overlap metric instead of Pearson correlation, 14 modes instead of 20, one-sided gradient
update instead of accept-or-scaled-reverse. delta is swept around the paper's value; eta is swept so
that the step eta * |C2 - C1| spans 0.01 .. 1 rad (the paper's 200 belongs to its unnormalised C).

Usage:  NN_RANGE=narrow python xie_spgd.py
Output: Results/22_xie_spgd/XIE_SPGD.md, xie_sweep.csv, xie_curves.png
"""
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("NN_RANGE", "narrow")
import spgd_test as st
from spgd import Camera, correlation, field_overlap, zernike_basis
from two_camera_correction import to_far
from seed_beam import results_dir

OUT = results_dir("22_xie_spgd")
G = st.G
N_TERMS = 14                                   # Zernike terms, as in the paper (Noll 2..15)
ITER_SWEEP, ITER_LONG = 100, 300               # the paper: 50-100 iterations suffice
DELTAS = (0.01, 0.05, 0.15)                    # perturbation per coefficient [rad]; paper ~0.01
ETAS = (3.0, 10.0, 30.0)                       # update gain; the paper's 200 belongs to its unnormalised C (camera counts):
                                               # here C is normalised, |C2 - C1| is 0.003 (delta 0.01) .. 0.035 (delta 0.15),
                                               # so eta 3 .. 30 gives steps of 0.01 .. 1 rad per coefficient (200 would give 0.6 .. 7 rad)


class FarFieldPlant:
    """The Step 14 plant seen through a far-field camera (f = 1 m lens, 25 um pixels, 12 bit)."""

    def __init__(self, plant):
        self.plant = plant
        self.cam = Camera(binning=1, blur=0.5, noise=plant.s["cam_noise"], bits=12, rng=np.random.default_rng(7))
        self.frames = 0

    def set_exposure(self, phase_map):
        """Fixed exposure from the uncorrected pattern (0.7 of full scale), as a lab camera would be set.
        Remembers the output power for the ideal image."""
        out = self.plant.output_nominal(phase_map)
        self.power = np.sum(np.abs(out) ** 2)
        self.cam._scale = None
        self.cam.image(to_far(out, self.plant.ref))
        self.cam._scale *= 0.7

    def ideal_image(self, target):
        """Noise-free camera image of the ideal mode carrying the output power, same exposure (the paper's I_ideal)."""
        noise, self.cam.noise = self.cam.noise, 0.0
        self.cam._scale /= 8                                           # the ideal mode is brighter than any frame: avoid clipping
        img = 8 * self.cam.image(to_far(target * np.sqrt(self.power / np.sum(np.abs(target) ** 2)), self.plant.ref))
        self.cam._scale *= 8
        self.cam.noise = noise
        return img

    def measure(self, phase_map):
        """One far-field camera frame of the displayed pattern (pump noise, pointing as in the scenario)."""
        pl = self.plant
        pl.set_state(pl.pump_now())
        self.frames += 1
        pl.frame += 1
        return self.cam.image(to_far(pl.amp.forward(pl.slm_field(phase_map)), pl.ref))


def paper_metric(image, ideal):
    """Eq. 2 of the paper: integral of I * I_ideal, normalised to 1 for I = I_ideal."""
    return float(np.sum(image * ideal) / np.sum(ideal * ideal))


def xie_loop(ff, base, basis, ideal, target, delta, eta, iterations, metric="paper", seed=1):
    """The paper's SPGD. Returns rows (frames, metric C1, true fidelity of the displayed pattern)."""
    rng = np.random.default_rng(seed)
    C = (lambda img: paper_metric(img, ideal)) if metric == "paper" else (lambda img: correlation(img, ideal))
    a = np.zeros(len(basis))
    phase = lambda a: base + np.tensordot(a, basis, axes=1)
    log = [(0, np.nan, field_overlap(ff.plant.output_nominal(phase(a)), target))]
    for _ in range(iterations):
        C1 = C(ff.measure(phase(a)))
        s = rng.choice([-1.0, 1.0], size=len(a))
        C2 = C(ff.measure(phase(a + delta * s)))
        a = a + eta * s * (C2 - C1)                                    # Eq. 4
        log.append((2 * (len(log)), C1, field_overlap(ff.plant.output_nominal(phase(a)), target)))
    return np.array(log)


def main():
    t0 = time.time()
    plant = st.Plant("static", 10000, seed=0)                          # the disk state of Steps 14/16
    ff = FarFieldPlant(plant)
    basis = zernike_basis(G, radius=3.0e-3, n_modes=N_TERMS)
    # the other methods on this very state (iteration_curves.py, Results/readme)
    other = {}
    try:
        for r in csv.DictReader(open(os.path.join(results_dir("readme"), "iterations.csv"))):
            other[r["target"]] = r
    except FileNotFoundError:
        pass

    rows, curves = [], {}
    for name, tgt in plant.targets.items():
        base = st.base_pattern(name)
        ff.set_exposure(base)
        ideal = ff.ideal_image(tgt)
        print(f"--- {name}: uncorrected fidelity {field_overlap(plant.output_nominal(base), tgt):.3f}", flush=True)
        best = None
        for delta in DELTAS:
            for eta in ETAS:
                plant.frame = 0
                log = xie_loop(ff, base, basis, ideal, tgt, delta, eta, ITER_SWEEP)
                F_end = log[-5:, 2].mean()                                 # mean of the last 5 iterations
                rows.append(dict(target=name, metric="paper", delta=delta, eta=eta, iterations=ITER_SWEEP, frames=int(log[-1, 0]),
                                 C_start=log[1, 1], C_end=np.nanmean(log[-5:, 1]), F_start=log[0, 2], F_end=F_end, F_best=log[:, 2].max()))
                print(f"delta {delta:.2f} eta {eta:4.0f}: C {log[1, 1]:.3f} -> {np.nanmean(log[-5:, 1]):.3f}, fidelity {log[0, 2]:.3f} -> {F_end:.3f} "
                      f"(best {log[:, 2].max():.3f}), {int(log[-1, 0])} frames, {time.time() - t0:.0f} s", flush=True)
                if best is None or F_end > best[0]:
                    best = (F_end, delta, eta)
        _, delta, eta = best
        print(f"best settings delta {delta}, eta {eta}; {ITER_LONG} iterations with both metrics", flush=True)
        for metric in ("paper", "pearson"):
            plant.frame = 0
            log = xie_loop(ff, base, basis, ideal, tgt, delta, eta, ITER_LONG, metric=metric)
            curves[(name, metric)] = log
            rows.append(dict(target=name, metric=metric, delta=delta, eta=eta, iterations=ITER_LONG, frames=int(log[-1, 0]),
                             C_start=log[1, 1], C_end=np.nanmean(log[-5:, 1]), F_start=log[0, 2], F_end=log[-5:, 2].mean(), F_best=log[:, 2].max()))
            print(f"{metric:8s}: C {log[1, 1]:.3f} -> {np.nanmean(log[-5:, 1]):.3f}, fidelity {log[0, 2]:.3f} -> {log[-5:, 2].mean():.3f} "
                  f"(best {log[:, 2].max():.3f}) after {int(log[-1, 0])} frames", flush=True)

    with open(os.path.join(OUT, "xie_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- table -------------------------------------------------------------------
    lines = ["# SPGD as in Xie et al. 2015 (Step 22)", "",
             "Disk state of Steps 14/16 (300 W, hot spots, nominal shape error), far-field camera, 14 Zernike terms,",
             "one-sided update a += eta s (C2 - C1), two frames per iteration. C = the loop's own feedback signal",
             "(paper: far-field intensity overlap with the ideal mode, normalised; pearson: image correlation of the",
             "slides). Fidelity = true field fidelity of the displayed pattern (mean of the last 5 iterations).", "",
             "| target | metric | delta | eta | iterations | frames | C start -> end | fidelity start -> end (best) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['target']} | {r['metric']} | {r['delta']} | {r['eta']:.0f} | {r['iterations']} | {r['frames']} | "
                     f"{r['C_start']:.3f} -> {r['C_end']:.3f} | {r['F_start']:.3f} -> {r['F_end']:.3f} ({r['F_best']:.3f}) |")
    if other:
        lines += ["", "Same state, other methods (Results/readme/iterations.csv):", "",
                  "| target | SPGD slides (Step 14) | response matrix (Step 16) | network, 2 frames | ceiling |", "|---|---|---|---|---|"]
        for name, r in other.items():
            lines.append(f"| {name} | {float(r['F_spgd']):.3f} ({r['frames_spgd']} frames) | {float(r['F_rm']):.3f} ({r['frames_rm']} frames) | "
                         f"{float(r['F_network']):.3f} | {float(r['F_ceiling']):.3f} |")
    with open(os.path.join(OUT, "XIE_SPGD.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figure ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for j, name in enumerate(plant.targets):
        A = ax[j]
        for metric, col in (("paper", "C3"), ("pearson", "C1")):
            log = curves[(name, metric)]
            A.plot(np.maximum(log[:, 0], 1), log[:, 2], color=col, lw=1.6, label=f"Xie SPGD, metric '{metric}': true fidelity")
            A.plot(np.maximum(log[1:, 0], 1), log[1:, 1], color=col, lw=1, ls=":", label=f"Xie SPGD, metric '{metric}': its own C")
        if name in other:
            A.plot([2], [float(other[name]["F_network"])], "*", color="C0", ms=17, label="neural network, one shot (2 frames)")
            A.axhline(float(other[name]["F_ceiling"]), color="k", ls="--", lw=1, label="ceiling (true disk state known)")
        A.set_xscale("log"); A.set_xlim(1, 700); A.set_ylim(0, 1)
        A.set_xlabel("camera frames"); A.set_title("Gaussian" if name == "gaussian" else "LG01 vortex"); A.grid(alpha=0.3, which="both")
    ax[0].set_ylabel("field fidelity with the target")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("SPGD of Xie et al. 2015 (far-field intensity overlap, 14 Zernike terms) on the Step 14/16 disk state")
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(OUT, "xie_curves.png"), dpi=110)
    print(f"saved {OUT} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
