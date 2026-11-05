"""
spgd_test.py - How well does the SPGD feedback of the slides correct the
amplifier, and does it survive a minimal "experimental zone": slight
temperature changes of the disk during the optimisation, camera noise, SLM
phase flicker and beam-pointing jitter?

Plant: the agreed operating point (pump over the whole disk, 300 W, relay
re-tuned to the hot disk) with a static shape error and 10 % hot spots as the
distortion to find. The disk OPD can change from frame to frame:
    OPD(t) = shape error + thermal OPD(300 W) * P(t) / 300 W
with P(t) following the scenario (white noise and/or a slow drift).

Scenarios
    static      : nothing moves, 1 % camera noise                (ideal lab)
    temp_noise  : pump / temperature white noise 1 % rms per frame
    temp_drift  : pump drifts +2 % over the run (the relay was tuned for 300 W)
                  plus 0.5 % noise
    jitter      : pointing jitter 10 urad rms, SLM flicker 0.03 rad rms, 2 % camera noise
    all         : everything together

For every scenario and target (Gaussian, vortex LG01) the loop runs N SPGD
iterations and reports
    correlation (CCD metric) start -> best,
    true field fidelity of the found pattern at the nominal state (P = 300 W,
    no jitter) and at the final drifted state,
    the physics-optimal phase-only fidelity of the benchmark as the reference,
    verdict: PASS if the SPGD fidelity reaches 90 % of the physics optimum.

usage: python spgd_test.py reference          (optimal patterns, once)
       python spgd_test.py <scenario> [iterations]
       python spgd_test.py plot
"""
import os
import sys
import csv
import warnings
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import limits_benchmark as lb
import thin_disk_system as tds
import slm
import correction as co
from correction_benchmark import hot_spot_field
from seed_beam import Grid, results_dir
from spgd import SPGD, Camera, correlation, field_overlap, zernike_basis
from gpu import make_amplifier

warnings.filterwarnings("ignore")
RESULTS = results_dir("14_spgd")
G = Grid(**tds.SIM_GRID)                    # fast grid (the limits benchmark itself uses the fine one)
P0 = 300.0
TARGETS = ("gaussian", "LG01")
SCENARIOS = {
    #            pump noise [rms]  drift [fraction over the run]  pointing [rad rms]  SLM flicker [rad rms]  camera noise
    "static":     dict(noise=0.0,   drift=0.0,  pointing=0.0,   flicker=0.0,  cam_noise=0.01),
    "temp_noise": dict(noise=0.01,  drift=0.0,  pointing=0.0,   flicker=0.0,  cam_noise=0.01),
    "temp_drift": dict(noise=0.005, drift=0.02, pointing=0.0,   flicker=0.0,  cam_noise=0.01),
    "jitter":     dict(noise=0.0,   drift=0.0,  pointing=10e-6, flicker=0.03, cam_noise=0.02),
    "all":        dict(noise=0.01,  drift=0.02, pointing=10e-6, flicker=0.03, cam_noise=0.02),
}


# --------------------------------------------------------------------------
# Plant with a time-dependent disk
# --------------------------------------------------------------------------
class Plant:
    def __init__(self, scenario, n_frames, seed=0):
        self.s = SCENARIOS[scenario]
        self.rng = np.random.default_rng(seed)
        self.n_frames = max(n_frames, 1)
        self.k0 = 2 * np.pi / tds.LAM
        th = lb.thermal(P0)
        self.opd_thermal = th.solve(G, Q=th.heat_flux(G) * (1 + hot_spot_field(G, rms=0.1)))["opd"]
        self.opd_shape = lb.shape_error(1, G)
        f_th, _ = lb.per_pass_lens(lb.thermal(P0).opd(G), G)
        tds.DISK.thermal_opd = None
        tds.MIRROR_ROC.update(lb.COLD_ROC)
        self.ref, _ = tds.trace(slm.flat(), G)                       # cold Gaussian output
        self.targets = {k: v for k, v in co.target_modes(G, self.ref).items() if k in TARGETS}
        tds.DISK.thermal_opd = self.opd(P0)
        tds.MIRROR_ROC.update(lb.retuned_mirrors(f_th))
        self.amp = make_amplifier(G)                   # GPU when available; disk screen updated per frame
        self.fin_beam = co.field_at_slm(G)
        self.fin = self.fin_beam.field
        self.cam = Camera(binning=int(round(50e-6 / G.dx)), blur=0.5, noise=self.s["cam_noise"], bits=12, rng=np.random.default_rng(seed + 1))
        self.frame = 0
        self.P_log = []

    def opd(self, P):
        return self.opd_shape + self.opd_thermal * P / P0

    def pump_now(self):
        """Pump power of the current frame: drift + white noise."""
        t = self.frame / self.n_frames
        return P0 * (1 + self.s["drift"] * t + self.s["noise"] * self.rng.standard_normal())

    def set_state(self, P, nominal=False):
        self.amp.set_thermal_opd(self.opd(P))
        self.jitter = (0.0, 0.0) if nominal else tuple(self.s["pointing"] * self.rng.standard_normal(2))

    def slm_field(self, phase_map, nominal=False):
        phi = phase_map
        if not nominal and self.s["flicker"] > 0:
            phi = phi + self.s["flicker"] * 8 * gaussian_filter(self.rng.standard_normal((G.n, G.n)), 8)   # smooth flicker, ~rms as given
        beam = self.fin_beam
        if any(self.jitter):
            tilt = self.k0 * (self.jitter[0] * G.X + self.jitter[1] * G.Y)
            beam = replace(beam, field=self.fin * np.exp(1j * tilt))
        return tds.SLM_DEVICE.modulate(beam, slm.map_pattern(phi, G), bias=tds.SLM_BIAS).field

    def measure(self, phase_map):
        """One camera frame with the disk, pointing and SLM as they are right now."""
        P = self.pump_now()
        self.P_log.append(P)
        self.set_state(P)
        img = self.cam.image(self.amp.forward(self.slm_field(phase_map)))
        self.frame += 1
        self.last_image = img                           # image of the last displayed pattern (for recordings)
        return img

    def output_nominal(self, phase_map, P=P0):
        """Noise-free output field for the pattern, disk at power P, no jitter."""
        self.set_state(P, nominal=True)
        return self.amp.forward(self.slm_field(phase_map, nominal=True))


def base_pattern(name):
    return np.zeros((G.n, G.n)) if name == "gaussian" else slm.vortex(1)(G.X, G.Y)


# --------------------------------------------------------------------------
def run_reference():
    """Physics-optimal phase-only patterns for the nominal state (benchmark method)."""
    plant = Plant("static", 1)
    plant.set_state(P0, nominal=True)
    for name, tgt in plant.targets.items():
        pattern, hist = co.optimise_pattern(plant.amp, plant.fin, tgt, 12)
        real = plant.output_nominal(pattern)
        F = field_overlap(real, tgt)
        np.save(os.path.join(RESULTS, f"reference_pattern_{name}.npy"), pattern)
        with open(os.path.join(RESULTS, f"reference_{name}.txt"), "w") as fh:
            fh.write(f"{F}\n")
        print(f"reference {name}: physics-optimal phase-only fidelity {hist[-1]:.3f} (ideal SLM), {F:.3f} on the real LCOS", flush=True)


def run_scenario(scenario, iterations):
    n_frames = int(iterations * 1.7)
    plant = Plant(scenario, n_frames, seed=hash(scenario) % 1000)
    basis = zernike_basis(G, radius=3.0e-3, n_modes=20)
    target_cam = Camera(binning=int(round(50e-6 / G.dx)), blur=0.5, noise=0.0, bits=16)
    rows = []
    for name, tgt in plant.targets.items():
        plant.cam._scale = None
        plant.frame = 0
        target_img = target_cam.image(tgt)
        opt = SPGD(plant.measure, lambda img: correlation(img, target_img), basis, sigma=0.2, gain=5.0,
                   rule="slides", phi_base=base_pattern(name), seed=1)
        F_start = field_overlap(plant.output_nominal(opt.phase()), tgt)
        hist = opt.run(iterations, verbose=20)
        phase_best = opt.phase(opt.best[1])
        F_nominal = field_overlap(plant.output_nominal(phase_best), tgt)
        P_end = P0 * (1 + plant.s["drift"])
        F_end = field_overlap(plant.output_nominal(phase_best, P_end), tgt)
        try:
            F_ref = float(open(os.path.join(RESULTS, f"reference_{name}.txt")).read())
        except FileNotFoundError:
            F_ref = np.nan
        verdict = "PASS" if F_nominal >= 0.9 * F_ref else "FAIL"
        rows.append(dict(scenario=scenario, target=name, iterations=iterations, frames=opt.frames,
                         corr_start=hist[0], corr_best=opt.best[0], F_start=F_start, F_nominal=F_nominal,
                         F_end_state=F_end, F_reference=F_ref, verdict=verdict))
        print(f"{scenario:10s} {name:8s}: correlation {hist[0]:.3f} -> {opt.best[0]:.3f}, fidelity {F_start:.3f} -> "
              f"{F_nominal:.3f} (at the final state {F_end:.3f}); physics optimum {F_ref:.3f}  {verdict}", flush=True)
        np.save(os.path.join(RESULTS, f"history_{scenario}_{name}.npy"), hist)
        np.save(os.path.join(RESULTS, f"pump_{scenario}_{name}.npy"), np.array(plant.P_log))
        # slide-style figure
        ext = np.array([-1, 1, -1, 1]) * G.size / 2 * 1e3
        before = plant.cam.image(plant.output_nominal(opt.phase(np.zeros(len(basis)))))
        after = plant.cam.image(plant.output_nominal(phase_best))
        fig, ax = plt.subplots(2, 3, figsize=(14, 8.5))
        ax[0, 0].imshow(np.angle(np.exp(1j * (opt.phase(np.zeros(len(basis))) + tds.SLM_BIAS))), extent=ext, origin="lower", cmap="gray")
        ax[0, 0].set_title("SLM phase, uncompensated")
        ax[0, 1].imshow(np.angle(np.exp(1j * (phase_best + tds.SLM_BIAS))), extent=ext, origin="lower", cmap="gray")
        ax[0, 1].set_title("SLM phase, compensated (SPGD)")
        ax[0, 2].plot(hist, lw=1); ax[0, 2].set_xlabel("# of SPGD iterations"); ax[0, 2].set_ylabel("correlation")
        ax[0, 2].grid(alpha=0.3); ax[0, 2].set_ylim(0, 1.02); ax[0, 2].set_title(f"{scenario}: {name}")
        for a, img, title in ((ax[1, 0], before, f"CCD, uncompensated (F = {F_start:.2f})"),
                              (ax[1, 1], after, f"CCD, compensated (C = {opt.best[0]:.2f}, F = {F_nominal:.2f})"),
                              (ax[1, 2], target_img, "target")):
            a.imshow(img, extent=ext, origin="lower", cmap="hot", vmin=0, vmax=max(img.max(), 1e-9)); a.set_title(title)
        for a in ax.ravel()[[0, 1, 3, 4, 5]]:
            a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"SPGD on the simulated amplifier, scenario '{scenario}', target {name}")
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f"spgd_{scenario}_{name}.png"), dpi=100); plt.close(fig)
    with open(os.path.join(RESULTS, f"result_{scenario}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def plot_all():
    rows = []
    for sc in SCENARIOS:
        f = os.path.join(RESULTS, f"result_{sc}.csv")
        if os.path.exists(f):
            rows += list(csv.DictReader(open(f)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for a, name in zip(ax, TARGETS):
        for i, sc in enumerate(SCENARIOS):
            f = os.path.join(RESULTS, f"history_{sc}_{name}.npy")
            if os.path.exists(f):
                a.plot(np.load(f), lw=1, color=f"C{i}", label=sc)
        a.set_title(f"target {name}"); a.set_xlabel("# of SPGD iterations"); a.grid(alpha=0.3); a.set_ylim(0, 1.02)
    ax[0].set_ylabel("correlation (CCD metric)"); ax[0].legend(fontsize=8)
    fig.suptitle("SPGD convergence under experimental disturbances (300 W, whole-disk pump, re-tuned relay)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "spgd_test_convergence.png"), dpi=110)
    lines = ["# SPGD test (Step 14)\n",
             "| scenario | target | frames | correlation start -> best | fidelity start -> SPGD | at final state | physics optimum | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['target']} | {r['frames']} | {float(r['corr_start']):.2f} -> {float(r['corr_best']):.2f} | "
                     f"{float(r['F_start']):.2f} -> {float(r['F_nominal']):.2f} | {float(r['F_end_state']):.2f} | {float(r['F_reference']):.2f} | {r['verdict']} |")
    open(os.path.join(RESULTS, "SPGD_TEST.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if what == "reference":
        run_reference()
    elif what == "plot":
        plot_all()
    else:
        run_scenario(what, int(sys.argv[2]) if len(sys.argv) > 2 else 120)
