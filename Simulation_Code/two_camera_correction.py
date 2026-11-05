"""
two_camera_correction.py - Step 16: iterative phase correction with two
cameras (near field + far field), the classical baselines that the neural
network will be compared with.

Both algorithms measure the complex output field from the two intensities
(Gerchberg-Saxton between the camera planes, started from the model's phase
or from the previous estimate) and then update the SLM pattern:

A. model-based (needs an accurate model of the amplifier at run time)
       dE_slm = A_model^dagger (E_target - E_meas),  phase-only projection.
   With the design model (retuned relay + nominal lens, no knowledge of the
   shape error / hot spots / doping) the update direction is poor and the
   loop crawls; with the true operator it converges in one step. Kept as a
   reference for what a perfect digital twin would give.

B. calibrated response matrix (classical adaptive optics, no model needed)
   1. calibration: poke every mode of a Zernike basis by +-eps on the SLM and
      measure the complex output field change -> Jacobian J (4 frames per mode);
   2. iteration: measure E_meas (2 frames), residual r = E_target - E_meas,
      solve the ridge least-squares problem  min |J a - r|^2 + lambda |a|^2,
      add sum_j a_j Z_j to the pattern; repeat a few times (and recalibrate
      once if the correction is large, since J is a linearisation).
   This learns the real amplifier's response, like the network does from
   data, but only around the current state and only in the chosen basis.

Cameras: 12-bit, 25 um pixels, 0.3 % rms noise (frame-averaged), dark-level
subtraction and 3-sigma threshold before the square root.
Plant and scenarios as in spgd_test.py. Output: Results/16_two_camera/.
"""
import os
import csv
import warnings
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

import thin_disk_system as tds
import correction as co
import limits_benchmark as lb
import slm
from gpu import make_amplifier
from optics import ideal_lens, _flip_x
from propagation import propagate
from seed_beam import results_dir
from spgd import Camera, correlation, field_overlap, zernike_basis
import spgd_test as st

warnings.filterwarnings("ignore")
RESULTS = results_dir("16_two_camera")
G = st.G
F_LENS = 1.0
ITERATIONS = 15
GAMMA = 1.0
GS_ITER = 20
CAM_NOISE = 0.003                # rms camera noise, fraction of full scale (averaged frames)


# --------------------------------------------------------------------------
# Propagation between the two camera planes (lens f, camera at its focus)
# --------------------------------------------------------------------------
def to_far(field, ref):
    return propagate(ideal_lens(replace(ref, field=field), F_LENS), F_LENS, "air").field


def to_near(field_far, ref):
    b = propagate(replace(ref, field=field_far), -F_LENS, "air")
    return ideal_lens(b, -F_LENS).field


def clean_camera(img, n_sigma=3.0):
    """Dark-level subtraction and noise threshold, estimated from the frame border
    (standard camera pre-processing; without it the noise floor of the whole
    frame carries as much amplitude as the beam after the square root)."""
    k = max(img.shape[0] // 10, 2)
    border = np.concatenate([img[:k].ravel(), img[-k:].ravel(), img[:, :k].ravel(), img[:, -k:].ravel()])
    bg, sigma = np.median(border), border.std()
    return np.clip(img - bg - n_sigma * sigma, 0, None)


def upsample(img, binning):
    """Camera image (binned pixels) back onto the simulation grid, non-negative."""
    img = clean_camera(img)
    full = img if binning == 1 else zoom(img, binning, order=1)
    out = np.zeros((G.n, G.n))
    n = min(G.n, full.shape[0])
    out[:n, :n] = full[:n, :n]
    return np.clip(out, 0, None)


def retrieve_field(I_near, I_far, phase_init, ref, n_iter=GS_ITER):
    """Gerchberg-Saxton between the near- and far-field intensities.
    Returns the complex near-field (output) estimate."""
    a_near = np.sqrt(I_near)
    a_far = np.sqrt(I_far)
    E = a_near * np.exp(1j * phase_init)
    for _ in range(n_iter):
        F = to_far(E, ref)
        scale = np.sqrt(np.sum(np.abs(F) ** 2) / (np.sum(a_far ** 2) + 1e-30))
        F = scale * a_far * np.exp(1j * np.angle(F))
        E = to_near(F, ref)
        E = a_near * np.exp(1j * np.angle(E))
    return E


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
class TwoCameraCorrector:
    def __init__(self, plant, model_amp, target, phi_base, near_cam, far_cam, gamma=GAMMA, gs_iter=GS_ITER, oracle=False):
        self.plant, self.model, self.tgt = plant, model_amp, target / np.sqrt(np.vdot(target, target).real)
        self.phi = phi_base.copy()
        self.near_cam, self.far_cam, self.gamma = near_cam, far_cam, gamma
        self.gs_iter, self.oracle = gs_iter, oracle          # oracle: use the true output field (tests the update rule alone)
        self.frames = 0
        self.history = []

    def grab(self, phase_map):
        """Two camera frames of the real plant for this pattern."""
        p = self.plant
        p.set_state(p.pump_now())
        out = p.amp.forward(p.slm_field(phase_map))
        self.frames += 2
        return self.near_cam.image(out), self.far_cam.image(to_far(out, p.ref)), out

    def step(self):
        p = self.plant
        I_n, I_f, out_true = self.grab(self.phi)
        # model prediction for this pattern (ideal SLM in the model, same pattern)
        E_slm = co.ideal_slm(p.fin, self.phi)
        E_model = self.model.forward(E_slm)
        # measured field: amplitude from the camera, phase retrieved, power from the model
        In = upsample(I_n, self.near_cam.binning); If = upsample(I_f, self.far_cam.binning)
        init = self.last_phase if getattr(self, "last_phase", None) is not None else np.angle(E_model)
        E_meas = out_true.copy() if self.oracle else retrieve_field(In, If, init, p.ref, n_iter=self.gs_iter)
        self.last_phase = np.angle(E_meas)
        E_meas *= np.sqrt(np.vdot(E_model, E_model).real / (np.vdot(E_meas, E_meas).real + 1e-30))
        # a camera measurement has no absolute phase: reference it to the model prediction
        E_meas *= np.exp(-1j * np.angle(np.vdot(E_model, E_meas)))
        # target with the same power and the free global phase aligned
        P = np.vdot(E_meas, E_meas).real
        align = np.exp(1j * np.angle(np.vdot(self.tgt, E_meas)))
        E_t = self.tgt * align * np.sqrt(P)
        # back-project the error with the design model and project onto a phase-only pattern
        dE = self.model.adjoint(E_t - E_meas)
        eta = np.vdot(E_model, E_model).real / np.vdot(E_slm, E_slm).real      # power throughput of the model, A^dagger A ~ eta
        E_new = E_slm + self.gamma / eta * dE
        self.phi = np.angle(_flip_x(E_new) * np.conj(p.fin))
        rec = dict(near_corr=correlation(I_n, self.near_target), far_corr=correlation(I_f, self.far_target),
                   retrieval=field_overlap(E_meas, out_true), fidelity_shown=field_overlap(out_true, self.tgt))
        self.history.append(rec)
        return rec

    def run(self, iterations, verbose=True):
        for i in range(iterations):
            r = self.step()
            if verbose:
                print(f"  iteration {i:2d} ({self.frames:3d} frames): fidelity of the displayed pattern {r['fidelity_shown']:.3f}, "
                      f"near corr {r['near_corr']:.3f}, far corr {r['far_corr']:.3f}, retrieval accuracy {r['retrieval']:.3f}", flush=True)
        # final: evaluate the last pattern
        I_n, I_f, out = self.grab(self.phi)
        self.history.append(dict(near_corr=correlation(I_n, self.near_target), far_corr=correlation(I_f, self.far_target),
                                 retrieval=np.nan, fidelity_shown=field_overlap(out, self.tgt)))
        return self.history


class ResponseMatrixCorrector:
    """Classical AO: measured complex-field response of each basis mode, then
    ridge least squares on the measured residual. No amplifier model needed."""

    def __init__(self, plant, target, phi_base, near_cam, far_cam, basis, eps=0.3, ridge=0.05,
                 gs_iter=GS_ITER, oracle=False, phase_init_model=None):
        self.plant, self.tgt = plant, target / np.sqrt(np.vdot(target, target).real)
        self.phi = phi_base.copy()
        self.near_cam, self.far_cam, self.basis, self.eps, self.ridge = near_cam, far_cam, basis, eps, ridge
        self.gs_iter, self.oracle, self.model = gs_iter, oracle, phase_init_model
        self.frames, self.history, self.last_phase = 0, [], None
        self.mask = np.abs(target) > 1e-2 * np.abs(target).max()           # pixels used in the fit

    def measure_field(self, phase_map):
        """Two frames -> complex output field, global phase referenced to the target."""
        p = self.plant
        p.set_state(p.pump_now())
        out = p.amp.forward(p.slm_field(phase_map))
        self.frames += 2
        if self.oracle:
            E = out.copy()
        else:
            In = upsample(self.near_cam.image(out), self.near_cam.binning)
            If = upsample(self.far_cam.image(to_far(out, p.ref)), self.far_cam.binning)
            if self.last_phase is None:
                init = np.angle(self.model.forward(co.ideal_slm(p.fin, phase_map))) if self.model is not None else np.zeros_like(In)
            else:
                init = self.last_phase
            E = retrieve_field(In, If, init, p.ref, n_iter=self.gs_iter)
            self.last_phase = np.angle(E)
        E *= np.exp(-1j * np.angle(np.vdot(self.tgt, E)))                  # phase reference: the target
        return E, out

    def calibrate(self):
        """Jacobian dE/da_j by +-eps pokes of every mode (4 frames per mode)."""
        cols = []
        for Z in self.basis:
            E_p, _ = self.measure_field(self.phi + self.eps * Z)
            E_m, _ = self.measure_field(self.phi - self.eps * Z)
            cols.append(((E_p - E_m) / (2 * self.eps))[self.mask])
        self.J = np.array(cols).T                                            # (pixels, modes)
        JtJ = self.J.conj().T @ self.J
        self.solve = np.linalg.inv(JtJ + self.ridge * np.trace(JtJ).real / len(self.basis) * np.eye(len(self.basis))) @ self.J.conj().T

    def step(self, gain=1.0):
        E, out = self.measure_field(self.phi)
        P = np.vdot(E, E).real
        r = (self.tgt * np.sqrt(P) - E)[self.mask]
        residual = np.vdot(r, r).real / P                                    # measured error, what the lab can see
        if residual < getattr(self, "best_residual", np.inf):                # keep the best pattern by the measured residual
            self.best_residual, self.best_phi = residual, self.phi.copy()
        a = gain * np.real(self.solve @ r)                                   # phase modes: real coefficients
        self.phi = self.phi + np.tensordot(a, self.basis, axes=1)
        rec = dict(fidelity_shown=field_overlap(out, self.tgt), retrieval=field_overlap(E, out), step_rms=np.sqrt(np.mean(a ** 2)),
                   residual=residual)
        self.history.append(rec)
        return rec

    def run(self, iterations, recalibrate_at=None, verbose=True):
        self.calibrate()
        self.frames_calibration = self.frames
        for i in range(iterations):
            if recalibrate_at is not None and i == recalibrate_at:
                self.calibrate()
            r = self.step(gain=1.0 if i == 0 else 0.5)                       # full first step, damped refinements
            if verbose:
                print(f"  iteration {i:2d} ({self.frames:3d} frames): fidelity of the displayed pattern {r['fidelity_shown']:.3f}, "
                      f"measured residual {r['residual']:.3f}, retrieval accuracy {r['retrieval']:.3f}, step {r['step_rms']:.2f} rad rms", flush=True)
        # final pattern: the best one by the measured residual (includes the last one)
        E, out = self.measure_field(self.phi)
        residual = np.vdot((self.tgt * np.sqrt(np.vdot(E, E).real) - E)[self.mask], (self.tgt * np.sqrt(np.vdot(E, E).real) - E)[self.mask]).real / np.vdot(E, E).real
        if residual >= self.best_residual:
            self.phi = self.best_phi
            _, out = self.measure_field(self.phi)
        self.history.append(dict(fidelity_shown=field_overlap(out, self.tgt), retrieval=np.nan, step_rms=0.0, residual=min(residual, self.best_residual)))
        return self.history


def model_amplifier(plant):
    """Design model: retuned relay + nominal parabolic thermal lens, nothing else."""
    f_th, _ = lb.per_pass_lens(lb.thermal(st.P0).opd(G), G)
    tds.DISK.thermal_opd = -G.R ** 2 / (2 * f_th)
    amp = make_amplifier(G)
    tds.DISK.thermal_opd = plant.opd(st.P0)
    return amp


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    scenarios = sys.argv[1:] or ["static", "temp_noise", "all"]
    N_MODES, ITER_RM = int(os.environ.get("N_MODES", 28)), int(os.environ.get("ITER_RM", 6))
    rows, curves = [], {}
    bin_near = 1
    basis = zernike_basis(G, radius=3.0e-3, n_modes=N_MODES)
    for scenario in scenarios:
        plant = st.Plant(scenario, 1000, seed=0)
        model = model_amplifier(plant)
        for name, tgt in plant.targets.items():
            print(f"--- {scenario}, target {name}", flush=True)
            cams = lambda: (Camera(binning=bin_near, blur=0.5, noise=CAM_NOISE, bits=12, rng=np.random.default_rng(3)),
                            Camera(binning=1, blur=0.5, noise=CAM_NOISE, bits=12, rng=np.random.default_rng(4)))
            near_t = Camera(binning=bin_near, blur=0.5, noise=0.0, bits=16).image(tgt)
            far_t = Camera(binning=1, blur=0.5, noise=0.0, bits=16).image(to_far(tgt, plant.ref))
            F_ref_file = os.path.join(results_dir("14_spgd"), f"reference_{name}.txt")
            F_ref = float(open(F_ref_file).read()) if os.path.exists(F_ref_file) else np.nan
            # A. model-based with the design model
            near_cam, far_cam = cams()
            mb = TwoCameraCorrector(plant, model, tgt, st.base_pattern(name), near_cam, far_cam)
            mb.near_target, mb.far_target = near_t, far_t
            h = mb.run(ITERATIONS, verbose=False)
            F = np.array([x["fidelity_shown"] for x in h])
            curves[(scenario, name, "model")] = (2 * np.arange(len(F)), F)
            rows.append(dict(scenario=scenario, target=name, method="model-based (design model)", frames=mb.frames, F_start=F[0], F_end=F[-1], F_best=F.max(), physics_optimum=F_ref))
            print(f"{scenario:10s} {name:8s} model-based    : {F[0]:.3f} -> {F[-1]:.3f} ({mb.frames} frames)", flush=True)
            # B. calibrated response matrix (one recalibration half way)
            for label, oracle in (("response matrix", False), ("response matrix, oracle field", True)):
                near_cam, far_cam = cams()
                rm = ResponseMatrixCorrector(plant, tgt, st.base_pattern(name), near_cam, far_cam, basis, oracle=oracle, phase_init_model=model)
                h = rm.run(ITER_RM, recalibrate_at=3, verbose=not oracle)
                F = np.array([x["fidelity_shown"] for x in h])
                frames = rm.frames_calibration + 2 * np.arange(len(F)) + (4 * N_MODES) * (np.arange(len(F)) > 3)
                curves[(scenario, name, label)] = (frames, F)
                rows.append(dict(scenario=scenario, target=name, method=label, frames=rm.frames, F_start=F[0], F_end=F[-1], F_best=F.max(), physics_optimum=F_ref))
                print(f"{scenario:10s} {name:8s} {label:30s}: {F[0]:.3f} -> {F[-1]:.3f} ({rm.frames} frames incl. {rm.frames_calibration} calibration), physics optimum {F_ref:.3f}", flush=True)
                np.save(os.path.join(RESULTS, f"pattern_{scenario}_{name}_{'oracle' if oracle else 'rm'}.npy"), rm.phi)

    with open(os.path.join(RESULTS, "two_camera_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    spgd = {}
    f_csv = os.path.join(results_dir("14_spgd"), "spgd_tuning.csv")
    if os.path.exists(f_csv):
        for r in csv.DictReader(open(f_csv)):
            spgd[(r["target"], r["config"])] = (float(r["F_best_pattern"]), int(r["frames"]))
    fig, ax = plt.subplots(len(scenarios), 2, figsize=(13, 4.2 * len(scenarios)), squeeze=False)
    for i, scenario in enumerate(scenarios):
        for j, name in enumerate(st.TARGETS):
            a = ax[i, j]
            for k, (label, mk) in enumerate((("model", "o-"), ("response matrix", "s-"), ("response matrix, oracle field", "^--"))):
                fr, F = curves[(scenario, name, label)]
                a.plot(fr, F, mk, color=f"C{k}", ms=4, label={"model": "model-based (design model)"}.get(label, label))
            for key, mk, lab in (("two_plane", "D", "SPGD, two cameras"), ("baseline", "v", "SPGD, one camera (slides)")):
                if (name, key) in spgd:
                    Fb, frn = spgd[(name, key)]
                    a.plot([frn], [Fb], mk, color="k", mfc="none", ms=8, label=f"{lab}: {Fb:.2f} @ {frn} frames")
            ref_file = os.path.join(results_dir("14_spgd"), f"reference_{name}.txt")
            if os.path.exists(ref_file):
                a.axhline(float(open(ref_file).read()), color="k", ls="--", lw=1, label="physics optimum (phase-only)")
            a.set_title(f"{scenario}: target {name}"); a.set_xlabel("camera frames used (incl. calibration)"); a.set_ylabel("true field fidelity")
            a.set_ylim(0, 1.02); a.grid(alpha=0.3); a.legend(fontsize=7, loc="lower right")
    fig.suptitle("Two-camera iterative correction vs SPGD (300 W, 5 mm disk, 1 mm beam)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "two_camera_vs_spgd.png"), dpi=110)
    lines = ["# Two-camera iterative correction (Step 16)\n",
             "| scenario | target | method | frames | fidelity start -> end | best | physics optimum |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['target']} | {r['method']} | {r['frames']} | {r['F_start']:.2f} -> {r['F_end']:.2f} | {r['F_best']:.2f} | {r['physics_optimum']:.2f} |")
    open(os.path.join(RESULTS, "TWO_CAMERA.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("saved", RESULTS)
