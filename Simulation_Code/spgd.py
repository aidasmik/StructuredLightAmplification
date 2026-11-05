"""
spgd.py - Stochastic parallel-gradient descent (SPGD) beam compensation, the
feedback method of the slides (slide 7, after Opt. Lett. 40, 1197 (2015)).

Loop (one camera frame per iteration):

    phi            current SLM phase  =  phi_base + sum_j a_j Z_j(x, y)
    d_phi          random trial step   =  sum_j da_j Z_j,   da_j = +-sigma
    C1 = metric(phi),  C2 = metric(phi + d_phi)
    slides rule :  C2 > C1  ->  a <- a + da
                   C2 < C1  ->  a <- a + s (C2 - C1) da        (s = gain)
    classic rule:  C+ = metric(phi + d_phi), C- = metric(phi - d_phi)
                   a <- a + s (C+ - C-) da                      (Vorontsov 1997)

The phase is parametrised by Zernike polynomials (Noll order, unit rms over
the control radius), so a few tens of numbers describe the SLM pattern and
one random draw perturbs all of them at once ("parallel").

The metric is the correlation between the camera image and the target
image. With a CCD only the intensity is measured, so by default
    C = Pearson correlation of  I_camera  and  I_target
(zero-mean normalised cross-correlation, 1 = identical shape, 0 = unrelated).
If the field is measured (interferometer, or the simulator), the complex
overlap |<E, E_target>|^2 can be used instead (metric="field").

The algorithm is independent of the "plant": `measure(phase_map)` is any
function that returns the camera image for a given SLM phase map, e.g.
the real amplifier simulation (spgd_demo.py) or later the lab hardware.
"""
from dataclasses import dataclass, field, replace
from math import factorial

import numpy as np
from scipy.ndimage import gaussian_filter


# --------------------------------------------------------------------------
# Zernike basis (Noll indexing, normalised to unit rms on the unit disk)
# --------------------------------------------------------------------------
def noll_to_nm(j):
    """Noll index j = 1, 2, ... -> radial order n, azimuthal frequency m (signed)."""
    n = 0
    while j > (n + 1) * (n + 2) // 2:
        n += 1
    k = j - n * (n + 1) // 2                 # position inside the order, 1-based
    ms = sorted(range(-n, n + 1, 2), key=abs)
    m = abs(ms[k - 1])
    if m != 0:                               # Noll: even j -> cos (m > 0), odd j -> sin (m < 0)
        m = m if j % 2 == 0 else -m
    return n, m


def zernike_mode(n, m, r, theta):
    """Zernike polynomial Z_n^m(r, theta), unit rms over the unit disk."""
    R = np.zeros_like(r)
    for s in range((n - abs(m)) // 2 + 1):
        c = ((-1) ** s * factorial(n - s)
             / (factorial(s) * factorial((n + abs(m)) // 2 - s) * factorial((n - abs(m)) // 2 - s)))
        R += c * r ** (n - 2 * s)
    norm = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
    if m > 0:
        return norm * R * np.cos(m * theta)
    if m < 0:
        return norm * R * np.sin(-m * theta)
    return norm * R


def zernike_basis(grid, radius, n_modes=15, skip_piston=True):
    """Stack of Zernike maps on `grid` inside `radius` (zero outside).
    Modes are Noll j = 2 .. n_modes+1 (piston skipped) -> tip, tilt, defocus,
    astigmatism x2, coma x2, trefoil x2, spherical, ..."""
    r, t = grid.R / radius, grid.PHI
    inside = r <= 1
    first = 2 if skip_piston else 1
    modes = []
    for j in range(first, first + n_modes):
        n, m = noll_to_nm(j)
        modes.append(np.where(inside, zernike_mode(n, m, np.minimum(r, 1), t), 0.0))
    return np.array(modes)


# --------------------------------------------------------------------------
# Camera model and metrics
# --------------------------------------------------------------------------
@dataclass
class Camera:
    """CCD: binning of the simulation grid to camera pixels, optical blur,
    shot/read noise and 8..16-bit quantisation. Exposure is normalised so that
    the brightest pixel of the first frame is at 80 % of full scale."""
    binning: int = 2                 # simulation pixels per camera pixel
    blur: float = 0.5                # optical blur [camera pixels]
    noise: float = 0.01              # rms noise, fraction of full scale
    bits: int = 12
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _scale: float = None

    def image(self, field_out):
        I = np.abs(field_out) ** 2
        n = I.shape[0] // self.binning * self.binning
        I = I[:n, :n].reshape(n // self.binning, self.binning, n // self.binning, self.binning).mean(axis=(1, 3))
        if self.blur > 0:
            I = gaussian_filter(I, self.blur)
        if self._scale is None:
            self._scale = 0.8 / I.max()
        I = I * self._scale + self.noise * self.rng.standard_normal(I.shape)
        levels = 2 ** self.bits - 1
        return np.clip(np.round(I * levels), 0, levels) / levels


def correlation(image, target):
    """Pearson correlation coefficient of two images (the 'Correlation' of the slides)."""
    a = image - image.mean()
    b = target - target.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-30))


def field_overlap(field_out, target_field):
    """Normalised complex overlap |<E, E_t>|^2 / (<E,E><E_t,E_t>) (needs the measured field)."""
    return float(abs(np.vdot(target_field, field_out)) ** 2
                 / (np.vdot(field_out, field_out).real * np.vdot(target_field, target_field).real + 1e-30))


# --------------------------------------------------------------------------
# The optimiser
# --------------------------------------------------------------------------
@dataclass
class SPGD:
    """
    measure : function(phase_map) -> camera image (or field) of the output
    metric  : function(measurement) -> scalar to maximise
    basis   : Zernike maps, shape (n_modes, n, n)
    sigma   : perturbation size per coefficient [rad rms]
    gain    : step scale s of the update rule
    rule    : "slides" (accept if better, else scaled reverse step) or "classic"
    phi_base: fixed part of the SLM phase (e.g. the vortex), same grid
    """
    measure: callable
    metric: callable
    basis: np.ndarray
    sigma: float = 0.3
    gain: float = 2.0
    rule: str = "slides"
    phi_base: np.ndarray = None
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.a = np.zeros(len(self.basis))
        if self.phi_base is None:
            self.phi_base = np.zeros(self.basis.shape[1:])
        self.history = []                      # metric of the accepted pattern per iteration
        self.best = (-np.inf, self.a.copy())
        self.frames = 0

    def phase(self, a=None):
        a = self.a if a is None else a
        return self.phi_base + np.tensordot(a, self.basis, axes=1)

    def evaluate(self, a):
        self.frames += 1
        return self.metric(self.measure(self.phase(a)))

    def step(self):
        da = self.sigma * self.rng.choice([-1.0, 1.0], size=len(self.a))     # Bernoulli perturbation
        if self.rule == "slides":
            C1 = self.evaluate(self.a) if not self.history else self.history[-1]
            C2 = self.evaluate(self.a + da)
            if C2 > C1:
                self.a = self.a + da
                C_now = C2
            else:
                self.a = self.a + self.gain * (C2 - C1) * da
                C_now = self.evaluate(self.a)
        elif self.rule == "classic":
            C_plus = self.evaluate(self.a + da)
            C_minus = self.evaluate(self.a - da)
            self.a = self.a + self.gain * (C_plus - C_minus) * da
            C_now = self.evaluate(self.a)
        else:
            raise ValueError(self.rule)
        self.history.append(C_now)
        if C_now > self.best[0]:
            self.best = (C_now, self.a.copy())
        return C_now

    def run(self, iterations, verbose=0, callback=None):
        """callback(i, self) is called after every iteration (e.g. to record frames)."""
        for i in range(iterations):
            C = self.step()
            if verbose and (i % verbose == 0 or i == iterations - 1):
                print(f"iteration {i:4d}: metric {C:.4f} (best {self.best[0]:.4f}), {self.frames} frames", flush=True)
            if callback is not None:
                callback(i, self)
        return np.array(self.history)


# --------------------------------------------------------------------------
# Self-test on a toy system: one unknown aberration screen, lens, camera at the focus
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seed_beam import Grid, gaussian, results_dir
    from propagation import propagate

    lam, w0 = 1030e-9, 1.0e-3
    grid = Grid(n=256, size=10e-3)
    seed = gaussian(grid, lam, w0)
    basis = zernike_basis(grid, radius=2.5e-3, n_modes=15)
    rng = np.random.default_rng(3)
    hidden = np.tensordot(rng.normal(0, 0.6, 15) * (np.arange(15) > 1), basis, axes=1)   # unknown aberration (no tip/tilt)
    cam = Camera(binning=2, noise=0.005)
    from optics import ideal_lens
    f = 1.0                                                  # the camera sits in the focal plane of a lens:
    target_field = propagate(ideal_lens(seed, f), f, "air").field   # there the intensity is sensitive to the phase
    target_img = Camera(binning=2, noise=0.0).image(target_field)

    def measure(phase_map):
        slm_out = replace(seed, field=seed.field * np.exp(1j * (phase_map + hidden)))   # SLM phase, then the "plant"
        out = propagate(ideal_lens(slm_out, f), f, "air")
        return cam.image(out.field)

    for rule in ("slides", "classic"):
        opt = SPGD(measure, lambda img: correlation(img, target_img), basis, sigma=0.15, gain=3.0, rule=rule)
        hist = opt.run(300)
        print(f"{rule:8s}: correlation {hist[0]:.3f} -> {hist[-1]:.3f} (best {opt.best[0]:.3f}) after 300 iterations, "
              f"{opt.frames} camera frames; residual rms phase error "
              f"{np.std((opt.phase(opt.best[1]) + hidden)[grid.R < 1.5e-3]):.2f} rad")
        if rule == "slides":
            keep = (hist, opt)
    hist, opt = keep
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    ax[0].imshow(hidden, cmap="twilight"); ax[0].set_title("hidden aberration [rad]")
    ax[1].imshow(measure(np.zeros_like(hidden)), cmap="inferno"); ax[1].set_title("camera, uncompensated")
    ax[2].imshow(measure(opt.phase(opt.best[1])), cmap="inferno"); ax[2].set_title(f"camera, SPGD (C = {opt.best[0]:.3f})")
    ax[3].plot(hist); ax[3].set_xlabel("# of SPGD iterations"); ax[3].set_ylabel("correlation"); ax[3].grid(alpha=0.3)
    for a in ax[:3]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    out = os.path.join(results_dir("14_spgd"), "spgd_selftest.png")
    fig.savefig(out, dpi=100)
    print("saved", os.path.abspath(out))
