"""
propagation.py - Scalar beam propagation through homogeneous and weakly
inhomogeneous media for the thin-disk laser simulator.

Method: angular spectrum of plane waves (exact solution of the scalar
Helmholtz equation in a homogeneous medium, no paraxial approximation),
with an optional split-step correction for a transverse index variation
dn(x, y) such as a thermal lens.

Materials: "vacuum", "air", "glass" (fused silica), "YbYAG".

Conventions follow seed_beam.py:
    * time dependence exp(-i*omega*t), propagation along +z,
    * beam.wavelength is the VACUUM wavelength; inside a medium of index n
      the wavenumber is k = n * 2*pi/lambda0,
    * the plane-wave phase exp(i*k*z) IS included here, so the absolute
      phase after propagation is physically correct (needed for a resonator).
"""

import warnings
from dataclasses import replace
from functools import lru_cache

import numpy as np
import scipy.fft as sfft

FFT_WORKERS = 4                              # threads for the FFTs (scipy.fft)


# --------------------------------------------------------------------------
# Refractive indices (Sellmeier / Edlen dispersion formulas, lambda in um)
# --------------------------------------------------------------------------
def refractive_index(material, wavelength):
    """Refractive index of `material` at vacuum wavelength `wavelength` [m]."""
    lam = wavelength * 1e6                       # [um]
    lam2 = lam ** 2

    if material == "vacuum":
        return 1.0

    if material == "air":
        # Standard air (15 C, 101325 Pa, dry), Birch & Downs 1994 (modified Edlen):
        # n - 1 = 8.06051e-5 + 2.480990e-2/(132.274 - 1/lam^2)
        #                    + 1.74557e-4/(39.32957 - 1/lam^2)
        s2 = 1 / lam2
        return 1 + 8.06051e-5 + 2.480990e-2 / (132.274 - s2) + 1.74557e-4 / (39.32957 - s2)

    if material == "glass":
        # Fused silica, Malitson 1965:
        # n^2 - 1 = sum_i  B_i lam^2 / (lam^2 - C_i^2)
        B = (0.6961663, 0.4079426, 0.8974794)
        C = (0.0684043, 0.1162414, 9.896161)
        return np.sqrt(1 + sum(b * lam2 / (lam2 - c ** 2) for b, c in zip(B, C)))

    if material == "YbYAG":
        # Undoped YAG, Zelmon et al. 1998 (Yb doping changes n by < 1e-3):
        # n^2 - 1 = 2.28200 lam^2/(lam^2 - 0.01185) + 3.27644 lam^2/(lam^2 - 282.734)
        return np.sqrt(1 + 2.28200 * lam2 / (lam2 - 0.01185)
                         + 3.27644 * lam2 / (lam2 - 282.734))

    raise ValueError(f"unknown material '{material}'")


# --------------------------------------------------------------------------
# Angular spectrum propagation in a homogeneous medium
# --------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _transfer_function(n, dx, k, distance):
    """H(kx, ky) = exp(i kz distance) on the n x n spectral grid, cached per
    (grid, medium, distance) because the same legs are traced thousands of
    times. Stored in complex64 (phase error ~1e-7 rad, half the memory)."""
    kx = 2 * np.pi * np.fft.fftfreq(n, dx)       # spatial frequencies [rad/m]
    KX, KY = np.meshgrid(kx, kx)
    kz2 = k ** 2 - KX ** 2 - KY ** 2
    kz = np.sqrt(np.maximum(kz2, 0.0))
    H = np.where(kz2 > 0, np.exp(1j * kz * distance), 0.0)
    return H.astype(np.complex64)


@lru_cache(maxsize=16)
def absorber(n, dx, edge=0.86, order=16):
    """Super-Gaussian window exp(-(r / r0)^order), r0 = edge x half the window size. Applied
    before every propagation step it removes the light that reaches the edge of the FFT
    window; without it that light wraps around (the FFT is periodic) and interferes with
    the beam - visible as a fine raster in the output images. Light leaving the window
    physically leaves the system."""
    x = (np.arange(n) - n / 2) * dx
    R = np.hypot(*np.meshgrid(x, x))
    return np.exp(-(R / (edge * n * dx / 2)) ** order).astype(np.float32)


def _angular_spectrum(field, dx, k, distance, absorb=True):
    """Propagate `field` by `distance` in a medium with wavenumber k.

    Each plane-wave component (kx, ky) of the field advances by the phase
        H(kx, ky) = exp( i * kz * distance ),   kz = sqrt(k^2 - kx^2 - ky^2).
    Components with kx^2 + ky^2 > k^2 are evanescent and are dropped. The
    absorbing window (see `absorber`) is applied first unless absorb=False.
    """
    H = _transfer_function(field.shape[0], float(dx), float(k), float(distance))
    if absorb:
        field = field * absorber(field.shape[0], float(dx))
    return sfft.ifft2(sfft.fft2(field, workers=FFT_WORKERS) * H, workers=FFT_WORKERS)


def _edge_power_fraction(field, border=8):
    """Fraction of power in the outer `border` pixels (should be ~0)."""
    I = np.abs(field) ** 2
    inner = I[border:-border, border:-border].sum()
    return 1 - inner / I.sum()


# --------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------
def propagate(beam, distance, material="vacuum", dn=None, steps=1):
    """Propagate `beam` by `distance` [m] through `material`.

    dn    : optional 2-D map of the refractive-index deviation dn(x, y)
            from the bulk value (e.g. a thermal lens). Treated as constant
            along z over `distance`.
    steps : number of split steps used when dn is given.

    Split-step (symmetric) scheme per step of length h:
        diffract h/2  ->  phase exp(i * k0 * dn * h)  ->  diffract h/2
    which is accurate to O(h^3) for smooth dn.

    Returns a new Beam (the input is not modified).
    """
    n_medium = refractive_index(material, beam.wavelength)
    k0 = 2 * np.pi / beam.wavelength             # vacuum wavenumber
    k = n_medium * k0                            # wavenumber in the medium
    dx = beam.grid.dx
    field = beam.field

    if dn is None:
        field = _angular_spectrum(field, dx, k, distance)
    else:
        h = distance / steps
        lens = np.exp(1j * k0 * dn * h)          # phase from index deviation
        for _ in range(steps):
            field = _angular_spectrum(field, dx, k, h / 2)
            field = field * lens
            field = _angular_spectrum(field, dx, k, h / 2)

    leak = _edge_power_fraction(field)
    if leak > 1e-3:
        warnings.warn(f"{leak:.1%} of the power reached the grid edge after "
                      f"{distance} m in {material}: enlarge Grid.size")

    return replace(beam, field=field)


def interface(beam, material_in, material_out, ar_coated=False):
    """Cross a flat interface at normal incidence from material_in to material_out.

    The transverse field is continuous; the transmitted amplitude is the
    Fresnel coefficient   t = 2 n1 / (n1 + n2)   (normal incidence),
    i.e. a power transmission of 1 - ((n1 - n2)/(n1 + n2))^2.
    With ar_coated=True the reflection loss is ignored (t = 1).
    """
    if ar_coated:
        return replace(beam, field=beam.field.copy())
    n1 = refractive_index(material_in, beam.wavelength)
    n2 = refractive_index(material_out, beam.wavelength)
    t = 2 * n1 / (n1 + n2)
    return replace(beam, field=beam.field * t)


# --------------------------------------------------------------------------
# Demo and check against the analytic Gaussian beam
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seed_beam import Grid, gaussian, donut, results_dir

    lam = 1030e-9
    w0 = 1.0e-3
    grid = Grid(n=512, size=10e-3)

    for m in ("vacuum", "air", "glass", "YbYAG"):
        print(f"n({m:6s}) at {lam*1e9:.0f} nm = {refractive_index(m, lam):.6f}")

    def compare(numeric, analytic, label):
        """Report amplitude and phase error between two beams (phase is
        compared up to a constant, i.e. ignoring the plane-wave term)."""
        a_err = np.abs(numeric.amplitude - analytic.amplitude).max() / analytic.amplitude.max()
        mask = analytic.intensity > 1e-3 * analytic.intensity.max()
        diff = numeric.field * np.conj(analytic.field)             # phase difference
        diff = diff * np.conj(np.mean(diff[mask]))                 # remove constant offset
        dphi = np.angle(diff)
        print(f"{label:34s} amplitude err {a_err:.1e}, phase err {np.abs(dphi[mask]).max():.1e} rad")

    # 1) 2 m in vacuum vs. analytic Gaussian at z = 2 m
    seed = gaussian(grid, lam, w0)
    prop = propagate(seed, 2.0, "vacuum")
    compare(prop, gaussian(grid, lam, w0, z=2.0), "2 m vacuum vs analytic")

    # 2) L = 0.5 m of glass: a Gaussian diffracts as in L/n of free space
    n_glass = refractive_index("glass", lam)
    prop_glass = propagate(seed, 0.5, "glass")
    compare(prop_glass, gaussian(grid, lam, w0, z=0.5 / n_glass), "0.5 m glass vs analytic (z=L/n)")

    # 3) Yb:YAG with a thermal lens: dn(r) = dn0 * (1 - r^2/a^2) over 10 mm
    #    Parabolic index -> focal length f = a^2 / (2 dn0 L)
    dn0, a, L = 1e-4, 2e-3, 10e-3
    dn = dn0 * (1 - (grid.R / a) ** 2)
    prop_lens = propagate(seed, L, "YbYAG", dn=dn, steps=10)
    f = a ** 2 / (2 * dn0 * L)
    print(f"thermal lens focal length f = {f:.2f} m, "
          f"power transmitted {prop_lens.power / seed.power:.4f}")
    focused = propagate(prop_lens, f, "air")
    w_num = np.sqrt(2 * np.sum(focused.intensity * grid.R ** 2) / np.sum(focused.intensity))
    # ABCD check: q = i*zR at the waist, thin lens 1/q' = 1/q - 1/f, then free space q'' = q' + f,
    # spot size w = sqrt(-lambda / (pi * Im(1/q'')))
    q = 1j * np.pi * w0 ** 2 / lam
    q = 1 / (1 / q - 1 / f) + f
    w_abcd = np.sqrt(-lam / (np.pi * np.imag(1 / q)))
    print(f"spot radius at z = f: numeric {w_num*1e3:.3f} mm, ABCD {w_abcd*1e3:.3f} mm")

    # 4) Donut through 1 m of air
    ring = propagate(donut(grid, lam, w0, l=1), 1.0, "air")

    beams = [(seed, "seed, z = 0"), (prop, "2 m vacuum"),
             (prop_glass, "0.5 m glass"), (prop_lens, "YbYAG + thermal lens"),
             (focused, "at thermal-lens focus"), (ring, "donut, 1 m air")]
    fig, axes = plt.subplots(len(beams), 2, figsize=(9, 4 * len(beams)))
    for (b, title), (ax_i, ax_p) in zip(beams, axes):
        b.name = title
        b.plot(ax_i, ax_p)
    fig.tight_layout()
    out = os.path.join(results_dir("2_propagation"), "propagation_demo.png")
    fig.savefig(out, dpi=100)
    print("saved", os.path.abspath(out))
