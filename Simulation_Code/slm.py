"""
slm.py - Physical model of a reflective phase-only LCOS spatial light modulator.

The ideal SLM would multiply the field by exp(i * phi(x, y)). A real device
differs in several ways, all included here:

    1. pixelation      phi is sampled at the pixel centres and is constant
                       over each pixel (pitch p)
    2. wrapping        only 0 .. 2 pi is available, phi is taken modulo 2 pi
    3. quantisation    `levels` grey levels (8 bit = 256) over 2 pi
    4. crosstalk       the liquid crystal cannot follow a sharp step between
                       pixels (fringing fields): the phase map is blurred
                       with a Gaussian of sigma = `crosstalk` * pitch
    5. fill factor     a fraction (1 - FF) of each pixel is dead space
                       (electrode gaps) that does not reflect into the
                       beam. A periodic amplitude mask with open fraction
                       FF has zero-order amplitude FF, so the retained
                       order sees  r = FF * exp(i phi)  (power FF^2); the
                       rest goes into the pixel diffraction orders.
    6. active area     light outside the n_pixels * pitch area is lost
    7. reflectivity    power reflectivity R, and the mirror image x -> -x
    8. bias            a constant phase offset added before wrapping, so a
                       small bipolar pattern does not straddle the 0/2pi
                       wrap (that would create a flyback line through it)
    9. diffraction     the pixel grid (and an optional blazed carrier
                       grating) sends light into orders at angles m*lambda/p
                       (m*lambda/Lambda). Only one order stays in the beam
                       path: the beam frame is centred on it and the other
                       orders are removed by an angular filter that
                       represents the acceptance of the downstream optics.

Grid requirement: Grid.dx <= pitch / 2 so that pixels are resolved, ideally
with pitch an exact multiple of dx (e.g. Grid(n=1024, size=10.24e-3) gives
dx = 10 um for a 20 um pitch).

Pattern functions take (X, Y) in metres and return the phase in radians.
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter

from optics import _flip_x


# --------------------------------------------------------------------------
# Phase patterns  (each returns a function of (X, Y))
# --------------------------------------------------------------------------
def flat():
    return lambda X, Y: np.zeros_like(X)


def vortex(l=1):
    """Helical phase l * phi: turns a Gaussian into a donut."""
    return lambda X, Y: l * np.arctan2(Y, X)


def lens(f, wavelength):
    """Fresnel lens: phi = -k (sqrt(f^2 + r^2) - f), wrapped by the SLM."""
    k = 2 * np.pi / wavelength
    return lambda X, Y: -np.sign(f) * k * (np.sqrt(f ** 2 + X ** 2 + Y ** 2) - abs(f))


def zernike_astigmatism(amplitude, radius, angle=0.0):
    """Primary astigmatism, amplitude in radians at r = radius:
       phi = A (r/radius)^2 cos(2 (theta - angle))."""
    return lambda X, Y: amplitude * (X ** 2 + Y ** 2) / radius ** 2 * np.cos(
        2 * (np.arctan2(Y, X) - angle))


def zernike(coeffs, radius):
    """Sum of low-order Zernike aberrations, e.g. zernike({"coma_x": 0.5}, 2e-3).
    coeffs: {name: amplitude [rad] at r = radius}; r, theta are polar coordinates."""
    terms = {
        "defocus":   lambda r, t: 2 * r ** 2 - 1,
        "astig0":    lambda r, t: r ** 2 * np.cos(2 * t),
        "astig45":   lambda r, t: r ** 2 * np.sin(2 * t),
        "coma_x":    lambda r, t: (3 * r ** 3 - 2 * r) * np.cos(t),
        "coma_y":    lambda r, t: (3 * r ** 3 - 2 * r) * np.sin(t),
        "trefoil0":  lambda r, t: r ** 3 * np.cos(3 * t),
        "trefoil30": lambda r, t: r ** 3 * np.sin(3 * t),
        "spherical": lambda r, t: 6 * r ** 4 - 6 * r ** 2 + 1,
    }

    def pattern(X, Y):
        r, t = np.hypot(X, Y) / radius, np.arctan2(Y, X)
        return sum(a * terms[name](r, t) for name, a in coeffs.items())
    return pattern


def map_pattern(phi_map, grid, wrapped=True):
    """Turn a phase map on `grid` into a pattern function (X, Y). A wrapped
    map is interpolated through cos/sin so that 2 pi jumps are not smeared."""
    from scipy.ndimage import map_coordinates

    def pattern(X, Y):
        coords = [Y / grid.dx + grid.n / 2, X / grid.dx + grid.n / 2]
        if not wrapped:
            return map_coordinates(phi_map, coords, order=1, mode="nearest")
        re = map_coordinates(np.cos(phi_map), coords, order=1, mode="nearest")
        im = map_coordinates(np.sin(phi_map), coords, order=1, mode="nearest")
        return np.arctan2(im, re)
    return pattern


def top_hat_shaper(w_in, w_out, wavelength, f):
    """Phase that maps a Gaussian (radius w_in) onto a flat-top of radius
    w_out in the focal plane of a lens f (geometric-optics beam shaping):
    the ray at r is sent to  rho(r) = w_out * sqrt(1 - exp(-2 r^2 / w_in^2)),
    so  d phi / dr = -k rho(r) / f  (integrated numerically)."""
    k = 2 * np.pi / wavelength
    r = np.linspace(0, 20 * w_in, 4000)
    rho = w_out * np.sqrt(1 - np.exp(-2 * r ** 2 / w_in ** 2))
    phi_r = -k / f * np.concatenate([[0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(r))])
    return lambda X, Y: np.interp(np.hypot(X, Y), r, phi_r)


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
@dataclass
class SLM:
    pitch: float = 20e-6              # pixel pitch [m]
    n_pixels: tuple = (792, 600)      # (columns along x, rows along y)
    fill_factor: float = 0.95
    levels: int = 256
    crosstalk: float = 0.35           # fringing-field blur, in units of pitch
    reflectivity: float = 0.90
    stroke: float = 1.0               # phase-stroke (look-up table) error: displayed phase x stroke

    def modulate(self, beam, pattern, blaze_period=None, order=1, bias=0.0, depth=None):
        """Reflect `beam` off the SLM displaying `pattern(X, Y)` + bias.

        blaze_period : optional carrier grating period along x [m]. The
                       beam frame then follows diffraction order `order`
                       (angle order * lambda / blaze_period).
        bias         : constant phase added before wrapping [rad].
        depth        : optional function (X, Y) -> [0, 1] multiplying the
                       wrapped phase (blaze depth). With a carrier this
                       encodes amplitude in the first order (Davis et al.
                       1999): first-order amplitude = sinc(1 - depth).
        Returns a new Beam.
        """
        g = beam.grid
        p = self.pitch
        if g.dx > p / 2:
            warnings.warn(f"grid dx = {g.dx*1e6:.1f} um does not resolve the "
                          f"{p*1e6:.1f} um pixels; use dx <= pitch/2")

        # 1. pixel centres and sampled phase (pixel grid centred on the axis)
        Xc = np.floor(g.X / p + 0.5) * p
        Yc = np.floor(g.Y / p + 0.5) * p
        phi = pattern(Xc, Yc) + bias
        if blaze_period is not None:
            phi = phi + 2 * np.pi * Xc / blaze_period

        # 2./3. wrap to 0..2pi, optional blaze depth, quantise
        phi = np.mod(phi, 2 * np.pi)
        if depth is not None:
            phi = phi * np.clip(depth(Xc, Yc), 0, 1)
        phi = np.floor(phi / (2 * np.pi) * self.levels) / self.levels * 2 * np.pi
        phi = phi * self.stroke

        # 4. crosstalk between pixels
        if self.crosstalk > 0:
            phi = gaussian_filter(phi, self.crosstalk * p / g.dx)

        # 5. dead space, 6. active area, 7. reflectivity and image flip
        r = self.fill_factor * np.exp(1j * phi)
        active = (np.abs(g.X) < self.n_pixels[0] * p / 2) & (np.abs(g.Y) < self.n_pixels[1] * p / 2)
        field = np.sqrt(self.reflectivity) * _flip_x(beam.field * r * active)

        # 9. follow the selected order and drop the others
        if blaze_period is not None:
            # after the image flip the carrier is -2 pi x / period, so order m sits at kx = -m 2 pi / period
            field = field * np.exp(1j * 2 * np.pi * order * g.X / blaze_period)
            period_x = blaze_period
        else:
            period_x = p
        field = _angular_filter(field, g.dx, np.pi / period_x, np.pi / p)
        return replace(beam, field=field)


def mirror_image(beam):
    """The same beam seen after one reflection (x -> -x)."""
    return replace(beam, field=_flip_x(beam.field))


def _angular_filter(field, dx, kx_max, ky_max):
    """Keep only plane-wave components with |kx| < kx_max and |ky| < ky_max
    (half the spacing of the neighbouring diffraction orders)."""
    k = 2 * np.pi * np.fft.fftfreq(field.shape[0], dx)
    KX, KY = np.meshgrid(k, k)
    keep = (np.abs(KX) < kx_max) & (np.abs(KY) < ky_max)
    import scipy.fft as sfft
    return sfft.ifft2(sfft.fft2(field, workers=4) * keep, workers=4)


# --------------------------------------------------------------------------
# Demo and checks
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seed_beam import Grid, gaussian, laguerre_gaussian, results_dir
    from propagation import propagate

    lam, w0 = 1030e-9, 1.0e-3
    grid = Grid(n=1024, size=10.24e-3)          # dx = 10 um = pitch / 2
    seed = gaussian(grid, lam, w0)
    ideal = SLM(fill_factor=1.0, crosstalk=0.0, reflectivity=1.0)   # for theory checks
    real = SLM()

    # 1) flat pattern: reflectivity and fill factor are lost
    out = real.modulate(seed, flat())
    print(f"flat pattern      : power {out.power:.4f}   (R * FF^2 = "
          f"{real.reflectivity * real.fill_factor**2:.4f})")

    # 2) blazed grating with N steps per period: efficiency sinc^2(1/N)
    for n_steps in (2, 4, 8, 16):
        out = ideal.modulate(seed, flat(), blaze_period=n_steps * ideal.pitch)
        print(f"blaze {n_steps:2d} px/period : 1st-order efficiency {out.power:.4f}   "
              f"(theory sinc^2(1/N) = {np.sinc(1 / n_steps)**2:.4f})")
    out = real.modulate(seed, flat(), blaze_period=8 * real.pitch)
    print(f"blaze 8 px, real  : 1st-order efficiency {out.power:.4f}   "
          f"(crosstalk, fill factor, R included)")

    # 3) Fresnel lens f = 1 m: efficiency into the ideal focused beam.
    #    (The flyback zones scatter a weak halo, so the overlap with the
    #    ideal_lens result is the meaningful figure of merit.)
    from optics import ideal_lens

    def overlap(a, b):
        """Fraction of the power of `a` that is in the mode of `b`."""
        return abs(np.sum(a.field * np.conj(b.field)) * grid.dx ** 2) ** 2 / (a.power * b.power)

    f = 1.0
    ref = propagate(mirror_image(ideal_lens(seed, f)), 0.9, "air")
    for name, dev in (("ideal", ideal), ("real", real)):
        out = propagate(dev.modulate(seed, lens(f, lam)), 0.9, "air")
        print(f"Fresnel lens f=1 m ({name:5s}): power {out.power:.4f}, "
              f"fraction in the ideal focused mode {overlap(out, ref):.4f}")

    # 4) vortex l=1: overlap with the LG01 mode after 1 m
    out = propagate(real.modulate(seed, vortex(1)), 1.0, "air")
    lg = laguerre_gaussian(grid, lam, w0, p=0, l=-1, z=1.0)        # l flips at the mirror
    print(f"vortex l=1        : LG01 mode purity after 1 m = {overlap(out, lg):.3f}, power {out.power:.4f}   "
          f"(Gaussian with helical phase: pi/4 = {np.pi/4:.3f})")

    # 5) Gaussian -> top-hat in the focal plane of a 0.5 m lens
    f_sh = 0.5
    shaped = real.modulate(seed, top_hat_shaper(w0, 0.3e-3, lam, f_sh))
    shaped = propagate(ideal_lens(shaped, f_sh), f_sh, "air")

    beams = [(real.modulate(seed, flat()), "flat pattern"),
             (real.modulate(seed, vortex(1)), "vortex l=1, at SLM"),
             (out, "vortex l=1, 1 m later"),
             (real.modulate(seed, lens(f, lam)), "Fresnel lens f=1 m, at SLM"),
             (shaped, "top-hat shaper, focal plane")]
    fig, axes = plt.subplots(len(beams), 2, figsize=(9, 4 * len(beams)))
    for (b, title), (ax_i, ax_p) in zip(beams, axes):
        b.name = title
        b.plot(ax_i, ax_p)
    fig.tight_layout()
    outp = os.path.join(results_dir("6_slm"), "slm_demo.png")
    fig.savefig(outp, dpi=90)
    print("saved", os.path.abspath(outp))
