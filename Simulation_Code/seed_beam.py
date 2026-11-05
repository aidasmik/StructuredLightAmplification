"""
seed_beam.py - Seed laser beam generator for the thin-disk laser simulator.

Generates the complex transverse field  E(x, y) = A(x, y) * exp(i * phi(x, y))
of a seed beam on a square grid. Both the amplitude A and the phase map phi
are physically correct paraxial solutions, so the beam can be propagated
directly (e.g. with an angular-spectrum / Fresnel propagator).

Supported beams
---------------
    gaussian            TEM00 Gaussian beam
    hermite_gaussian    HG_mn rectangular modes (TEM_mn)
    laguerre_gaussian   LG_pl cylindrical modes (p radial, l azimuthal index)
    donut               LG_0l vortex "donut" beam (l = topological charge)
    top_hat             flat-top (super-Gaussian) beam, planar phase

Conventions (all SI units: metres, radians, watts)
--------------------------------------------------
    * Time dependence exp(-i*omega*t), propagation along +z (Goodman).
      A diverging beam (z > 0, after the waist) therefore has a phase
      front  exp(+i*k*r^2 / (2R))  with R > 0, and Gouy phase exp(-i*psi).
    * The beam waist is at z = 0. Passing z != 0 gives the field at that
      distance from the waist (larger spot, curved phase, Gouy phase).
    * The constant plane-wave phase exp(i*k*z) is omitted.
    * field[iy, ix]: first index is y, second is x (numpy image convention).
    * The field is normalised so that  sum(|E|^2) * dx * dy = power  [W],
      hence |E|^2 is an intensity in W/m^2.
"""

import os
from dataclasses import dataclass

import numpy as np
from scipy.special import eval_hermite, eval_genlaguerre


def results_dir(name):
    """Folder Results/<name>/ for the figures of one step (created if needed)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", name)
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------
@dataclass
class Grid:
    """Square computational grid centred on the optical axis.

    n     : number of points per side
    size  : physical side length [m]
    """
    n: int = 32
    size: float = 4e-3

    def __post_init__(self):
        self.dx = self.size / self.n
        self.x = (np.arange(self.n) - self.n / 2) * self.dx       # 1-D axis [m]
        self.y = self.x.copy()
        self.X, self.Y = np.meshgrid(self.x, self.y)             # 2-D maps
        self.R = np.hypot(self.X, self.Y)                        # radius
        self.PHI = np.arctan2(self.Y, self.X)                    # azimuth


# --------------------------------------------------------------------------
# Beam container
# --------------------------------------------------------------------------
@dataclass
class Beam:
    """Complex field on a grid plus the wavelength needed to propagate it."""
    grid: Grid
    field: np.ndarray          # complex, shape (n, n)
    wavelength: float          # [m]
    name: str = "beam"

    # -- derived quantities --------------------------------------------------
    @property
    def k(self):
        return 2 * np.pi / self.wavelength

    @property
    def amplitude(self):
        return np.abs(self.field)

    @property
    def intensity(self):
        """Intensity |E|^2 [W/m^2]."""
        return np.abs(self.field) ** 2

    @property
    def phase(self):
        """Wrapped phase map in (-pi, pi]."""
        return np.angle(self.field)

    @property
    def power(self):
        return float(np.sum(self.intensity) * self.grid.dx ** 2)

    def widths(self):
        """Second-moment (D4sigma) beam radii (wx, wy) [m].
        For a Gaussian this equals the 1/e^2 radius w:  w = 2 * sigma."""
        I = self.intensity
        g = self.grid
        x0 = np.sum(I * g.X) / I.sum()
        y0 = np.sum(I * g.Y) / I.sum()
        wx = 2 * np.sqrt(np.sum(I * (g.X - x0) ** 2) / I.sum())
        wy = 2 * np.sqrt(np.sum(I * (g.Y - y0) ** 2) / I.sum())
        return wx, wy

    # -- modifiers (each returns self so calls can be chained) ---------------
    def normalize(self, power=1.0):
        """Scale the field to the given total power [W]."""
        self.field *= np.sqrt(power / self.power)
        return self

    def add_phase(self, phi):
        """Multiply by an arbitrary phase map phi(x, y) [rad]."""
        self.field = self.field * np.exp(1j * phi)
        return self

    def add_tilt(self, theta_x=0.0, theta_y=0.0):
        """Tilt the wavefront by small angles [rad] about the y and x axes."""
        g = self.grid
        return self.add_phase(self.k * (np.sin(theta_x) * g.X + np.sin(theta_y) * g.Y))

    def add_curvature(self, radius):
        """Impose a spherical wavefront of curvature radius R [m].

        R > 0: diverging, R < 0: converging (like a thin lens of focal length -R).
        """
        return self.add_phase(self.k * self.grid.R ** 2 / (2 * radius))

    # -- I/O ------------------------------------------------------------------
    def save(self, filename):
        np.savez(filename, field=self.field, x=self.grid.x,
                 wavelength=self.wavelength, name=self.name)

    @staticmethod
    def load(filename):
        d = np.load(filename)
        x = d["x"]
        grid = Grid(n=len(x), size=float(x[-1] - x[0] + (x[1] - x[0])))
        return Beam(grid, d["field"], float(d["wavelength"]), str(d["name"]))

    # -- plotting -------------------------------------------------------------
    def plot(self, ax_int=None, ax_phase=None, phase_mask=1e-3):
        """Plot intensity and phase side by side.

        The phase is blanked where the intensity is below `phase_mask`
        times its maximum, because the phase of (almost) zero field is noise.
        """
        import matplotlib.pyplot as plt

        if ax_int is None or ax_phase is None:
            _, (ax_int, ax_phase) = plt.subplots(1, 2, figsize=(9, 4))

        ext = np.array([-1, 1, -1, 1]) * self.grid.size / 2 * 1e3   # mm
        I = self.intensity
        phase = np.where(I > phase_mask * I.max(), self.phase, np.nan)

        im = ax_int.imshow(I, extent=ext, origin="lower", cmap="inferno")
        ax_int.set_title(f"{self.name}: intensity")
        ax_int.set_xlabel("x [mm]"); ax_int.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax_int, label="W/m$^2$")

        im = ax_phase.imshow(phase, extent=ext, origin="lower",
                             cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title(f"{self.name}: phase")
        ax_phase.set_xlabel("x [mm]"); ax_phase.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax_phase, label="rad")
        return ax_int, ax_phase


# --------------------------------------------------------------------------
# Gaussian-beam parameters at distance z from the waist
# --------------------------------------------------------------------------
def _beam_parameters(w0, wavelength, z):
    """Return spot size w(z), inverse curvature 1/R(z) and Gouy phase psi(z)."""
    z_r = np.pi * w0 ** 2 / wavelength              # Rayleigh range
    w = w0 * np.sqrt(1 + (z / z_r) ** 2)
    inv_r = z / (z ** 2 + z_r ** 2)                 # 1/R, finite at z = 0
    gouy = np.arctan(z / z_r)
    return w, inv_r, gouy


# --------------------------------------------------------------------------
# Beam generators
# --------------------------------------------------------------------------
def hermite_gaussian(grid, wavelength, w0, m=0, n=0, z=0.0, power=1.0):
    """Hermite-Gaussian mode HG_mn (TEM_mn) with waist w0 [m] at z = 0.

    E = (w0/w) * H_m(sqrt2 x/w) * H_n(sqrt2 y/w) * exp(-r^2/w^2)
        * exp(+i k r^2 / 2R) * exp(-i (m+n+1) psi)
    """
    k = 2 * np.pi / wavelength
    w, inv_r, gouy = _beam_parameters(w0, wavelength, z)

    amplitude = (w0 / w
                 * eval_hermite(m, np.sqrt(2) * grid.X / w)
                 * eval_hermite(n, np.sqrt(2) * grid.Y / w)
                 * np.exp(-grid.R ** 2 / w ** 2))
    phase = k * grid.R ** 2 * inv_r / 2 - (m + n + 1) * gouy

    beam = Beam(grid, amplitude * np.exp(1j * phase), wavelength, f"HG{m}{n}")
    return beam.normalize(power)


def laguerre_gaussian(grid, wavelength, w0, p=0, l=0, z=0.0, power=1.0):
    """Laguerre-Gaussian mode LG_pl with waist w0 [m] at z = 0.

    p : radial index (number of rings = p + 1 for l != 0, p dark rings)
    l : azimuthal index / topological charge (sign = handedness)

    E = (w0/w) * (sqrt2 r/w)^|l| * L_p^|l|(2 r^2/w^2) * exp(-r^2/w^2)
        * exp(i l phi) * exp(+i k r^2 / 2R) * exp(-i (2p+|l|+1) psi)
    """
    k = 2 * np.pi / wavelength
    w, inv_r, gouy = _beam_parameters(w0, wavelength, z)
    rho2 = 2 * grid.R ** 2 / w ** 2

    amplitude = (w0 / w
                 * np.sqrt(rho2) ** abs(l)
                 * eval_genlaguerre(p, abs(l), rho2)
                 * np.exp(-rho2 / 2))
    phase = l * grid.PHI + k * grid.R ** 2 * inv_r / 2 - (2 * p + abs(l) + 1) * gouy

    beam = Beam(grid, amplitude * np.exp(1j * phase), wavelength, f"LG{p}{l}")
    return beam.normalize(power)


def gaussian(grid, wavelength, w0, z=0.0, power=1.0):
    """Fundamental TEM00 Gaussian beam with 1/e^2 intensity radius w0 [m]."""
    beam = hermite_gaussian(grid, wavelength, w0, 0, 0, z, power)
    beam.name = "Gaussian"
    return beam


def donut(grid, wavelength, w0, l=1, z=0.0, power=1.0):
    """Donut (optical vortex) beam = LG_0l with helical phase exp(i l phi).

    The intensity ring peaks at r = w * sqrt(|l| / 2).
    """
    beam = laguerre_gaussian(grid, wavelength, w0, 0, l, z, power)
    beam.name = f"Donut l={l}"
    return beam


def top_hat(grid, wavelength, radius, order=10, power=1.0):
    """Flat-top beam with a planar wavefront (super-Gaussian profile).

    radius : 1/e^2 intensity radius [m]
    order  : super-Gaussian exponent; 2 = Gaussian, large = sharp edge
    """
    amplitude = np.exp(-(grid.R / radius) ** order)   # intensity ~ exp(-2 (r/a)^n)
    beam = Beam(grid, amplitude.astype(complex), wavelength, f"Top-hat n={order}")
    return beam.normalize(power)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelength = 1030e-9          # Yb:YAG
    w0 = 1.0e-3                   # 1 mm waist
    grid = Grid(n=512, size=8e-3)

    beams = [
        gaussian(grid, wavelength, w0),
        gaussian(grid, wavelength, w0, z=2.0),            # 2 m after the waist
        hermite_gaussian(grid, wavelength, w0, m=1, n=0),
        hermite_gaussian(grid, wavelength, w0, m=2, n=1),
        laguerre_gaussian(grid, wavelength, w0, p=1, l=0),
        donut(grid, wavelength, w0, l=1),
        donut(grid, wavelength, w0, l=3),
        top_hat(grid, wavelength, radius=1.5e-3, order=12),
    ]
    beams[1].name = "Gaussian z=2m"

    fig, axes = plt.subplots(len(beams), 2, figsize=(9, 4 * len(beams)))
    for beam, (ax_i, ax_p) in zip(beams, axes):
        beam.plot(ax_i, ax_p)
        print(f"{beam.name:16s} power = {beam.power:.3f} W, "
              f"peak = {beam.intensity.max():.3e} W/m^2")
    fig.tight_layout()

    out = os.path.join(results_dir("1_seed_beam"), "seed_beam_demo.png")
    fig.savefig(out, dpi=100)
    print("saved", os.path.abspath(out))
