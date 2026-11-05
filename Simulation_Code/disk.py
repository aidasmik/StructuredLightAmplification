"""
disk.py - Physical model of the Yb:YAG thin disk (no pump, no gain yet).

The disk is a thin Yb:YAG plate, AR coated on the front, HR coated on the
back, bent to a radius of curvature ROC (concave towards the beam). One
reflection on the disk is modelled as the sequence

    1. AR front face     curved refraction air -> YAG, residual reflection R_AR
    2. crystal           propagation over thickness / cos(theta2), with the
                         ground-state absorption of unpumped Yb:YAG at 1030 nm
    3. HR back face      curved mirror of radius ROC inside the crystal
                         (k = n k0), reflectivity R_HR
    4. crystal           propagation back
    5. AR front face     curved refraction YAG -> air

Paraxially the three curved surfaces add up to  2n/ROC - 2(n-1)/ROC = 2/ROC,
i.e. the bent plate focuses like a mirror of radius ROC in air, while the
oblique incidence gives the usual astigmatism. Without pump, Yb:YAG absorbs
at the laser wavelength (quasi-three-level system):
    alpha = sigma_abs * N_Yb,   N_Yb = doping * 1.38e28 m^-3 (Y sites in YAG)
With the pump on (next step) this absorption turns into gain.
"""

from dataclasses import dataclass

import numpy as np

from propagation import propagate, refractive_index
from optics import mirror, aperture
from media import refract, snell


@dataclass
class ThinDisk:
    thickness: float = 0.5e-3        # [m]
    roc: float = 2.0                 # radius of curvature [m], > 0 concave towards the beam
    diameter: float = 12e-3          # clear aperture [m]
    doping: float = 0.10             # Yb fraction on Y sites (10 at.%)
    sigma_abs: float = 1.2e-25       # ground-state absorption cross-section at 1030 nm [m^2]
    r_ar: float = 1e-3               # residual power reflectivity of the AR coating
    r_hr: float = 0.999              # power reflectivity of the HR coating
    material: str = "YbYAG"
    thermal_opd: np.ndarray = None   # optional reflection OPD map [m] from thermal.py (same grid)
    gain: np.ndarray = None          # optional power gain per reflection G(x, y) from gain.py (same grid)

    @property
    def n_yb(self):
        return self.doping * 1.38e28                     # [m^-3]

    @property
    def alpha(self):
        """Unpumped absorption coefficient [1/m]."""
        return self.sigma_abs * self.n_yb

    def reflect(self, beam, aoi=0.0):
        """One reflection of `beam` on the disk at angle of incidence `aoi` [rad].
        Returns the reflected beam in its own frame."""
        n = refractive_index(self.material, beam.wavelength)
        theta2 = snell(1.0, n, aoi)
        path = self.thickness / np.cos(theta2)            # single pass length inside
        transmission = np.sqrt((1 - self.r_ar) * np.exp(-self.alpha * path))   # amplitude, one way

        b = aperture(beam, self.diameter)
        b, _ = refract(b, "air", self.material, aoi, radius=-self.roc, ar_coated=True)
        b = propagate(b, path, self.material)
        b.field *= transmission
        b = mirror(b, radius=self.roc, reflectivity=self.r_hr, incidence_angle=theta2,
                   diameter=self.diameter, n_medium=n)
        b = propagate(b, path, self.material)
        b.field *= transmission
        b, _ = refract(b, self.material, "air", theta2, radius=self.roc, ar_coated=True)
        if self.thermal_opd is not None:                   # thermal deformation + dn/dT, thin phase screen
            b.field = b.field * np.exp(1j * 2 * np.pi / beam.wavelength * self.thermal_opd)
        if self.gain is not None:                          # small-signal gain (or re-absorption) per reflection
            b.field = b.field * np.sqrt(self.gain)
        return b

    def power_reflectivity(self, aoi=0.0, wavelength=1030e-9):
        """Expected power reflectivity of one disk reflection (losses only)."""
        n = refractive_index(self.material, wavelength)
        path = self.thickness / np.cos(snell(1.0, n, aoi))
        return (1 - self.r_ar) ** 2 * self.r_hr * np.exp(-2 * self.alpha * path)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seed_beam import Grid, gaussian, results_dir
    from optics import ideal_lens

    lam, w0 = 1030e-9, 1.0e-3
    grid = Grid(n=1024, size=10.24e-3)
    seed = gaussian(grid, lam, w0)
    disk = ThinDisk()
    n = refractive_index("YbYAG", lam)
    z_r = np.pi * w0 ** 2 / lam
    print(f"Yb:YAG {disk.doping*100:.0f} at.%: alpha = {disk.alpha/100:.2f} 1/cm, "
          f"double-pass transmission {np.exp(-2*disk.alpha*disk.thickness):.3f}")

    def focus(beam, zs):
        w = np.array([propagate(beam, z, "air").widths() for z in zs])
        return zs[w[:, 0].argmin()], zs[w[:, 1].argmin()]

    def abcd_focus(f):
        return f / (1 + (f / z_r) ** 2)

    # 1) curved refracting surface at 30 deg vs Coddington (new physics in media.refract)
    R, th = 50e-3, np.deg2rad(30)
    n_g = refractive_index("glass", lam)
    c1, c2 = np.cos(th), np.cos(snell(1.0, n_g, th))
    s_sag = n_g * R / (n_g * c2 - c1)                 # Coddington sagittal image distance
    s_tan = s_sag * c2 ** 2                           # Coddington tangential
    b, _ = refract(seed, "air", "glass", th, radius=R, ar_coated=True)
    zs = np.linspace(0.11, 0.16, 101)
    w = np.array([propagate(b, z, "glass").widths() for z in zs])
    print(f"curved surface 30deg: tangential focus {zs[w[:,0].argmin()]*1e3:.1f} mm (Coddington {s_tan*1e3:.1f}), "
          f"sagittal {zs[w[:,1].argmin()]*1e3:.1f} mm (Coddington {s_sag*1e3:.1f})")

    # 2) disk at normal incidence: focuses like a mirror of radius ROC, f = ROC/2
    f = disk.roc / 2
    zs = np.linspace(0.85, 0.95, 41)
    zt, zs_ = focus(disk.reflect(seed, 0.0), zs)
    print(f"disk, normal incidence: focus at {zt:.3f} m   (ABCD f = ROC/2: {abcd_focus(f):.3f} m)")

    # 3) disk at 10 deg: astigmatic foci
    th = np.deg2rad(10)
    zt, zs_ = focus(disk.reflect(seed, th), np.linspace(0.80, 0.98, 91))
    print(f"disk, 10 deg: tangential focus {zt:.3f} m (ABCD {abcd_focus(f*np.cos(th)):.3f}), "
          f"sagittal {zs_:.3f} m (ABCD {abcd_focus(f/np.cos(th)):.3f})")

    # 4) power budget
    out = disk.reflect(seed, th)
    print(f"disk, 10 deg: reflected power {out.power:.4f}   "
          f"(AR^2 * HR * exp(-2 alpha L) = {disk.power_reflectivity(th):.4f})")

    # 5) phase after the disk vs an ideal mirror of radius ROC (residual = thickness effects)
    ref = mirror(seed, radius=disk.roc)
    out0 = disk.reflect(seed, 0.0)
    cross = out0.field * np.conj(ref.field)
    cross *= np.conj(np.sum(cross))
    m = seed.intensity > 1e-3 * seed.intensity.max()
    print(f"disk vs ideal mirror ROC: max phase difference {np.abs(np.angle(cross[m])).max():.3f} rad")

    beams = [(disk.reflect(seed, 0.0), "disk, normal incidence"),
             (out, "disk, 10 deg incidence"),
             (propagate(out, 0.89, "air"), "10 deg, 0.89 m later (between foci)")]
    fig, axes = plt.subplots(len(beams), 2, figsize=(9, 4 * len(beams)))
    for (bb, title), (ax_i, ax_p) in zip(beams, axes):
        bb.name = title
        bb.plot(ax_i, ax_p)
    fig.tight_layout()
    outp = os.path.join(results_dir("7_disk"), "disk_demo.png")
    fig.savefig(outp, dpi=90)
    print("saved", os.path.abspath(outp))
