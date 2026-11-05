"""
media.py - Transition of a beam from one medium into another.

Models a planar (or spherical, at normal incidence) interface between two
media at an arbitrary angle of incidence:

    * Snell's law                 n1 sin(theta1) = n2 sin(theta2)
    * Fresnel coefficients        amplitude r, t for s or p polarisation,
                                  including Brewster angle and total
                                  internal reflection (complex r)
    * footprint change            in the plane of incidence the beam is
                                  stretched by  A = cos(theta2)/cos(theta1)
    * curved surface              focusing power (n2 - n1)/R

The field lives on a plane perpendicular to the beam axis (the "beam
frame"). After refraction the axis is bent by Snell's law; the field is
expressed again in the new beam frame. The plane of incidence is x-z
(x = tangential), as in optics.mirror().

Power bookkeeping: |E|^2 is the intensity in W/m^2, so the transmitted
field is  E2(x', y) = sqrt(n2/n1) * t * E1(x'/A, y),  which gives the
Fresnel power transmission  T = n2 cos(theta2) / (n1 cos(theta1)) |t|^2.
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import map_coordinates

from propagation import refractive_index, propagate
from optics import _sag, _flip_x


# --------------------------------------------------------------------------
# Angles and Fresnel coefficients
# --------------------------------------------------------------------------
def snell(n1, n2, theta1):
    """Refraction angle theta2 [rad]. Returns a complex number beyond the
    critical angle (total internal reflection): cos(theta2) is then imaginary."""
    sin2 = n1 / n2 * np.sin(theta1)
    return np.arcsin(sin2 + 0j) if abs(sin2) > 1 else np.arcsin(sin2)


def fresnel(n1, n2, theta1, polarization="s"):
    """Amplitude reflection r and transmission t for s or p polarisation.

        r_s = (n1 c1 - n2 c2) / (n1 c1 + n2 c2)    t_s = 2 n1 c1 / (n1 c1 + n2 c2)
        r_p = (n2 c1 - n1 c2) / (n2 c1 + n1 c2)    t_p = 2 n1 c1 / (n2 c1 + n1 c2)
    with c1 = cos(theta1), c2 = cos(theta2). For total internal reflection
    c2 is imaginary, |r| = 1 and r carries the TIR phase shift.
    """
    c1 = np.cos(theta1)
    c2 = np.cos(snell(n1, n2, theta1))
    if polarization == "s":
        r = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
        t = 2 * n1 * c1 / (n1 * c1 + n2 * c2)
    elif polarization == "p":
        r = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
        t = 2 * n1 * c1 / (n2 * c1 + n1 * c2)
    else:
        raise ValueError("polarization must be 's' or 'p'")
    return r, t


def brewster_angle(n1, n2):
    """tan(theta_B) = n2 / n1  (p-polarised reflection vanishes)."""
    return np.arctan2(n2, n1)


def critical_angle(n1, n2):
    """sin(theta_c) = n2 / n1  (total internal reflection for n1 > n2)."""
    return np.arcsin(n2 / n1) if n1 > n2 else np.nan


# --------------------------------------------------------------------------
# Field resampling for the footprint change
# --------------------------------------------------------------------------
def _stretch_x(field, factor):
    """Return E2(x, y) = E1(x / factor, y) by linear interpolation
    (the grid centre, index n/2, stays fixed)."""
    n = field.shape[0]
    iy, ix = np.mgrid[0:n, 0:n].astype(float)
    ix = (ix - n / 2) / factor + n / 2
    coords = [iy, ix]
    re = map_coordinates(field.real, coords, order=1, mode="constant", cval=0.0)
    im = map_coordinates(field.imag, coords, order=1, mode="constant", cval=0.0)
    return re + 1j * im


# --------------------------------------------------------------------------
# Interfaces
# --------------------------------------------------------------------------
def refract(beam, material_in, material_out, angle=0.0, polarization="s",
            radius=np.inf, ar_coated=False):
    """Beam crossing an interface from material_in to material_out.

    angle     : angle of incidence [rad] (plane of incidence = x-z)
    radius    : curvature radius of the surface [m]. R > 0: centre of
                curvature inside material_out (surface convex towards the
                incoming beam). Flat: np.inf.
    ar_coated : ignore the Fresnel reflection (t = 1, r = 0)

    Returns (transmitted, reflected) beams, each in its own beam frame.

    Curved surface: a surface displaced by the sag s along its normal
    changes the optical path by  s (n1 cos(theta1) - n2 cos(theta2)), so
        phi = k0 (n1 c1 - n2 c2) sag( (x/c1)^2 + y^2 )
    evaluated at the footprint on the surface (x stretched by 1/c1).
    At normal incidence this is a lens of focal length f = n2 R / (n2 - n1)
    in medium 2; at oblique incidence it reproduces Coddington's sagittal
    and tangential powers.
    """
    n1 = refractive_index(material_in, beam.wavelength)
    n2 = refractive_index(material_out, beam.wavelength)
    k0 = 2 * np.pi / beam.wavelength
    theta2 = snell(n1, n2, angle)
    r, t = (0.0, 1.0) if ar_coated else fresnel(n1, n2, angle, polarization)

    # reflected beam: same footprint, mirrored image
    reflected = replace(beam, field=r * _flip_x(beam.field))

    # transmitted beam: total internal reflection -> nothing propagates
    if np.iscomplexobj(theta2):
        return replace(beam, field=np.zeros_like(beam.field)), reflected

    c1, c2 = np.cos(angle), np.cos(theta2)
    field = beam.field
    if not np.isinf(radius):                       # curved surface: sag phase
        g = beam.grid
        phi = k0 * (n1 * c1 - n2 * c2) * _sag((g.X / c1) ** 2 + g.Y ** 2, radius)
        field = np.where(np.isfinite(phi), field * np.exp(1j * np.nan_to_num(phi)), 0.0)
    if angle != 0:                                 # footprint change
        field = _stretch_x(field, c2 / c1)

    transmitted = replace(beam, field=np.sqrt(n2 / n1) * t * field)
    return transmitted, reflected


def plate(beam, thickness, material, angle=0.0, polarization="s",
          surround="air", ar_coated=False):
    """Plane-parallel plate (window, Brewster plate, uncoated disk substrate).

    refract in -> propagate thickness / cos(theta2) along the internal axis
    -> refract out. Returns the transmitted beam in its own frame. (The axis
    is laterally displaced by  d = thickness sin(angle - theta2) / cos(theta2),
    which does not show up in the beam frame.)

    Equivalent free-space lengths (ABCD, Kogelnik):
        tangential  d_t = thickness cos^2(theta1) / (n cos^3(theta2))
        sagittal    d_s = thickness / (n cos(theta2))
    """
    n_s = refractive_index(surround, beam.wavelength)
    n_p = refractive_index(material, beam.wavelength)
    theta2 = snell(n_s, n_p, angle)
    inside, _ = refract(beam, surround, material, angle, polarization, ar_coated=ar_coated)
    inside = propagate(inside, thickness / np.cos(theta2), material)
    out, _ = refract(inside, material, surround, theta2, polarization, ar_coated=ar_coated)
    return out


# --------------------------------------------------------------------------
# Demo and checks
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seed_beam import Grid, gaussian, _beam_parameters, results_dir

    lam, w0 = 1030e-9, 1.0e-3
    grid = Grid(n=512, size=10e-3)
    seed = gaussian(grid, lam, w0)
    n_air = refractive_index("air", lam)
    n_yag = refractive_index("YbYAG", lam)
    n_gl = refractive_index("glass", lam)

    # 1) air -> YAG at 45 deg, s and p: power conservation and footprint
    th = np.deg2rad(45)
    th2 = snell(n_air, n_yag, th)
    for pol in ("s", "p"):
        tr, rf = refract(seed, "air", "YbYAG", th, pol)
        r, t = fresnel(n_air, n_yag, th, pol)
        T = n_yag * np.cos(th2) / (n_air * np.cos(th)) * abs(t) ** 2
        print(f"air->YAG 45deg {pol}: T={tr.power:.4f} R={rf.power:.4f} sum={tr.power+rf.power:.4f}"
              f"   (Fresnel T={T:.4f} R={abs(r)**2:.4f})")
    print(f"footprint        : wx/w0 = {tr.widths()[0]/w0:.4f}   "
          f"(cos(theta2)/cos(theta1) = {np.cos(th2)/np.cos(th):.4f})")

    # 2) Brewster angle, p polarisation: no reflection
    thb = brewster_angle(n_air, n_gl)
    _, rf = refract(seed, "air", "glass", thb, "p")
    print(f"Brewster {np.rad2deg(thb):.2f} deg p: reflected power {rf.power:.1e}")

    # 3) Total internal reflection YAG -> air beyond the critical angle
    thc = critical_angle(n_yag, n_air)
    tr, rf = refract(seed, "YbYAG", "air", np.deg2rad(45), "s")
    r, _ = fresnel(n_yag, n_air, np.deg2rad(45), "s")
    print(f"TIR YAG->air 45deg (critical {np.rad2deg(thc):.1f} deg): R={rf.power:.4f}, "
          f"T={tr.power:.1e}, phase shift {np.rad2deg(np.angle(r)):.1f} deg")

    # 4) Curved surface air -> glass, R = 50 mm: f = n2 R / (n2 - n1)
    Rs = 50e-3
    f = n_gl * Rs / (n_gl - n_air)
    tr, _ = refract(seed, "air", "glass", radius=Rs, ar_coated=True)
    zs = np.linspace(0.9, 1.0, 21) * f
    wx = [propagate(tr, z, "glass").widths()[0] for z in zs]
    print(f"curved surface   : focus at z={zs[np.argmin(wx)]*1e3:.1f} mm in glass   (f={f*1e3:.1f} mm)")

    # 5) Tilted glass plate, 10 mm at 45 deg (AR coated): compare with an
    #    astigmatic Gaussian propagated by d_t (x) and d_s (y) in air
    L = 10e-3
    th2 = snell(n_air, n_gl, th)
    d_t = L * np.cos(th) ** 2 / (n_gl * np.cos(th2) ** 3)
    d_s = L / (n_gl * np.cos(th2))
    out = plate(seed, L, "glass", th, ar_coated=True)

    def gauss_1d(u, z):
        """1-D Gaussian beam factor, same conventions as seed_beam."""
        w, inv_r, gouy = _beam_parameters(w0, lam, z)
        k = 2 * np.pi * n_air / lam
        return np.sqrt(w0 / w) * np.exp(-u ** 2 / w ** 2) * np.exp(1j * (k * u ** 2 * inv_r / 2 - gouy / 2))

    ana = replace(seed, field=gauss_1d(grid.X, d_t) * gauss_1d(grid.Y, d_s)).normalize(seed.power)
    mask = ana.intensity > 1e-3 * ana.intensity.max()
    diff = out.field * np.conj(ana.field)
    diff = diff * np.conj(np.mean(diff[mask]))
    print(f"tilted plate     : d_t={d_t*1e3:.2f} mm d_s={d_s*1e3:.2f} mm, power {out.power:.4f}, "
          f"amplitude err {np.abs(out.amplitude-ana.amplitude).max()/ana.amplitude.max():.1e}, "
          f"phase err {np.abs(np.angle(diff[mask])).max():.1e} rad")

    # Figure
    tr_s, rf_s = refract(seed, "air", "YbYAG", th, "s")
    beams = [(tr_s, "air->YAG 45deg, transmitted"), (rf_s, "air->YAG 45deg, reflected"),
             (propagate(tr_s, 50e-3, "YbYAG"), "50 mm later inside YAG"),
             (propagate(refract(seed, "air", "glass", radius=Rs, ar_coated=True)[0], f, "glass"),
              "curved surface, at focus"),
             (plate(seed, 10e-3, "glass", thb, "p"), "Brewster plate 10 mm, p")]
    fig, axes = plt.subplots(len(beams), 2, figsize=(9, 4 * len(beams)))
    for (b, title), (ax_i, ax_p) in zip(beams, axes):
        b.name = title
        b.plot(ax_i, ax_p)
    fig.tight_layout()
    outp = os.path.join(results_dir("4_media"), "media_demo.png")
    fig.savefig(outp, dpi=100)
    print("saved", os.path.abspath(outp))
