"""
optics.py - Standard optical components for the thin-disk laser simulator.

Every component is a thin element: it multiplies the field by a complex
transmission  T(x, y) = a(x, y) * exp(i * phi(x, y))  in one plane. The
phase phi is computed from the exact geometry (spherical sag, oblique
incidence), not from the paraxial r^2 approximation, so the components
are "perfect" but still show the aberrations that follow from physics:

    * spherical aberration   from true spherical lens / mirror surfaces,
    * astigmatism            from oblique incidence on a curved mirror,
    * diffraction            from finite apertures,
    * reflection losses      from coatings / Fresnel reflection.

Propagation between components is done with propagation.propagate().

Sign conventions (same as seed_beam.py / propagation.py, exp(-i*omega*t)):
    * a converging (focusing) element imprints  exp(-i k r^2 / 2f),  f > 0,
    * mirror radius R > 0 = concave (focusing), R < 0 = convex,
    * lens surface radius > 0 if its centre of curvature lies downstream.
"""

from dataclasses import replace

import numpy as np

from propagation import refractive_index, interface


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _sag(r2, radius):
    """Sag (depth) of a spherical surface of curvature radius `radius`:
        s(r) = R - sqrt(R^2 - r^2) = r^2 / (R + sqrt(R^2 - r^2))
    The second form keeps the sign of R and is exact for R -> inf (s = 0).
    Returns nan where r is outside the sphere (r > |R|)."""
    if np.isinf(radius):
        return np.zeros_like(r2)
    root = np.sqrt(radius ** 2 - r2)
    return r2 / (radius + np.sign(radius) * root)


def _flip_x(field):
    """Mirror image x -> -x, keeping the grid centre (index n/2) fixed."""
    return np.roll(field[:, ::-1], 1, axis=1)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------
def aperture(beam, diameter):
    """Hard circular aperture: field set to zero for r > diameter / 2."""
    mask = beam.grid.R <= diameter / 2
    return replace(beam, field=beam.field * mask)


def ideal_lens(beam, f):
    """Aberration-free lens of focal length f [m] (f < 0: diverging).

    A plane wave must become a spherical wave converging to the focus:
        phi(r) = -k * ( sqrt(f^2 + r^2) - |f| ) * sign(f)
    Expanding to second order gives the familiar -k r^2 / 2f.
    """
    r2 = beam.grid.R ** 2
    phi = -np.sign(f) * beam.k * (np.sqrt(f ** 2 + r2) - abs(f))
    return replace(beam, field=beam.field * np.exp(1j * phi))


def spherical_lens(beam, r1, r2, thickness, material="glass",
                   diameter=25.4e-3, surround="air", ar_coated=True):
    """Thin lens with exact spherical surfaces (shows spherical aberration).

    r1, r2    : surface radii [m]; biconvex is r1 > 0, r2 < 0; use np.inf for flat
    thickness : centre thickness [m]
    The glass thickness profile is
        t(r) = thickness - sag1(r) + sag2(r)
    and the optical path difference relative to the surround medium gives
        phi(r) = k0 * (n_lens - n_surround) * t(r).
    Paraxially this is -k r^2 / 2f with the lensmaker's equation
        1/f = (n - 1) * (1/r1 - 1/r2).
    The field is clipped at the lens diameter. Propagation inside the lens
    thickness is neglected (thin-lens approximation).
    """
    k0 = 2 * np.pi / beam.wavelength
    n_lens = refractive_index(material, beam.wavelength)
    n_out = refractive_index(surround, beam.wavelength)
    rr = beam.grid.R ** 2
    t = thickness - _sag(rr, r1) + _sag(rr, r2)
    phi = k0 * (n_lens - n_out) * t
    inside = (beam.grid.R <= diameter / 2) & np.isfinite(phi)
    field = np.where(inside, beam.field * np.exp(1j * np.nan_to_num(phi)), 0.0)
    out = replace(beam, field=field)
    if not ar_coated:                          # Fresnel loss at both surfaces
        out = interface(out, surround, material)
        out = interface(out, material, surround)
    return out


def mirror(beam, radius=np.inf, reflectivity=1.0, incidence_angle=0.0,
           diameter=25.4e-3, n_medium=1.0):
    """Spherical (or flat) mirror, possibly at oblique incidence.

    radius          : R > 0 concave (focusing, f = R/2), R < 0 convex, inf = flat
    reflectivity    : power reflectivity of the coating
    incidence_angle : angle between beam axis and mirror normal [rad].
                      The plane of incidence is the x-z plane (x = tangential).
    n_medium        : refractive index of the medium in front of the mirror
                      (e.g. an HR coating on the back of a crystal), k = n k0.

    A ray at transverse position (x, y) hits the mirror at (x / cos(theta), y)
    because the footprint is stretched along x. The round-trip path is
    shortened by 2 * cos(theta) * sag, so
        phi(x, y) = -2 k cos(theta) * sag( (x/cos(theta))^2 + y^2 )
    which paraxially gives the two astigmatic focal lengths
        f_tangential = (R/2) cos(theta),   f_sagittal = (R/2) / cos(theta).
    The reflection also mirrors the image, x -> -x, which flips the
    handedness of e.g. a vortex beam.
    """
    c = np.cos(incidence_angle)
    g = beam.grid
    r2_mirror = (g.X / c) ** 2 + g.Y ** 2
    phi = -2 * n_medium * beam.k * c * _sag(r2_mirror, radius)
    inside = (r2_mirror <= (diameter / 2) ** 2) & np.isfinite(phi)
    field = np.where(inside, beam.field * np.exp(1j * np.nan_to_num(phi)), 0.0)
    field = np.sqrt(reflectivity) * _flip_x(field)
    return replace(beam, field=field)


def beam_splitter(beam, reflectivity):
    """Flat partially reflecting coating (output coupler, beam splitter).

    Returns (reflected, transmitted). For a lossless coating the reflected
    and transmitted fields are in quadrature (Stokes relations), so
        r = i * sqrt(R),   t = sqrt(1 - R).
    """
    reflected = replace(beam, field=1j * np.sqrt(reflectivity) * _flip_x(beam.field))
    transmitted = replace(beam, field=np.sqrt(1 - reflectivity) * beam.field)
    return reflected, transmitted


# --------------------------------------------------------------------------
# Demo and checks
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seed_beam import Grid, gaussian, donut, results_dir
    from propagation import propagate

    lam, w0 = 1030e-9, 1.0e-3
    grid = Grid(n=512, size=10e-3)
    seed = gaussian(grid, lam, w0)
    z_r = np.pi * w0 ** 2 / lam

    def focus_distance(f):
        """ABCD: a lens of focal length f at the waist puts the new waist at
           z = f / (1 + (f/zR)^2)  with size  w0 / sqrt(1 + (zR/f)^2)."""
        return f / (1 + (f / z_r) ** 2), w0 / np.sqrt(1 + (z_r / f) ** 2)

    def scan_widths(beam, zs):
        """Beam radii (wx, wy) along z, found by propagating from `beam`."""
        return np.array([propagate(beam, z, "air").widths() for z in zs])

    # 1) Ideal lens f = 1 m: waist position and size vs ABCD
    f = 1.0
    z_abcd, w_abcd = focus_distance(f)
    zs = np.linspace(0.8, 1.0, 41) * z_abcd
    wx = scan_widths(ideal_lens(seed, f), zs)[:, 0]
    print(f"ideal lens f=1 m : waist {wx.min()*1e3:.4f} mm at z={zs[wx.argmin()]:.3f} m   "
          f"(ABCD {w_abcd*1e3:.4f} mm at {z_abcd:.3f} m)")

    # 2) Spherical biconvex lens with the same paraxial f vs ideal lens:
    #    the difference is the spherical aberration (r^4 wavefront error)
    n_g = refractive_index("glass", lam) - refractive_index("air", lam)   # n relative to air
    r_surf = 2 * n_g * f                          # 1/f = (n-1)(1/r1 - 1/r2), r2 = -r1
    big = gaussian(grid, lam, 3e-3)               # wide beam to see the r^4 term
    diff = spherical_lens(big, r_surf, -r_surf, 5e-3).field * np.conj(ideal_lens(big, f).field)
    diff = diff * np.conj(diff[grid.n // 2, grid.n // 2])     # reference to the axis
    r = 3e-3
    wf_num = np.angle(diff[grid.n // 2, np.argmin(abs(grid.x - r))]) / (2 * np.pi)
    # analytic r^4 terms: sag = r^2/2R + r^4/8R^3 and sqrt(f^2+r^2) = f + r^2/2f - r^4/8f^3
    # give  dphi = -k r^4 [ (n-1)/(4 R^3) + 1/(8 f^3) ]
    wf_ana = -(r ** 4 / lam) * (n_g / (4 * r_surf ** 3) + 1 / (8 * f ** 3))
    print(f"spherical lens   : wavefront error at r={r*1e3:.0f} mm = {wf_num:.2e} waves   "
          f"(analytic r^4 term {wf_ana:.2e} waves)")

    # 3) Concave mirror R = 2 m at 10 deg incidence: astigmatic foci
    R, th = 2.0, np.deg2rad(10)
    zs = np.linspace(0.5, 1.0, 101)
    w = scan_widths(mirror(seed, R, incidence_angle=th), zs)
    for i, (name, f_ab) in enumerate([("tangential (x)", R / 2 * np.cos(th)),
                                      ("sagittal   (y)", R / 2 / np.cos(th))]):
        print(f"tilted mirror    : {name} focus at z={zs[w[:, i].argmin()]:.3f} m   "
              f"(ABCD {focus_distance(f_ab)[0]:.3f} m)")

    # 4) Aperture of diameter 2 w0: transmitted power = 1 - exp(-2 a^2 / w0^2)
    clipped = aperture(seed, 2 * w0)
    print(f"aperture d=2w0   : transmitted {clipped.power:.4f}   "
          f"(analytic {1 - np.exp(-2):.4f})")
    far = propagate(clipped, 3.0, "air")

    # 5) Output coupler R = 0.9
    refl, trans = beam_splitter(seed, 0.9)
    print(f"output coupler   : reflected {refl.power:.3f} W, transmitted {trans.power:.3f} W")

    # 6) Donut on a flat mirror: handedness flips (l -> -l)
    ring = mirror(donut(grid, lam, w0, l=1))

    beams = [(propagate(ideal_lens(seed, f), z_abcd, "air"), "ideal lens, at focus"),
             (spherical_lens(big, r_surf, -r_surf, 5e-3), "spherical lens phase"),
             (propagate(mirror(seed, R, incidence_angle=th), 0.75, "air"), "tilted mirror, between foci"),
             (far, "aperture d=2w0, 3 m later"),
             (ring, "donut after flat mirror")]
    fig, axes = plt.subplots(len(beams), 2, figsize=(9, 4 * len(beams)))
    for (b, title), (ax_i, ax_p) in zip(beams, axes):
        b.name = title
        b.plot(ax_i, ax_p)
    fig.tight_layout()
    out = os.path.join(results_dir("3_optics"), "optics_demo.png")
    fig.savefig(out, dpi=100)
    print("saved", os.path.abspath(out))
