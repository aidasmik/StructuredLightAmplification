"""
thermal.py - Temperature field and thermally induced shape change of the
Yb:YAG thin disk (no gain yet; the pump only heats the crystal).

Model (steady state, pumped region of radius r_p, thickness h, cooled through
the HR back face into a heat sink):

    heat load            Q(r) = eta_heat * P_abs * p(r)          [W/m^2]
                         p(r): normalised super-Gaussian pump profile,
                         smoothed over ~h for the radial heat spreading
    back-face jump       T_back(r) - T_sink = Q(r) / h_contact
    axial profile        T(r,z) - T_back(r) = (Q/(k h)) (h z - z^2/2)
                         (uniform heat source, adiabatic front face, z from
                         the back face), so  front-back difference  Q h/(2k)
                         and thickness-mean rise                    Q h/(3k)
    bulge                dh(r) = alpha * <T - T0> * h   (front face moves
                         towards the beam, back face fixed)
    bending              the warm disk is bonded to a cold heat sink: the
                         radial expansion mismatch  d_eps = alpha_disk <T-T0>
                         - alpha_sink dT_sink  bends the composite plate like
                         a bimetal strip (Timoshenko 1925),
                           kappa = 6 E1 E2 t1 t2 (t1+t2) d_eps / D,
                           D = E1^2 t1^4 + 4 E1 E2 t1^3 t2 + 6 E1 E2 t1^2 t2^2
                               + 4 E1 E2 t1 t2^3 + E2^2 t2^4,
                         plus the thermal moment of the gradient inside the
                         disk, E1 alpha dT t1^2 / 12, acting on the stiffness
                         of the composite. The deflection w(r) is integrated
                         radially from kappa(r). For a 140 um disk on 1.5 mm
                         diamond this reproduces the measured
                         -0.009 .. -0.017 dioptres per kW/cm^2 of pump
                         intensity (Schuhmann et al. 2016, arXiv:1607.08873).
    optical path (2 passes, reflection):
        OPD(r) = 2 (n-1) dh(r) + 2 (dn/dT) <T - T0> h + 2 w_bend(r)
    (bulge: longer path in the crystal; dn/dT: index rise; bending: the
     hotter front face expands more, the disk becomes convex towards the
     beam and its edges recede by w_bend(r) -> longer air path at the
     edges -> defocusing)

Material data for YAG (approximate literature values, temperature slopes
marked PLACEHOLDER where no reliable measurement was available):
    k(300 K)   = 11 W/m/K undoped, ~6.5 W/m/K for 10 at.% Yb (doping
                 scattering), k ~ T^-0.7 above 300 K          [slope PLACEHOLDER]
    alpha(300) = 6.9e-6 /K, + 1.3e-8 /K per K                  [slope PLACEHOLDER]
    dn/dT(300) = 7.3e-6 /K at 1 um, + 1.0e-8 /K per K           [slope PLACEHOLDER]
    stress / photo-elastic term: reduces the dn/dT lens by ~15 % [PLACEHOLDER]
    contact conductance solder/diamond: 3e5 W/m^2/K              [PLACEHOLDER]
    E_YAG = 280 GPa; heat sinks: diamond E = 1050 GPa, alpha = 1.0e-6 /K;
    copper E = 120 GPa, alpha = 17e-6 /K; CuW E = 300 GPa, alpha = 7e-6 /K
    heat-sink mean temperature rise under the pump: half the back-face rise [PLACEHOLDER]
    density 4560 kg/m^3, heat capacity 590 J/kg/K  ->  diffusivity ~2.5e-6 m^2/s

Transient (pump switched on at t = 0): the 1-D heat equation through the
thickness is solved explicitly for a unit heat flux (uniform source, Robin
condition to the sink at the back, adiabatic front). Because the problem is
linear, T(r, z, t) = T_sink + Q(r) g(z, t). The thickness-mean rise and the
front-back difference per unit flux, g_mean(t) and g_diff(t), give the
deformation at any time. Axial time constant ~ h^2 / (pi^2 D) ~ 10 ms for
0.5 mm; the slower radial spreading (~ r_p^2 / 4D ~ 0.4 s) is not time
resolved: the smoothed radial profile is used at all times.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


# --------------------------------------------------------------------------
# Temperature dependent material properties of Yb:YAG
# --------------------------------------------------------------------------
RHO = 4560.0          # density [kg/m^3]
C_P = 590.0           # specific heat [J/kg/K]
E_YAG = 280e9         # Young's modulus [Pa]
HEAT_SINKS = {        # Young's modulus [Pa], expansion coefficient [1/K]
    "diamond": (1050e9, 1.0e-6),
    "copper": (120e9, 17e-6),
    "CuW": (300e9, 7.0e-6),
}


def conductivity(T, doping=0.10):
    """Thermal conductivity [W/m/K]. Yb doping adds phonon scattering:
    k300 = 11 / (1 + 6.5 * doping) gives 11 (undoped) and ~6.7 (10 at.%)."""
    k300 = 11.0 / (1 + 6.5 * doping)
    return k300 * (300.0 / T) ** 0.7                     # exponent PLACEHOLDER


def expansion(T):
    """Linear thermal expansion coefficient [1/K]."""
    return 6.9e-6 + 1.3e-8 * (T - 300.0)                 # slope PLACEHOLDER


def dndT(T):
    """Thermo-optic coefficient at 1030 nm [1/K], stress term included as a
    15 % reduction (PLACEHOLDER)."""
    return 0.85 * (7.3e-6 + 1.0e-8 * (T - 300.0))


# --------------------------------------------------------------------------
# Thermal state of the disk
# --------------------------------------------------------------------------
@dataclass
class DiskThermal:
    thickness: float = 0.5e-3          # [m]
    doping: float = 0.10
    n_index: float = 1.815             # refractive index at 1030 nm
    pump_power: float = 200.0          # incident pump [W]
    pump_radius: float = 2.0e-3        # 1/e^2-like radius of the pump spot [m]
    pump_order: int = 8                # super-Gaussian order of the pump profile
    eta_abs: float = 0.90              # absorbed fraction of the pump (multi-pass pump)
    eta_heat: float = 0.10             # heat fraction of the absorbed pump (quantum defect + losses)
    sink_temperature: float = 293.0    # [K]
    contact_conductance: float = 3e5   # [W/m^2/K]  PLACEHOLDER
    heat_sink: str = "diamond"         # "diamond", "copper" or "CuW"
    sink_thickness: float = 1.5e-3     # [m]

    def heat_flux(self, grid):
        """Heat flux density towards the sink [W/m^2], smoothed over the
        thickness for the radial heat spreading."""
        p = np.exp(-2 * (grid.R / self.pump_radius) ** self.pump_order)
        p = p / (p.sum() * grid.dx ** 2)                              # normalised to 1 [1/m^2]
        Q = self.eta_heat * self.eta_abs * self.pump_power * p
        return gaussian_filter(Q, self.thickness / grid.dx)

    def solve(self, grid, Q=None):
        """Steady-state temperature and deformation maps on `grid`.
        Q: optional heat-flux map [W/m^2] (default: from the pump profile)."""
        h = self.thickness
        Q = self.heat_flux(grid) if Q is None else Q
        # temperatures, iterating the temperature dependence of k
        T_back = self.sink_temperature + Q / self.contact_conductance
        T_mean = T_back.copy()
        for _ in range(3):
            k = conductivity(T_mean, self.doping)
            dT_axial = Q * h / (2 * k)                                # front - back
            T_mean = T_back + Q * h / (3 * k)                         # thickness average
        return self._deform(grid, Q, T_mean, dT_axial)

    def transient(self, times, k=None):
        """Thickness-mean temperature rise and front-back difference per unit
        heat flux, g_mean(t), g_diff(t) [K m^2/W], after switching the pump
        on at t = 0 (1-D explicit finite differences through the thickness)."""
        h = self.thickness
        if k is None:                                                 # k at the steady centre temperature
            k = conductivity(self.solve_centre_temperature(), self.doping)
        D = k / (RHO * C_P)
        nz = 50
        dz = h / nz
        dt = 0.4 * dz ** 2 / D
        z_nodes = nz + 1
        T = np.zeros(z_nodes)                                        # rise per unit flux, z = 0 at the back
        q = 1.0 / h                                                   # volumetric source per unit flux
        g_mean, g_diff = [], []
        t, i_out = 0.0, 0
        times = np.asarray(times, float)
        while i_out < len(times):
            if t >= times[i_out]:
                g_mean.append(T.mean()); g_diff.append(T[-1] - T[0]); i_out += 1
                continue
            lap = np.empty_like(T)
            lap[1:-1] = T[2:] - 2 * T[1:-1] + T[:-2]
            ghost_back = T[1] - 2 * dz * self.contact_conductance * T[0] / k   # -k dT/dz = h_c T at the back
            lap[0] = T[1] - 2 * T[0] + ghost_back
            lap[-1] = 2 * T[-2] - 2 * T[-1]                           # adiabatic front
            T = T + dt * (D * lap / dz ** 2 + q / (RHO * C_P))
            t += dt
        return np.array(g_mean), np.array(g_diff)

    def solve_centre_temperature(self):
        """Steady thickness-mean temperature at the pump centre (scalar)."""
        Q0 = self.eta_heat * self.eta_abs * self.pump_power / (np.pi * self.pump_radius ** 2)
        T = self.sink_temperature + Q0 / self.contact_conductance
        for _ in range(3):
            T = self.sink_temperature + Q0 / self.contact_conductance + Q0 * self.thickness / (3 * conductivity(T, self.doping))
        return T

    def solve_at(self, grid, g_mean, g_diff):
        """Temperature and deformation maps at one instant, from the unit
        responses returned by transient()."""
        Q = self.heat_flux(grid)
        T_mean = self.sink_temperature + Q * g_mean
        return self._deform(grid, Q, T_mean, Q * g_diff)

    def _deform(self, grid, Q, T_mean, dT_axial):
        h = self.thickness
        T_back = T_mean - 2 * dT_axial / 3                            # parabolic profile: mean = back + 2/3 (front-back)
        T_front = T_back + dT_axial
        rise = T_mean - self.sink_temperature                         # relative to the cold disk

        # deformations
        alpha = expansion(T_mean)
        bulge = alpha * rise * h                                      # front face displacement [m]
        kappa = self.bending_curvature(alpha, rise, dT_axial, T_back - self.sink_temperature)
        w_bend = _integrate_axisymmetric(kappa, grid)                 # edge recession, 0 on axis

        opd = (2 * (self.n_index - 1) * bulge
               + 2 * dndT(T_mean) * rise * h
               + 2 * w_bend)

        return {"Q": Q, "T_front": T_front, "T_back": T_back, "T_mean": T_mean,
                "bulge": bulge, "kappa": kappa, "w_bend": w_bend, "opd": opd,
                "opd_bulge": 2 * (self.n_index - 1) * bulge,
                "opd_dndT": 2 * dndT(T_mean) * rise * h,
                "opd_bend": 2 * w_bend}

    def bending_curvature(self, alpha, rise, dT_axial, sink_rise):
        """Curvature [1/m] of the disk-heat-sink plate (positive = convex
        towards the beam): bimetal mismatch term plus the moment of the
        temperature gradient inside the disk (see module docstring)."""
        E1, t1 = E_YAG, self.thickness
        E2, alpha2 = HEAT_SINKS[self.heat_sink]
        t2 = self.sink_thickness
        D = (E1 ** 2 * t1 ** 4 + 4 * E1 * E2 * t1 ** 3 * t2 + 6 * E1 * E2 * t1 ** 2 * t2 ** 2
             + 4 * E1 * E2 * t1 * t2 ** 3 + E2 ** 2 * t2 ** 4)
        d_eps = alpha * rise - alpha2 * 0.5 * sink_rise              # sink rise: PLACEHOLDER estimate
        kappa_mismatch = 6 * E1 * E2 * t1 * t2 * (t1 + t2) * d_eps / D
        moment = E1 * alpha * dT_axial * t1 ** 2 / 12                 # thermal moment of the gradient
        stiffness = D / (12 * (E1 * t1 + E2 * t2))                    # bending stiffness of the composite
        return kappa_mismatch + moment / stiffness

    def opd(self, grid):
        """Reflection optical path difference map [m] on `grid`."""
        return self.solve(grid)["opd"]


def _integrate_axisymmetric(kappa, grid):
    """Deflection w(r) with w'' = kappa(r), w(0) = w'(0) = 0, from the radial
    profile of kappa (azimuthal average), mapped back to 2-D. The plate bending
    is a stiff, long-range response, so only the azimuthal mean of a
    non-axisymmetric heat load (hot spots, decentred pump) bends the disk;
    their local structure enters the OPD through the bulge and dn/dT terms."""
    r = np.arange(0, grid.size / np.sqrt(2) + grid.dx, grid.dx)
    k = np.clip(np.round(grid.R / grid.dx).astype(int), 0, len(r) - 1)        # ring index of every pixel
    counts = np.bincount(k.ravel(), minlength=len(r))
    prof = np.bincount(k.ravel(), kappa.ravel(), minlength=len(r)) / np.maximum(counts, 1)
    empty = counts == 0
    if empty.any():
        prof[empty] = np.interp(r[empty], r[~empty], prof[~empty])
    slope = np.concatenate([[0], np.cumsum(0.5 * (prof[1:] + prof[:-1]) * np.diff(r))])
    w = np.concatenate([[0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(r))])
    return np.interp(grid.R, r, w)


def fit_lens(opd, grid, radius, wavelength):
    """Fit OPD = a + b r^2 inside `radius`. A focusing element has
    OPD = -r^2 / (2 f), so f = -1/(2b) (f < 0: defocusing). Also returns the
    rms of the non-parabolic residual in waves (the aberration)."""
    m = grid.R < radius
    A = np.stack([np.ones(m.sum()), grid.R[m] ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, opd[m], rcond=None)
    resid = opd[m] - A @ coef
    f = np.inf if coef[1] == 0 else -1 / (2 * coef[1])
    return f, np.sqrt(np.mean(resid ** 2)) / wavelength


# --------------------------------------------------------------------------
# Demo and checks
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seed_beam import Grid, results_dir

    lam = 1030e-9
    grid = Grid(n=1024, size=10.24e-3)
    c = grid.n // 2

    # analytic check: top-hat pump without smoothing -> known centre temperatures
    th = DiskThermal(pump_power=200.0, pump_order=40)
    s = th.solve(grid)
    Q0 = th.eta_heat * th.eta_abs * th.pump_power / (np.pi * th.pump_radius ** 2)
    k0 = conductivity(s["T_mean"][c, c], th.doping)
    print(f"200 W pump, 2 mm radius: heat flux {Q0/1e4:.1f} W/cm^2")
    print(f"  centre: T_back-T_sink = {s['T_back'][c,c]-th.sink_temperature:.1f} K (analytic {Q0/th.contact_conductance:.1f}), "
          f"front-back = {s['T_front'][c,c]-s['T_back'][c,c]:.1f} K (analytic {Q0*th.thickness/(2*k0):.1f})")
    print(f"  centre bulge {s['bulge'][c,c]*1e6:.3f} um, bending curvature {s['kappa'][c,c]:.4f} 1/m "
          f"(HR radius {1/s['kappa'][c,c]:.1f} m, convex)")

    # literature check: 140 um disk on 1.5 mm diamond, dioptric power per pump intensity
    # (Schuhmann et al. 2016: -0.0092 1/m per kW/cm^2 in laser operation, -0.017 in fluorescence)
    lit = DiskThermal(thickness=140e-6, doping=0.07, pump_power=200.0, pump_radius=2e-3)
    s_lit = lit.solve(grid)
    f_lit, _ = fit_lens(s_lit["opd"], grid, 1.0e-3, lam)
    intensity = lit.pump_power / (np.pi * (lit.pump_radius * 1e2) ** 2) / 1e3       # kW/cm^2
    print(f"literature check, 140 um disk on diamond: {1/f_lit/intensity:+.4f} 1/m per kW/cm^2 "
          f"(measured -0.009 .. -0.017), front-back {s_lit['T_front'][c,c]-s_lit['T_back'][c,c]:.1f} K")
    for sink in ("diamond", "copper"):
        d = DiskThermal(pump_power=200.0, heat_sink=sink)
        f_s, _ = fit_lens(d.solve(grid)["opd"], grid, 1.0e-3, lam)
        print(f"0.5 mm disk on {sink:7s}: thermal lens f = {f_s:7.2f} m at 200 W")

    # pump power series: thermal lens and aberration seen by a 0.7 mm seed
    print(f"\n{'pump [W]':>8} {'T_front max [K]':>15} {'bulge [um]':>10} {'OPD centre [waves]':>18} "
          f"{'f_thermal [m]':>13} {'residual [waves rms]':>20}")
    powers = (50, 100, 200, 400)
    states = {}
    for P in powers:
        th = DiskThermal(pump_power=P)
        s = th.solve(grid); states[P] = s
        f, res = fit_lens(s["opd"], grid, 1.0e-3, lam)
        print(f"{P:8.0f} {s['T_front'].max():15.1f} {s['bulge'].max()*1e6:10.3f} {s['opd'][c,c]/lam:18.2f} "
              f"{f:13.2f} {res:20.4f}")

    # figure
    s = states[200]
    xmm = grid.x * 1e3
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax[0, 0].plot(xmm, s["T_front"][c], label="front face"); ax[0, 0].plot(xmm, s["T_back"][c], label="back face (HR)")
    ax[0, 0].set_title("temperature, 200 W pump"); ax[0, 0].set_ylabel("K"); ax[0, 0].legend()
    ax[0, 1].plot(xmm, s["bulge"][c] * 1e6, label="bulge of front face")
    ax[0, 1].plot(xmm, -s["w_bend"][c] * 1e6, label="bending (edges recede)")
    ax[0, 1].set_title("surface deformation"); ax[0, 1].set_ylabel("um"); ax[0, 1].legend()
    for key, lab in (("opd", "total"), ("opd_bulge", "bulge"), ("opd_dndT", "dn/dT"), ("opd_bend", "bending")):
        ax[0, 2].plot(xmm, s[key][c] / lam, label=lab)
    ax[0, 2].set_title("optical path difference in reflection"); ax[0, 2].set_ylabel("waves"); ax[0, 2].legend()
    ext = np.array([-1, 1, -1, 1]) * grid.size / 2 * 1e3
    im = ax[1, 0].imshow(s["T_front"], extent=ext, origin="lower", cmap="inferno"); plt.colorbar(im, ax=ax[1, 0], label="K")
    ax[1, 0].set_title("front face temperature")
    im = ax[1, 1].imshow(s["opd"] / lam, extent=ext, origin="lower", cmap="viridis"); plt.colorbar(im, ax=ax[1, 1], label="waves")
    ax[1, 1].set_title("OPD map")
    for P in powers:
        ax[1, 2].plot(xmm, states[P]["opd"][c] / lam, label=f"{P} W")
    ax[1, 2].set_title("OPD vs pump power"); ax[1, 2].set_ylabel("waves"); ax[1, 2].legend()
    for a in ax.ravel():
        a.set_xlabel("x [mm]"); a.grid(alpha=0.3)
    for a in ax[0]:
        a.set_xlim(-4, 4)
    fig.tight_layout()
    out = os.path.join(results_dir("8_thermal"), "disk_thermal.png")
    fig.savefig(out, dpi=110)
    print("saved", os.path.abspath(out))
