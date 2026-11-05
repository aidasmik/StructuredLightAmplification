"""
doping.py - Non-uniform Yb doping of the thin disk and its consequences.

Doping field:  c(x, y, z) = c0 (1 + delta),  |delta| <= amplitude (default 2 %),
delta a clustered random field (filtered Gaussian noise, correlation length
`cluster_size` in all three directions). doping_map() gives the depth-uniform
2-D version, doping_map_3d() the full volume (z = 0 at the HR back face).
With 3-D clusters every column (x, y) has its own absorption, conductivity
and index profile along z; the optics sees the depth integrals.

Properties that depend on the local Yb concentration c:
    ion density          N = c * 1.38e28 m^-3           (Y sites in YAG)
    pump absorption      alpha_p = sigma_p N,  sigma_p = 0.8e-24 m^2 at 969 nm
                         -> absorbed fraction for n_p double passes and, more
                         important for a thick disk, the depth profile of the
                         heat deposition (more doping = heat nearer the front)
    thermal conductivity k(T, c) = 11 / (1 + 6.5 c) (300/T)^0.7 W/m/K
    refractive index     n(c) = n_YAG + DN_DC (c - c0),  DN_DC = 0.02 per unit
                         doping fraction (2e-4 per at.%)        [PLACEHOLDER]
    1030 nm absorption / gain  alpha = sigma N (unpumped), g = (sigma_e N2 - sigma_a N1):
                         both scale with N -> +-2 % doping = +-2 % local gain
                         per pass once the gain model exists
    lattice strain       Yb3+ is smaller than Y3+: strain ~ -6.7e-5 per at.%,
                         photo-elastic OPD ~ 1e-9 m for +-0.2 at.% -> neglected

Thermal consequence: the axial temperature profile is solved per doping value
with the exponential pump deposition (first pass front->back, reflected at
the HR, remaining passes) instead of a uniform source. The thickness-mean rise
and the front-back difference per unit heat flux, g_mean(c), g_diff(c), feed
the same deformation model as thermal.py, plus the direct index term
    OPD_index = 2 h (n(c) - n(c0)).
"""

from dataclasses import dataclass, field

import numpy as np
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # numpy 2 / numpy 1 compatibility
from scipy.ndimage import gaussian_filter

from thermal import DiskThermal, conductivity, dndT, expansion, _integrate_axisymmetric

N_SITES = 1.38e28          # Y sites per m^3 in YAG
SIGMA_PUMP = 0.8e-24       # absorption cross-section at 969 nm [m^2]
DN_DC = 0.02               # dn per unit doping fraction, PLACEHOLDER (2e-4 per at.%)


def doping_map(grid, nominal=0.10, amplitude=0.02, cluster_size=0.3e-3, seed=0):
    """Clustered random doping field c(x, y); returns (c, delta)."""
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((grid.n, grid.n)), cluster_size / grid.dx)
    delta = np.clip(f / (2.5 * f.std()), -1, 1) * amplitude      # |delta| <= amplitude, ~0.8 % rms
    return nominal * (1 + delta), delta


def doping_map_3d(grid, thickness, nz=50, nominal=0.10, amplitude=0.02, cluster_size=0.3e-3, seed=0):
    """Clustered random doping field c(z, y, x) on grid x grid x nz; returns (c, delta, z)."""
    rng = np.random.default_rng(seed)
    dz = thickness / nz
    f = gaussian_filter(rng.standard_normal((nz, grid.n, grid.n)),
                        (cluster_size / dz, cluster_size / grid.dx, cluster_size / grid.dx), mode="nearest")
    delta = np.clip(f / (2.5 * f.std()), -1, 1) * amplitude
    return nominal * (1 + delta), delta, (np.arange(nz) + 0.5) * dz


def column_response(c3, z, thickness, n_passes, k3):
    """Steady 1-D temperature rise per unit heat flux for every column of a
    3-D doping field with its own absorption alpha(z) and conductivity k(z).
    c3, k3: arrays (nz, ny, nx); returns (g_mean, g_diff, absorbed fraction)
    with the contact term excluded (add 1 / h_contact to g_mean separately)."""
    h = thickness
    dz = z[1] - z[0]
    a = SIGMA_PUMP * N_SITES * c3                               # alpha(z) per column
    # optical depth from the front face (z = h) down to each z, and total
    tau_from_front = np.cumsum(a[::-1], axis=0)[::-1] * dz - a * dz / 2
    tau_total = a.sum(axis=0) * dz
    q = np.zeros_like(a)
    remaining = np.ones_like(tau_total)
    for _ in range(n_passes):
        down = np.exp(-tau_from_front)                          # pump travelling front -> back
        up = np.exp(-tau_total) * np.exp(-(tau_total - tau_from_front))   # reflected at HR, back -> front
        q += remaining * a * (down + up)
        remaining = remaining * np.exp(-2 * tau_total)
    absorbed = 1 - remaining
    q = q / (q.sum(axis=0) * dz)                               # normalised deposition per unit flux
    flow = np.cumsum(q[::-1], axis=0)[::-1] * dz               # heat flowing towards the back at z
    T = np.cumsum(flow / k3, axis=0) * dz                      # rise above the back face
    return T.mean(axis=0), T[-1] - T[0], absorbed


def absorbed_fraction(c, thickness, n_passes):
    """Pump fraction absorbed after n_passes double passes through the disk."""
    return 1 - np.exp(-2 * SIGMA_PUMP * N_SITES * c * thickness * n_passes)


def axial_response(c, thickness, n_passes, k, contact_conductance, nz=200):
    """Steady 1-D temperature rise per unit heat flux with the pump deposition
    profile of doping c: returns (g_mean, g_diff) [K m^2 / W].
    z = 0 at the HR back face (cooled), z = h adiabatic front face."""
    h = thickness
    z = np.linspace(0, h, nz)
    a = SIGMA_PUMP * N_SITES * c
    q = np.zeros(nz)
    remaining = 1.0
    for _ in range(n_passes):                     # each pass: front -> back, HR, back -> front
        q += remaining * a * (np.exp(-a * (h - z)) + np.exp(-a * (h + z)))
        remaining *= np.exp(-2 * a * h)
    q /= _trapz(q, z)                       # normalised deposition per unit flux [1/m]
    # heat flow towards the back: k T'(z) = int_z^h q dz'
    flow = _trapz(q, z) - np.concatenate([[0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(z))])
    T = 1 / contact_conductance + np.concatenate([[0], np.cumsum(0.5 * (flow[1:] + flow[:-1]) * np.diff(z))]) / k
    return _trapz(T, z) / h, T[-1] - T[0]


@dataclass
class DopedDiskThermal(DiskThermal):
    """DiskThermal with a doping map: local absorption, conductivity, axial
    deposition profile and the direct index change."""
    doping_field: np.ndarray = None       # c(x, y) on the grid
    n_pump_passes: int = 24
    n_index: float = 1.815

    def solve(self, grid, Q=None):
        h = self.thickness
        c = self.doping_field if self.doping_field is not None else np.full((grid.n, grid.n), self.doping)
        c0 = self.doping
        # heat flux: pump profile x local absorbed fraction (relative to nominal absorption eta_abs)
        if Q is None:
            p = np.exp(-2 * (grid.R / self.pump_radius) ** self.pump_order)
            p = p / (p.sum() * grid.dx ** 2)
            eta_local = absorbed_fraction(c, h, self.n_pump_passes) / absorbed_fraction(c0, h, self.n_pump_passes)
            Q = self.eta_heat * self.eta_abs * self.pump_power * p * eta_local
            Q = gaussian_filter(Q, h / grid.dx)
        # axial response as a function of doping (table), evaluated at the local k(T, c)
        T_mean = self.sink_temperature + Q / self.contact_conductance
        for _ in range(3):
            k = conductivity(T_mean, c)
            cs = np.linspace(c.min(), c.max(), 15) if c.max() > c.min() else np.array([c0])
            gm = np.array([axial_response(ci, h, self.n_pump_passes, 1.0, 1e30)[0] for ci in cs])   # per unit k
            gd = np.array([axial_response(ci, h, self.n_pump_passes, 1.0, 1e30)[1] for ci in cs])
            g_mean = np.interp(c, cs, gm) / k                                                      # scale by local k
            g_diff = np.interp(c, cs, gd) / k
            T_mean = self.sink_temperature + Q / self.contact_conductance + Q * g_mean
        dT_axial = Q * g_diff
        out = self._deform(grid, Q, T_mean, dT_axial)
        out["opd_index"] = 2 * h * DN_DC * (c - c0)                  # direct index term (cold effect)
        out["opd"] = out["opd"] + out["opd_index"]
        out["doping"] = c
        return out

    def solve_3d(self, grid, c3, z, heat_scale=None):
        """Like solve(), for a 3-D doping field c3(z, y, x): column-wise
        absorption, deposition and conductivity profiles; index term
        integrated through the thickness. heat_scale: optional map
        multiplying the pump profile (e.g. 1 + hot spots)."""
        h = self.thickness
        c0 = self.doping
        p = np.exp(-2 * (grid.R / self.pump_radius) ** self.pump_order)
        p = p / (p.sum() * grid.dx ** 2)
        if heat_scale is not None:
            p = p * heat_scale
        T_mean = np.full((grid.n, grid.n), self.sink_temperature + 30.0)
        for _ in range(3):
            k3 = conductivity(T_mean[None, :, :], c3)
            gm, gd, absorbed = column_response(c3, z, h, self.n_pump_passes, k3)
            eta_local = absorbed / absorbed_fraction(c0, h, self.n_pump_passes)
            Q = gaussian_filter(self.eta_heat * self.eta_abs * self.pump_power * p * eta_local, h / grid.dx)
            T_mean = self.sink_temperature + Q / self.contact_conductance + Q * gm
        dT_axial = Q * gd
        out = self._deform(grid, Q, T_mean, dT_axial)
        out["opd_index"] = 2 * DN_DC * (c3 - c0).sum(axis=0) * (z[1] - z[0])   # depth integral
        out["opd"] = out["opd"] + out["opd_index"]
        out["doping"] = c3.mean(axis=0)
        return out


# --------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seed_beam import Grid, results_dir
    from thermal import fit_lens
    import thin_disk_system as tds
    import slm

    warnings.filterwarnings("ignore")
    results = results_dir("11_doping")
    lam = tds.LAM
    grid = Grid(n=512, size=10.24e-3)
    c = grid.n // 2
    c0, h, n_p = 0.10, 0.5e-3, 24

    cmap, delta = doping_map(grid, c0, 0.02, 0.3e-3, seed=1)
    print(f"doping field: nominal {c0*100:.0f} at.%, deviation {delta.min()*100:+.2f} .. {delta.max()*100:+.2f} %, "
          f"rms {delta.std()*100:.2f} %, cluster size 0.3 mm")

    # property sensitivities at +-2 %
    k0 = conductivity(350.0, c0)
    print("\nlocal property changes for +2 % / -2 % doping (at 350 K):")
    for name, fn, unit in (
        ("Yb density", lambda cc: cc * N_SITES, "m^-3"),
        ("pump absorption, 1 double pass", lambda cc: absorbed_fraction(cc, h, 1), ""),
        (f"pump absorption, {n_p} double passes", lambda cc: absorbed_fraction(cc, h, n_p), ""),
        ("thermal conductivity", lambda cc: conductivity(350.0, cc), "W/m/K"),
        ("refractive index - n(c0)", lambda cc: DN_DC * (cc - c0), ""),
        ("unpumped 1030 nm absorption alpha", lambda cc: 1.2e-25 * cc * N_SITES / 100, "1/cm"),
    ):
        v0, vp, vm = fn(c0), fn(c0 * 1.02), fn(c0 * 0.98)
        if v0 == 0:
            print(f"  {name:36s}: {v0:10.4g} {unit:6s}  {vp:+.2e} / {vm:+.2e} (absolute)")
        else:
            print(f"  {name:36s}: {v0:10.4g} {unit:6s}  {(vp/v0-1)*100:+.2f} % / {(vm/v0-1)*100:+.2f} %")
    gm0, gd0 = axial_response(c0, h, n_p, k0, 3e5)
    gmp, gdp = axial_response(c0 * 1.02, h, n_p, k0, 3e5)
    gmu, gdu = 1 / 3e5 + h / (3 * k0), h / (2 * k0)
    print(f"  axial profile (pump deposition vs uniform source): mean rise x{gm0/gmu:.3f}, front-back x{gd0/gdu:.3f}; "
          f"+2 % doping -> front-back {(gdp/gd0-1)*100:+.2f} %, mean {(gmp/gm0-1)*100:+.2f} %")

    # thermal + optical effect at 200 W: doped vs uniform
    P = 200.0
    uni = DopedDiskThermal(pump_power=P, doping=c0, doping_field=np.full((grid.n, grid.n), c0), n_pump_passes=n_p)
    dop = DopedDiskThermal(pump_power=P, doping=c0, doping_field=cmap, n_pump_passes=n_p)
    su, sd = uni.solve(grid), dop.solve(grid)
    beam = grid.R < 1.0e-3
    d_opd = sd["opd"] - su["opd"]
    d_th = d_opd - sd["opd_index"]
    print(f"\n200 W pump: uniform disk T_front max {su['T_front'].max():.1f} K, OPD centre {su['opd'][c,c]/lam:.3f} waves")
    print(f"  doped disk: T_front differs by {np.abs(sd['T_front']-su['T_front']).max():.2f} K max")
    print(f"  OPD difference inside the 1 mm beam: thermal part rms {d_th[beam].std()/lam:.4f} waves, "
          f"index part rms {sd['opd_index'][beam].std()/lam:.4f} waves, total p-v {np.ptp(d_opd[beam])/lam:.4f} waves")
    f_u, r_u = fit_lens(su["opd"], grid, 1e-3, lam); f_d, r_d = fit_lens(sd["opd"], grid, 1e-3, lam)
    print(f"  thermal lens {f_u:.2f} -> {f_d:.2f} m, non-spherical residual {r_u:.4f} -> {r_d:.4f} waves rms")

    # beam through the multipass
    def fid(a, b):
        return abs(np.vdot(a.field, b.field)) ** 2 / (np.vdot(a.field, a.field).real * np.vdot(b.field, b.field).real)
    outs = {}
    for tag, opd in (("cold uniform", None), ("cold doped (index term only)", sd["opd_index"]),
                     ("200 W uniform", su["opd"]), ("200 W doped", sd["opd"])):
        tds.DISK.thermal_opd = opd
        outs[tag], _ = tds.trace(slm.flat(), grid)
    tds.DISK.thermal_opd = None
    print("\nmultipass output (Gaussian seed):")
    for tag, b in outs.items():
        ref = outs["cold uniform"] if "cold" in tag else outs["200 W uniform"]
        wx, wy = b.widths()
        print(f"  {tag:28s}: w = {wx*1e3:.3f}/{wy*1e3:.3f} mm, fidelity to the uniform case {fid(b, ref):.5f}")

    # 3-D clusters: the same statistics, varying through the thickness as well
    c3, delta3, z = doping_map_3d(grid, h, nz=50, nominal=c0, amplitude=0.02, cluster_size=0.3e-3, seed=1)
    s3 = dop.solve_3d(grid, c3, z)
    d3 = s3["opd"] - su["opd"]
    print(f"\n3-D clusters (0.3 mm in x, y and z; {h/0.3e-3:.1f} cluster lengths through the thickness):")
    print(f"  column-mean doping deviation: rms {delta3.mean(axis=0).std()*100:.2f} % (2-D field: {delta.std()*100:.2f} %)")
    print(f"  index OPD rms inside the beam: {s3['opd_index'][beam].std()/lam:.4f} waves (2-D: {sd['opd_index'][beam].std()/lam:.4f})")
    print(f"  thermal OPD change rms: {(d3 - s3['opd_index'])[beam].std()/lam:.4f} waves, total p-v {np.ptp(d3[beam])/lam:.4f} waves")
    f_3, r_3 = fit_lens(s3["opd"], grid, 1e-3, lam)
    print(f"  thermal lens {f_u:.2f} -> {f_3:.2f} m, non-spherical residual {r_3:.4f} waves rms")
    tds.DISK.thermal_opd = s3["opd"]
    out3, _ = tds.trace(slm.flat(), grid)
    tds.DISK.thermal_opd = s3["opd_index"]
    out3c, _ = tds.trace(slm.flat(), grid)
    tds.DISK.thermal_opd = None
    print(f"  multipass output: cold 3-D doped fidelity {fid(out3c, outs['cold uniform']):.5f}, "
          f"200 W 3-D doped fidelity {fid(out3, outs['200 W uniform']):.5f}")
    np.save(os.path.join(results, "doping_map_3d.npy"), c3)

    # figure
    ext = np.array([-1, 1, -1, 1]) * grid.size / 2 * 1e3
    fig, ax = plt.subplots(3, 3, figsize=(15, 13))
    im = ax[0, 0].imshow(delta * 100, extent=ext, origin="lower", cmap="RdBu_r", vmin=-2, vmax=2); plt.colorbar(im, ax=ax[0, 0], label="%")
    ax[0, 0].set_title("doping deviation from 10 at.% (random clusters)")
    im = ax[0, 1].imshow(sd["T_front"] - su["T_front"], extent=ext, origin="lower", cmap="RdBu_r"); plt.colorbar(im, ax=ax[0, 1], label="K")
    ax[0, 1].set_title("front-face temperature change, 200 W")
    im = ax[0, 2].imshow(d_th / lam, extent=ext, origin="lower", cmap="RdBu_r"); plt.colorbar(im, ax=ax[0, 2], label="waves")
    ax[0, 2].set_title("thermal OPD change (absorption depth, k)")
    im = ax[1, 0].imshow(sd["opd_index"] / lam, extent=ext, origin="lower", cmap="RdBu_r"); plt.colorbar(im, ax=ax[1, 0], label="waves")
    ax[1, 0].set_title("direct index OPD, 2 h dn/dc (c - c0)  [placeholder dn/dc]")
    im = ax[1, 1].imshow(d_opd / lam, extent=ext, origin="lower", cmap="RdBu_r"); plt.colorbar(im, ax=ax[1, 1], label="waves")
    ax[1, 1].set_title("total OPD change, doped - uniform")
    I_u = outs["200 W uniform"].intensity / outs["200 W uniform"].power
    I_d = outs["200 W doped"].intensity / outs["200 W doped"].power
    im = ax[1, 2].imshow((I_d - I_u) / I_u.max() * 100, extent=ext, origin="lower", cmap="RdBu_r"); plt.colorbar(im, ax=ax[1, 2], label="% of peak")
    ax[1, 2].set_title("multipass output intensity change, 200 W")
    # third row: the 3-D field
    im = ax[2, 0].imshow(delta3[:, c, :] * 100, extent=[-grid.size/2*1e3, grid.size/2*1e3, 0, h*1e3], origin="lower",
                         cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto"); plt.colorbar(im, ax=ax[2, 0], label="%")
    ax[2, 0].set_title("3-D doping: x-z section through the disk (y = 0)"); ax[2, 0].set_xlabel("x [mm]"); ax[2, 0].set_ylabel("z [mm] (0 = HR)")
    im = ax[2, 1].imshow(delta3.mean(axis=0) * 100, extent=ext, origin="lower", cmap="RdBu_r", vmin=-2, vmax=2); plt.colorbar(im, ax=ax[2, 1], label="%")
    ax[2, 1].set_title("3-D doping: column mean (what the beam integrates)")
    im = ax[2, 2].imshow(d3 / lam, extent=ext, origin="lower", cmap="RdBu_r"); plt.colorbar(im, ax=ax[2, 2], label="waves")
    ax[2, 2].set_title("total OPD change, 3-D doped - uniform, 200 W")
    for a in list(ax[0]) + list(ax[1]) + [ax[2, 1], ax[2, 2]]:
        a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
    ax[2, 0].set_xlim(-4, 4)
    fig.suptitle("Non-uniform Yb doping (+-2 %, 0.3 mm clusters), depth-uniform (rows 1-2) and 3-D (row 3)")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "doping_effects.png"), dpi=110)
    np.save(os.path.join(results, "doping_map.npy"), cmap)
    print("saved", os.path.abspath(results))
