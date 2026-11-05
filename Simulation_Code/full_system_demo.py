"""
full_system_demo.py - Realistic scenario at moderate pump: the complete
passive system with every non-uniformity and thermal effect of the crystal.

    disk        0.5 mm Yb:YAG, 10 at.% with +-2 % 3-D doping clusters,
                static shape error (30 nm astigmatism, 20 nm trefoil, 15 nm coma),
                diamond heat sink
    pump        300 W (slides), 2 mm radius (assumed), mild 5 % rms hot-spot structure
    thermal     axial profile per column, bulge, bimaterial bending, dn/dT,
                doping index term, all as one reflection OPD map
    system      confocal Fourier-transform multipass (flat array, concave M1/M2),
                mode-matched 0.70 mm seed, flat SLM

Figures (Results/12_full_system/): layout, disk state with the OPD
contributions, beam radius along the path, intensity at every element,
intensity and phase at the disk passes, input vs output.
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seed_beam import Grid, results_dir
import thin_disk_system as tds
import slm
from propagation import propagate
from thermal import fit_lens
from doping import DopedDiskThermal, doping_map_3d
from correction_benchmark import hot_spot_field

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    results = results_dir("12_full_system")
    grid = Grid(n=512, size=12.8e-3)              # dx = 25 um (flat SLM pattern: pixels irrelevant)
    lam = tds.LAM
    c = grid.n // 2
    ext = np.array([-1, 1, -1, 1]) * grid.size / 2 * 1e3

    # --- the crystal state ---------------------------------------------------
    P = 300.0                                     # slides: 300 W pump
    th = DopedDiskThermal(pump_power=P, doping=0.10, n_pump_passes=24)
    c3, delta3, z = doping_map_3d(grid, th.thickness, nz=50, nominal=0.10, amplitude=0.02, cluster_size=0.3e-3, seed=1)
    hot = 1 + hot_spot_field(grid, rng_seed=5, rms=0.05)
    state = th.solve_3d(grid, c3, z, heat_scale=hot)
    shape = 2 * slm.zernike({"astig0": 30e-9, "trefoil30": 20e-9, "coma_x": 15e-9}, 2e-3)(grid.X, grid.Y)
    uniform = DopedDiskThermal(pump_power=P, doping=0.10, doping_field=np.full((grid.n, grid.n), 0.10), n_pump_passes=24).solve(grid)
    opd_total = state["opd"] + shape
    contributions = {
        "thermal bulge": state["opd_bulge"], "thermal dn/dT": state["opd_dndT"], "thermal bending": state["opd_bend"],
        "doping index term": state["opd_index"], "static shape error": shape,
        "hot-spot structure": state["opd"] - state["opd_index"] - uniform["opd"],
    }
    beam = grid.R < 1e-3
    f_th, resid = fit_lens(opd_total, grid, 1e-3, lam)
    print(f"disk at {P:.0f} W: T_front max {state['T_front'].max():.1f} K, thermal lens {f_th:.1f} m, "
          f"non-spherical residual {resid:.4f} waves rms inside the beam")
    for k, v in contributions.items():
        print(f"  {k:20s}: inside the beam p-v {np.ptp(v[beam])/lam:7.4f} waves, rms {v[beam].std()/lam:7.4f} waves")

    # --- figure: disk state ---------------------------------------------------
    fig, ax = plt.subplots(3, 3, figsize=(15, 13))
    panels = [
        (delta3.mean(axis=0) * 100, "doping deviation, column mean [%]", "RdBu_r", (-2, 2)),
        ((hot - 1) * 100, "pump structure [%]", "RdBu_r", (-15, 15)),
        (state["T_front"], "front-face temperature [K]", "inferno", None),
        ((state["bulge"] - state["w_bend"]) * 1e6, "front-face height towards the beam [um]", "coolwarm", None),
        (contributions["thermal bending"] / lam, "OPD: bending [waves]", "viridis", None),
        ((contributions["thermal bulge"] + contributions["thermal dn/dT"]) / lam, "OPD: bulge + dn/dT [waves]", "viridis", None),
        (contributions["doping index term"] / lam, "OPD: doping index term [waves]", "RdBu_r", None),
        ((contributions["static shape error"] + contributions["hot-spot structure"]) / lam, "OPD: shape error + hot spots [waves]", "RdBu_r", None),
        (opd_total / lam, "OPD total [waves]", "viridis", None),
    ]
    for a, (img, title, cmap, lim) in zip(ax.ravel(), panels):
        kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
        im = a.imshow(img, extent=ext, origin="lower", cmap=cmap, **kw)
        a.set_title(title, fontsize=10); a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
        a.add_patch(plt.Circle((0, 0), 0.7, fill=False, color="w", lw=0.8, ls="--"))
        plt.colorbar(im, ax=a)
    fig.suptitle(f"Disk state at {P:.0f} W: 3-D doping clusters, 5 % pump structure, static shape error, thermal response "
                 f"(dashed: 0.7 mm seed)")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "disk_state.png"), dpi=110)

    # --- the system ------------------------------------------------------------
    tds.DISK.thermal_opd = opd_total
    tds.plot_layout(results)
    out = tds.run(slm.flat(), "realistic", grid, results)
    tds.DISK.thermal_opd = None
    cold, _ = tds.trace(slm.flat(), grid)
    fid = abs(np.vdot(out.field, cold.field)) ** 2 / (np.vdot(out.field, out.field).real * np.vdot(cold.field, cold.field).real)
    print(f"output: w = {out.widths()[0]*1e3:.3f}/{out.widths()[1]*1e3:.3f} mm (cold {cold.widths()[0]*1e3:.3f}), "
          f"fidelity to the cold output {fid:.4f}")

    # --- montage: the beam at every element ----------------------------------------
    tds.DISK.thermal_opd = opd_total
    snaps = {}
    lengths, aois = tds.path_geometry()
    b = tds.seed_beam(grid)
    fields = []
    for i, name in enumerate(tds.PATH):
        if i > 0:
            b = propagate(b, lengths[i - 1], "air")
        b = tds.apply(name, b, aois[i], slm.flat())
        fields.append((name, b.intensity.copy(), b.widths()))
    tds.DISK.thermal_opd = None
    ncol = 7
    nrow = int(np.ceil(len(fields) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.8 * nrow))
    for a, (name, I, (wx, wy)) in zip(ax.ravel(), fields):
        half = 3 * max(wx, wy) * 1e3
        a.imshow(I, extent=ext, origin="lower", cmap="inferno")
        a.set_xlim(-half, half); a.set_ylim(-half, half); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{name}  w={wx*1e3:.2f}/{wy*1e3:.2f} mm", fontsize=8)
    for a in ax.ravel()[len(fields):]:
        a.axis("off")
    fig.suptitle("Beam intensity at every element (each panel scaled to 3 beam radii)")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "beam_at_every_element.png"), dpi=110)
    print("saved figures to", os.path.abspath(results))
