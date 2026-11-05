"""
correction_benchmark.py - Step 10: upper bound of phase-only pre-correction.

For several target structured modes and several distortion classes of the
amplifier, compute
    naive      : SLM pattern designed for the undistorted amplifier, applied
                 to the distorted one (what happens without correction)
    corrected  : physics-optimal pattern for the distorted amplifier
                 (alternating projections through the amplifier), ideal SLM
    corrected, real SLM : the same pattern through the pixelated LCOS model
    complex ideal : complex-amplitude modulation (phase + amplitude), ideal
    complex Davis : complex-amplitude encoding on the real LCOS with a blazed
                    carrier (first order), including its power cost
and report the output fidelity |<E_out, E_target>|^2 (normalised) and, for
the complex cases, the output power relative to the flat-pattern phase-only case.
Optimisation and evaluation on the 1024 grid (the relay design focuses to
0.1 mm on M1/M2, which a coarser grid cannot represent).
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
import correction as co
from thermal import DiskThermal

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Distortions (OPD maps in reflection, metres), defined as functions of the grid
# --------------------------------------------------------------------------
def hot_spot_field(grid, rng_seed=5, n_spots=12, sigma=0.7e-3, rms=0.2):
    """Grid-independent random hot-spot pattern inside the pump area."""
    rng = np.random.default_rng(rng_seed)
    eps = np.zeros((grid.n, grid.n))
    for _ in range(n_spots):
        x0, y0 = rng.uniform(-1.8e-3, 1.8e-3, 2)
        eps += rng.standard_normal() * np.exp(-((grid.X - x0) ** 2 + (grid.Y - y0) ** 2) / (2 * sigma ** 2))
    inside = grid.R < 2e-3
    eps = eps - eps[inside].mean()                     # zero mean: structure only, the total heat load is unchanged
    return rms * eps / eps[inside].std()


def disk_shape(grid):
    """Static surface error of the disk: 60 nm astigmatism, 40 nm trefoil,
    30 nm coma at r = 2 mm (OPD = 2 x surface in reflection)."""
    surf = slm.zernike({"astig0": 60e-9, "trefoil30": 40e-9, "coma_x": 30e-9}, 2e-3)(grid.X, grid.Y)
    return 2 * surf


DISTORTIONS = {
    "none": lambda g: None,
    "thermal_400W": lambda g: DiskThermal(pump_power=400).opd(g),
    "disk_shape": disk_shape,
    "hot_spots_200W": lambda g: (lambda th: th.solve(g, Q=th.heat_flux(g) * (1 + hot_spot_field(g)))["opd"])(DiskThermal(pump_power=200)),
    "all": lambda g: (lambda th: th.solve(g, Q=th.heat_flux(g) * (1 + hot_spot_field(g)))["opd"])(DiskThermal(pump_power=400)) + disk_shape(g),
}


def setup(grid, distortion):
    """Set the disk state and return (amplifier, field at SLM, reference output)."""
    tds.DISK.thermal_opd = None
    ref, _ = tds.trace(slm.flat(), grid)                 # undistorted Gaussian output = reference
    tds.DISK.thermal_opd = DISTORTIONS[distortion](grid)
    return co.Amplifier(grid), co.field_at_slm(grid).field, ref


def resample(pattern, grid_from, grid_to):
    return slm.map_pattern(pattern, grid_from)(grid_to.X, grid_to.Y)


if __name__ == "__main__":
    results = results_dir("10_correction")
    g_opt, g_eval = Grid(**tds.FINE_GRID), Grid(**tds.FINE_GRID)   # relay design: the 0.1 mm spots on M1/M2 need the full grid
    iters = 15

    # undistorted amplifier: naive patterns for every target
    amp0, fin0, ref0 = setup(g_opt, "none")
    targets0 = co.target_modes(g_opt, ref0)
    naive = {}
    for name, tgt in targets0.items():
        naive[name], hist = co.optimise_pattern(amp0, fin0, tgt, iters)
        print(f"undistorted amplifier, target {name:8s}: fidelity {hist[0]:.3f} -> {hist[-1]:.4f} after {iters} iterations")

    rows, pictures = [], {}
    for dist in DISTORTIONS:
        amp, fin, _ = setup(g_opt, dist)
        amp_e, fin_e, ref_e = setup(g_eval, dist)
        fin_e_beam = co.field_at_slm(g_eval)
        targets_e = co.target_modes(g_eval, ref_e)
        for name, tgt in targets0.items():
            corrected, hist = co.optimise_pattern(amp, fin, tgt, iters)
            tgt_e = targets_e[name]
            p_naive = resample(naive[name], g_opt, g_eval)
            p_corr = resample(corrected, g_opt, g_eval)
            out_naive = amp_e.forward(co.ideal_slm(fin_e, p_naive))
            out_corr = amp_e.forward(co.ideal_slm(fin_e, p_corr))
            realised = tds.SLM_DEVICE.modulate(fin_e_beam, slm.map_pattern(p_corr, g_eval), bias=tds.SLM_BIAS).field
            out_real = amp_e.forward(realised)
            # complex-amplitude modulation: direct back-projection of the target
            out_flat = amp_e.forward(co.ideal_slm(fin_e, np.zeros_like(p_corr)))
            p_flat = np.vdot(out_flat, out_flat).real
            align = np.exp(1j * np.angle(np.vdot(tgt_e, out_corr)))
            desired = amp_e.adjoint(tgt_e * align)
            f_cplx, eff_cplx, ratio, phase = co.complex_modulation(fin_e, desired)
            out_cplx = amp_e.forward(f_cplx)
            out_davis = amp_e.forward(co.davis_realise(fin_e_beam, ratio, phase))
            eff_c = np.vdot(out_cplx, out_cplx).real / p_flat
            eff_d = np.vdot(out_davis, out_davis).real / p_flat
            f_naive, f_corr, f_real, f_c, f_d = (co.fidelity(o, tgt_e) for o in (out_naive, out_corr, out_real, out_cplx, out_davis))
            rows.append((dist, name, f_naive, f_corr, f_real, f_c, eff_c, f_d, eff_d))
            pictures[(dist, name)] = (tgt_e, out_naive, out_real, out_davis)
            print(f"{dist:15s} {name:8s}: phase-only naive {f_naive:.3f} | optimal {f_corr:.3f} real {f_real:.3f} || "
                  f"complex ideal {f_c:.3f} (eff {eff_c:.2f}) | Davis real {f_d:.3f} (eff {eff_d:.2f})")

    with open(os.path.join(results, "fidelity_table.csv"), "w") as fh:
        fh.write("distortion,target,phase_only_naive,phase_only_optimal_ideal_slm,phase_only_optimal_real_slm,"
                 "complex_ideal_fidelity,complex_ideal_efficiency,complex_davis_real_fidelity,complex_davis_real_efficiency\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")

    # figure: the "all" distortion, one row per target
    ext = np.array([-1, 1, -1, 1]) * g_eval.size / 2 * 1e3
    c = g_eval.n // 2
    names = list(targets0)
    fig, ax = plt.subplots(len(names), 6, figsize=(21, 3.6 * len(names)))
    for i, name in enumerate(names):
        tgt, o_naive, o_real, o_davis = pictures[("all", name)]
        fields = [tgt, o_naive, o_real, o_davis]
        titles = ["target", "phase-only, naive pattern", "phase-only, optimal (real SLM)", "complex Davis (real SLM)"]
        for j, (f, t) in enumerate(zip(fields, titles)):
            I = np.abs(f) ** 2 / np.abs(tgt).max() ** 2
            ax[i, j].imshow(I, extent=ext, origin="lower", cmap="inferno", vmin=0, vmax=1.05 * (np.abs(f).max() / np.abs(tgt).max()) ** 2)
            fid = "" if j == 0 else f", F = {co.fidelity(f, tgt):.3f}"
            ax[i, j].set_title(f"{name}: {t}{fid}", fontsize=9)
        for j, f in ((4, tgt), (5, o_davis)):
            m = np.abs(f) ** 2 > 1e-2 * np.abs(f).max() ** 2
            ph = np.where(m, np.angle(f * np.conj(f[c, c + 1])), np.nan)
            ax[i, j].imshow(ph, extent=ext, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax[i, j].set_title(f"{name}: phase, {'target' if j == 4 else 'complex Davis output'}", fontsize=9)
        for a in ax[i]:
            a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Distortion 'all' (400 W thermal lens + hot spots + static disk shape): pre-correction at the SLM")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "correction_all_distortions.png"), dpi=100)

    # fidelity bar chart
    fig, ax = plt.subplots(figsize=(12, 4.5))
    dists = list(DISTORTIONS)
    width = 0.2
    for k, name in enumerate(names):
        vals_n = [next(r[2] for r in rows if r[0] == d and r[1] == name) for d in dists]
        vals_c = [next(r[4] for r in rows if r[0] == d and r[1] == name) for d in dists]
        vals_d = [next(r[7] for r in rows if r[0] == d and r[1] == name) for d in dists]
        x = np.arange(len(dists)) + (k - 1.5) * width
        ax.bar(x, vals_d, width, color=f"C{k}", alpha=0.25)
        ax.bar(x, vals_c, width * 0.6, color=f"C{k}", alpha=1.0, label=name)
        ax.plot(x, vals_n, "kx", ms=5)
    ax.set_xticks(np.arange(len(dists))); ax.set_xticklabels(dists); ax.set_ylabel("output fidelity"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left", title="x: naive | bar: phase-only optimal | faint: complex Davis"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Pre-correction at the SLM: achievable output fidelity per target mode and distortion (real LCOS)")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "fidelity_summary.png"), dpi=110)
    print("saved results to", os.path.abspath(results))
