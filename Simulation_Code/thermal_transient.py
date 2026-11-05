"""
thermal_transient.py - OPD of the disk after a change of its temperature and
the resulting beam change:

    1. pump switched on at t = 0: temperature, OPD, thermal lens vs time
    2. pump switched off from steady state: decay of the OPD
    3. heat-sink temperature sweep at constant pump
    4. the multipass output at selected instants after switch-on
"""

import os
import contextlib
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seed_beam import Grid, results_dir
import thin_disk_system as tds
import slm
from thermal import DiskThermal, fit_lens, conductivity, RHO, C_P


def overlap(a, b):
    return abs(np.sum(a.field * np.conj(b.field)) * a.grid.dx ** 2) ** 2 / (a.power * b.power)


if __name__ == "__main__":
    results = results_dir("9_thermal_transient")
    grid = Grid(**tds.SIM_GRID)
    lam = tds.LAM
    c = grid.n // 2
    P = 200.0
    th = DiskThermal(pump_power=P)
    k = conductivity(th.solve_centre_temperature(), th.doping)
    tau = th.thickness ** 2 / (np.pi ** 2 * k / (RHO * C_P))
    print(f"axial thermal time constant h^2/(pi^2 D) = {tau*1e3:.1f} ms (k = {k:.1f} W/m/K)")

    # 1) switch-on transient
    times = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    g_mean, g_diff = th.transient(times)
    steady = th.solve(grid)
    f_ss, _ = fit_lens(steady["opd"], grid, 1e-3, lam)
    print(f"steady state: g_mean {g_mean[-1]:.3e}, g_diff {g_diff[-1]:.3e} K m^2/W  "
          f"(analytic {1/th.contact_conductance + th.thickness/(3*k):.3e}, {th.thickness/(2*k):.3e}), f_thermal {f_ss:.2f} m")
    print(f"\n{'t [s]':>7} {'T_front centre [K]':>18} {'front-back [K]':>14} {'OPD centre [waves]':>18} {'f_thermal [m]':>13}")
    states = {}
    for t, gm, gd in zip(times, g_mean, g_diff):
        s = th.solve_at(grid, gm, gd); states[t] = s
        f, _ = fit_lens(s["opd"], grid, 1e-3, lam)
        print(f"{t:7.4f} {s['T_front'][c,c]:18.1f} {s['T_front'][c,c]-s['T_back'][c,c]:14.1f} "
              f"{s['opd'][c,c]/lam:18.3f} {f:13.2f}")

    # 2) switch-off: linear system -> response = steady - switch-on response
    print("\nswitch-off from steady state:")
    off = {}
    for t, gm, gd in zip(times, g_mean, g_diff):
        s = th.solve_at(grid, g_mean[-1] - gm, g_diff[-1] - gd); off[t] = s
        f, _ = fit_lens(s["opd"], grid, 1e-3, lam)
        print(f"  t = {t*1e3:6.1f} ms: OPD centre {s['opd'][c,c]/lam:6.3f} waves, f_thermal {f:8.2f} m")

    # 3) heat-sink temperature sweep (steady state, same pump)
    print("\nheat-sink temperature sweep at 200 W:")
    for Ts in (273.0, 283.0, 293.0, 303.0, 313.0, 333.0):
        s = DiskThermal(pump_power=P, sink_temperature=Ts).solve(grid)
        f, res = fit_lens(s["opd"], grid, 1e-3, lam)
        print(f"  T_sink = {Ts:5.0f} K: T_front max {s['T_front'].max():6.1f} K, "
              f"OPD centre {s['opd'][c,c]/lam:6.3f} waves, f_thermal {f:7.2f} m, aberration {res:.4f} waves rms")

    # 4) beam through the multipass at selected instants after switch-on
    print("\nmultipass output after switch-on:")
    cold = None
    rows = []
    for t in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 1.0):
        tds.DISK.thermal_opd = None if t == 0 else states[t]["opd"]
        with contextlib.redirect_stdout(io.StringIO()):
            out = tds.run(slm.flat(), f"t_{t*1e3:.0f}ms", grid, results)
        if cold is None:
            cold = out
        wx, wy = out.widths()
        rows.append((t, wx * 1e3, wy * 1e3, overlap(out, cold)))
        print(f"  t = {t*1e3:6.1f} ms: output wx/wy = {wx*1e3:.3f}/{wy*1e3:.3f} mm, overlap with cold {overlap(out, cold):.4f}")
    tds.DISK.thermal_opd = None

    # figure
    xmm = grid.x * 1e3
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for t in (0.001, 0.005, 0.02, 0.1, 1.0):
        ax[0, 0].plot(xmm, states[t]["T_front"][c], label=f"{t*1e3:.0f} ms")
    ax[0, 0].set_title("front-face temperature after switch-on (200 W)"); ax[0, 0].set_ylabel("K"); ax[0, 0].legend()
    for t in (0.001, 0.005, 0.02, 0.1, 1.0):
        ax[0, 1].plot(xmm, states[t]["opd"][c] / lam, label=f"{t*1e3:.0f} ms")
    ax[0, 1].set_title("OPD after switch-on"); ax[0, 1].set_ylabel("waves"); ax[0, 1].legend()
    tt = times * 1e3
    ax[1, 0].semilogx(tt, [states[t]["T_front"][c, c] - th.sink_temperature for t in times], "o-", label="front-face rise, on")
    ax[1, 0].semilogx(tt, [states[t]["T_front"][c, c] - states[t]["T_back"][c, c] for t in times], "s-", label="front-back, on")
    ax[1, 0].semilogx(tt, [off[t]["T_front"][c, c] - th.sink_temperature for t in times], "o--", label="front-face rise, off")
    ax[1, 0].set_xlabel("time [ms]"); ax[1, 0].set_ylabel("K"); ax[1, 0].legend(); ax[1, 0].set_title("centre temperatures vs time")
    ax[1, 1].semilogx(tt, [1 / fit_lens(states[t]["opd"], grid, 1e-3, lam)[0] for t in times], "o-", label="pump on")
    ax[1, 1].semilogx(tt, [1 / fit_lens(off[t]["opd"], grid, 1e-3, lam)[0] for t in times], "s--", label="pump off")
    ax[1, 1].set_xlabel("time [ms]"); ax[1, 1].set_ylabel("thermal lens power 1/f [1/m]"); ax[1, 1].legend()
    ax[1, 1].set_title("thermal lens vs time")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    for a in ax[0]:
        a.set_xlim(-4, 4); a.set_xlabel("x [mm]")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "opd_transient.png"), dpi=110)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r[0] * 1e3 for r in rows[1:]], [r[1] for r in rows[1:]], "o-", label="output wx [mm]")
    ax.plot([r[0] * 1e3 for r in rows[1:]], [r[3] for r in rows[1:]], "k^--", label="overlap with cold output")
    ax.set_xscale("log"); ax.set_xlabel("time after pump switch-on [ms]"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("multipass output during the thermal transient (200 W)")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "output_vs_time.png"), dpi=110)
    print("saved figures to", os.path.abspath(results))
