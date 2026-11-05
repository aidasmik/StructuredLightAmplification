"""
thermal_study.py - Run the multipass system with the thermally deformed disk
for several pump powers (no gain yet) and compare the output with the cold case.
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
from thermal import DiskThermal, fit_lens


def overlap(a, b):
    return abs(np.sum(a.field * np.conj(b.field)) * a.grid.dx ** 2) ** 2 / (a.power * b.power)


if __name__ == "__main__":
    results = results_dir("8_thermal")
    grid = Grid(**tds.SIM_GRID)
    powers = (0, 100, 200, 300, 400)
    rows, outputs, spots = [], {}, {}
    for P in powers:
        th = DiskThermal(pump_power=P)
        state = th.solve(grid)
        tds.DISK.thermal_opd = state["opd"]
        f_th, resid = fit_lens(state["opd"], grid, 1.0e-3, tds.LAM)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            out = tds.run(slm.flat(), f"pump_{P}W", grid, results)
        # spot on the disk at each pass from the printed log
        disk_w = [float(l.split()[3]) for l in buf.getvalue().splitlines() if l.split()[1:2] == ["Disk"]]
        outputs[P] = out; spots[P] = disk_w
        wx, wy = out.widths()
        rows.append((P, state["T_front"].max(), f_th, resid, min(disk_w), max(disk_w), wx * 1e3, wy * 1e3, out.power))

    print(f"{'pump[W]':>7} {'T_front[K]':>10} {'f_thermal[m]':>12} {'aberr[waves]':>12} "
          f"{'disk spot min/max[mm]':>22} {'out wx/wy[mm]':>14} {'P[W]':>7} {'overlap cold':>12}")
    for r in rows:
        ov = overlap(outputs[r[0]], outputs[0])
        print(f"{r[0]:7.0f} {r[1]:10.1f} {r[2]:12.2f} {r[3]:12.4f} {r[4]:10.3f}/{r[5]:.3f}{'':>5} "
              f"{r[6]:6.3f}/{r[7]:.3f} {r[8]:7.4f} {ov:12.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for P in powers:
        ax[0].plot(range(1, len(spots[P]) + 1), spots[P], "o-", label=f"{P} W")
    ax[0].set_xlabel("disk pass"); ax[0].set_ylabel("spot radius on disk [mm]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("spot on the disk vs pump power")
    c = grid.n // 2
    xmm = grid.x * 1e3
    for P in powers:
        I = outputs[P].intensity / outputs[P].power
        ax[1].plot(xmm, I[c], label=f"{P} W")
    ax[1].set_xlim(-3, 3); ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("W/m$^2$ (1 W)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("output intensity profile")
    ax[2].plot([r[0] for r in rows], [r[6] for r in rows], "o-", label="output wx")
    ax[2].plot([r[0] for r in rows], [r[7] for r in rows], "s-", label="output wy")
    ax[2].plot([r[0] for r in rows], [overlap(outputs[r[0]], outputs[0]) for r in rows], "k^--", label="overlap with cold output")
    ax[2].set_xlabel("pump power [W]"); ax[2].legend(); ax[2].grid(alpha=0.3)
    ax[2].set_title("output vs pump power")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "system_vs_pump.png"), dpi=110)
    print("saved", os.path.join(os.path.abspath(results), "system_vs_pump.png"))
