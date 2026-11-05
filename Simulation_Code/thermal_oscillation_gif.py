"""
thermal_oscillation_gif.py - The whole crystal changes temperature in time:
the heat load oscillates and carries noise, P(t) = P0 (1 + a sin(2 pi f t) + noise),
and the disk follows with its thermal inertia. Animated: pump and crystal
temperature traces, disk OPD, output intensity and phase.

The heat equation is linear, so the response to an arbitrary P(t) is the
superposition of step responses g(t) from DiskThermal.transient():
    G(t) = sum_k (dP_k / P0) * g(t - t_k)      (dP_k: increments of P at t_k)
and the maps at time t follow from solve_at(grid, G_mean(t), G_diff(t)).
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from seed_beam import Grid, results_dir
import thin_disk_system as tds
import slm
from thermal import DiskThermal, fit_lens

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    results = results_dir("9_thermal_transient")
    grid = Grid(n=512, size=12.8e-3)
    lam = tds.LAM
    c = grid.n // 2

    # --- heat-load history: 3 Hz oscillation (30 %) + random noise (10 % rms, 20 ms correlation)
    P0, amp, f_osc, noise_rms = 200.0, 0.30, 3.0, 0.10
    dt, t_end = 2e-3, 0.6
    t = np.arange(0, t_end + dt, dt)
    rng = np.random.default_rng(3)
    rho = np.exp(-dt / 0.02)
    noise = np.zeros_like(t)
    for i in range(1, len(t)):
        noise[i] = rho * noise[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    P = P0 * (1 + amp * np.sin(2 * np.pi * f_osc * t) + noise_rms * noise)
    P = np.clip(P, 0, None)

    # --- disk response by superposition of step responses
    th = DiskThermal(pump_power=P0)
    g_mean, g_diff = th.transient(t)                  # unit step responses on the time grid
    dP = np.diff(np.concatenate([[0.0], P]))          # increments, first one = switch-on
    G_mean = np.array([np.sum(dP[:i + 1] / P0 * g_mean[i - np.arange(i + 1)]) for i in range(len(t))])
    G_diff = np.array([np.sum(dP[:i + 1] / P0 * g_diff[i - np.arange(i + 1)]) for i in range(len(t))])
    T_front = np.array([th.solve_at(grid, gm, gd)["T_front"][c, c] for gm, gd in zip(G_mean, G_diff)])

    # --- frames
    frame_every = 5                                   # 10 ms per frame
    ext = np.array([-1, 1, -1, 1]) * grid.size / 2 * 1e3
    frames = []
    for i in range(0, len(t), frame_every):
        state = th.solve_at(grid, G_mean[i], G_diff[i])
        tds.DISK.thermal_opd = state["opd"]
        out, _ = tds.trace(slm.flat(), grid)
        wx, wy = out.widths()
        f_th, _ = fit_lens(state["opd"], grid, 1e-3, lam)
        I = out.intensity / out.power
        mask = I > 1e-3 * I.max()
        ph = np.where(mask, np.angle(out.field * np.conj(out.field[c, c])), np.nan)

        fig, ax = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
        ax[0].plot(t * 1e3, P, color="tab:red", lw=1, label="pump heat load P(t)")
        ax[0].set_ylabel("pump power [W]", color="tab:red"); ax[0].set_xlabel("time [ms]")
        a2 = ax[0].twinx()
        a2.plot(t * 1e3, T_front, color="tab:blue", lw=1.5, label="crystal front-face T")
        a2.set_ylabel("T_front centre [K]", color="tab:blue")
        ax[0].axvline(t[i] * 1e3, color="k", lw=1)
        ax[0].set_title(f"t = {t[i]*1e3:.0f} ms, T = {T_front[i]:.1f} K, f_thermal = {f_th:.1f} m")
        im = ax[1].imshow(state["opd"] / lam, extent=ext, origin="lower", cmap="viridis", vmin=0, vmax=3)
        ax[1].set_title("disk OPD [waves]"); plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(I, extent=ext, origin="lower", cmap="inferno", vmin=0, vmax=6e5)
        ax[2].set_title(f"output intensity, w = {wx*1e3:.2f}/{wy*1e3:.2f} mm"); plt.colorbar(im, ax=ax[2])
        im = ax[3].imshow(ph, extent=ext, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[3].set_title("output phase [rad]"); plt.colorbar(im, ax=ax[3])
        for a in ax[1:]:
            a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
        fig.suptitle("whole-disk temperature oscillation: 200 W pump, 30 % at 3 Hz + 10 % noise")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
        print(f"t = {t[i]*1e3:5.0f} ms: P = {P[i]:6.1f} W, T_front = {T_front[i]:6.1f} K, "
              f"f_th = {f_th:7.2f} m, output w = {wx*1e3:.3f}/{wy*1e3:.3f} mm")

    tds.DISK.thermal_opd = None
    out_path = os.path.join(results, "beam_thermal_oscillation.gif")
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print("saved", os.path.abspath(out_path))
