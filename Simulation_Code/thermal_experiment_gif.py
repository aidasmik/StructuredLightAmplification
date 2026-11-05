"""
thermal_experiment_gif.py - What a camera at the amplifier output would show
when only realistic pump noise heats the disk (whole-crystal temperature
noise, no hot spots, no deliberate modulation):

    fast noise : 1.5 % rms, 5 ms correlation   (diode current / driver noise)
    slow drift : 3 % rms, 0.5 s correlation    (diode and cooling-water temperature)

The disk follows through its thermal inertia (superposition of step
responses). Panels: pump and crystal temperature traces, output radius trace, the disk itself
(front-face temperature, surface shape as map and cross-section, HR radius of
curvature), output intensity, output phase, and the intensity difference from
the time-averaged beam, which is how such small changes are detected in practice.
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


def ar1(t, rms, tau, rng):
    """AR(1) noise with rms amplitude and correlation time tau on the grid t."""
    rho = np.exp(-(t[1] - t[0]) / tau)
    x = np.zeros_like(t)
    for i in range(1, len(t)):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    return rms * x


if __name__ == "__main__":
    results = results_dir("9_thermal_transient")
    grid = Grid(n=512, size=12.8e-3)
    lam = tds.LAM
    c = grid.n // 2

    P0, dt, t_end = 200.0, 2e-3, 2.0
    t = np.arange(0, t_end + dt, dt)
    rng = np.random.default_rng(7)
    P = P0 * (1 + ar1(t, 0.015, 0.005, rng) + ar1(t, 0.03, 0.5, rng))

    th = DiskThermal(pump_power=P0)
    g_mean, g_diff = th.transient(t)
    # start from the steady state at P0 (pump has been on for a long time), then add the noise increments
    dP = np.diff(np.concatenate([[P0], P]))
    G_mean = g_mean[-1] + np.array([np.sum(dP[:i + 1] / P0 * g_mean[i - np.arange(i + 1)]) for i in range(len(t))])
    G_diff = g_diff[-1] + np.array([np.sum(dP[:i + 1] / P0 * g_diff[i - np.arange(i + 1)]) for i in range(len(t))])
    T_front = np.array([th.solve_at(grid, gm, gd)["T_front"][c, c] for gm, gd in zip(G_mean, G_diff)])

    frame_every = 10                                  # 20 ms per frame
    idx = list(range(0, len(t), frame_every))
    outs, radii, f_th, states = [], [], [], []
    for i in idx:
        state = th.solve_at(grid, G_mean[i], G_diff[i])
        tds.DISK.thermal_opd = state["opd"]
        out, _ = tds.trace(slm.flat(), grid)
        outs.append(out); radii.append(np.mean(out.widths())); f_th.append(fit_lens(state["opd"], grid, 1e-3, lam)[0])
        states.append(state)
        print(f"t = {t[i]*1e3:5.0f} ms: P = {P[i]:6.1f} W, T_front = {T_front[i]:6.2f} K, "
              f"f_th = {f_th[-1]:6.2f} m, w = {radii[-1]*1e3:.4f} mm")
    tds.DISK.thermal_opd = None
    radii = np.array(radii)
    f_th = np.array(f_th)
    I_mean = np.mean([o.intensity / o.power for o in outs], axis=0)
    roc_disk = np.array([1 / st["kappa"][c, c] for st in states])          # bent HR surface at the centre
    surf_max = max((st["bulge"] - st["w_bend"]).max() for st in states) * 1e6
    surf_min = min((st["bulge"] - st["w_bend"]).min() for st in states) * 1e6

    ext = np.array([-1, 1, -1, 1]) * grid.size / 2 * 1e3
    frames = []
    for k, i in enumerate(idx):
        out = outs[k]
        I = out.intensity / out.power
        mask = I > 1e-3 * I.max()
        ph = np.where(mask, np.angle(out.field * np.conj(out.field[c, c])), np.nan)
        st = states[k]
        front = (st["bulge"] - st["w_bend"]) * 1e6          # front-face height towards the beam [um]
        back = -st["w_bend"] * 1e6                          # HR back face [um]
        fig, ax = plt.subplots(3, 3, figsize=(15, 12))
        ax[0, 0].plot(t * 1e3, P, color="tab:red", lw=0.8)
        ax[0, 0].set_ylabel("pump power [W]", color="tab:red"); ax[0, 0].set_xlabel("time [ms]")
        a2 = ax[0, 0].twinx(); a2.plot(t * 1e3, T_front, color="tab:blue", lw=1.5); a2.set_ylabel("T_front [K]", color="tab:blue")
        ax[0, 0].axvline(t[i] * 1e3, color="k", lw=1)
        ax[0, 0].set_title(f"t = {t[i]*1e3:.0f} ms: P = {P[i]:.1f} W, T = {T_front[i]:.1f} K, f_th = {f_th[k]:.2f} m")
        ax[0, 1].plot(t[idx] * 1e3, radii * 1e3, "-", color="tab:green"); ax[0, 1].plot(t[i] * 1e3, radii[k] * 1e3, "ko")
        ax[0, 1].set_xlabel("time [ms]"); ax[0, 1].set_ylabel("output radius [mm]")
        ax[0, 1].set_title(f"output radius: {radii.mean()*1e3:.3f} mm ± {radii.std()/radii.mean()*100:.2f} %")
        ax[0, 2].plot(grid.x * 1e3, I[c], label="now"); ax[0, 2].plot(grid.x * 1e3, I_mean[c], "k--", lw=0.8, label="time average")
        ax[0, 2].set_xlim(-3, 3); ax[0, 2].set_xlabel("x [mm]"); ax[0, 2].set_title("intensity cross-section"); ax[0, 2].legend()
        im = ax[1, 0].imshow(I, extent=ext, origin="lower", cmap="inferno", vmin=0, vmax=I_mean.max() * 1.1)
        ax[1, 0].set_title("output intensity (camera)"); plt.colorbar(im, ax=ax[1, 0])
        im = ax[1, 1].imshow(ph, extent=ext, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[1, 1].set_title("output phase [rad]"); plt.colorbar(im, ax=ax[1, 1])
        im = ax[1, 2].imshow((I - I_mean) / I_mean.max() * 100, extent=ext, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3)
        ax[1, 2].set_title("difference from time average [% of peak]"); plt.colorbar(im, ax=ax[1, 2])
        # --- disk row ---
        im = ax[2, 0].imshow(st["T_front"], extent=ext, origin="lower", cmap="inferno",
                             vmin=th.sink_temperature, vmax=T_front.max())
        ax[2, 0].set_title(f"disk front-face temperature [K], max {st['T_front'].max():.1f} K"); plt.colorbar(im, ax=ax[2, 0])
        im = ax[2, 1].imshow(front, extent=ext, origin="lower", cmap="coolwarm", vmin=surf_min, vmax=surf_max)
        ax[2, 1].set_title("disk front-face height towards the beam [um]"); plt.colorbar(im, ax=ax[2, 1])
        xmm = grid.x * 1e3
        ax[2, 2].plot(xmm, front[c], label="front (AR) face")
        ax[2, 2].plot(xmm, back[c], label="back (HR) face")
        ax[2, 2].fill_between(xmm, back[c], front[c], color="gold", alpha=0.3, label="crystal (thickness exaggerated)")
        ax[2, 2].set_ylim(surf_min * 1.1, surf_max * 1.3); ax[2, 2].set_xlim(-5, 5)
        ax[2, 2].set_xlabel("x [mm]"); ax[2, 2].set_ylabel("height [um]"); ax[2, 2].legend(loc="lower center", fontsize=8)
        ax[2, 2].set_title(f"disk (convex): R_HR = {roc_disk[k]:.2f} m, f_th = {f_th[k]:.2f} m", fontsize=10)
        for a in ax[1]:
            a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
        for a in ax[2, :2]:
            a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
        for a in list(ax[0, :2]) + [ax[2, 2]]:
            a.grid(alpha=0.3)
        fig.suptitle("Expected view with realistic pump noise only (1.5 % fast + 3 % slow drift), 200 W, no modulation")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)

    out_path = os.path.join(results, "beam_experiment_noise.gif")
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"output radius {radii.mean()*1e3:.3f} mm, rms variation {radii.std()/radii.mean()*100:.2f} %, "
          f"thermal lens {min(f_th):.2f} .. {max(f_th):.2f} m, T_front {T_front.min():.1f} .. {T_front.max():.1f} K")
    print("saved", os.path.abspath(out_path))
