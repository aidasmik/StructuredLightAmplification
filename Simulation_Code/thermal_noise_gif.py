"""
thermal_noise_gif.py - Animated output beam of the multipass while the disk
is distorted by a fluctuating temperature field.

Noise model: the heat load of the 200 W pump is multiplied by (1 + eps(x, y, t)),
where eps is a Gaussian random field with correlation length `corr` (hot
spots from pump inhomogeneity) and rms amplitude `rms`, evolving in time as
an AR(1) process (frame-to-frame correlation `rho`). The thermal model turns
each heat-flux map into an OPD map, the multipass is traced, and the output
intensity and phase are drawn.
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

from seed_beam import Grid, results_dir
import thin_disk_system as tds
import slm
from thermal import DiskThermal

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    results = results_dir("9_thermal_transient")
    grid = Grid(n=512, size=12.8e-3)          # coarser grid for speed (flat SLM pattern)
    lam = tds.LAM
    c = grid.n // 2
    n_frames, rms, corr, rho, frame_dt = 40, 0.20, 0.7e-3, 0.85, 0.01   # 10 ms per frame

    th = DiskThermal(pump_power=200.0)
    Q0 = th.heat_flux(grid)
    rng = np.random.default_rng(1)

    def noise_field():
        f = gaussian_filter(rng.standard_normal((grid.n, grid.n)), corr / grid.dx)
        return f / f.std()

    eps = noise_field()
    cold, _ = tds.trace(slm.flat(), grid)
    ext = np.array([-1, 1, -1, 1]) * grid.size / 2 * 1e3
    frames = []
    for k in range(n_frames):
        eps = rho * eps + np.sqrt(1 - rho ** 2) * noise_field()
        state = th.solve(grid, Q=Q0 * (1 + rms * eps))
        tds.DISK.thermal_opd = state["opd"]
        out, _ = tds.trace(slm.flat(), grid)
        wx, wy = out.widths()
        I = out.intensity / out.power
        mask = I > 1e-3 * I.max()
        ph = np.where(mask, np.angle(out.field * np.conj(out.field[c, c])), np.nan)
        T = state["T_front"]
        opd = state["opd"] / lam

        fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
        im = ax[0].imshow(opd, extent=ext, origin="lower", cmap="viridis", vmin=0, vmax=3)
        ax[0].set_title(f"disk OPD [waves], T_front max {T.max():.0f} K"); plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(I, extent=ext, origin="lower", cmap="inferno", vmin=0, vmax=6e5)
        ax[1].set_title(f"output intensity, wx/wy = {wx*1e3:.2f}/{wy*1e3:.2f} mm"); plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(ph, extent=ext, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[2].set_title("output phase [rad]"); plt.colorbar(im, ax=ax[2])
        for a in ax:
            a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
        fig.suptitle(f"200 W pump with {rms*100:.0f}% rms heat-load noise, t = {k*frame_dt*1e3:.0f} ms")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
        print(f"frame {k+1}/{n_frames}: wx={wx*1e3:.3f} wy={wy*1e3:.3f} mm, T_front max {T.max():.1f} K")

    tds.DISK.thermal_opd = None
    out = os.path.join(results, "beam_thermal_noise.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=150, loop=0)
    print("saved", os.path.abspath(out))
