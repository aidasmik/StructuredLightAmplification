"""
slm_correction.py - Correct a deformed seed beam back to a Gaussian with the SLM.

A phase-only SLM can remove wavefront (phase) errors; amplitude errors it
can only redistribute. The procedure mimics an SLM + wavefront-sensor loop:

    1. the deformed seed arrives at the SLM plane,
    2. its phase error relative to the ideal Gaussian is measured there,
    3. the SLM displays the conjugate phase,
    4. the residual after the (real, imperfect) SLM is measured and the
       pattern is updated; a few iterations remove the SLM's own
       quantisation and crosstalk errors.

The corrected and uncorrected beams are then sent through the complete
multipass system (thin_disk_system) and compared with the ideal output.
"""

import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seed_beam import Grid, gaussian, results_dir
from propagation import propagate
from optics import beam_splitter, _flip_x
import slm
import thin_disk_system as tds


def overlap(a, b):
    """Fraction of the power of beam a that is in the mode of beam b."""
    return abs(np.sum(a.field * np.conj(b.field)) * a.grid.dx ** 2) ** 2 / (a.power * b.power)


def phase_error(beam, reference):
    """Wrapped phase difference beam - reference (piston removed) and its
    intensity-weighted rms."""
    cross = beam.field * np.conj(reference.field)
    cross = cross * np.conj(np.sum(cross))            # remove the constant piston
    dphi = np.angle(cross)
    I = beam.intensity
    rms = np.sqrt(np.sum(I * dphi ** 2) / I.sum())
    return dphi, rms


if __name__ == "__main__":
    results = results_dir("6.5_slm_correction")
    grid = Grid(**tds.SIM_GRID)
    lam, w0 = tds.LAM, tds.W0

    # --- deformed seed: wavefront aberrations + 8 % amplitude ellipticity ---
    ideal = tds.seed_beam(grid)                     # mode-matched seed of the current design
    aberration = slm.zernike({"astig0": 0.8, "coma_x": 0.6, "trefoil30": 0.5,
                              "spherical": 0.3, "defocus": 0.4}, radius=1.5e-3)
    deformed = replace(ideal, field=ideal.field * np.exp(-0.08 * (grid.X / w0) ** 2))
    deformed = deformed.add_phase(aberration(grid.X, grid.Y)).normalize(1.0)
    deformed.name = "deformed seed"

    # --- fields arriving at the SLM (BS reflection + 0.3 m), ideal and deformed ---
    d_bs_slm = np.linalg.norm(tds.POS["SLM"] - tds.POS["BS"])
    at_slm = propagate(beam_splitter(deformed, 0.5)[0], d_bs_slm, "air")
    ref_slm = propagate(beam_splitter(ideal, 0.5)[0], d_bs_slm, "air")
    ref_after = replace(ref_slm, field=_flip_x(ref_slm.field))          # SLM flips the image

    # --- closed-loop correction ---
    pattern_map = np.zeros((grid.n, grid.n))
    light = at_slm.intensity > 1e-3 * at_slm.intensity.max()   # measure only where there is light
    gain = 0.8                                                  # loop gain < 1 for stability
    best = (np.inf, None, None)
    print("iteration  rms wavefront error [rad]  overlap with ideal Gaussian")
    for it in range(6):
        out = tds.SLM_DEVICE.modulate(at_slm, slm.map_pattern(pattern_map, grid), bias=tds.SLM_BIAS)
        resid, rms = phase_error(out, ref_after)
        print(f"{it:9d}  {rms:25.3f}  {overlap(out, ref_after):27.4f}")
        if rms < best[0]:
            best = (rms, pattern_map.copy(), out)
        pattern_map -= gain * _flip_x(resid) * light            # back to pre-flip coordinates
    rms, pattern_map, out = best                                # keep the best iteration
    correction = slm.map_pattern(pattern_map, grid)
    dphi_in, rms_in = phase_error(at_slm, ref_slm)
    print(f"input at SLM: rms wavefront error {rms_in:.3f} rad, overlap {overlap(at_slm, ref_slm):.4f}")
    # --- figure: incoming beam | SLM correction | outgoing beam ---
    ext = np.array([-1, 1, -1, 1]) * grid.size / 2 * 1e3
    c = grid.n // 2

    def axis_phase(b):
        """Phase relative to the beam axis, blanked where there is no light."""
        m = b.intensity > 1e-3 * b.intensity.max()
        return np.where(m, np.angle(b.field * np.conj(b.field[c, c])), np.nan)

    shown = light                                                    # SLM area that matters
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (at_slm.intensity, "incoming beam at SLM: intensity", "inferno", None, ""),
        (np.where(shown, np.mod(pattern_map + tds.SLM_BIAS, 2 * np.pi), np.nan),
         "SLM pattern displayed (wrapped)", "twilight", (0, 2 * np.pi), "rad"),
        (out.intensity, "outgoing beam: intensity", "inferno", None, ""),
        (axis_phase(at_slm), "incoming beam: phase", "twilight", (-np.pi, np.pi), "rad"),
        (np.where(shown, pattern_map - pattern_map[c, c], np.nan),
         "SLM correction phase (unwrapped)", "RdBu_r", "sym", "rad"),
        (axis_phase(out), "outgoing beam: phase", "twilight", (-np.pi, np.pi), "rad"),
    ]
    for a, (img, title, cmap, lim, unit) in zip(ax.ravel(), panels):
        if lim == "sym":
            lim = (-np.nanmax(np.abs(img)), np.nanmax(np.abs(img)))
        kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
        im = a.imshow(img, extent=ext, origin="lower", cmap=cmap, **kw)
        a.set_title(title); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
        a.set_xlim(-3, 3); a.set_ylim(-3, 3)
        plt.colorbar(im, ax=a, label=unit)
    fig.suptitle(f"SLM wavefront correction: rms error {rms_in:.3f} rad -> {rms:.3f} rad, "
                 f"Gaussian overlap {overlap(at_slm, ref_slm):.3f} -> {overlap(out, ref_after):.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "slm_in_correction_out.png"), dpi=110)

    # --- through the full system ---
    out_ideal = tds.run(slm.flat(), "ideal_seed", grid, results)
    out_uncorr = tds.run(slm.flat(), "deformed_uncorrected", grid, results, beam_in=deformed)
    out_corr = tds.run(correction, "deformed_corrected", grid, results, beam_in=deformed)
    for name, b in (("ideal seed", out_ideal), ("deformed, SLM flat", out_uncorr),
                    ("deformed, SLM corrected", out_corr)):
        wx, wy = b.widths()
        print(f"system output, {name:24s}: wx={wx*1e3:.3f} wy={wy*1e3:.3f} mm, "
              f"P={b.power:.4f} W, fraction in ideal output mode {overlap(b, out_ideal):.4f}")

    # --- figure ---
    ext = np.array([-1, 1, -1, 1]) * grid.size / 2 * 1e3
    mask = ideal.intensity > 1e-3 * ideal.intensity.max()
    resid_corr, _ = phase_error(out, ref_after)
    fig, ax = plt.subplots(3, 3, figsize=(13, 11))
    panels = [
        (deformed.intensity, "deformed seed: intensity", "inferno", None),
        (np.where(mask, np.angle(deformed.field), np.nan), "deformed seed: phase", "twilight", np.pi),
        (np.where(mask, dphi_in, np.nan), "measured wavefront error at SLM", "twilight", np.pi),
        (np.mod(pattern_map + tds.SLM_BIAS, 2 * np.pi) * mask, "SLM pattern (wrapped)", "twilight", None),
        (out.intensity, "after SLM: intensity", "inferno", None),
        (np.where(mask, resid_corr, np.nan), "after SLM: residual phase", "twilight", 0.5),
        (out_uncorr.intensity, "system output, uncorrected", "inferno", None),
        (out_corr.intensity, "system output, corrected", "inferno", None),
        (out_ideal.intensity, "system output, ideal seed", "inferno", None),
    ]
    for a, (img, title, cmap, lim) in zip(ax.ravel(), panels):
        kw = {} if lim is None else {"vmin": -lim, "vmax": lim}
        im = a.imshow(img, extent=ext, origin="lower", cmap=cmap, **kw)
        a.set_title(title); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
        a.set_xlim(-3, 3); a.set_ylim(-3, 3)
        plt.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(os.path.join(results, "slm_correction.png"), dpi=110)
    print("saved", os.path.join(os.path.abspath(results), "slm_correction.png"))
