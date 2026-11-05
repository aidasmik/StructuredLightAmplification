"""
thin_disk_system.py - Passive (no gain) optical model of the thin-disk
multipass system: seed -> beam splitter / SLM arm -> M0 -> multipass
amplifier (mirror array, relay mirrors M1/M2, thin disk) -> output.

The thin disk is the physical model from disk.py: 0.5 mm Yb:YAG, AR front
face, HR back face; here flat and without absorption (no gain yet).

Beam routing (same topology as the PSI/ETH design, Schuhmann et al. 2015):
A1 -> Disk -> A2 -> M1 -> A3 -> Disk -> A4 -> M1 -> ... with the same end
mirrors M1 / M2 used for all passes. The disk is flat (any curvature is
thermal), so the mirrors must provide the imaging. Two designs (DESIGN):

  "relay"   : each unit disk -> A -> M -> A' -> disk is a -I relay (4f-type),
              f_array = L2,  f_M = L2^2 / (2 (L2 - L1)).  Spot on the disk is
              reproduced for any input beam, but a thermal lens on the disk
              accumulates linearly with the number of passes.
  "fourier" : each unit is an optical Fourier transform (A = D = 0,
              B = pi w0^2 / lambda), i.e. an unfolded stable resonator.
              The seed must be mode matched (waist W_DISK on the disk); in
              return the spot on the disk is insensitive to the thermal lens
              to first order. W_DISK = None gives the confocal special case
              with flat array mirrors and R_M = 2 (L1 + L2).
The entrance and exit mirrors A1 / A12 only steer the beam and are flat.
All radii are collected in MIRROR_ROC and can be overridden.

The SLM is the physical LCOS model from slm.py; the pattern it displays is
chosen in __main__ (flat, vortex, lens, astigmatism, ...).

All dimensions are ASSUMPTIONS (edit the GEOMETRY section):
    disk ROC 2 m, mirror array on an arc of radius 1 m centred on the disk,
    12 array mirrors 5 deg apart, relay mirrors M1/M2 at 0.5 m, 6 disk passes.

The beam is simulated in its own frame (seed_beam / propagation / optics /
media / slm modules) and checked against an ABCD Gaussian-beam trace.
"""

import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

from seed_beam import Grid, gaussian, results_dir
from propagation import propagate
from optics import mirror, beam_splitter
import slm
from disk import ThinDisk

# --------------------------------------------------------------------------
# GEOMETRY  (top view, metres; the disk sits at the origin facing -x)
# --------------------------------------------------------------------------
LAM = 1030e-9
W0 = 1.0e-3                          # beam 1/e^2 radius on the disk [m] (user: 1 mm; slides' 1.5 mm would clip the 5 mm disk)
SEED_POWER = 0.1                     # CW seed power [W] (slides: 0.1 .. 0.2 W)
SIM_GRID = dict(n=512, size=12.8e-3)      # default grid, dx = 25 um: traces agree with the 12.5 um grid to >= 0.998 fidelity, 6x faster
FINE_GRID = dict(n=1024, size=12.8e-3)    # dx = 12.5 um = one SLM pixel: for pixel-level SLM studies (blazed carrier, Davis encoding)
DISK = ThinDisk(thickness=0.5e-3, roc=np.inf, doping=0.0, diameter=5e-3)   # flat, 5 mm diameter, gain/absorption via DISK.gain
DESIGN = "relay"                     # "relay" or "fourier" (see docstring); the relay reproduces the
                                     # 1.5 mm seed at the output with practical mirror radii
W_DISK = None                        # fourier design only: spot radius on the disk [m]; None = confocal
ARRAY_RADIUS = 1.0                   # disk -> array mirrors
RELAY_RADIUS = 0.5                   # disk -> M1 / M2
N_PASSES = 10                        # reflections on the disk (slides: 1, 2, 5, 10 passes)
ARRAY_SPACING_DEG = 2.75             # angular spacing of the 2 N_PASSES array mirrors
SLM_DEVICE = slm.SLM(pitch=12.5e-6, n_pixels=(1272, 1024), fill_factor=0.96, reflectivity=0.97)  # Hamamatsu X15213
SLM_BIAS = np.pi                     # centre patterns in the 0..2pi range

N_ARRAY = 2 * N_PASSES
angles = np.deg2rad(ARRAY_SPACING_DEG * ((N_ARRAY - 1) / 2 - np.arange(N_ARRAY)))   # A1..A20, top to bottom
POS = {f"A{i+1}": -ARRAY_RADIUS * np.array([np.cos(a), -np.sin(a)])
       for i, a in enumerate(angles)}
# relay links: after pass k the beam goes A(2k) -> M -> A(2k+1); M1 serves the first
# half of the links, M2 the rest. Each M sits on the symmetry axis of its array mirrors.
_half = N_PASSES // 2
RELAY_GROUPS = {"M1": [], "M2": []}
for k in range(1, N_PASSES):
    RELAY_GROUPS["M1" if k <= _half else "M2"] += [f"A{2*k}", f"A{2*k+1}"]
_mean_angle = {m: np.mean([angles[int(a[1:]) - 1] for a in grp]) for m, grp in RELAY_GROUPS.items()}
POS.update({
    "Disk": np.array([0.0, 0.0]),
    "M1":   -RELAY_RADIUS * np.array([np.cos(_mean_angle["M1"]), -np.sin(_mean_angle["M1"])]),
    "M2":   -RELAY_RADIUS * np.array([np.cos(_mean_angle["M2"]), -np.sin(_mean_angle["M2"])]),
    "M0":   np.array([-0.60, 1.30]),     # placed so that A1 / A12 work at ~50 deg incidence
    "BS":   np.array([-1.30, 1.30]),
    "SLM":  np.array([-1.00, 1.30]),
    "Out":  np.array([-0.60, -1.30]),
})

# Beam path: element names in order. Pass k: A(2k-1) -> Disk -> A(2k),
# then relay M1 (upper half) or M2 (lower half) to A(2k+1).
PATH = ["BS", "SLM", "BS", "M0"]
for k in range(N_PASSES):
    PATH += [f"A{2*k+1}", "Disk", f"A{2*k+2}"]
    if k < N_PASSES - 1:
        PATH.append("M1" if k + 1 <= _half else "M2")
PATH.append("Out")


def _path_length(*names):
    return sum(np.linalg.norm(POS[b] - POS[a]) for a, b in zip(names[:-1], names[1:]))


def _abcd_T(d):
    return np.array([[1, d], [0, 1.0]])


def _abcd_L(f):
    return np.array([[1, 0], [-1 / f, 1.0]])


def _unit_matrix(fa, fm, L1, L2):
    """ABCD matrix of one unit disk -> A -> M -> A' -> disk."""
    return (_abcd_T(L1) @ _abcd_L(fa) @ _abcd_T(L2) @ _abcd_L(fm)
            @ _abcd_T(L2) @ _abcd_L(fa) @ _abcd_T(L1))


def design_focal_lengths(design, L1, L2, w_disk=None):
    """Focal lengths (f_array, f_M) of one unit for the chosen design."""
    if design == "relay":                               # unit = -I
        return L2, L2 ** 2 / (2 * (L2 - L1))
    if design == "fourier":
        if w_disk is None:                              # confocal: flat array, R_M = 2 (L1 + L2)
            return np.inf, L1 + L2
        B = np.pi * w_disk ** 2 / LAM                   # Fourier parameter
        def eqs(p):
            M = _unit_matrix(p[0], p[1], L1, L2)
            return [M[0, 0], M[0, 1] - B]
        for guess in ((0.7, -0.15), (0.55, -0.05), (0.5, -0.02), (0.8, 0.5), (1.5, 0.5)):
            sol, _, ier, _ = fsolve(eqs, guess, full_output=True)
            if ier == 1 and abs(eqs(sol)[0]) < 1e-6:
                return float(sol[0]), float(sol[1])
        raise RuntimeError("no Fourier-transform solution found for this geometry")
    raise ValueError(design)


def design_mirrors(design=DESIGN, w_disk=W_DISK):
    """Radii of curvature of all mirrors (np.inf = flat) for the design."""
    roc = {name: np.inf for name in POS}
    roc["Disk"] = DISK.roc
    for m, arrays in RELAY_GROUPS.items():
        L1 = _path_length("Disk", arrays[0])
        L2 = np.mean([_path_length(a, m) for a in arrays])
        fa, fm = design_focal_lengths(design, L1, L2, w_disk)
        roc[m] = 2 * fm
        for a in arrays:
            roc[a] = 2 * fa
    roc["A1"] = roc[f"A{N_ARRAY}"] = np.inf                   # entrance / exit mirrors: flat
    return roc


MIRROR_ROC = design_mirrors()


def seed_q():
    """Complex beam parameter of the seed at the BS. The waist sits on the first
    disk reflection (the -I relay then reproduces it on every pass): radius W0
    for the relay design, W_DISK (mode matched) for the fourier design."""
    L = _path_length(*PATH[:PATH.index("Disk") + 1])         # BS -> first disk reflection
    if DESIGN == "fourier":
        w = W_DISK if W_DISK is not None else np.sqrt(LAM * (_path_length("Disk", "A2") + _path_length("A2", "M1")) / np.pi)
        return 1j * np.pi * w ** 2 / LAM - L
    return 1j * np.pi * W0 ** 2 / LAM - L                    # converging seed, waist W0 on the disk


def seed_beam(grid):
    """Seed field at the BS consistent with seed_q()."""
    q = seed_q()
    w0 = np.sqrt(np.imag(q) * LAM / np.pi)
    return gaussian(grid, LAM, w0, z=np.real(q), power=SEED_POWER)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------
def incidence_angle(p_prev, p_here, p_next):
    """Half the angle between the incoming and outgoing beam directions."""
    d_in = p_here - p_prev
    d_out = p_next - p_here
    cos_dev = np.dot(-d_in, d_out) / (np.linalg.norm(d_in) * np.linalg.norm(d_out))
    return 0.5 * np.arccos(np.clip(cos_dev, -1, 1))


def apply(name, beam, aoi, pattern):
    """Apply the component `name` to the beam."""
    if name == "Disk":
        return DISK.reflect(beam, aoi)
    if name == "BS":                      # 50/50 splitter: keep the reflected arm
        return beam_splitter(beam, 0.5)[0]
    if name == "SLM":
        return SLM_DEVICE.modulate(beam, pattern, bias=SLM_BIAS)
    if name == "Out":
        return beam
    return mirror(beam, radius=MIRROR_ROC[name], incidence_angle=aoi)   # array / relay mirrors


# --------------------------------------------------------------------------
# ABCD reference (tangential and sagittal q-parameter, flat SLM)
# --------------------------------------------------------------------------
def abcd_widths(path, lengths, aois):
    """Gaussian beam radii (wx, wy) at every element from the ABCD matrices:
    free space [[1, L], [0, 1]], curved mirror of radius R = lens with
    f_t = (R/2) cos(aoi), f_s = (R/2) / cos(aoi). Returns a list aligned with `path`."""
    q = [seed_q()] * 2
    out = []
    for i, name in enumerate(path):
        if i > 0:
            q = [qq + lengths[i - 1] for qq in q]
        R = MIRROR_ROC.get(name, np.inf)
        if np.isfinite(R):
            f = [R / 2 * np.cos(aois[i]), R / 2 / np.cos(aois[i])]
            q = [1 / (1 / qq - 1 / ff) for qq, ff in zip(q, f)]
        out.append([np.sqrt(-LAM / (np.pi * np.imag(1 / qq))) for qq in q])
    return np.array(out)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def path_geometry():
    """Element positions, leg lengths and incidence angles along PATH."""
    pts = [POS[n] for n in PATH]
    lengths = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    aois = [0.0] + [incidence_angle(pts[i - 1], pts[i], pts[i + 1])
                    for i in range(1, len(pts) - 1)] + [0.0]
    return lengths, aois


def trace(pattern, grid, beam_in=None, snapshots=None):
    """Trace the seed through PATH; returns (output beam, log of (wx, wy, power)
    per element). If `snapshots` is a dict, the beams at the disk, SLM and
    output are stored in it."""
    beam = seed_beam(grid) if beam_in is None else beam_in
    lengths, aois = path_geometry()
    log = []
    for i, name in enumerate(PATH):
        if i > 0:
            beam = propagate(beam, lengths[i - 1], "air")
        beam = apply(name, beam, aois[i], pattern)
        wx, wy = beam.widths()
        log.append((wx, wy, beam.power))
        if snapshots is not None and name in ("Disk", "SLM", "Out"):
            snapshots[f"{i:02d} {name}"] = replace(beam, name=f"{name} (step {i})")
    return beam, np.array(log)


def run(pattern, tag, grid, results, beam_in=None):
    """Trace the seed (default: mode-matched Gaussian, or `beam_in`) through
    PATH with the SLM showing `pattern`. Writes *_<tag>.png and returns the output beam."""
    beam_in = seed_beam(grid) if beam_in is None else beam_in
    snapshots = {}
    beam, log = trace(pattern, grid, beam_in, snapshots)
    lengths, aois = path_geometry()
    ref = abcd_widths(PATH, lengths, aois)

    print(f"\n=== SLM pattern: {tag} ===")
    print(f"{'step':>4} {'element':>7} {'AOI[deg]':>9} {'wx[mm]':>8} {'wy[mm]':>8} "
          f"{'ABCD wx':>8} {'ABCD wy':>8} {'power[W]':>9}")
    for i, name in enumerate(PATH):
        print(f"{i:4d} {name:>7} {np.rad2deg(aois[i]):9.2f} {log[i,0]*1e3:8.4f} {log[i,1]*1e3:8.4f} "
              f"{ref[i,0]*1e3:8.4f} {ref[i,1]*1e3:8.4f} {log[i,2]:9.4f}")
    print(f"total path length {sum(lengths):.2f} m, max width deviation from Gaussian ABCD "
          f"{np.abs(log[:, :2] / ref - 1).max():.1e}")

    # --- Figure: beam radius along the path ---
    fig, ax = plt.subplots(figsize=(11, 4))
    steps = np.arange(len(PATH))
    ax.plot(steps, log[:, 0] * 1e3, "o-", label="wx simulated")
    ax.plot(steps, log[:, 1] * 1e3, "s-", label="wy simulated")
    ax.plot(steps, ref[:, 0] * 1e3, "k--", lw=1, label="ABCD wx (Gaussian, flat SLM)")
    ax.plot(steps, ref[:, 1] * 1e3, "k:", lw=1, label="ABCD wy")
    ax.set_xticks(steps); ax.set_xticklabels(PATH, rotation=90, fontsize=8)
    ax.set_ylabel("beam radius [mm]"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"Beam radius at each element, SLM pattern: {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(results, f"system_beam_radius_{tag}.png"), dpi=120)

    # --- Figure: intensity and phase at SLM, disk passes and output ---
    keys = sorted(snapshots)
    fig, axes = plt.subplots(len(keys), 2, figsize=(9, 4 * len(keys)))
    for key, (ax_i, ax_p) in zip(keys, axes):
        snapshots[key].plot(ax_i, ax_p)
    fig.tight_layout()
    fig.savefig(os.path.join(results, f"system_beam_snapshots_{tag}.png"), dpi=90)

    plot_in_vs_out(beam_in, beam, grid, os.path.join(results, f"system_in_vs_out_{tag}.png"), tag)
    return beam


def plot_in_vs_out(beam_in, beam_out, grid, filename, tag):
    """Input vs output: intensity, phase, residuals and centre cross-sections."""
    ext = np.array([-1, 1, -1, 1]) * grid.size / 2 * 1e3
    I_in = beam_in.intensity / beam_in.power          # normalised to 1 W each
    I_out = beam_out.intensity / beam_out.power
    mask = I_out > 1e-3 * I_out.max()
    c = grid.n // 2

    def rel_phase(b):
        """Phase referenced to a point next to the axis (the axis itself may
        be a vortex singularity), blanked outside the beam."""
        ph = np.angle(b.field * np.conj(b.field[c, c + 1]))
        return np.where(mask, ph, np.nan)

    ph_in, ph_out = rel_phase(beam_in), rel_phase(beam_out)
    ph_res = np.where(mask, np.angle(np.exp(1j * (ph_out - ph_in))), np.nan)

    fig, ax = plt.subplots(3, 3, figsize=(13, 11))
    panels = [(I_in, "input intensity", "inferno", None),
              (I_out, "output intensity", "inferno", None),
              ((I_out - I_in) / I_in.max(), "residual (out - in) / peak", "RdBu_r", "sym"),
              (ph_in, "input phase", "twilight", np.pi),
              (ph_out, "output phase", "twilight", np.pi),
              (ph_res, "residual phase (out - in)", "twilight", np.pi)]
    for a, (img, title, cmap, lim) in zip(ax[:2].ravel(), panels):
        if lim == "sym":
            lim = np.nanmax(np.abs(img))
        kw = {} if lim is None else {"vmin": -lim, "vmax": lim}
        im = a.imshow(img, extent=ext, origin="lower", cmap=cmap, **kw)
        a.set_title(title); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
        plt.colorbar(im, ax=a, label="rad" if "phase" in title else "")

    xmm = grid.x * 1e3
    ax[2, 0].plot(xmm, I_in[c], label="in"); ax[2, 0].plot(xmm, I_out[c], label="out, x")
    ax[2, 0].plot(xmm, I_out[:, c], "--", label="out, y")
    ax[2, 0].set_title("intensity cross-section (1 W each)"); ax[2, 0].set_ylabel("W/m$^2$")
    ax[2, 1].plot(xmm, ph_out[c], label="x"); ax[2, 1].plot(xmm, ph_out[:, c], "--", label="y")
    ax[2, 1].set_title("output phase cross-section"); ax[2, 1].set_ylabel("rad")
    ax[2, 2].plot(xmm, (I_out[c] - I_in[c]) / I_in.max(), label="x")
    ax[2, 2].plot(xmm, (I_out[:, c] - I_in[:, c]) / I_in.max(), "--", label="y")
    ax[2, 2].set_title("intensity residual cross-section")
    for a in ax[2]:
        a.set_xlabel("position [mm]"); a.grid(alpha=0.3); a.legend(); a.set_xlim(-4, 4)
    wi, wo = beam_in.widths(), beam_out.widths()
    fig.suptitle(f"SLM: {tag}.  Input w = {wi[0]*1e3:.3f} mm, {beam_in.power:.3f} W   ->   "
                 f"output wx = {wo[0]*1e3:.3f} mm, wy = {wo[1]*1e3:.3f} mm, {beam_out.power:.3f} W")
    fig.tight_layout()
    fig.savefig(filename, dpi=110)


def plot_layout(results):
    pts = [POS[n] for n in PATH]
    lengths = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    fig, ax = plt.subplots(figsize=(10, 8))
    xy = np.array(pts)
    ax.plot(xy[:, 0], xy[:, 1], "r-", lw=0.8, alpha=0.7)
    for name, p in POS.items():
        ax.plot(*p, "ks", ms=6 if name != "Disk" else 10)
        ax.annotate(name, p, textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Thin-disk multipass layout, {N_PASSES} disk passes, path {sum(lengths):.1f} m\n"
                 f"{DESIGN} design, flat disk, array ROC = {MIRROR_ROC['A2']:.2f} m, M1/M2 ROC = "
                 f"{MIRROR_ROC['M1']:.2f} / {MIRROR_ROC['M2']:.2f} m")
    fig.tight_layout()
    fig.savefig(os.path.join(results, "system_layout.png"), dpi=120)


def thermal_sensitivity(rocs=(np.inf, 20, 10, 5, 2)):
    """ABCD table: spot on the disk and output beam vs thermal disk curvature."""
    pts = [POS[n] for n in PATH]
    lengths = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    aois = [0.0] + [incidence_angle(pts[i - 1], pts[i], pts[i + 1]) for i in range(1, len(pts) - 1)] + [0.0]
    idx_disk = [i for i, n in enumerate(PATH) if n == "Disk"]
    idx_M = [i for i, n in enumerate(PATH) if n in ("M1", "M2")]
    print(f"{DESIGN} design: ABCD sensitivity to a thermal curvature of the disk")
    print(f"{'disk ROC':>9} {'spot on disk, min..max [mm]':>28} {'spot on M1/M2, min..max [mm]':>30} {'output wx/wy':>14}")
    for R in rocs:
        MIRROR_ROC["Disk"] = R
        w = abcd_widths(PATH, lengths, aois) * 1e3
        print(f"{str(R):>9} {w[idx_disk,1].min():12.2f} .. {w[idx_disk,1].max():5.2f}{'':>7} "
              f"{w[idx_M,1].min():12.2f} .. {w[idx_M,1].max():5.2f}{'':>9} {w[-1,0]:6.2f}/{w[-1,1]:.2f}")
    MIRROR_ROC["Disk"] = DISK.roc


if __name__ == "__main__":
    results = results_dir("5_system")
    grid = Grid(**SIM_GRID)
    print("mirror radii [m]:", {k: round(float(v), 3) for k, v in MIRROR_ROC.items() if np.isfinite(v)})
    thermal_sensitivity(rocs=(np.inf, 50, 20, 10, 5))
    plot_layout(results)

    patterns = {"flat": slm.flat(), "vortex_l1": slm.vortex(1)}
    for tag, pattern in patterns.items():
        run(pattern, tag, grid, results)
    print("saved figures to", os.path.abspath(results))
