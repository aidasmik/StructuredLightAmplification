"""
correction.py - Physics-optimal pre-correction of structured light with the SLM.

The amplifier from the SLM output plane to the output is a linear operator T
on the field (for a fixed disk state). It is represented here as a list of
elementary operations (propagation, phase screen, mirror image, scaling,
aperture, footprint stretch), each with its adjoint, so that T and T^dagger
can be applied. The best phase-only SLM pattern for a target output mode is
found by alternating projections (Gerchberg-Saxton through the amplifier):

    E_out  = T  E_slm                   forward
    E_slm' = T^dagger (E_target e^{i a}) back-projection, a = global phase
    phi    = arg( E_slm' / E_in )       phase-only constraint on the seed

The resulting fidelity |<E_out, E_target>|^2 / (P_out P_target) is the upper
bound any correction scheme (including a neural network driving the SLM)
can reach with a phase-only SLM at this position.
"""

import numpy as np
import thin_disk_system as tds
import slm
from seed_beam import gaussian
from propagation import propagate, refractive_index, _angular_spectrum, absorber
from optics import _sag, _flip_x, beam_splitter
from media import snell


# --------------------------------------------------------------------------
# Elementary operations with adjoints
# --------------------------------------------------------------------------
class Prop:
    def __init__(self, z, k, dx):
        self.z, self.k, self.dx = z, k, dx

    def apply(self, f):
        return _angular_spectrum(f, self.dx, self.k, self.z)                 # absorbing window, then propagate

    def adjoint(self, f):                                                    # exact transpose: propagate back, then window
        return _angular_spectrum(f, self.dx, self.k, -self.z, absorb=False) * absorber(f.shape[0], float(self.dx))


class Phase:
    def __init__(self, phi):
        self.e = np.exp(1j * phi)

    def apply(self, f):
        return f * self.e

    def adjoint(self, f):
        return f * np.conj(self.e)


class Flip:
    def apply(self, f):
        return _flip_x(f)

    adjoint = apply


class Scale:
    def __init__(self, s):
        self.s = s                      # scalar or real mask

    def apply(self, f):
        return f * self.s

    def adjoint(self, f):
        return f * np.conj(self.s)


class Stretch:
    """E2(x, y) = E1(x / a, y) by linear interpolation along x, written as a
    sparse bidiagonal matrix W so that the adjoint is the exact transpose."""
    def __init__(self, a, n):
        from scipy.sparse import csr_matrix
        src = (np.arange(n) - n / 2) / a + n / 2          # source position for each output index
        i0 = np.floor(src).astype(int)
        t = src - i0
        rows, cols, vals = [], [], []
        for j in range(n):
            if 0 <= i0[j] < n:
                rows.append(j); cols.append(i0[j]); vals.append(1 - t[j])
            if 0 <= i0[j] + 1 < n:
                rows.append(j); cols.append(i0[j] + 1); vals.append(t[j])
        self.W = csr_matrix((vals, (rows, cols)), shape=(n, n))
        self.WT = self.W.T.tocsr()

    def apply(self, f):
        return (self.W @ f.T).T

    def adjoint(self, f):
        return (self.WT @ f.T).T


# --------------------------------------------------------------------------
# Building the operator from the system definition
# --------------------------------------------------------------------------
def _mirror_ops(grid, radius, reflectivity, aoi, diameter=25.4e-3, n_medium=1.0, k0=None):
    c = np.cos(aoi)
    r2 = (grid.X / c) ** 2 + grid.Y ** 2
    phi = -2 * n_medium * k0 * c * _sag(r2, radius)
    inside = (r2 <= (diameter / 2) ** 2) & np.isfinite(phi)
    return [Phase(np.nan_to_num(phi)), Scale(inside.astype(float)), Scale(np.sqrt(reflectivity)), Flip()]


def _disk_ops(grid, aoi, k0, disk):
    """Same sequence as ThinDisk.reflect (flat disk, AR coated, optional thermal OPD)."""
    n = refractive_index(disk.material, tds.LAM)
    th2 = snell(1.0, n, aoi)
    c1, c2 = np.cos(aoi), np.cos(th2)
    path = disk.thickness / c2
    transmission = np.sqrt((1 - disk.r_ar) * np.exp(-disk.alpha * path))
    ops = [Scale((grid.R <= disk.diameter / 2).astype(float))]
    # refract in (flat surface, AR): stretch and amplitude factor
    ops += [Stretch(c2 / c1, grid.n), Scale(np.sqrt(n / 1.0))]
    ops += [Prop(path, n * k0, grid.dx), Scale(transmission)]
    ops += _mirror_ops(grid, disk.roc, disk.r_hr, th2, disk.diameter, n_medium=n, k0=k0)
    ops += [Prop(path, n * k0, grid.dx), Scale(transmission)]
    ops += [Stretch(c1 / c2, grid.n), Scale(np.sqrt(1.0 / n))]
    if disk.thermal_opd is not None:
        screen = Phase(k0 * disk.thermal_opd)
        screen.thermal = True                           # marks the disk screen (see Amplifier.set_thermal_opd)
        ops.append(screen)
    if disk.gain is not None:
        g = Scale(np.sqrt(disk.gain))                   # amplitude gain per reflection
        g.gain = True                                   # marks the gain map (see Amplifier.set_gain)
        ops.append(g)
    return ops


class Amplifier:
    """Linear operator from the field right after the SLM to the output plane."""

    def __init__(self, grid):
        self.grid = grid
        k0 = 2 * np.pi / tds.LAM
        lengths, aois = tds.path_geometry()
        start = tds.PATH.index("SLM")
        self.ops = []
        for i in range(start + 1, len(tds.PATH)):
            name = tds.PATH[i]
            self.ops.append(Prop(lengths[i - 1], refractive_index("air", tds.LAM) * k0, grid.dx))
            if name == "BS":
                self.ops += [Scale(1j * np.sqrt(0.5)), Flip()]
            elif name == "Disk":
                self.ops += _disk_ops(grid, aois[i], k0, tds.DISK)
            elif name == "Out":
                pass
            else:
                mops = _mirror_ops(grid, tds.MIRROR_ROC[name], 1.0, aois[i], k0=k0)
                mops[0].mirror = name                   # marks the relay mirror (see set_mirror_tilts)
                self.ops += mops

    def set_gain(self, gain):
        """Replace the power-gain map per reflection (all passes); needs DISK.gain set when built."""
        s = np.sqrt(gain)
        for op in self.ops:
            if getattr(op, "gain", False):
                op.s = s

    def set_mirror_tilts(self, tilts):
        """Tilt the relay mirrors, tilts = {name: (theta_x, theta_y)} [rad]: a reflection off a
        mirror tilted by theta deviates the beam by 2 theta (linear phase 2 k0 theta x)."""
        k0 = 2 * np.pi / tds.LAM
        for op in self.ops:
            name = getattr(op, "mirror", None)
            if name is not None:
                if not hasattr(op, "e0"):
                    op.e0 = op.e                        # the aligned mirror
                tx, ty = tilts.get(name, (0.0, 0.0))
                op.e = op.e0 * np.exp(1j * 2 * k0 * (tx * self.grid.X + ty * self.grid.Y))

    def set_thermal_opd(self, opd):
        """Replace the disk phase screen (reflection OPD in metres) for all passes."""
        e = np.exp(1j * 2 * np.pi / tds.LAM * opd)
        screens = [op for op in self.ops if getattr(op, "thermal", False)]
        if not screens:
            raise RuntimeError("amplifier was built without a thermal screen (DISK.thermal_opd was None)")
        for op in screens:
            op.e = e                                    # shared array

    def forward(self, field):
        for op in self.ops:
            field = op.apply(field)
        return field

    def adjoint(self, field):
        for op in reversed(self.ops):
            field = op.adjoint(field)
        return field


# --------------------------------------------------------------------------
# SLM input field, ideal SLM, fidelity, optimisation
# --------------------------------------------------------------------------
def field_at_slm(grid):
    """Seed field arriving at the SLM (before the SLM's own image flip)."""
    d = np.linalg.norm(tds.POS["SLM"] - tds.POS["BS"])
    return propagate(beam_splitter(tds.seed_beam(grid), 0.5)[0], d, "air")


def ideal_slm(field_in, pattern, device=tds.SLM_DEVICE):
    """Continuous phase-only SLM with the device's amplitude factor."""
    return np.sqrt(device.reflectivity) * device.fill_factor * _flip_x(field_in * np.exp(1j * pattern))


def fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


def optimise_pattern(amp, field_in, target, iterations=25):
    """Alternating projections for the best phase-only SLM pattern.
    Returns (pattern map in SLM coordinates, fidelity history)."""
    pattern = np.zeros_like(field_in, dtype=float)
    history = []
    for _ in range(iterations):
        out = amp.forward(ideal_slm(field_in, pattern))
        history.append(fidelity(out, target))
        align = np.exp(1j * np.angle(np.vdot(target, out)))     # free global phase
        desired = amp.adjoint(target * align)                     # field wanted after the SLM
        pattern = np.angle(_flip_x(desired) * np.conj(field_in))  # back to SLM coordinates
    out = amp.forward(ideal_slm(field_in, pattern))
    history.append(fidelity(out, target))
    return pattern, np.array(history)


def target_modes(grid, reference):
    """Structured target fields at the output plane built on the Gaussian
    output `reference` (same size and wavefront curvature)."""
    w = np.mean(reference.widths())
    g = grid
    r2 = g.R ** 2
    gauss = reference.field
    modes = {
        "gaussian": gauss,
        "LG01": gauss * (np.sqrt(2) * g.R / w) * np.exp(1j * g.PHI),
        "HG10": gauss * (2 * g.X / w),
        "tophat": np.abs(gauss).max() * np.exp(-(g.R / (0.8 * w)) ** 12) * np.exp(1j * np.angle(gauss)),
    }
    return {k: v / np.sqrt(np.vdot(v, v).real) for k, v in modes.items()}


# --------------------------------------------------------------------------
# Complex-amplitude modulation
# --------------------------------------------------------------------------
def complex_modulation(field_in, desired, device=tds.SLM_DEVICE, min_fidelity=0.99):
    """Ideal complex modulator: the field after the SLM is  a(x,y) e^{i psi}
    with a <= device amplitude * |field_in|. `desired` is the wanted field
    after the SLM (output coordinates). The achievable field is the desired
    one scaled by s and clipped to the seed envelope. Without clipping the
    shape is exact but the power is low; with strong clipping the field
    degenerates to phase-only. s is chosen as the largest scale for which the
    overlap with the wanted field in the SLM plane stays >= `min_fidelity`,
    i.e. the most efficient encoding that still reproduces the target.
    Returns (field after SLM, efficiency, amplitude ratio map, phase map)
    with the maps in SLM (pre-flip) coordinates."""
    want = _flip_x(desired)
    env = np.sqrt(device.reflectivity) * device.fill_factor * np.abs(field_in)
    best = None
    for s in np.geomspace(0.02, 50, 120) * env.max() / (np.abs(want).max() + 1e-30):
        amp = np.minimum(s * np.abs(want), env)
        f = amp * np.exp(1j * np.angle(want))
        ov = fidelity(f, want)
        if ov >= min_fidelity or best is None:
            best = (ov, f, amp)
    ov, f, amp = best
    efficiency = np.vdot(f, f).real / np.vdot(env, env).real
    ratio = amp / (env + 1e-30)                                  # in [0, 1]
    pattern = np.angle(want * np.conj(field_in))                 # phase relative to the incoming field
    return _flip_x(f), efficiency, ratio, pattern


def davis_realise(field_in_beam, ratio, pattern, device=tds.SLM_DEVICE, period_pixels=8):
    """Complex-amplitude encoding on the real LCOS: blazed carrier whose
    depth sets the first-order amplitude. Returns the field after the SLM."""
    grid = field_in_beam.grid
    M = davis_depth(ratio)
    phi = pattern - davis_phase_correction(M)
    return device.modulate(field_in_beam, slm.map_pattern(phi, grid), blaze_period=period_pixels * device.pitch,
                           depth=slm.map_pattern(M, grid, wrapped=False)).field


def davis_depth(ratio):
    """Blaze depth M in [0, 1] giving first-order amplitude `ratio`:
    inverts  A(M) = sinc(1 - M)  (numpy sinc = sin(pi x)/(pi x))."""
    M = np.linspace(0, 1, 1001)
    A = np.sinc(1 - M)
    return np.interp(np.clip(ratio, 0, 1), A, M)


def davis_phase_correction(depth):
    """Phase of the first order of a blazed grating with depth M relative
    to a full-depth grating: pi (M - 1). Subtract it from the pattern."""
    return np.pi * (depth - 1)
