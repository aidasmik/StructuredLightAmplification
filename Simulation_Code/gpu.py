"""
gpu.py - GPU (PyTorch) version of the amplifier operator of correction.py.

TorchAmplifier builds the same chain of elementary operations as
correction.Amplifier (propagation, phase screens, apertures, image flips,
footprint stretches) and runs it on the GPU with complex64 tensors. Inputs
and outputs are numpy arrays, so it is a drop-in replacement:

    amp = make_amplifier(grid)          # TorchAmplifier if CUDA is available, else the numpy one
    out = amp.forward(field)            # numpy in, numpy out
    amp.set_thermal_opd(opd)            # update the disk phase screen (all passes) without rebuilding
    amp.set_gain(G)                     # per-reflection power gain map (built with DISK.gain set)
    amp.set_mirror_tilts(tilts)         # relay mirror tilts {name: (theta_x, theta_y)} or (B, M, 2) tensor

The screens, gain and tilts may be torch tensors with a leading batch axis, one
per training example; the chain then runs batched and stays differentiable.

One 10-pass forward trace at 1024 x 1024 takes ~1.4 s on the CPU and
~30 ms on an RTX 2080 (FFT 0.5 ms). The SLM and camera models stay in numpy.
"""
import numpy as np

import correction as co
from propagation import _transfer_function, absorber

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:                      # torch not installed: numpy fallback only
    torch = None
    HAVE_CUDA = False


class TorchAmplifier:
    def __init__(self, grid, device="cuda", dtype=None):
        self.grid = grid
        self.device = torch.device(device)
        self.dtype = dtype or torch.complex64
        base = co.Amplifier(grid)                      # the numpy operator chain
        self.ops = []
        self.screen = None                             # shared thermal phase screen
        self.mirror_e = {}                             # aligned phase of each physical relay mirror
        for op in base.ops:
            self.ops.append(self._convert(op))
        self.mirror_names = list(self.mirror_e)        # order used by set_mirror_tilts with a tensor
        self.k0 = 2 * np.pi / co.tds.LAM
        self.X, self.Y = self._t(grid.X), self._t(grid.Y)

    def _t(self, a):
        """numpy -> torch tensor on the device (complex64 unless real)."""
        a = np.asarray(a)
        if np.iscomplexobj(a):
            return torch.as_tensor(a.astype(np.complex64), device=self.device)
        return torch.as_tensor(a.astype(np.float32), device=self.device)

    def _convert(self, op):
        if isinstance(op, co.Prop):
            H = _transfer_function(self.grid.n, float(op.dx), float(op.k), float(op.z))
            return ("prop", (self._t(H), self._t(absorber(self.grid.n, float(op.dx)))))   # transfer function, absorbing window
        if isinstance(op, co.Phase):
            e = self._t(op.e)
            if getattr(op, "thermal", False):          # all passes share one screen
                if self.screen is None:
                    self.screen = e
                return ("mul", self.screen, "thermal")
            name = getattr(op, "mirror", None)
            if name is not None:                       # one physical mirror, possibly several reflections
                self.mirror_e.setdefault(name, e)
                return ("mul", self.mirror_e[name], ("mirror", name))
            return ("mul", e)
        if isinstance(op, co.Scale):
            s = op.s
            t = self._t(s) if np.ndim(s) else complex(s)
            return ("mul", t, "gain") if getattr(op, "gain", False) else ("mul", t)
        if isinstance(op, co.Flip):
            return ("flip",)
        if isinstance(op, co.Stretch):
            W = op.W.tocoo()                           # out[:, row] += val * in[:, col]
            return ("sparse", torch.as_tensor(W.row, device=self.device, dtype=torch.long),
                    torch.as_tensor(W.col, device=self.device, dtype=torch.long),
                    torch.as_tensor(W.data.astype(np.complex64), device=self.device))
        raise TypeError(type(op))

    def set_thermal_opd(self, opd):
        """Replace the disk phase screen (reflection OPD in metres) for all passes.
        opd: numpy (n, n), or a torch tensor (n, n) / (B, n, n) for one screen per
        example of a training batch (the chain then runs batched)."""
        k0 = 2 * np.pi / co.tds.LAM
        if self.screen is None:
            raise RuntimeError("amplifier was built without a thermal screen (DISK.thermal_opd was None)")
        if torch.is_tensor(opd):
            phi = (k0 * opd).to(self.device, torch.float32)
            e = torch.polar(torch.ones_like(phi), phi)
        else:
            e = torch.as_tensor(np.exp(1j * k0 * opd).astype(np.complex64), device=self.device)
        self._replace("thermal", e)
        self.screen = e

    def _replace(self, tag, value):
        self.ops = [("mul", value, tag) if len(op) > 2 and op[2] == tag else op for op in self.ops]

    def set_gain(self, gain):
        """Replace the per-reflection power gain map: numpy (n, n) or torch (n, n) / (B, n, n)."""
        if not any(len(op) > 2 and op[2] == "gain" for op in self.ops):
            raise RuntimeError("amplifier was built without a gain map (set DISK.gain before building)")
        s = torch.sqrt(gain.to(self.device, torch.float32)) if torch.is_tensor(gain) else self._t(np.sqrt(gain))
        self._replace("gain", s)

    def set_mirror_tilts(self, tilts):
        """Tilt the relay mirrors: {name: (theta_x, theta_y)} [rad], or a tensor (M, 2) / (B, M, 2)
        in the order of self.mirror_names. A reflection off a mirror tilted by theta deviates
        the beam by 2 theta, i.e. a linear phase 2 k0 (theta_x x + theta_y y)."""
        for i, name in enumerate(self.mirror_names):
            if torch.is_tensor(tilts):
                t = tilts[..., i, :].to(self.device, torch.float32)
                tx, ty = t[..., 0], t[..., 1]
            else:
                tx, ty = (torch.as_tensor(float(v), device=self.device) for v in tilts.get(name, (0.0, 0.0)))
            phi = 2 * self.k0 * (tx[..., None, None] * self.X + ty[..., None, None] * self.Y)
            self._replace(("mirror", name), self.mirror_e[name] * torch.polar(torch.ones_like(phi), phi))

    def forward_t(self, f):
        """Torch in, torch out, differentiable; f may be batched (B, n, n)."""
        for op in self.ops:
            f = self._apply(op, f)
        return f

    # -- the chain -----------------------------------------------------------
    @staticmethod
    def _flip(f):
        return torch.roll(f.flip(-1), 1, dims=-1)    # x -> -x keeping the centre pixel (last axis: works batched)

    def _apply(self, op, f, adjoint=False):
        kind = op[0]
        if kind == "prop":
            H, W = op[1]
            if adjoint:                                                      # exact transpose of (propagate o window)
                return torch.fft.ifft2(torch.fft.fft2(f) * H.conj()) * W
            return torch.fft.ifft2(torch.fft.fft2(f * W) * H)
        if kind == "mul":
            s = op[1]
            if adjoint:
                s = s.conj() if torch.is_tensor(s) else np.conj(s)
            return f * s
        if kind == "flip":
            return self._flip(f)
        if kind == "sparse":
            _, rows, cols, vals = op
            out = torch.zeros_like(f)
            if adjoint:
                out.index_add_(-1, cols, f[..., rows] * vals)
            else:
                out.index_add_(-1, rows, f[..., cols] * vals)
            return out
        raise ValueError(kind)

    def forward(self, field):
        f = torch.as_tensor(np.asarray(field).astype(np.complex64), device=self.device)
        for op in self.ops:
            f = self._apply(op, f)
        return f.cpu().numpy()

    def adjoint(self, field):
        f = torch.as_tensor(np.asarray(field).astype(np.complex64), device=self.device)
        for op in reversed(self.ops):
            f = self._apply(op, f, adjoint=True)
        return f.cpu().numpy()


def make_amplifier(grid):
    """TorchAmplifier on the GPU when available, otherwise the numpy Amplifier."""
    if HAVE_CUDA:
        return TorchAmplifier(grid)
    return co.Amplifier(grid)


if __name__ == "__main__":
    # consistency check against the numpy operator (needs torch; GPU optional)
    import time
    import thin_disk_system as tds
    from seed_beam import Grid
    G = Grid(**tds.SIM_GRID)
    rng = np.random.default_rng(0)
    tds.DISK.thermal_opd = 1e-7 * rng.standard_normal((G.n, G.n))
    cpu = co.Amplifier(G)
    dev = "cuda" if HAVE_CUDA else "cpu"
    gpu = TorchAmplifier(G, device=dev)
    fin = co.field_at_slm(G).field
    t0 = time.perf_counter(); a = cpu.forward(fin); t_cpu = time.perf_counter() - t0
    gpu.forward(fin)
    t0 = time.perf_counter(); b = gpu.forward(fin); t_gpu = time.perf_counter() - t0
    print(f"forward : max |diff| / max |cpu| = {np.abs(a - b).max() / np.abs(a).max():.2e}   cpu {t_cpu:.2f} s, {dev} {t_gpu*1e3:.0f} ms")
    a2, b2 = cpu.adjoint(a), gpu.adjoint(a)
    print(f"adjoint : max |diff| / max |cpu| = {np.abs(a2 - b2).max() / np.abs(a2).max():.2e}")
    x, y = fin, a
    print(f"<Ax, y> vs <x, A^T y> on {dev}: {np.vdot(gpu.forward(x), y):.6e} vs {np.vdot(x, gpu.adjoint(y)):.6e}")
    gpu.set_thermal_opd(np.zeros((G.n, G.n)))
    tds.DISK.thermal_opd = np.zeros((G.n, G.n))
    c = co.Amplifier(G).forward(fin)
    print(f"set_thermal_opd: max |diff| = {np.abs(c - gpu.forward(fin)).max() / np.abs(c).max():.2e}")
