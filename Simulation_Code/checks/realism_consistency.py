"""Consistency of the new amplifier features: mirror tilts and gain, numpy vs torch, single vs batched."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the simulation modules
import numpy as np, torch
import thin_disk_system as tds, correction as co, limits_benchmark as lb
from seed_beam import Grid
from gpu import TorchAmplifier
G = Grid(**tds.SIM_GRID)
th = lb.thermal(300); tds.DISK.thermal_opd = th.opd(G); tds.DISK.gain = np.ones((G.n, G.n))
f_th, _ = lb.per_pass_lens(th.opd(G), G); tds.MIRROR_ROC.update(lb.retuned_mirrors(f_th))
cpu = co.Amplifier(G); gpu = TorchAmplifier(G)
fin = co.field_at_slm(G).field
a = gpu.forward(fin)
rng = np.random.default_rng(0)
tilts = {n: tuple(20e-6 * rng.standard_normal(2)) for n in gpu.mirror_names}
gpu.set_mirror_tilts(tilts); cpu.set_mirror_tilts(tilts)
b, c = gpu.forward(fin), cpu.forward(fin)
print(f"{len(gpu.mirror_names)} mirrors; tilts torch vs numpy {np.abs(b - c).max() / np.abs(c).max():.2e}; fidelity aligned vs 20 urad tilts {co.fidelity(a, b):.3f}")
gain = 1 + 0.2 * np.exp(-(G.R / 2e-3) ** 2); gpu.set_gain(gain); cpu.set_gain(gain)
d, e = gpu.forward(fin), cpu.forward(fin)
print(f"gain torch vs numpy {np.abs(d - e).max() / np.abs(e).max():.2e}; power ratio {(np.abs(d) ** 2).sum() / (np.abs(b) ** 2).sum():.2f} (<= {1.2 ** 10:.2f})")
T = torch.zeros(2, len(gpu.mirror_names), 2); T[0] = torch.tensor([tilts[n] for n in gpu.mirror_names])
gpu.set_mirror_tilts(T); gpu.set_gain(torch.stack([torch.as_tensor(gain, dtype=torch.float32), torch.ones(G.n, G.n)]))
out = gpu.forward_t(torch.as_tensor(np.stack([fin, fin]).astype(np.complex64), device=gpu.device)).cpu().numpy()
print(f"batched: example 0 vs single {np.abs(out[0] - d).max() / np.abs(d).max():.2e}, example 1 vs aligned/unit gain {np.abs(out[1] - a).max() / np.abs(a).max():.2e}")
