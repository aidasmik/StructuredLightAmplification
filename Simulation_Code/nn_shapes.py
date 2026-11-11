"""
nn_shapes.py - Step 23: one network for any beam shape ("target as image").

The networks of Steps 18-21 know two targets by a 0/1 flag. Here the target is given as images,
so one network can be asked for shapes it never saw:

    shape S        wanted intensity in the work plane = focus of the f = 1 m lens (the far-field
                   camera plane), as the application forms flat-tops, rings and spot arrays
    base hologram  phi_S, designed once for the NOMINAL system (tuned relay, 300 W disk) by gradient
                   ascent on the overlap of the far-field intensity with S (bank, GPU)
    target field   T_S = nominal output of fin * exp(i phi_S): what the undistorted amplifier makes
                   from the hologram. A perfect correction always exists (zero on the nominal disk).
    network input  (13 channels, 128 x 128)
                   near / far camera image of the distorted beam (base hologram displayed) minus
                   the camera images of T_S, five temperatures and the pump reading minus the
                   reference, doping map, the camera images of T_S themselves, cos / sin phi_S
    output         correction added to phi_S (256 x 256 at 25 um, the residual formulation:
                   the reference correction is zero by construction)
    loss           1 - (field fidelity with T_S) x (power kept), see delivered(): physics loss only,
                   ideal SLM, the batched torch amplifier; no labels needed. shapes_v1 / v2 were
                   trained on the fidelity alone and learned to throw light away (kept ~36 %).

Layout (--relay). As modelled, the SLM is 2.9 m of free space before the disk: a shaped beam is already
spread over the disk aperture there (the bank's shapes keep ~14 % of the seed power, a Gaussian 43 %),
and the correction is different for every shape (nn_shapes_diag.py, nn_shapes_diag2.py). With --relay
the SLM is imaged onto the disk (ideal 1:1 telescope): one correction per disk state serves all shapes.

Disk states: the 20 000 states of the wide-range dataset (Step 19: 240-360 W, hot spots, decentre,
shape error, doping clusters; no gain / lab nuisances yet). Camera images are simulated on the GPU
for every batch (12 bit, 1 % noise, the dataset's fixed exposures).

Shape families of the bank (sizes in units of the far-field Gaussian spot radius w):
    train / test   elliptical spot, ring, flat-top (super-Gaussian), 2-4 spot array, line (bank.npz)
                   + spiral (charge 1-3 vortex / axicon phases, no design), polygon (3, 5, 6 sides),
                   strokes (thick polylines with corners), blob (random outlines) (bank_v2.npz)
    unseen         letters (A, F, T, H, L, X) and squares: held out of training completely
    named          Gaussian (phi = 0) and the charge-1 vortex (phi = spiral)

Usage:  NN_RANGE=wide python nn_shapes.py bank            (target bank, ~1 h)
        NN_RANGE=wide python nn_shapes.py extend          (adds the new families: bank_v2.npz, ~1 h)
        NN_RANGE=wide python nn_shapes.py train NAME [--bank bank_v2.npz --width 128 --epochs 16 --steps 3000]
        add --one-camera to train for the far-field camera alone (intensity score, phase not rated)
        add --relay to bank / extend / train for the SLM imaged onto the disk (bank_relay.npz, bank_v2_relay.npz)
        NN_RANGE=wide python nn_shapes.py eval NAME        (table + figures)
Output: Results/23_nn_shapes/
"""
import os
import sys
import csv
import glob
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("NN_RANGE", "wide")
import nn_dataset as nd
import nn_train as nt
import thin_disk_system as tds
import slm
from propagation import _transfer_function, absorber, refractive_index
from two_camera_correction import F_LENS
from seed_beam import results_dir

warnings.filterwarnings("ignore")
DEV = nt.DEV
OUT = results_dir("23_nn_shapes")
G = nd.G
N, C = G.n, nd.CROP_LABEL                       # 512 grid, 256 correction crop
CAM = nd.CROP_CAM                               # 128 camera pixels
FAMILIES = ("spot", "ring", "flattop", "spots", "line")
UNSEEN = ("letter", "square")
NEW_FAMILIES = ("spiral", "polygon", "strokes", "blob")
N_TRAIN, N_TEST, N_UNSEEN = 1536, 128, 48       # bank.npz
N_NEW_TRAIN, N_NEW_TEST = 1536, 104             # added by `extend`


# --------------------------------------------------------------------------
# Optics on the GPU: amplifier, far field, cameras
# --------------------------------------------------------------------------
class Optics:
    def __init__(self, relay=False):
        self.meta = np.load(os.path.join(nd.RESULTS, "meta.npz"))
        self.phys = nt.Physics(self.meta)                                 # amplifier (relay tuned to the nominal disk)
        self.amp, self.fin, self.slm_gain = self.phys.amp, self.phys.fin, self.phys.slm_gain
        r = np.load(os.path.join(nd.RESULTS, "reference.npz"))
        self.ref_T, self.ref_P = torch.as_tensor(r["T_pts"].astype(np.float32), device=DEV), float(r["P"])
        opd0 = nd.crop(nd.zoom(r["opd"], 0.5, order=1), C // 2).astype(np.float32)   # nominal disk, stored like the states
        self.screen0 = self.phys.screens(torch.as_tensor(opd0[None], device=DEV))[0]
        # far field: ideal lens f, then f of air (the far-field camera plane), as two_camera_correction.to_far
        lam = tds.LAM
        k = 2 * np.pi / lam * refractive_index("air", lam)
        lens = np.exp(-1j * 2 * np.pi / lam * (np.sqrt(F_LENS ** 2 + G.R ** 2) - F_LENS))
        self.lens = torch.as_tensor((lens * absorber(N, float(G.dx))).astype(np.complex64), device=DEV)
        self.H = torch.as_tensor(_transfer_function(N, float(G.dx), float(k), float(F_LENS)), device=DEV)
        # SLM -> disk: free space over the layout's path (flat mirrors only), for the disk-plane correction (to_slm)
        z = tds._path_length("SLM", "BS", "M0", "A1", "Disk")
        self.H_disk = torch.as_tensor(_transfer_function(N, float(G.dx), float(k), float(z)), device=DEV)
        self.fin_phase = self.fin / (self.fin.abs() + 1e-30)              # the seed's own phase factor
        self.relay = relay
        if relay:                                                         # what-if: a 1:1 telescope images the SLM onto the disk,
            self.fin = torch.fft.ifft2(torch.fft.fft2(self.fin) * self.H_disk)   # lit by the seed as it is on the disk (1 mm radius)
            self.fin_phase = self.fin / (self.fin.abs() + 1e-30)
        self.scale = {"near": float(self.meta["near_scale"]), "far": float(self.meta["far_scale"])}
        x = torch.arange(-2, 3, device=DEV, dtype=torch.float32)
        g = torch.exp(-x ** 2 / (2 * 0.5 ** 2)); g = g / g.sum()
        self.blur = (g[:, None] * g[None, :])[None, None]                 # camera blur, sigma 0.5 px
        self.spiral = torch.as_tensor(slm.vortex(1)(G.X, G.Y).astype(np.float32), device=DEV)

    def pad(self, phi_crop):
        """(B, 256, 256) phase -> full grid (zero outside the SLM crop, where the seed is ~0)."""
        return Fn.pad(phi_crop, ((N - C) // 2,) * 4)

    def output(self, phi_crop, screens):
        """Amplifier output for SLM phase phi_crop (B, 256, 256) with disk screens (B, n, n) or (n, n)."""
        phi = self.pad(phi_crop)
        field = self.fin * torch.polar(torch.ones_like(phi), phi)
        if self.relay:                                                    # ideal imaging of the SLM plane onto the disk
            field = torch.fft.ifft2(torch.fft.fft2(field) * self.H_disk.conj())   # = undo the 2.91 m of free space in front of the disk
        field = self.slm_gain * torch.roll(field.flip(-1), 1, dims=-1)        # SLM image flip, as nt.Physics
        self.amp.set_thermal_opd(screens)
        return self.amp.forward_t(field)

    def to_slm(self, phi_crop, psi):
        """SLM pattern (B, 256, 256) that pre-corrects a phase map psi at the DISK plane (B or 1, 256, 256)
        for the hologram phi_crop. The relay images the disk onto itself, so the disk error is one phase
        screen there; the field that cancels it is
            P(-z) [ exp(-i psi) P(z) [ seed exp(i phi) ] ],    P(z): free space over the SLM-disk distance,
        and the SLM shows its phase. A fixed Fresnel step (four FFTs, no disk model); psi = 0 returns phi."""
        phi = self.pad(phi_crop)
        at_disk = torch.fft.ifft2(torch.fft.fft2(self.fin * torch.polar(torch.ones_like(phi), phi)) * self.H_disk)
        psi = self.pad(psi)
        back = torch.fft.ifft2(torch.fft.fft2(at_disk * torch.polar(torch.ones_like(psi), -psi)) * self.H_disk.conj())
        back = back * self.fin_phase.conj()                               # the pattern is the phase on top of the seed's
        c = (N - C) // 2
        return torch.atan2(back.imag, back.real + 1e-30)[..., c:c + C, c:c + C]

    def far(self, out):
        return torch.fft.ifft2(torch.fft.fft2(out * self.lens) * self.H)

    def camera(self, field, which, noise=0.01, quantize=True):
        """12-bit camera crop (B, 128, 128) of a field: binning (near 2, far 1), blur, fixed exposure, noise.
        quantize=False: no 12-bit rounding / clipping (differentiable, for the hologram design)."""
        I = field.abs() ** 2
        if which == "near":
            I = Fn.avg_pool2d(I[:, None], 2)[:, 0]
        I = Fn.conv2d(I[:, None], self.blur, padding=2)[:, 0] * self.scale[which]
        if noise:
            I = I + noise * torch.randn_like(I)
        if quantize:
            I = torch.clamp(torch.round(I * 4095), 0, 4095) / 4095
        c = I.shape[-1] // 2
        return I[:, c - CAM // 2:c + CAM // 2, c - CAM // 2:c + CAM // 2]

    def images(self, out, noise=0.01):
        return self.camera(out, "near", noise), self.camera(self.far(out), "far", noise)


def upsample(m):
    """(B, r, r) map -> (B, 256, 256), bilinear."""
    return m if m.shape[-1] == C else Fn.interpolate(m[:, None], size=(C, C), mode="bilinear", align_corners=False)[:, 0]


def fidelity(out, T):
    num = (T.conj() * out).sum(dim=(-2, -1)).abs() ** 2
    return num / ((out.abs() ** 2).sum(dim=(-2, -1)) * (T.abs() ** 2).sum(dim=(-2, -1)))


def delivered(out, T):
    """Fidelity x power kept:  |<T, out>|^2 / (|T|^2 max(|out|^2, |T|^2)).
    The fidelity is normalised by the output power, so a pattern that scatters most of the light out
    of the disk aperture and keeps a small clean remainder scores well (the networks and "ceilings"
    trained on it kept 1-40 % of the power). This measure multiplies the fidelity by the output power
    relative to the target's (the undistorted system's) when power is lost, and equals it otherwise.
    Used as the training loss, for the ceiling and in the tables."""
    num = (T.conj() * out).sum(dim=(-2, -1)).abs() ** 2
    Po, Pt = (out.abs() ** 2).sum(dim=(-2, -1)), (T.abs() ** 2).sum(dim=(-2, -1))
    return num / (Pt * torch.maximum(Po, Pt))


def power(f):
    return (f.abs() ** 2).sum(dim=(-2, -1))


def scorer(opt, one_camera=False):
    """The measure a network is trained and scored on, as a function (out, T) -> (B,).
    Two cameras: `delivered` (field fidelity x power kept).
    One camera (far field only): the phase cannot be seen, so it is not rated. Intensity similarity
    of the far-field (work-plane) images x power kept:  (sum sqrt(I I_T))^2 / (P_T max(P, P_T)).
    It equals `delivered` when the phases agree and is larger otherwise."""
    if not one_camera:
        return delivered
    amp = lambda f: (f.real ** 2 + f.imag ** 2 + 1e-20).sqrt()            # |f| with a finite gradient at 0
    def intensity(out, T):
        A, B = amp(opt.far(out)), amp(opt.far(T))
        Po, Pt = (A ** 2).sum(dim=(-2, -1)), (B ** 2).sum(dim=(-2, -1))
        return (A * B).sum(dim=(-2, -1)) ** 2 / (Pt * torch.maximum(Po, Pt))
    return intensity


# --------------------------------------------------------------------------
# Shapes (far-field intensity, 128 x 128 at 25 um) and the hologram bank
# --------------------------------------------------------------------------
def letter_mask(ch, size):
    """Binary image of a capital letter (DejaVu Sans Bold), size x size."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    p = TextPath((0, 0), ch, size=1, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    v = p.vertices; lo, hi = v.min(0), v.max(0)
    yy, xx = np.mgrid[0:size, 0:size]
    span = (hi - lo).max()
    pts = np.stack([lo[0] + (xx.ravel() + 0.5) / size * span - (span - (hi - lo)[0]) / 2,
                    lo[1] + (size - yy.ravel() - 0.5) / size * span - (span - (hi - lo)[1]) / 2], 1)
    return p.contains_points(pts).reshape(size, size).astype(float)


def make_shape(family, rng, w):
    """Wanted far-field intensity (128 x 128, sum 1). w: Gaussian far-field spot radius [px]."""
    y, x = np.mgrid[0:CAM, 0:CAM] - CAM / 2 + 0.5
    rot = lambda a: (np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y)
    if family == "spot":
        u, v = rot(rng.uniform(0, np.pi))
        S = np.exp(-2 * ((u / (w * rng.uniform(0.9, 1.8))) ** 2 + (v / (w * rng.uniform(0.9, 1.8))) ** 2))
    elif family == "ring":
        r0, d = w * rng.uniform(1.0, 2.2), w * rng.uniform(0.45, 0.8)
        S = np.exp(-2 * ((np.hypot(x, y) - r0) / d) ** 2)
    elif family == "flattop":
        u, v = rot(rng.uniform(0, np.pi))
        a, b, m = w * rng.uniform(1.2, 2.4), w * rng.uniform(1.2, 2.4), rng.uniform(6, 12)
        S = np.exp(-np.hypot(u / a, v / b) ** m)
    elif family == "spots":
        S = np.zeros_like(x)
        k = rng.integers(2, 5)
        a0 = rng.uniform(0, 2 * np.pi)
        r = w * rng.uniform(1.4, 2.4)
        for j in range(k):
            a = a0 + 2 * np.pi * j / k
            S += np.exp(-2 * ((x - r * np.cos(a)) ** 2 + (y - r * np.sin(a)) ** 2) / (0.9 * w) ** 2)
    elif family == "line":
        u, v = rot(rng.uniform(0, np.pi))
        S = np.exp(-(np.abs(u) / (w * rng.uniform(2.0, 3.2))) ** 10 - 2 * (v / (0.9 * w)) ** 2)
    elif family == "polygon":
        from matplotlib.path import Path
        k, a0, r = rng.choice([3, 5, 6]), rng.uniform(0, 2 * np.pi), w * rng.uniform(2.0, 3.2)
        a = a0 + 2 * np.pi * np.arange(k) / k
        S = Path(np.stack([r * np.cos(a), r * np.sin(a)], 1)).contains_points(np.stack([x.ravel(), y.ravel()], 1)).reshape(x.shape).astype(float)
        S = gaussian_filter(S, w * rng.uniform(0.35, 0.6))
    elif family == "strokes":
        pts = np.stack([rng.uniform(-3 * w, 3 * w, 2) for _ in range(rng.integers(3, 5))])   # 3-4 corner points (x, y)
        d = np.full_like(x, np.inf, dtype=float)
        for A, B in zip(pts[:-1], pts[1:]):                                # distance to each segment of the polyline
            e = B - A
            t = np.clip(((x - A[0]) * e[0] + (y - A[1]) * e[1]) / (e @ e + 1e-9), 0, 1)
            d = np.minimum(d, np.hypot(x - A[0] - t * e[0], y - A[1] - t * e[1]))
        S = gaussian_filter((d < w * rng.uniform(0.45, 0.75)).astype(float), w * rng.uniform(0.35, 0.6))
    elif family == "blob":
        inside = np.hypot(x, y) < 3.2 * w
        f = gaussian_filter(rng.standard_normal(x.shape), w * rng.uniform(0.9, 1.6))
        S = gaussian_filter(((f > np.quantile(f[inside], rng.uniform(0.35, 0.65))) & inside).astype(float), w * rng.uniform(0.35, 0.6))
    elif family == "square":
        u, v = rot(rng.uniform(0, np.pi / 2))
        a = w * rng.uniform(1.4, 2.2)
        S = np.exp(-(np.abs(u) / a) ** 10 - (np.abs(v) / a) ** 10)
    elif family == "letter":
        ch = "AFTHLX"[rng.integers(6)]
        size = int(round(w * rng.uniform(4.5, 6.0)))
        m = letter_mask(ch, size)
        S = np.zeros_like(x); o = CAM // 2 - size // 2
        S[o:o + size, o:o + size] = m
        S = gaussian_filter(S, 0.5 * w)                                   # what a phase-only beam can draw
    else:
        raise ValueError(family)
    return (S / S.sum()).astype(np.float32)


def design(opt, S, steps=100, lr=0.2, seed=0):
    """Base holograms for wanted far-field intensities S (B, 128, 128): gradient ascent on
    (cosine similarity of the nominal system's far-field camera intensity with S) x sqrt(power kept),
    power kept = output power relative to the plain Gaussian's. Without the power term the design
    is free to clip most of the beam at the disk aperture (the banks without _relay were made so)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    phi = (0.3 * Fn.interpolate(torch.randn(len(S), 1, 16, 16, generator=g, device=DEV), size=(C, C), mode="bicubic")[:, 0]).requires_grad_()
    Adam = torch.optim.Adam([phi], lr=lr)
    S = S / S.norm(dim=(-2, -1), keepdim=True)
    with torch.no_grad():
        P_gauss = power(opt.output(torch.zeros(1, C, C, device=DEV), opt.screen0))
    for _ in range(steps):
        out = opt.output(phi, opt.screen0)
        I = opt.camera(opt.far(out), "far", noise=0, quantize=False)
        sim, kept = (I * S).sum(dim=(-2, -1)) / (I.norm(dim=(-2, -1)) + 1e-12), power(out) / P_gauss
        Adam.zero_grad(); (-(sim * kept.clamp(max=1).sqrt()).sum()).backward(); Adam.step()
    return torch.remainder(phi.detach() + np.pi, 2 * np.pi) - np.pi, sim.detach(), kept.detach()


def bank_file(name, relay):
    """bank.npz / bank_v2.npz for the layout as modelled, *_relay.npz for the SLM imaged onto the disk."""
    return os.path.join(OUT, name + ("_relay" if relay else "") + ".npz")


def build_bank(relay=False):
    opt = Optics(relay)
    with torch.no_grad():
        I0 = opt.camera(opt.far(opt.output(torch.zeros(1, C, C, device=DEV), opt.screen0)), "far", noise=0)[0].cpu().numpy()
    y, x = np.mgrid[0:CAM, 0:CAM]
    p = I0 / I0.sum(); cy, cx = (p * y).sum(), (p * x).sum()
    w = float(np.sqrt(((p * ((x - cx) ** 2 + (y - cy) ** 2)).sum())))      # 1/e^2 radius = 2 x rms radius / sqrt(2)... rms radius per axis x sqrt(2)
    print(f"far-field Gaussian: rms radius {w:.1f} px = {w * G.dx * 1e3:.2f} mm", flush=True)
    sets = {"train": (FAMILIES, N_TRAIN, 1), "test": (FAMILIES, N_TEST, 2), "unseen": (UNSEEN, N_UNSEEN, 3)}
    bank = {}
    t0 = time.time()
    for name, (fams, n, seed) in sets.items():
        rng = np.random.default_rng(seed)
        fam = [fams[i % len(fams)] for i in range(n)]
        S = np.stack([make_shape(f, rng, w) for f in fam])
        phis, sims, kept = [], [], []
        for b in range(0, n, 16):
            ph, sm, kp = design(opt, torch.as_tensor(S[b:b + 16], device=DEV), seed=seed * 1000 + b)
            phis.append(ph.cpu().numpy().astype(np.float16)); sims.append(sm.cpu().numpy()); kept.append(kp.cpu().numpy())
        sims, kept = np.concatenate(sims), np.concatenate(kept)
        bank[f"{name}_S"], bank[f"{name}_phi"], bank[f"{name}_sim"], bank[f"{name}_family"] = S, np.concatenate(phis), sims, np.array(fam)
        print(f"{name}: {n} holograms, design similarity {sims.mean():.3f}, power kept {kept.mean():.2f} (per family "
              + ", ".join(f"{f} {sims[np.array(fam) == f].mean():.2f} / {kept[np.array(fam) == f].mean():.2f}" for f in fams) + f"), {time.time() - t0:.0f} s", flush=True)
    # named: Gaussian and charge-1 vortex (no design)
    bank["named_phi"] = np.stack([np.zeros((C, C)), nd.crop(slm.vortex(1)(G.X, G.Y), C)]).astype(np.float16)
    bank["named_family"] = np.array(["gaussian", "vortex"])
    np.savez_compressed(bank_file("bank", relay), w=w, **bank)
    print("saved", bank_file("bank", relay), flush=True)


def spiral_phase(rng):
    """Charge 1-3 vortex (either sign) with random defocus, axicon and tilt terms on the correction crop."""
    x = (np.arange(C) - C / 2) * G.dx
    X, Y = np.meshgrid(x, x)
    r, R = np.hypot(X, Y), 1.5e-3
    l = rng.choice([-3, -2, -1, 1, 2, 3])
    tilt = 2 * np.pi / tds.LAM * rng.uniform(-3e-4, 3e-4, 2)               # up to 0.3 mm shift at the 1 m focus
    return np.angle(np.exp(1j * (l * np.arctan2(Y, X) + rng.uniform(-1.5, 1.5) * (r / R) ** 2 + rng.uniform(-4, 4) * r / R + tilt[0] * X + tilt[1] * Y)))


def extend_bank(relay=False):
    """bank + the new families -> bank_v2. Spirals are not designed: their shape is what the nominal system makes."""
    opt = Optics(relay)
    bank = dict(np.load(bank_file("bank", relay)))
    w = float(bank["w"])
    t0 = time.time()
    for name, n, seed in (("train", N_NEW_TRAIN, 11), ("test", N_NEW_TEST, 12)):
        rng = np.random.default_rng(seed)
        fam = np.array([NEW_FAMILIES[i % len(NEW_FAMILIES)] for i in range(n)])
        phi, S, sim, kept = np.zeros((n, C, C), np.float32), np.zeros((n, CAM, CAM), np.float32), np.ones(n, np.float32), np.ones(n, np.float32)
        spiral, designed = np.where(fam == "spiral")[0], np.where(fam != "spiral")[0]
        for i in spiral:
            phi[i] = spiral_phase(rng)
        for b in range(0, len(spiral), 16):
            idx = spiral[b:b + 16]
            with torch.no_grad():
                I = opt.camera(opt.far(opt.output(torch.as_tensor(phi[idx], device=DEV), opt.screen0)), "far", noise=0, quantize=False)
            S[idx] = (I / I.sum(dim=(-2, -1), keepdim=True)).cpu().numpy()
        for i in designed:
            S[i] = make_shape(fam[i], rng, w)
        for b in range(0, len(designed), 16):
            idx = designed[b:b + 16]
            ph, sm, kp = design(opt, torch.as_tensor(S[idx], device=DEV), seed=seed * 1000 + b)
            phi[idx], sim[idx], kept[idx] = ph.cpu().numpy(), sm.cpu().numpy(), kp.cpu().numpy()
        print(f"{name}: {n} new shapes, design similarity / power kept " + ", ".join(f"{f} {sim[fam == f].mean():.2f} / {kept[fam == f].mean():.2f}" for f in NEW_FAMILIES)
              + f", {time.time() - t0:.0f} s", flush=True)
        for key, new in (("S", S), ("phi", phi.astype(np.float16)), ("sim", sim), ("family", fam)):
            bank[f"{name}_{key}"] = np.concatenate([bank[f"{name}_{key}"], new])
    np.savez_compressed(bank_file("bank_v2", relay), **bank)
    print("saved", bank_file("bank_v2", relay), flush=True)


# --------------------------------------------------------------------------
# Disk states (Step 19 dataset) and batches
# --------------------------------------------------------------------------
def load_states():
    f = os.path.join(OUT, f"states_{os.environ['NN_RANGE']}.npz")
    if not os.path.exists(f):
        keys = {"opd": [], "T_pts": [], "P_meas": [], "dop_map": [], "index": []}
        for s in sorted(glob.glob(os.path.join(nd.RESULTS, "shards", "shard_*.npz"))):
            d = np.load(s)
            for k in keys:
                keys[k].append(d[k])
        np.savez(f, **{k: np.concatenate(v) for k, v in keys.items()})
    d = dict(np.load(f))
    print(f"{len(d['index'])} disk states from {nd.RESULTS}", flush=True)
    return d


class Batches:
    """Pairs of (disk state, shape) -> network input, target field, screens. All on the GPU."""

    def __init__(self, opt, states, bank, one_camera=False):
        self.opt, self.one_camera = opt, one_camera
        self.opd = torch.as_tensor(states["opd"])                          # large arrays stay in RAM, batches go to the GPU
        self.T = torch.as_tensor(states["T_pts"].astype(np.float32), device=DEV)
        self.P = torch.as_tensor(states["P_meas"].astype(np.float32), device=DEV)
        self.dop = torch.as_tensor(states["dop_map"].astype(np.float32))
        self.phi = {k[:-4]: torch.as_tensor(v.astype(np.float32)) for k, v in bank.items() if k.endswith("_phi")}

    def make(self, si, set_name, ti, noise=0.01):
        """si: state indices, ti: hologram indices of bank set `set_name`. Returns (x, phi_base, screens, T)."""
        o = self.opt
        phi = self.phi[set_name][ti].to(DEV)
        screens = o.phys.screens(self.opd[si].to(DEV))
        with torch.no_grad():
            T = o.output(phi, o.screen0)                                   # target field (nominal system)
            out0 = o.output(phi, screens)                                  # what the cameras see now
            near0, far0 = o.images(out0, noise)
            nearT, farT = o.images(T, 0)                                   # the target's own camera images
        B = len(si)
        const = lambda v: v.reshape(B, -1, 1, 1).expand(-1, -1, CAM, CAM)
        cs = Fn.avg_pool2d(torch.stack([torch.cos(phi), torch.sin(phi)], 1), 2)
        x = torch.cat([nt.IMG_SCALE * (near0 - nearT)[:, None], nt.IMG_SCALE * (far0 - farT)[:, None],
                       const((self.T[si] - o.ref_T) / nt.T_SCALE), const((self.P[si] - o.ref_P) / nt.P_SCALE),
                       nt.DOP_SCALE * self.dop[si].to(DEV)[:, None], nt.IMG_SCALE * nearT[:, None], nt.IMG_SCALE * farT[:, None], cs], 1)
        if self.one_camera:                                                # far-field camera only: the two near-field channels are empty
            x[:, [0, 9]] = 0
        return x, phi, screens, T


# --------------------------------------------------------------------------
def train(args):
    opt = Optics(args.relay)
    states, bank = load_states(), dict(np.load(os.path.join(OUT, args.bank)))
    bt, score = Batches(opt, states, bank, args.one_camera), scorer(opt, args.one_camera)
    what = "intensity similarity x power kept, far-field camera only" if args.one_camera else "fidelity x power kept"
    n_train, n_test, n_unseen = (len(bank[f"{k}_phi"]) for k in ("train", "test", "unseen"))
    val_s = np.where(states["index"] % 20 == 0)[0]
    train_s = np.setdiff1d(np.arange(len(states["index"])), val_s)
    out_dir = results_dir(f"23_nn_shapes/{args.name}")
    torch.manual_seed(0)
    net = nt.build("resnet", 13, args.width)
    print(f"resnet width {args.width}, {'SLM imaged onto the disk' if args.relay else 'layout as modelled'}: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters; "
          f"{len(train_s)} training states x {n_train} shapes ({args.bank})", flush=True)
    adam = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(adam, max_lr=args.lr, total_steps=args.epochs * args.steps + 1, pct_start=0.05)
    rng = np.random.default_rng(0)
    vrng = np.random.default_rng(123)                                      # fixed validation pairs
    vpairs = {"test": (vrng.choice(val_s, n_test), np.arange(n_test)),
              "unseen": (vrng.choice(val_s, n_unseen), np.arange(n_unseen)),
              "named": (vrng.choice(val_s, 32), np.arange(32) % 2)}
    history = []
    for epoch in range(args.epochs):
        net.train(); t0 = time.time(); losses = []
        for _ in range(args.steps):
            si, ti = rng.choice(train_s, args.batch), rng.integers(n_train, size=args.batch)
            x, phi, screens, T = bt.make(si, "train", ti)
            out = opt.output(phi + net(x), screens)
            loss = (1 - score(out, T)).mean()
            adam.zero_grad(); loss.backward(); adam.step(); sched.step()
            losses.append(loss.item())
        net.eval(); row = dict(epoch=epoch, loss=float(np.mean(losses)))
        with torch.no_grad():
            for name, (si, ti) in vpairs.items():
                D0, D1 = [], []
                for b in range(0, len(si), args.batch):
                    x, phi, screens, T = bt.make(si[b:b + args.batch], name, ti[b:b + args.batch])
                    D0.append(score(opt.output(phi, screens), T)); D1.append(score(opt.output(phi + net(x), screens), T))
                row[f"D0_{name}"], row[f"D_{name}"] = torch.cat(D0).mean().item(), torch.cat(D1).mean().item()
        row["seconds"] = time.time() - t0
        history.append(row)
        print(f"epoch {epoch:2d} loss {row['loss']:.4f}  val {what} (uncorrected -> network): "
              + "  ".join(f"{k} {row[f'D0_{k}']:.3f} -> {row[f'D_{k}']:.3f}" for k in vpairs) + f"  ({row['seconds']:.0f} s)", flush=True)
        torch.save(dict(state=net.state_dict(), args=vars(args)), os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "history.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    print("saved", out_dir, flush=True)


def ceiling(opt, phi, screens, T, start, steps=100, score=delivered):
    """Best correction with the true disk state known (Adam on the correction, from `start`,
    maximising the score, by default fidelity x power kept). Returns the corrected output fields."""
    corr = start.clone().requires_grad_()
    adam = torch.optim.Adam([corr], lr=0.05)
    for _ in range(steps):
        loss = (1 - score(opt.output(phi + corr, screens), T)).sum()
        adam.zero_grad(); loss.backward(); adam.step()
    with torch.no_grad():
        return opt.output(phi + corr, screens)


def evaluate(args):
    ck = torch.load(os.path.join(OUT, args.name, "model.pt"), map_location=DEV)
    opt = Optics(ck["args"].get("relay", False))
    states, bank = load_states(), dict(np.load(os.path.join(OUT, ck["args"].get("bank", "bank.npz"))))
    one = ck["args"].get("one_camera", False)
    bt, score = Batches(opt, states, bank, one), scorer(opt, one)
    what = "intensity similarity x power kept" if one else "fidelity x power kept"
    net = nt.build("resnet", 13, ck["args"]["width"]); net.load_state_dict(ck["state"]); net.eval()
    val_s = np.where(states["index"] % 20 == 0)[0]
    rng = np.random.default_rng(7)
    rows, ex = [], {}
    for set_name, n in (("test", len(bank["test_phi"])), ("unseen", len(bank["unseen_phi"])), ("named", 2)):
        fam = bank[f"{set_name}_family"]
        for b in range(0, n if set_name != "named" else 32, 8):
            ti = np.arange(b, b + 8) % n
            si = rng.choice(val_s, len(ti))
            x, phi, screens, T = bt.make(si, set_name, ti)
            with torch.no_grad():
                corr = net(x)
                o0, o1 = opt.output(phi, screens), opt.output(phi + corr, screens)
            oc = ceiling(opt, phi, screens, T, corr.detach(), args.ceiling_steps, score)
            F0, F1 = score(o0, T), score(o1, T)                   # in the figure: fidelity x power kept
            for k in range(len(ti)):
                rows.append(dict(set=set_name, family=fam[ti[k]], state=int(si[k]), shape=int(ti[k]),
                                 **{f"{m}_{tag}": f(o[k:k + 1], T[k:k + 1]).item() for tag, o in (("0", o0), ("net", o1), ("ceiling", oc))
                                    for m, f in (("D", score), ("F", fidelity), ("P", lambda a, b: power(a) / power(b)))}))
            with torch.no_grad():                                         # far-field images for the figure
                for k in range(len(ti)):
                    key = (set_name, fam[ti[k]])
                    if key not in ex:
                        far = lambda f: opt.camera(opt.far(f), "far", 0, quantize=False)[0].cpu().numpy()
                        S = bank[f"{set_name}_S"][ti[k]] if f"{set_name}_S" in bank else far(T[k:k + 1])
                        ex[key] = dict(S=S, F0=F0[k].item(), F1=F1[k].item(), u=far(opt.output(phi[k:k + 1], screens[k:k + 1])),
                                       c=far(opt.output(phi[k:k + 1] + corr[k:k + 1], screens[k:k + 1])), t=far(T[k:k + 1]))
        print(f"{set_name}: done ({len(rows)} rows)", flush=True)
    with open(os.path.join(OUT, f"eval_{args.name}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # ---- table ---------------------------------------------------------------------------------
    lines = ["# One network for any shape (Step 23)", "",
             f"Network '{args.name}', {'SLM imaged onto the disk' if opt.relay else 'layout as modelled (SLM 2.9 m before the disk)'}. "
             "Held-out disk states of the wide range (240-360 W, no lab nuisances), ideal SLM. "
             + ("One camera (far field): the phase is not rated." if one else ""),
             f"Target = what the nominal system makes from the shape's base hologram. Each cell: {what} "
             "(field fidelity alone / output power relative to the target's), mean.", "",
             "| shapes | family | n | uncorrected | network, one shot | ceiling (disk known) |", "|---|---|---|---|---|---|"]
    for set_name in ("test", "unseen", "named"):
        fams = sorted(set(r["family"] for r in rows if r["set"] == set_name))
        for f in fams + ["all"]:
            rr = [r for r in rows if r["set"] == set_name and (f == "all" or r["family"] == f)]
            cell = lambda k: f"{np.mean([r[f'D_{k}'] for r in rr]):.2f} ({np.mean([r[f'F_{k}'] for r in rr]):.2f} / {np.mean([r[f'P_{k}'] for r in rr]):.2f})"
            lines.append(f"| {set_name} | {f} | {len(rr)} | {cell('0')} | {cell('net')} | {cell('ceiling')} |")
    open(os.path.join(OUT, f"NN_SHAPES_{args.name}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    # ---- figure: far-field (work plane) images -------------------------------------------------------
    keys = [k for k in ex if k[0] == "test"] + [k for k in ex if k[0] == "unseen"] + [k for k in ex if k[0] == "named"]
    fig, ax = plt.subplots(len(keys), 4, figsize=(10.5, 2.55 * len(keys)))
    for r, k in enumerate(keys):
        e = ex[k]
        panels = ((e["S"], "wanted shape"), (e["u"], f"uncorrected, {e['F0']:.2f}"),
                  (e["c"], f"network, one shot, {e['F1']:.2f}"), (e["t"], "target (undistorted system)"))
        for c, (img, t) in enumerate(panels):
            a = ax[r, c]
            if img is not None:
                a.imshow(img / img.max(), cmap="inferno", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(t, fontsize=9)
        ax[r, 0].set_ylabel(f"{k[1]}\n({'never seen' if k[0] == 'unseen' else k[0]})", fontsize=9)
    fig.suptitle("One network for any shape: work-plane intensity (focus of the 1 m lens), held-out disk states\n"
                 f"numbers: {what}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(os.path.join(OUT, f"shapes_{args.name}.png"), dpi=110); plt.close(fig)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("bank", "extend", "train", "eval", "smoke"))
    ap.add_argument("--bank", default="bank.npz", help="bank.npz (5 families), bank_v2.npz (9 families), or their *_relay.npz versions")
    ap.add_argument("name", nargs="?", default="shapes_v1")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--relay", action="store_true", help="what-if layout: the SLM is imaged onto the disk (banks and results get the suffix _relay)")
    ap.add_argument("--one-camera", action="store_true", help="far-field camera only; trained and scored on the far-field intensity (phase not rated)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ceiling-steps", type=int, default=100)
    a = ap.parse_args()
    if a.mode == "bank":
        build_bank(a.relay)
    elif a.mode == "extend":
        extend_bank(a.relay)
    elif a.mode == "train":
        train(a)
    elif a.mode == "eval":
        evaluate(a)
    else:                                                                    # quick timing / consistency check
        opt = Optics(a.relay)
        with torch.no_grad():
            phi = torch.zeros(1, C, C, device=DEV)
            T = opt.output(phi, opt.screen0)
            print("nominal system, zero correction, against itself:", fidelity(opt.output(phi, opt.screen0), T).item())
            near, far = opt.images(T, 0)
            print("camera peaks near %.2f far %.2f" % (near.max().item(), far.max().item()))
        S = torch.as_tensor(np.stack([make_shape(f, np.random.default_rng(0), 13.0) for f in FAMILIES + UNSEEN]), device=DEV)
        t0 = time.time(); ph, sim, kept = design(opt, S, steps=20)
        print(f"design 20 steps x {len(S)}: {time.time() - t0:.1f} s, similarity {sim.cpu().numpy().round(2)}, power kept {kept.cpu().numpy().round(2)}")
