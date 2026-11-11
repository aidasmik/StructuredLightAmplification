"""
nn_train.py - Step 18: train the one-shot pre-correction network on the Step 17 dataset.

Residual formulation (default). A reference state is recorded once: the tuned
system with the nominal 300 W disk, its two camera images per target, the
pyrometer readings and the SLM correction that is optimal for it
(nn_dataset.py reference -> reference.npz). The network sees only the
DIFFERENCES to that reference and outputs the CHANGE to add to the reference
correction. With no change in the images it outputs nothing and the reference
correction is displayed. (--formulation direct: raw inputs, full correction.)

Input, 10 channels of 128 x 128:
    near-field image - reference, far-field image - reference
    five pyrometer readings - reference, pump power - reference (constant channels)
    the known doping map of the crystal (deviation from nominal)
    target flag (0 Gaussian, 1 LG01)
    --inputs cams zeroes the temperatures, the power and the doping map.
Output: phase change on 256 x 256 (6.4 mm at the SLM) as (cos, sin) -> atan2.

Backbones: --arch resnet (default): full-resolution residual blocks with
dilations 1,2,4,8 repeated, so the receptive field covers the whole image;
--arch unet: the 4-level U-Net.

Losses:  supervised  piston-free phase error against the physics-optimal label,
                     weighted with the seed intensity on the SLM,
         physics     1 - fidelity x power kept of the output field, through the
                     differentiable torch amplifier with the example's own disk state.
                     Fidelity alone is normalised by the output power, so a network
                     trained on it throws light out of the disk aperture and keeps a
                     small clean remainder (Step 23 finding). Power kept = output power
                     / output power with the reference correction, at most 1.
Schedule: EPOCHS_SUP supervised epochs, then EPOCHS_PHYS epochs physics + 0.1 supervised.

Usage:  python nn_train.py NAME [--arch resnet|unet] [--formulation residual|direct]
                               [--inputs all|cams] [--epochs-sup N] [--epochs-phys N]
Output: Results/18_nn/NAME/model.pt, history.csv, training.png
"""
import os
import csv
import glob
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint_sequential
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thin_disk_system as tds
import slm
from seed_beam import Grid, results_dir
from gpu import TorchAmplifier
import nn_dataset as nd

warnings.filterwarnings("ignore")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = nd.RESULTS                  # dataset of the selected range (NN_RANGE), see nn_dataset.py
IMG_SCALE = 5.0                    # image differences (fraction of full scale) -> ~unity
T_SCALE, P_SCALE = nd.T_SCALE, nd.P_SCALE   # temperature differences [K], pump power differences [W]
DOP_SCALE = 50.0                   # doping deviation (+-2 %) -> ~unity


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
STATE_KEYS = ("opd", "gain", "seed", "tilts", "stroke")   # what the physics loss needs of a state


def state_tensors(data, idx):
    """The stored state of rows idx as tensors (keys present in the dataset only)."""
    return {k: torch.as_tensor(data[k][idx], device=DEV) for k in STATE_KEYS if k in data}


BIG = {"near": np.float16, "far": np.float16, "label": np.float16, "dop": np.float16, "opd": np.float32, "gain": np.float16}


def load_dataset(max_examples=None):
    """All shards -> dict of arrays. Per-target rows (near, far, label, F0, Fopt, tgt, index) and
    per-example arrays (T, P, dop and the state) kept once; data["ex"] maps a row to its example.
    The large arrays are written once into Results/<dataset>/cache/*.npy and memory-mapped, so
    20 000 examples (~13 GB) do not have to fit in RAM next to the user's desktop."""
    files = sorted(glob.glob(os.path.join(DATA, "shards", "shard_*.npz")))
    counts = [len(np.load(f)["index"]) for f in files]
    if max_examples:
        keep = int(np.searchsorted(np.cumsum(counts), max_examples)) + 1
        files, counts = files[:keep], counts[:keep]
    n_ex, n_rows = sum(counts), 2 * sum(counts)
    cache = os.path.join(DATA, "cache" + (f"_{max_examples}" if max_examples else ""))
    if not os.path.exists(os.path.join(cache, "done")):
        os.makedirs(cache, exist_ok=True)
        keys = [k for k in BIG if k in ("near", "far", "label", "dop", "opd") or k in np.load(files[0]).files]
        side = {"label": nd.CROP_LABEL}
        mm = {k: np.lib.format.open_memmap(os.path.join(cache, f"{k}.npy"), "w+", BIG[k],
                                           (n_rows if k in ("near", "far", "label") else n_ex, side.get(k, nd.CROP_CAM), side.get(k, nd.CROP_CAM)))
              for k in keys}
        small = {k: [] for k in ("F0", "Fopt", "tgt", "index", "ex", "T", "P", "seed", "tilts", "stroke", "setup")}
        o = 0
        for f, m in zip(files, counts):
            d = np.load(f)
            for t, name in enumerate(nd.TARGETS):
                r = slice(2 * o + t * m, 2 * o + (t + 1) * m)                       # rows of this shard and target
                mm["near"][r], mm["far"][r], mm["label"][r] = d[f"near_{name}"], d[f"far_{name}"], d[f"label_{name}"]
                small["F0"].append(d[f"F0_{name}"]); small["Fopt"].append(d[f"Fopt_{name}"])
                small["tgt"].append(np.full(m, t)); small["index"].append(d["index"]); small["ex"].append(o + np.arange(m))
            for k in ("dop", "opd", "gain"):
                if k in mm:
                    mm[k][o:o + m] = d["dop_map" if k == "dop" else k]
            small["T"].append(d["T_pts"]); small["P"].append(d["P_meas"])
            for k in ("seed", "tilts", "stroke", "setup"):
                if k in d.files:
                    small[k].append(d[k])
            o += m
        for k, v in small.items():
            if v:
                np.save(os.path.join(cache, f"{k}.npy"), np.concatenate(v))
        for a in mm.values():
            a.flush()
        open(os.path.join(cache, "done"), "w").close()
    data = {}
    for f in sorted(glob.glob(os.path.join(cache, "*.npy"))):
        k = os.path.basename(f)[:-4]
        data[k] = np.load(f, mmap_mode="r") if k in BIG else np.load(f)
    print(f"{len(files)} shards, {n_rows} rows ({n_ex} examples), cache {cache}", flush=True)
    return data


def load_reference():
    """reference.npz -> tensors indexed by (setup, target id): near/far (S,2,128,128), corr (S,2,256,256),
    T_pts (S,5), P (S,). A single reference (ranges without setups) gets S = 1."""
    r = np.load(os.path.join(DATA, "reference.npz"))
    multi = r[f"near_{nd.TARGETS[0]}"].ndim == 3
    def stack(key):
        a = np.stack([r[f"{key}_{t}"] for t in nd.TARGETS], axis=-3).astype(np.float32)
        return torch.as_tensor(a if multi else a[None], device=DEV)
    return dict(near=stack("near"), far=stack("far"), corr=stack("corr"),
                T_pts=torch.as_tensor(np.atleast_2d(r["T_pts"]).astype(np.float32), device=DEV),
                P=torch.as_tensor(np.atleast_1d(r["P"]).astype(np.float32), device=DEV))


def make_input(near, far, T_pts, P, dop_map, tgt, ref, inputs="all", formulation="residual", setup=None, ref_images=False):
    """numpy arrays (B, ...) of the measurements -> network input tensor (B, 10, 128, 128).
    setup: the setup of each example (its reference is subtracted); None = setup 0.
    ref_images: also give the setup's reference images themselves (channels 10, 11), so the
    network sees the fixed seed / mirror / camera errors of the setup and not only the change."""
    f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=DEV)
    near, far, dop, T_pts, P = f(near), f(far), f(dop_map), f(T_pts), f(P)
    t = torch.as_tensor(np.asarray(tgt, np.int64), device=DEV)
    s = torch.zeros_like(t) if setup is None else torch.as_tensor(np.asarray(setup, np.int64), device=DEV)
    if formulation == "residual":                         # differences to the recorded reference state of this setup
        near, far = near - ref["near"][s, t], far - ref["far"][s, t]
        T_pts, P = T_pts - ref["T_pts"][s], P - ref["P"][s]
    else:                                                 # raw measurements, centred roughly
        T_pts, P = T_pts - 340.0, P - 300.0
    B, n = near.shape[0], near.shape[-1]
    const = lambda v: v.reshape(B, -1, 1, 1).expand(-1, -1, n, n)
    x = torch.cat([IMG_SCALE * near[:, None], IMG_SCALE * far[:, None], const(T_pts / T_SCALE),
                   const(P / P_SCALE), DOP_SCALE * dop[:, None], const(t.float())], dim=1)
    if inputs == "cams":                                  # cameras and target flag only
        x[:, 2:9] = 0
    if ref_images:
        x = torch.cat([x, IMG_SCALE * ref["near"][s, t][:, None], IMG_SCALE * ref["far"][s, t][:, None]], dim=1)
    return x


def batch_tensors(data, idx, ref, inputs, formulation, ref_images=False):
    """-> network input, label, stored state, target id, setup id (zeros with a single reference)."""
    e = data["ex"][idx]                                            # the example of each row
    setup = data["setup"][e] if "setup" in data else np.zeros(len(idx), np.int64)
    x = make_input(data["near"][idx], data["far"][idx], data["T"][e], data["P"][e], data["dop"][e], data["tgt"][idx],
                   ref, inputs, formulation, setup, ref_images)
    label = torch.as_tensor(data["label"][idx].astype(np.float32), device=DEV)
    tgt = torch.as_tensor(data["tgt"][idx].astype(np.int64), device=DEV)
    return x, label, state_tensors(data, e), tgt, torch.as_tensor(setup.astype(np.int64), device=DEV)


# --------------------------------------------------------------------------
# Networks. Both end in a (cos, sin) head at 128 x 128, upsampled to 256 x 256,
# initialised to output zero phase (the identity: reference correction only).
# --------------------------------------------------------------------------
def phase_head(width):
    head = nn.Conv2d(width, 2, 1)
    nn.init.zeros_(head.weight)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([1.0, 0.0]))         # cos = 1, sin = 0 -> phase 0
    return head


def to_phase(y):
    y = Fn.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
    return torch.atan2(y[:, 1], y[:, 0])


class ResBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation), nn.GroupNorm(8, width), nn.GELU(),
                                  nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation), nn.GroupNorm(8, width))

    def forward(self, x):
        return Fn.gelu(x + self.body(x))


class ResNet(nn.Module):
    """Full-resolution ResNet with dilated blocks (receptive field ~180 px > the 128 px image)."""

    def __init__(self, cin=10, width=64, blocks=12):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, width, 3, padding=1), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(width, (1, 2, 4, 8)[i % 4]) for i in range(blocks)])
        self.head = phase_head(width)

    def forward(self, x):
        x = self.stem(x)
        if self.training and x.requires_grad is False:
            x.requires_grad_(True)                        # checkpointing needs a grad-tracking input
        h = checkpoint_sequential(self.blocks, 4, x, use_reentrant=False) if self.training else self.blocks(x)
        return to_phase(self.head(h))                     # activations recomputed in backward: ~4x less memory


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class UNet(nn.Module):
    def __init__(self, cin=10, width=32, depth=4):
        super().__init__()
        ch = [width * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([conv_block(cin, ch[0])] + [conv_block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([conv_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(depth))])
        self.head = phase_head(ch[0])

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else Fn.max_pool2d(x, 2))
            skips.append(x)
        for u, s in zip(self.up, reversed(skips[:-1])):
            x = u(torch.cat([Fn.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), s], dim=1))
        return to_phase(self.head(x))


def build(arch, cin=10, width=64):
    return (ResNet(cin, width) if arch == "resnet" else UNet(cin)).to(DEV)


# --------------------------------------------------------------------------
# Physics: correction -> output field -> fidelity (ideal SLM, torch amplifier)
# --------------------------------------------------------------------------
class Physics:
    def __init__(self, meta):
        self.G = Grid(n=int(meta["grid_n"]), size=float(meta["grid_size"]))
        n, c = self.G.n, int(meta["crop_label"])
        tds.DISK.thermal_opd = np.zeros((n, n))                      # placeholder screen; set per batch
        import limits_benchmark as lb
        f_th, _ = lb.per_pass_lens(lb.thermal(nd.P0).opd(self.G), self.G)
        tds.MIRROR_ROC.update(lb.retuned_mirrors(f_th))
        tds.DISK.gain = np.ones((n, n)) if nd.REALISM else None     # slot for the per-example gain map
        self.amp = TorchAmplifier(self.G, device=DEV)
        tds.DISK.gain = None
        self.fin = torch.as_tensor(meta["fin"], device=DEV)
        self.seed_basis = torch.as_tensor(np.stack([slm.zernike({t: 1.0}, nd.R_SEED)(self.G.X, self.G.Y) for t in nd.SEED_TERMS]
                                                   + [self.G.X / nd.R_SEED, self.G.Y / nd.R_SEED]).astype(np.float32), device=DEV)
        self.targets = torch.stack([torch.as_tensor(meta[f"target_{t}"], device=DEV) for t in nd.TARGETS])
        self.base = torch.stack([torch.as_tensor(meta[f"base_{t}"], device=DEV) for t in nd.TARGETS])
        self.slm_gain = np.sqrt(tds.SLM_DEVICE.reflectivity) * tds.SLM_DEVICE.fill_factor
        w = (self.fin.abs() ** 2)[n // 2 - c // 2:n // 2 + c // 2, n // 2 - c // 2:n // 2 + c // 2]
        self.w = w / w.sum()                                          # seed intensity on the SLM (label crop)
        self.pad = (n - c) // 2

    def screens(self, crop, outside=0.0):
        """Stored 50 um crop (B, 128, 128) -> full grid (B, n, n), `outside` beyond the crop (behind the aperture)."""
        up = Fn.interpolate(crop[:, None].float(), scale_factor=2, mode="bilinear", align_corners=False)[:, 0]
        return Fn.pad(up, (self.pad,) * 4, value=outside)

    def output(self, corr, st, tgt):
        """corr: (B, 256, 256) full phase correction; st: the stored state (state_tensors).
        Output field per example, ideal SLM apart from the stroke error."""
        phi = self.base[tgt] + Fn.pad(corr, (self.pad,) * 4)
        fin = self.fin
        if "seed" in st:                                                             # this example's seed beam
            s = st["seed"]
            fin = torch.stack([torch.roll(self.fin, (int(s[b, 8]), int(s[b, 7])), dims=(0, 1)) for b in range(len(s))])
            fin = fin * torch.polar(torch.sqrt(s[:, 9])[:, None, None] * torch.ones_like(phi), torch.einsum("bk,kxy->bxy", s[:, :7], self.seed_basis))
        if "stroke" in st:                                                           # LUT error on the wrapped phase
            phi = torch.remainder(phi + np.pi, 2 * np.pi) * (1 + st["stroke"])[:, None, None]
        field = fin * torch.polar(torch.ones_like(phi), phi)
        field = self.slm_gain * torch.roll(field.flip(-1), 1, dims=-1)                 # SLM image flip
        self.amp.set_thermal_opd(self.screens(st["opd"]))
        if "gain" in st:
            self.amp.set_gain(self.screens(st["gain"], 1.0))
        if "tilts" in st:
            self.amp.set_mirror_tilts(st["tilts"])
        return self.amp.forward_t(field)

    def fidelity(self, corr, st, tgt, with_power=False):
        """Field fidelity |<T, out>|^2 / (|T|^2 |out|^2) per example; with_power: also the output power."""
        out, t = self.output(corr, st, tgt), self.targets[tgt]
        P = (out.abs() ** 2).sum(dim=(-2, -1))
        F = (t.conj() * out).sum(dim=(-2, -1)).abs() ** 2 / (P * (t.abs() ** 2).sum(dim=(-2, -1)))
        return (F, P) if with_power else F

    def delivered(self, corr, st, tgt, ref_corr):
        """Fidelity x power kept: the share of the available power that arrives in the target mode.
        Power kept = P_out / P_out(reference correction, same disk state), at most 1.
        Returns (delivered, fidelity, power kept) per example."""
        F, P = self.fidelity(corr, st, tgt, with_power=True)
        with torch.no_grad():
            P_ref = self.fidelity(ref_corr, st, tgt, with_power=True)[1]
        kept = torch.clamp(P / P_ref, max=1)
        return F * kept, F, kept

    def supervised(self, corr, label):
        d = corr - label
        return 1 - (self.w * torch.polar(torch.ones_like(d), d)).sum(dim=(-2, -1)).abs()


def full_correction(net_out, tgt, ref, formulation, setup=None):
    """Network output -> correction to display: reference correction of this setup + change (residual) or as is."""
    s = torch.zeros_like(tgt) if setup is None else torch.as_tensor(setup, device=DEV)
    return ref["corr"][s, tgt] + net_out if formulation == "residual" else net_out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--arch", default="resnet", choices=("resnet", "unet"))
    ap.add_argument("--formulation", default="residual", choices=("residual", "direct"))
    ap.add_argument("--inputs", default="all", choices=("all", "cams"))
    ap.add_argument("--epochs-sup", type=int, default=15)
    ap.add_argument("--epochs-phys", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)          # 8 x 512^2 complex activations through ~100 ops fit the 8 GB card
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--init", default=None, help="model.pt to start from (e.g. the wide-range network for the realism set)")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (one-cycle); 3e-4 for fine-tuning")
    ap.add_argument("--width", type=int, default=64, help="ResNet channels (64: 0.9 M parameters, 128: 3.6 M)")
    ap.add_argument("--ref-images", action="store_true", help="also input the setup's reference images")
    ap.add_argument("--sup-weight", type=float, default=0.1, help="weight of the label term during the physics phase")
    args = ap.parse_args()
    out_dir = results_dir(f"{nd.TRAIN_RESULTS}/{args.name}")
    torch.manual_seed(0); np.random.seed(0)

    data = load_dataset(args.max_examples)
    meta = np.load(os.path.join(DATA, "meta.npz"))
    ref = load_reference()
    phys = Physics(meta)
    n_rows = len(data["index"])
    val = np.where(data["index"] % 20 == 0)[0]                         # 5 % held out by example index
    train = np.setdiff1d(np.arange(n_rows), val)
    print(f"train rows {len(train)}, validation rows {len(val)}; device {DEV}; {args.arch}, {args.formulation}, inputs {args.inputs}", flush=True)

    # sanity: the label through the reconstructed 50 um screens should give the stored optimum,
    # and the reference correction alone is the starting point of the residual network
    idx = val[:16]
    _, label, st, tgt, setup = batch_tensors(data, idx, ref, args.inputs, args.formulation, args.ref_images)
    with torch.no_grad():
        F_lab = phys.fidelity(label, st, tgt).cpu().numpy()
        F_ref = phys.fidelity(ref["corr"][setup, tgt], st, tgt).cpu().numpy()
    print(f"label check: fidelity through reconstructed screens {F_lab.mean():.3f} vs stored optimum {data['Fopt'][idx].mean():.3f}; "
          f"reference correction alone {F_ref.mean():.3f}, uncorrected {data['F0'][idx].mean():.3f}", flush=True)

    net = build(args.arch, 12 if args.ref_images else 10, args.width)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=DEV)["state"])
        print(f"initialised from {args.init}", flush=True)
    print(f"{args.arch} with {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters", flush=True)
    n_epochs = args.epochs_sup + args.epochs_phys
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=n_epochs * (len(train) // args.batch) + 1, pct_start=0.1)
    history = []
    rng = np.random.default_rng(0)
    for epoch in range(n_epochs):
        physics_on = epoch >= args.epochs_sup
        net.train(); t0 = time.time(); losses = []
        for b in range(len(train) // args.batch):
            idx = rng.choice(train, args.batch, replace=False)
            x, label, st, tgt, setup = batch_tensors(data, idx, ref, args.inputs, args.formulation, args.ref_images)
            corr = full_correction(net(x), tgt, ref, args.formulation, setup)
            L_sup = phys.supervised(corr, label).mean()
            if physics_on:
                loss = (1 - phys.delivered(corr, st, tgt, ref["corr"][setup, tgt])[0]).mean() + args.sup_weight * L_sup
            else:
                loss = L_sup
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        # validation: ideal-SLM fidelity x power kept (D), fidelity (F) and power kept (K) on held-out rows, per target
        net.eval(); V = {t: [] for t in range(len(nd.TARGETS))}
        with torch.no_grad():
            for s in range(0, min(len(val), 512), args.batch):
                idx = val[s:s + args.batch]
                x, label, st, tgt, setup = batch_tensors(data, idx, ref, args.inputs, args.formulation, args.ref_images)
                dfk = phys.delivered(full_correction(net(x), tgt, ref, args.formulation, setup), st, tgt, ref["corr"][setup, tgt])
                for ti, v in zip(tgt.cpu().numpy(), torch.stack(dfk, dim=1).cpu().numpy()):
                    V[int(ti)].append(v)
        V = {t: np.mean(V[t], axis=0) for t in V}                       # per target: D, F, K
        row = dict(epoch=epoch, phase="phys" if physics_on else "sup", loss=float(np.mean(losses)),
                   **{f"{k}_{nd.TARGETS[t]}": float(V[t][j]) for t in V for j, k in enumerate("DFK")}, seconds=time.time() - t0)
        history.append(row)
        print(f"epoch {epoch:3d} [{row['phase']}] loss {row['loss']:.4f}  val fidelity x power kept (fidelity / power kept) "
              + "  ".join(f"{nd.TARGETS[t]} {V[t][0]:.3f} ({V[t][1]:.3f} / {V[t][2]:.2f})" for t in V) + f"  ({row['seconds']:.0f} s)", flush=True)
        torch.save(dict(state=net.state_dict(), args=vars(args)), os.path.join(out_dir, "model.pt"))

    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    vi = val[:512]
    refF = {t: (data["F0"][vi][data["tgt"][vi] == i].mean(), data["Fopt"][vi][data["tgt"][vi] == i].mean()) for i, t in enumerate(nd.TARGETS)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([h["loss"] for h in history]); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].grid(alpha=0.3)
    ax[0].axvline(args.epochs_sup - 0.5, color="k", ls=":", label="physics loss switched on"); ax[0].legend()
    for i, t in enumerate(nd.TARGETS):
        ax[1].plot([h[f"D_{t}"] for h in history], color=f"C{i}", label=f"{t}: network, fidelity x power kept")
        ax[1].plot([h[f"F_{t}"] for h in history], color=f"C{i}", lw=0.8, alpha=0.5, label=f"{t}: network, fidelity only")
        ax[1].axhline(refF[t][1], color=f"C{i}", ls="--", label=f"{t}: label, fidelity only")
        ax[1].axhline(refF[t][0], color=f"C{i}", ls=":", label=f"{t}: uncorrected, fidelity only")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation (ideal SLM)"); ax[1].set_ylim(0, 1); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle(f"Pre-correction network '{args.name}': {args.arch}, {args.formulation}, inputs {args.inputs}")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "training.png"), dpi=110)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
