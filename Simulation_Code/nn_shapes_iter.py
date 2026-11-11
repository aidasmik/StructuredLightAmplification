"""
nn_shapes_iter.py - Step 24: the single-shot networks against the best iterative loops, like for like.

Same held-out disk states, same shapes, same camera model and the same two scores for every method:

    I   one camera:  intensity similarity of the far-field images x power kept (phase not rated)
    D   two cameras: field fidelity x power kept

Iterative loops (they see camera frames only, 1 % noise, as the networks do):
    SPGD        Xie et al., Opt. Lett. 40, 1197 (2015): one far-field camera, 14 Zernike terms,
                a += eta s (C2 - C1), two frames per iteration (xie_spgd.py). Perturbation, gain and
                feedback signal (paper's intensity overlap / image correlation) are chosen by a sweep
                on a few pairs, by the true score: the best case for the loop.
    response    two cameras, calibrated response matrix of 28 Zernike modes and ridge least squares
    matrix      (two_camera_correction.py, method B). The complex field is taken as measured exactly
                (no phase-retrieval error, no noise): again the best case for the loop.
Networks:       shapes_onecam (far-field frame only) and shapes_free (both frames); with --relay
                (SLM imaged onto the disk) shapes_relay.
Ceiling:        best phase-only pattern with the disk known, for each score (nn_shapes.ceiling).

Usage:  NN_RANGE=wide python nn_shapes_iter.py [--relay] [--quick]        (GPU PC, ~1 h)
Output: Results/23_nn_shapes/ITER_<layout>.md, iter_<layout>.csv, iter_<layout>.npz (mean curves)
"""
import os
import csv
import time
import argparse

import numpy as np
import torch

import nn_shapes as ns
import nn_dataset as nd
import nn_train as nt
from spgd import zernike_basis

ap = argparse.ArgumentParser()
ap.add_argument("--relay", action="store_true")
ap.add_argument("--quick", action="store_true", help="tiny run to test the script")
args = ap.parse_args()
DEV, C, G = ns.DEV, ns.C, ns.G
LAYOUT = "relay" if args.relay else "free"
B = 12                                                   # pairs per batch
N_SPGD, N_RM = 14, 28                                    # Zernike terms (paper: 14; response matrix: 28)
ITER = 10 if args.quick else 300                         # SPGD iterations, 2 frames each
SWEEP_ITER = 6 if args.quick else 100
DELTAS, ETAS = ((0.05,), (10.0,)) if args.quick else ((0.01, 0.05, 0.15), (3.0, 10.0, 30.0))
LOG_EVERY = 5 if args.quick else 10
t0 = time.time()

opt = ns.Optics(args.relay)
states = ns.load_states()
bank = dict(np.load(ns.bank_file("bank_v2", args.relay)))
bt = ns.Batches(opt, states, bank)
SCORE = {"I": ns.scorer(opt, True), "D": ns.delivered}

# Zernike basis on the SLM crop. Radius 3 mm for the seed of the layout as modelled (as in Steps 14-22),
# scaled with the beam radius when the SLM is imaged onto the disk (smaller beam on the SLM).
rms = lambda f: float(((f.abs() ** 2 * torch.as_tensor(G.R ** 2, device=DEV)).sum() / (f.abs() ** 2).sum()).sqrt())
RADIUS = 3.0e-3 * rms(opt.fin) / rms(opt.phys.fin)
basis = torch.as_tensor(np.stack([nd.crop(z, C) for z in zernike_basis(G, radius=RADIUS, n_modes=N_RM)]).astype(np.float32), device=DEV)
print(f"layout {LAYOUT}: Zernike radius {RADIUS * 1e3:.2f} mm", flush=True)


def gather(pairs):
    """pairs: list of (set name, state indices, hologram indices) -> network input, hologram, screens, target."""
    parts = [bt.make(si, name, ti) for name, si, ti in pairs]
    return [torch.cat(p) for p in zip(*parts)]


def scores(corr, phi, screens, T):
    out = opt.output(phi + corr, screens)
    return {k: f(out, T) for k, f in SCORE.items()}


@torch.no_grad()
def spgd(phi, screens, T, delta, eta, iterations, metric, seed=1):
    """Published SPGD, batched. Returns the final correction and the log [(frames, scores)]."""
    Z = basis[:N_SPGD]
    frame = lambda a: opt.camera(opt.far(opt.output(phi + torch.einsum("bn,nxy->bxy", a, Z), screens)), "far", noise=0.01)
    ideal = opt.camera(opt.far(T), "far", noise=0)                      # far-field image of the undistorted beam
    if metric == "paper":                                               # Eq. 2 of the paper, 1 for the ideal frame
        Cm = lambda img: (img * ideal).sum(dim=(-2, -1)) / (ideal * ideal).sum(dim=(-2, -1))
    else:                                                               # Pearson correlation of the two images
        zm = lambda x: x - x.mean(dim=(-2, -1), keepdim=True)
        Cm = lambda img: (zm(img) * zm(ideal)).sum(dim=(-2, -1)) / (zm(img).norm(dim=(-2, -1)) * zm(ideal).norm(dim=(-2, -1)) + 1e-12)
    g = torch.Generator(device=DEV).manual_seed(seed)
    a = torch.zeros(len(phi), N_SPGD, device=DEV)
    corr = lambda: torch.einsum("bn,nxy->bxy", a, Z)
    log = [(0, scores(corr(), phi, screens, T))]
    for k in range(1, iterations + 1):
        s = torch.randint(0, 2, a.shape, generator=g, device=DEV) * 2.0 - 1
        C1, C2 = Cm(frame(a)), Cm(frame(a + delta * s))
        a = a + eta * s * (C2 - C1)[:, None]                            # Eq. 4: one-sided update, no accept / reject
        if k % LOG_EVERY == 0 or k == iterations:
            log.append((2 * k, scores(corr(), phi, screens, T)))
    return corr(), log


@torch.no_grad()
def response_matrix(phi, screens, T, eps=0.3, ridge=0.05, steps=6, recalibrate_at=3):
    """Calibrated response matrix with two cameras (field measured exactly). Returns correction and log."""
    n, Bn = len(basis), len(phi)
    tgt = T / T.flatten(1).norm(dim=1)[:, None, None]
    mask = T.abs() > 1e-2 * T.abs().amax(dim=(-2, -1), keepdim=True)    # pixels used in the fit
    frames = 0

    def measure(corr):                                                  # 2 frames -> field, global phase referenced to the target
        nonlocal frames
        frames += 2
        E = opt.output(phi + corr, screens)
        ov = (tgt.conj() * E).sum(dim=(-2, -1))
        return E * (ov.conj() / (ov.abs() + 1e-30))[:, None, None]

    def calibrate(corr):                                                # J[b, mode, pixel] = dE / da_mode by +-eps pokes
        J = torch.stack([((measure(corr + eps * Zj) - measure(corr - eps * Zj)) / (2 * eps) * mask).flatten(1) for Zj in basis], 1)
        JtJ = J.conj() @ J.transpose(1, 2)
        reg = ridge * torch.diagonal(JtJ, dim1=1, dim2=2).real.sum(1) / n
        return torch.linalg.solve(JtJ + reg[:, None, None] * torch.eye(n, device=DEV), J.conj())

    corr = torch.zeros_like(phi)
    log = [(0, scores(corr, phi, screens, T))]
    best_res, best = torch.full((Bn,), np.inf, device=DEV), corr.clone()
    solve = calibrate(corr)
    for i in range(steps + 1):
        if i == recalibrate_at:
            solve = calibrate(corr)                                     # J is a linearisation: once more around the new pattern
        E = measure(corr)
        P = (E.abs() ** 2).sum(dim=(-2, -1))
        r = ((tgt * P.sqrt()[:, None, None] - E) * mask).flatten(1)     # residual against the target at the measured power
        res = (r.abs() ** 2).sum(1) / P                                 # what the lab can see
        better = res < best_res
        best_res, best = torch.where(better, res, best_res), torch.where(better[:, None, None], corr, best)
        log.append((frames, scores(corr, phi, screens, T)))
        if i < steps:                                                   # phase modes: real coefficients; damped after the first step
            a = (1.0 if i == 0 else 0.5) * (solve @ r[:, :, None])[:, :, 0].real
            corr = corr + torch.einsum("bn,nxy->bxy", a, basis)
    log.append((frames, scores(best, phi, screens, T)))                 # shown at the end: the best pattern by the measured residual
    return best, log


def load_net(name):
    ck = torch.load(os.path.join(ns.OUT, name, "model.pt"), map_location=DEV)
    net = nt.build("resnet", 13, ck["args"]["width"]); net.load_state_dict(ck["state"]); net.eval()
    return net, ck["args"].get("one_camera", False)


# ---- pairs (disk state, shape): Gaussian, vortex, shapes of the test families, never-seen shapes -------
val_s = np.where(states["index"] % 20 == 0)[0]
rng = np.random.default_rng(11)
n_named, n_test, n_unseen = (4, 4, 4) if args.quick else (32, 36, 16)
fam_t = bank["test_family"]
test_i = np.concatenate([np.where(fam_t == f)[0][:max(1, n_test // 9)] for f in sorted(set(fam_t))])[:n_test]
todo = [("named", rng.choice(val_s, n_named), np.arange(n_named) % 2),
        ("test", rng.choice(val_s, len(test_i)), test_i),
        ("unseen", rng.choice(val_s, n_unseen), np.arange(n_unseen))]
group = lambda name, ti: {"named": ("Gaussian", "LG01 vortex")[ti % 2] if name == "named" else None, "test": "shapes", "unseen": "shapes never seen"}[name]

# ---- SPGD settings: sweep on a few pairs, the best mean one-camera score wins ---------------------------
sub = [(name, si[:4], ti[:4]) for name, si, ti in todo]
_, phi, screens, T = gather(sub)
best = None
for metric in ("paper", "pearson"):
    for delta in DELTAS:
        for eta in ETAS:
            _, log = spgd(phi, screens, T, delta, eta, SWEEP_ITER, metric)
            I, D = log[-1][1]["I"].mean().item(), log[-1][1]["D"].mean().item()
            print(f"sweep {metric:7s} delta {delta:.2f} eta {eta:4.0f}: I {log[0][1]['I'].mean().item():.3f} -> {I:.3f}, D {log[0][1]['D'].mean().item():.3f} -> {D:.3f}  ({time.time() - t0:.0f} s)", flush=True)
            if best is None or I > best[0]:
                best = (I, metric, delta, eta)
_, METRIC, DELTA, ETA = best
print(f"SPGD settings: signal '{METRIC}', delta {DELTA}, eta {ETA}", flush=True)

# ---- all methods on all pairs -------------------------------------------------------------------------
nets = {n: load_net(n) for n in (("shapes_relay",) if args.relay else ("shapes_onecam", "shapes_free")) if os.path.exists(os.path.join(ns.OUT, n, "model.pt"))}
rows, curves = [], {}
for name, si_all, ti_all in todo:
    for b in range(0, len(si_all), B):
        si, ti = si_all[b:b + B], ti_all[b:b + B]
        x, phi, screens, T = gather([(name, si, ti)])
        res = {}
        with torch.no_grad():
            res["uncorrected"] = scores(torch.zeros_like(phi), phi, screens, T)
            for n, (net, one) in nets.items():
                xi = x.clone()
                if one:
                    xi[:, [0, 9]] = 0                                   # far-field camera only
                res[n] = scores(net(xi), phi, screens, T)
        for k, f in SCORE.items():                                      # ceiling for each score
            out = ns.ceiling(opt, phi, screens, T, torch.zeros_like(phi), 30 if args.quick else 100, f)
            with torch.no_grad():
                res[f"ceiling_{k}"] = {kk: ff(out, T) for kk, ff in SCORE.items()}
        _, log_s = spgd(phi, screens, T, DELTA, ETA, ITER, METRIC)
        _, log_r = response_matrix(phi, screens, T)
        res["spgd_200"] = min(log_s, key=lambda l: abs(l[0] - 200))[1]
        res["spgd_end"], res["rm"] = log_s[-1][1], log_r[-1][1]
        for j in range(len(si)):
            g = group(name, int(ti[j]))
            rows.append(dict(set=name, group=g, family=str(bank[f"{name}_family"][ti[j]]), state=int(si[j]), shape=int(ti[j]),
                             **{f"{k}_{m}": v[k][j].item() for m, v in res.items() for k in SCORE}))
            for tag, log in (("spgd", log_s), ("rm", log_r)):           # curves: frames, I, D per pair
                curves.setdefault((g, tag), []).append([[fr, s["I"][j].item(), s["D"][j].item()] for fr, s in log])
        print(f"{name} {b + len(si)}/{len(si_all)}  ({(time.time() - t0) / 60:.0f} min)", flush=True)

with open(os.path.join(ns.OUT, f"iter_{LAYOUT}.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
np.savez(os.path.join(ns.OUT, f"iter_{LAYOUT}.npz"), **{f"{g}|{tag}": np.mean(v, 0) for (g, tag), v in curves.items()})   # mean over pairs: frames, I, D

# ---- tables ---------------------------------------------------------------------------------------------
groups = ["Gaussian", "LG01 vortex", "shapes", "shapes never seen"]
mean = lambda g, col: np.mean([r[col] for r in rows if r["group"] == g])
fr_s, fr_r = 2 * ITER, int(np.mean(curves[(groups[0], "rm")], 0)[-1, 0])
lines = [f"# Networks against the best iterative loops, like for like (Step 24, layout: {'SLM imaged onto the disk' if args.relay else 'SLM 2.9 m before the disk'})", "",
         f"Held-out disk states (240-360 W), {len(rows)} pairs of disk state and shape, means. SPGD: Xie et al. 2015, far-field camera, 14 Zernike terms, "
         f"signal '{METRIC}', delta {DELTA}, eta {ETA} (best of a sweep). Response matrix: 28 Zernike modes, field measured exactly, {fr_r} frames.", ""]
for k, what, its in (("I", "One camera: intensity similarity x power kept", ("spgd_200", "spgd_end")),
                     ("D", "Two cameras: field fidelity x power kept", ("spgd_200", "spgd_end", "rm"))):
    label = {"spgd_200": "SPGD, 200 frames", "spgd_end": f"SPGD, {fr_s} frames", "rm": f"response matrix, {fr_r} frames"}
    cols = ["uncorrected"] + list(its) + [n for n in nets if (k == "I") == nets[n][1] or args.relay] + [f"ceiling_{k}"]
    lines += [f"## {what}", "", "| target | n | " + " | ".join(label.get(c, c.replace(f"ceiling_{k}", "ceiling (disk known)").replace("shapes_", "network ")) for c in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
    for g in groups:
        lines.append(f"| {g} | {sum(r['group'] == g for r in rows)} | " + " | ".join(f"{mean(g, f'{k}_{c}'):.2f}" for c in cols) + " |")
    lines.append("")
open(os.path.join(ns.OUT, f"ITER_{LAYOUT}.md"), "w").write("\n".join(lines))
print("\n".join(lines), flush=True)
print(f"saved {ns.OUT} ({(time.time() - t0) / 60:.0f} min)", flush=True)
