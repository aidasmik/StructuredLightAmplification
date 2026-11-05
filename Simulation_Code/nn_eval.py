"""
nn_eval.py - Step 18: test the trained pre-correction network(s) on fresh disk
states in the exact simulated plant (real LCOS pixel model, 12-bit cameras).

For N_TEST new random states (different seeds from the training set, drawn with
the same tolerances incl. the LENS_TOL rejection) and both targets:
    uncorrected      base pattern on the SLM
    reference only   the correction recorded for the nominal disk, no network
    network          one shot: two camera images (+ thermal camera, pump meter)
                     plus five pyrometer readings, pump meter and the known doping map
                     -> correction -> displayed, nothing iterated
    (with NN_RANGE=real the fresh states carry the same nuisance layer as the training
    set: gain, seed aberrations, mirror tilts, SLM stroke error, camera misregistration)
    physics optimum  the ceiling: alternating projections (the label recipe) followed by
                     gradient ascent on the fidelity through the differentiable amplifier
                     with the true disk state (phase-only, through the same LCOS)
Scenarios:
    static           images and correction with the same disk
    temp_noise       pump power moves 1 % rms between the images and the correction
    temp_drift       pump power 2 % higher when the correction is displayed
    cam_noise        camera noise 3 % instead of 1 %
Reference (static scenario, thermal model after the bending fix): Step 16
response-matrix loop 0.80 / 0.59 after ~240 frames, SPGD 0.41 / 0.15 after
~390 frames, physics optimum 0.94 / 0.77.

Usage:  python nn_eval.py NAME [NAME2 ...] [--n-test 100] [--opt-steps 100]
Output: Results/<18_nn|19_nn_wide|20_nn_real>/eval_<names>.csv, NN_EVAL.md, nn_eval.png, nn_examples.png, nn_vs_power.png
"""
import os
import sys
import csv
import argparse
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

import thin_disk_system as tds
import correction as co
import slm
import limits_benchmark as lb
from seed_beam import Grid, results_dir
from gpu import TorchAmplifier
from spgd import Camera
from two_camera_correction import to_far
import nn_dataset as nd
import nn_train as nt

warnings.filterwarnings("ignore")
RESULTS = results_dir(nd.TRAIN_RESULTS)        # 18_nn (narrow), 19_nn_wide or 20_nn_real
SCENARIOS = {"static": dict(noise=0.0, drift=0.0, cam=0.01),
             "temp_noise": dict(noise=0.01, drift=0.0, cam=0.01),
             "temp_drift": dict(noise=0.0, drift=0.02, cam=0.01),
             "cam_noise": dict(noise=0.0, drift=0.0, cam=0.03)}
REFERENCE = {"response matrix (Step 16, ~240 frames)": (0.80, 0.59), "SPGD (slides loop, ~390 frames)": (0.41, 0.15)}   # static, fixed thermal model


def load_model(name):
    """model.pt -> (network, its training arguments: arch, formulation, inputs)."""
    ck = torch.load(os.path.join(RESULTS, name, "model.pt"), map_location=nt.DEV)
    a = ck["args"]
    net = nt.build(a["arch"], 12 if a.get("ref_images") else 10, a.get("width", 64)); net.load_state_dict(ck["state"]); net.eval()
    return net, ck["args"]


def refine_optimum(phys, pattern, rec, t_id, base, steps=100, lr=0.05):
    """Gradient ascent (Adam) on the output fidelity through the differentiable amplifier with the
    true state (rec: what the shards store of it), started from the alternating-projection pattern
    (which stalls short of the optimum: the trained network beats it). The pattern is parametrised
    as (cos, sin) at 50 um and upsampled x2 like the network head: pixel-level (25 um) structure
    is not resolved by the propagation grid (it looked 0.1-0.5 better on the 25 um grid than on the
    12.5 um grid) and would make the ceiling a numerical artefact. Returns the refined full pattern."""
    pad = (nd.G.n - nd.CROP_LABEL) // 2
    st = nt.state_tensors({k: np.asarray(v)[None] for k, v in rec.items()}, [0])
    z = np.exp(1j * nd.crop(pattern - base, nd.CROP_LABEL))
    z = z.reshape(nd.CROP_LABEL // 2, 2, nd.CROP_LABEL // 2, 2).mean(axis=(1, 3))                  # phasor to 50 um
    y = torch.as_tensor(np.stack([z.real, z.imag]), dtype=torch.float32, device=nt.DEV)[None].requires_grad_()
    opt = torch.optim.Adam([y], lr=lr)
    tgt = torch.as_tensor([t_id], device=nt.DEV)
    for _ in range(steps):
        F = phys.fidelity(nt.to_phase(y), st, tgt).sum()
        opt.zero_grad(); (-F).backward(); opt.step()
    return base + np.pad(nt.to_phase(y.detach())[0].cpu().numpy(), pad)


class FineGrid:
    """The same system on the 12.5 um design grid (ideal SLM) to score patterns free of the 25 um
    sampling: structure near the 25 um pixel scale is propagated wrongly there (it can look 0.1-0.5
    better than on the fine grid), so the static-scenario fidelities are reported on both grids."""
    def __init__(self):
        self.G2 = G2 = Grid(n=2 * nd.G.n, size=nd.G.size)
        th = lb.thermal(nd.P0); f_th, _ = lb.per_pass_lens(th.opd(G2), G2)
        tds.DISK.thermal_opd = None; tds.MIRROR_ROC.update(lb.COLD_ROC)
        ref2, _ = tds.trace(slm.flat(), G2)
        self.targets = co.target_modes(G2, ref2)
        tds.DISK.thermal_opd = th.opd(G2); tds.MIRROR_ROC.update(lb.retuned_mirrors(f_th))
        tds.DISK.gain = np.ones((G2.n, G2.n)) if nd.REALISM else None
        self.amp = TorchAmplifier(G2, device=nt.DEV); tds.DISK.gain = None
        self.fin = co.field_at_slm(G2).field
        self.base = {"gaussian": np.zeros((G2.n, G2.n)), "LG01": slm.vortex(1)(G2.X, G2.Y)}
        self.seed_basis = np.stack([slm.zernike({t: 1.0}, nd.R_SEED)(G2.X, G2.Y) for t in nd.SEED_TERMS] + [G2.X / nd.R_SEED, G2.Y / nd.R_SEED])

    def fidelity(self, pattern, st, p, name):
        """Fidelity of the full 25 um pattern (phasor-interpolated to 12.5 um) with the state (st, p)."""
        up = lambda a: zoom(a, 2, order=1)
        self.amp.set_thermal_opd(up(st["opd"]))
        fin = self.fin
        if nd.REALISM:
            self.amp.set_gain(up(st["gain"])); self.amp.set_mirror_tilts(dict(zip(nd.MIRROR_NAMES, p["tilts"])))
            s = p["seed"]
            fin = np.roll(fin, (2 * int(s[8]), 2 * int(s[7])), axis=(0, 1)) * np.exp(1j * np.tensordot(s[:7], self.seed_basis, 1)) * np.sqrt(s[9])
        z = np.exp(1j * pattern)
        phi = np.angle(up(z.real) + 1j * up(z.imag))
        if nd.REALISM:
            phi = np.mod(phi + np.pi, 2 * np.pi) * (1 + p["stroke"])
        return co.fidelity(self.amp.forward(co.ideal_slm(fin, phi)), self.targets[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--opt-steps", type=int, default=100, help="gradient steps refining the physics optimum")
    args = ap.parse_args()
    nets = {n: load_model(n) for n in args.names}
    ref = nt.load_reference()
    sysm = nd.System()
    phys = nt.Physics(np.load(os.path.join(nt.DATA, "meta.npz")))
    fine = FineGrid()
    pad = (nd.G.n - nd.CROP_LABEL) // 2
    rows, examples = [], {}
    for i in range(args.n_test):
        rng = np.random.default_rng([i, 99])                                 # never used for training
        setup = i % nd.N_SETUPS if nd.N_SETUPS else 0                         # known setups: their references were recorded
        p, st = sysm.draw_state(rng, setup=setup if nd.N_SETUPS else None)   # same tolerances and nuisance layer as the training set
        T_pts = sysm.temperature_points(st["T_mean"], rng); P_meas = p["P"] * (1 + 0.01 * rng.standard_normal())
        dop_map = sysm.doping_input(p)
        for t_id, name in enumerate(nd.TARGETS):
            tgt = sysm.targets[name]
            pat_gs, _ = sysm.optimum(name, st, p)                            # label recipe (12 alternating projections)
            pat_opt = refine_optimum(phys, pat_gs, sysm.record(st, p), t_id, sysm.base[name], args.opt_steps) if args.opt_steps else pat_gs
            if i < 3:
                print(f"state {i} {name}: optimum {co.fidelity(sysm.output(pat_gs, st, p), tgt):.3f} (projections) -> "
                      f"{co.fidelity(sysm.output(pat_opt, st, p), tgt):.3f} (+{args.opt_steps} gradient steps)", flush=True)
            for sc, s in SCENARIOS.items():
                # the disk when the correction is displayed
                p2 = dict(p, P=p["P"] * (1 + s["drift"] + s["noise"] * rng.standard_normal()))
                st2 = st if sc == "static" else sysm.state(p2)
                out0 = sysm.output(sysm.base[name], st, p)                   # images taken now
                sysm.near_cam.noise = sysm.far_cam.noise = s["cam"]
                near, far = sysm.frames(out0, p)
                row = dict(index=i, target=name, scenario=sc, P=p["P"], lens_change=p["lens_change"], F0=co.fidelity(sysm.output(sysm.base[name], st2, p2), tgt),
                           Fref=co.fidelity(sysm.output(sysm.base[name] + np.pad(ref["corr"][setup, t_id].cpu().numpy(), pad), st2, p2), tgt),
                           Fopt=co.fidelity(sysm.output(pat_opt, st2, p2), tgt))
                if sc == "static":
                    row["Fopt_fine"] = fine.fidelity(pat_opt, st, p, name)
                for mname, (net, a) in nets.items():
                    x = nt.make_input(near[None], far[None], T_pts[None], [P_meas], dop_map[None], [t_id], ref, a["inputs"], a["formulation"], [setup], a.get("ref_images", False))
                    with torch.no_grad():
                        tgt_t = torch.as_tensor([t_id], device=nt.DEV)
                        corr = nt.full_correction(net(x), tgt_t, ref, a["formulation"], [setup])[0].cpu().numpy()
                    pattern = sysm.base[name] + np.pad(corr, pad)
                    out = sysm.output(pattern, st2, p2)
                    row[f"F_{mname}"] = co.fidelity(out, tgt)
                    if sc == "static":
                        row[f"F_{mname}_fine"] = fine.fidelity(pattern, st, p, name)
                    if sc == "static" and i < 2:
                        examples[(i, name, mname)] = dict(near=near, far=far, corr=corr, out0=out0, out=out, label=np.angle(np.exp(1j * (pat_opt - sysm.base[name]))))
                rows.append(row)
        if i % 10 == 9:
            st = [r for r in rows if r["scenario"] == "static"]
            print(f"{i + 1}/{args.n_test}: static mean " + "  ".join(
                f"{t}: F0 {np.mean([r['F0'] for r in st if r['target'] == t]):.3f} "
                + " ".join(f"{m} {np.mean([r[f'F_{m}'] for r in st if r['target'] == t]):.3f}" for m in nets)
                + f" opt {np.mean([r['Fopt'] for r in st if r['target'] == t]):.3f}" for t in nd.TARGETS), flush=True)

    tag = "_".join(args.names)
    with open(os.path.join(RESULTS, f"eval_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    # ---- table ---------------------------------------------------------------
    methods = ["F0", "Fref"] + [f"F_{m}" for m in nets] + ["Fopt"]
    labels = {"F0": "uncorrected", "Fref": "reference correction only (no network)", "Fopt": "physics optimum (projections + gradient ascent, phase-only, LCOS)",
              **{f"F_{m}": f"network '{m}' ({a['arch']}, {a['formulation']}, {a['inputs']}; 1 shot)" for m, (_, a) in nets.items()}}
    lines = ["# Neural-network pre-correction (Step 18)", "",
             f"Fresh random disk states ({args.n_test}), true field fidelity through the real LCOS model, mean (10th percentile). Gaussian / LG01.", "",
             "| method | " + " | ".join(SCENARIOS) + " |", "|---|" + "---|" * len(SCENARIOS)]
    for m in methods:
        cells = []
        for sc in SCENARIOS:
            v = []
            for t in nd.TARGETS:
                f = np.array([r[m] for r in rows if r["scenario"] == sc and r["target"] == t])
                v.append(f"{f.mean():.2f} ({np.percentile(f, 10):.2f})")
            cells.append(" / ".join(v))
        lines.append(f"| {labels[m]} | " + " | ".join(cells) + " |")
    for m in [f"F_{m}" for m in nets] + ["Fopt"]:                      # the static scenario scored on the 12.5 um grid
        v = []
        for t in nd.TARGETS:
            f = np.array([r[f"{m}_fine"] for r in rows if r["scenario"] == "static" and r["target"] == t])
            v.append(f"{f.mean():.2f} ({np.percentile(f, 10):.2f})")
        lines.append(f"| {labels[m]}, scored on the 12.5 um grid | " + " / ".join(v) + " | | | |")
    for k, v in REFERENCE.items():
        lines.append(f"| {k} | {v[0]:.2f} / {v[1]:.2f} | | | |")
    with open(os.path.join(RESULTS, "NN_EVAL.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figures --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(len(SCENARIOS))
    for j, t in enumerate(nd.TARGETS):
        for k, m in enumerate(methods):
            means = [np.mean([r[m] for r in rows if r["scenario"] == sc and r["target"] == t]) for sc in SCENARIOS]
            ax[j].bar(xs + (k - len(methods) / 2 + 0.5) * 0.8 / len(methods), means, 0.8 / len(methods), label=labels[m])
        for k, (rn, rv) in enumerate(REFERENCE.items()):
            ax[j].axhline(rv[j], color="k", ls=("--", ":")[k], lw=1, label=rn)
        ax[j].set_xticks(xs); ax[j].set_xticklabels(SCENARIOS); ax[j].set_ylim(0, 1); ax[j].set_ylabel("field fidelity"); ax[j].set_title(t); ax[j].grid(alpha=0.3, axis="y")
    ax[0].legend(fontsize=7)
    fig.suptitle("One-shot network correction vs iterative loops on fresh disk states")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "nn_eval.png"), dpi=110)

    # fidelity against pump power (static scenario): does the correction hold over the operating range?
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    P_all = np.array([r["P"] for r in rows if r["scenario"] == "static"])
    edges = np.linspace(P_all.min(), P_all.max() + 1e-9, 7)
    for j, t in enumerate(nd.TARGETS):
        st = [r for r in rows if r["scenario"] == "static" and r["target"] == t]
        P = np.array([r["P"] for r in st]); mid = 0.5 * (edges[1:] + edges[:-1])
        for m in methods:
            f = np.array([r[m] for r in st])
            ax[j].plot(mid, [f[(P >= a) & (P < b)].mean() if ((P >= a) & (P < b)).any() else np.nan for a, b in zip(edges[:-1], edges[1:])], "o-", label=labels[m])
        ax[j].set_xlabel("pump power [W]"); ax[j].set_ylabel("field fidelity"); ax[j].set_ylim(0, 1); ax[j].set_title(t); ax[j].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("Fidelity over the pump range (relay tuned to 300 W), bin means of the fresh states")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "nn_vs_power.png"), dpi=110)

    m0 = args.names[0]
    keys = [k for k in examples if k[2] == m0]
    fig, ax = plt.subplots(len(keys), 6, figsize=(19, 3.3 * len(keys)))
    for r, k in enumerate(keys):
        e = examples[k]
        ax[r, 0].imshow(e["near"], cmap="hot"); ax[r, 0].set_title(f"near-field camera ({k[1]})")
        ax[r, 1].imshow(e["far"], cmap="hot"); ax[r, 1].set_title("far-field camera")
        ax[r, 2].imshow(e["label"][pad:-pad, pad:-pad], cmap="twilight", vmin=-np.pi, vmax=np.pi); ax[r, 2].set_title("physics optimum (projections + gradient)")
        ax[r, 3].imshow(e["corr"], cmap="twilight", vmin=-np.pi, vmax=np.pi); ax[r, 3].set_title("network correction")
        c = nd.G.n // 2; w = 60
        ax[r, 4].imshow(np.abs(e["out0"][c - w:c + w, c - w:c + w]) ** 2, cmap="inferno"); ax[r, 4].set_title("output, uncorrected")
        ax[r, 5].imshow(np.abs(e["out"][c - w:c + w, c - w:c + w]) ** 2, cmap="inferno"); ax[r, 5].set_title("output, network-corrected")
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "nn_examples.png"), dpi=100)
    print("saved", RESULTS)


if __name__ == "__main__":
    main()
