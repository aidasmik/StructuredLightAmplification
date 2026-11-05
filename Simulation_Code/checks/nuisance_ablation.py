"""Which nuisance effect does the realism network fail on? Fresh states as in nn_eval; the trained
network is run with single nuisance terms switched off (the images are then taken with that term
off too, as the lab would see them). Mean fidelity through the real LCOS model."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the simulation modules
import os, sys, numpy as np, torch
os.environ.setdefault("NN_RANGE", "real")
import nn_dataset as nd, nn_train as nt, nn_eval as ne, correction as co

sysm = nd.System(); ref = nt.load_reference(); net, a = ne.load_model(sys.argv[1] if len(sys.argv) > 1 else "resnet_all")
pad = (nd.G.n - nd.CROP_LABEL) // 2
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
M = len(nd.MIRROR_NAMES)
configs = {
    "all nuisances (as trained)": lambda p: p,
    "no camera misregistration": lambda p: dict(p, cam=np.zeros(13)),
    "no mirror tilts": lambda p: dict(p, tilts=np.zeros((M, 2))),
    "no seed aberration / pointing": lambda p: dict(p, seed=np.array([0.0] * 9 + [1.0])),
    "no SLM stroke error": lambda p: dict(p, stroke=0.0),
    "none (disk state + gain only)": lambda p: dict(p, cam=np.zeros(13), tilts=np.zeros((M, 2)), seed=np.array([0.0] * 9 + [1.0]), stroke=0.0),
}
F = {k: {t: [] for t in nd.TARGETS} for k in configs}; F0 = {k: {t: [] for t in nd.TARGETS} for k in configs}
for i in range(N):
    rng = np.random.default_rng([i, 99]); p0, st = sysm.draw_state(rng)
    T_pts = sysm.temperature_points(st["T_mean"], rng); P_meas = p0["P"] * (1 + 0.01 * rng.standard_normal()); dop = sysm.doping_input(p0)
    for k, cfg in configs.items():
        p = cfg(p0)
        for t_id, name in enumerate(nd.TARGETS):
            out0 = sysm.output(sysm.base[name], st, p)
            near, far = sysm.frames(out0, p)
            x = nt.make_input(near[None], far[None], T_pts[None], [P_meas], dop[None], [t_id], ref, a["inputs"], a["formulation"])
            with torch.no_grad():
                corr = nt.full_correction(net(x), torch.as_tensor([t_id], device=nt.DEV), ref, a["formulation"])[0].cpu().numpy()
            F0[k][name].append(co.fidelity(out0, sysm.targets[name]))
            F[k][name].append(co.fidelity(sysm.output(sysm.base[name] + np.pad(corr, pad), st, p), sysm.targets[name]))
    if i % 10 == 9:
        print(f"{i + 1}/{N}", flush=True)
print(f"\n{'configuration':34s} uncorrected G / LG01   network G / LG01")
for k in configs:
    print(f"{k:34s} {np.mean(F0[k]['gaussian']):.3f} / {np.mean(F0[k]['LG01']):.3f}        {np.mean(F[k]['gaussian']):.3f} / {np.mean(F[k]['LG01']):.3f}")
