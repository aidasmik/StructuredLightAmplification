"""Step 21 network: how much of the gap to the ceiling is the shot-to-shot drift of the nuisances?
Same fresh states, once with drift (as trained) and once with the setup's fixed nuisances only."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the simulation modules
import os, numpy as np, torch
os.environ["NN_RANGE"] = "setup"
import nn_dataset as nd, nn_train as nt, nn_eval as ne, correction as co
sysm = nd.System(); ref = nt.load_reference(); net, a = ne.load_model("resnet_ft")
pad = (nd.G.n - nd.CROP_LABEL) // 2
F = {k: {t: [] for t in nd.TARGETS} for k in ("with drift", "no drift")}
for i in range(40):
    s = i % nd.N_SETUPS
    rng = np.random.default_rng([i, 99]); p0, st = sysm.draw_state(rng, setup=s)
    T_pts = sysm.temperature_points(st["T_mean"], rng); P_meas = p0["P"] * (1 + 0.01 * rng.standard_normal()); dop = sysm.doping_input(p0)
    for k, p in (("with drift", p0), ("no drift", dict(p0, **sysm.setup_nuisance(s)))):
        for t_id, name in enumerate(nd.TARGETS):
            near, far = sysm.frames(sysm.output(sysm.base[name], st, p), p)
            x = nt.make_input(near[None], far[None], T_pts[None], [P_meas], dop[None], [t_id], ref, a["inputs"], a["formulation"], [s])
            with torch.no_grad():
                corr = nt.full_correction(net(x), torch.as_tensor([t_id], device=nt.DEV), ref, a["formulation"], [s])[0].cpu().numpy()
            F[k][name].append(co.fidelity(sysm.output(sysm.base[name] + np.pad(corr, pad), st, p), sysm.targets[name]))
for k in F:
    print(f"{k:12s} network {np.mean(F[k]['gaussian']):.3f} / {np.mean(F[k]['LG01']):.3f}")
