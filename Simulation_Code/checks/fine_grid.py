"""Is the fine structure of the corrections a 25 um grid artefact? Re-evaluate the same
patterns (network, label, gradient optimum) on the 12.5 um design grid (1024 x 12.8 mm)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the simulation modules
import os, sys, numpy as np, torch
from scipy.ndimage import zoom
os.environ.setdefault("NN_RANGE", "wide")
import nn_dataset as nd, nn_train as nt, nn_eval as ne, correction as co, thin_disk_system as tds, slm, limits_benchmark as lb
from seed_beam import Grid
from gpu import TorchAmplifier

G = nd.G; pad = (G.n - nd.CROP_LABEL) // 2
sysm = nd.System(); phys = nt.Physics(np.load(os.path.join(nt.DATA, "meta.npz"))); ref = nt.load_reference()
net, a = ne.load_model(sys.argv[1] if len(sys.argv) > 1 else "resnet_all")

# the same system on the fine grid
G2 = Grid(n=1024, size=12.8e-3)
th = lb.thermal(nd.P0); f_th, _ = lb.per_pass_lens(th.opd(G2), G2)
tds.DISK.thermal_opd = None; tds.MIRROR_ROC.update(lb.COLD_ROC)
ref2, _ = tds.trace(slm.flat(), G2); targets2 = co.target_modes(G2, ref2)
tds.DISK.thermal_opd = th.opd(G2); tds.MIRROR_ROC.update(lb.retuned_mirrors(f_th))
amp2 = TorchAmplifier(G2); fin2 = co.field_at_slm(G2).field
base2 = {"gaussian": np.zeros((G2.n, G2.n)), "LG01": slm.vortex(1)(G2.X, G2.Y)}
print(f"fine grid: dx {G2.dx*1e6:.1f} um; nominal lens {f_th:.3f} m vs {sysm.f0:.3f} m on the 25 um grid")

def up(phase):                                   # wrapped phase map on G -> G2 (through the phasor)
    z = np.exp(1j * phase)
    return np.angle(zoom(z.real, 2, order=1) + 1j * zoom(z.imag, 2, order=1))

def fid_fine(pattern_coarse, opd_coarse, name):
    amp2.set_thermal_opd(zoom(opd_coarse, 2, order=1))
    out = amp2.forward(co.ideal_slm(fin2, up(pattern_coarse)))
    return co.fidelity(out, targets2[name])

def fid_coarse(pattern, opd, name):
    sysm.amp.set_thermal_opd(opd)
    return co.fidelity(sysm.amp.forward(co.ideal_slm(sysm.fin, pattern)), sysm.targets[name])

print(f"{'state / pattern':44s} 25 um grid  12.5 um grid")
for i in range(3):
    rng = np.random.default_rng([i, 99]); p, st = sysm.draw_state(rng)
    T_pts = sysm.temperature_points(st["T_mean"], rng); P_meas = p["P"] * (1 + 0.01 * rng.standard_normal()); dop = sysm.doping_input(p)
    rec = sysm.record(st, p)
    for t_id, name in enumerate(nd.TARGETS):
        tgt_t = torch.as_tensor([t_id], device=nt.DEV)
        near, far = sysm.frames(sysm.output(sysm.base[name], st, p), p)
        x = nt.make_input(near[None], far[None], T_pts[None], [P_meas], dop[None], [t_id], ref, a["inputs"], a["formulation"])
        with torch.no_grad():
            corr = nt.full_correction(net(x), tgt_t, ref, a["formulation"])[0].cpu().numpy()
        pat_gs, _ = sysm.optimum(name, st, p)
        pat_ref = ne.refine_optimum(phys, pat_gs, rec, t_id, sysm.base[name], 100)
        for k, pat in (("uncorrected", sysm.base[name]), ("network", sysm.base[name] + np.pad(corr, pad)),
                       ("label (12 projections)", pat_gs), ("gradient optimum", pat_ref)):
            print(f"  {i} {name:9s} {k:30s} {fid_coarse(pat, st['opd'], name):8.3f}    {fid_fine(pat, st['opd'], name):8.3f}", flush=True)
