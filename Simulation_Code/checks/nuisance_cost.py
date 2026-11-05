"""How much each nuisance term costs on the nominal disk: uncorrected fidelity and the
12-projection optimum (what the SLM can still fix), Gaussian target, 3 random draws each."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the simulation modules
import os, numpy as np
os.environ["NN_RANGE"] = "real"
import nn_dataset as nd, correction as co
sysm = nd.System()
tgt = sysm.targets["gaussian"]
def run(label, modify):
    F0, Fo = [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        p = sysm.nominal_params(); modify(p, rng); st = sysm.state(p)
        F0.append(co.fidelity(sysm.output(sysm.base["gaussian"], st, p), tgt))
        Fo.append(sysm.optimum("gaussian", st, p)[1][-1])
    print(f"{label:38s} uncorrected {np.mean(F0):.3f}   12-projection optimum {np.mean(Fo):.3f}", flush=True)
run("nominal (gain on)", lambda p, r: None)
for s in (2e-6, 5e-6, 10e-6, 20e-6):
    run(f"mirror tilts {s*1e6:.0f} urad rms", lambda p, r, s=s: p.update(tilts=s * r.standard_normal((len(nd.MIRROR_NAMES), 2))))
run("seed aberrations 0.2 rad x5", lambda p, r: p["seed"].__setitem__(slice(0, 5), 0.2 * r.standard_normal(5)))
run("seed tilt 0.4 rad + shift 1.5 px", lambda p, r: p["seed"].__setitem__(slice(5, 9), [*(0.4 * r.standard_normal(2)), *np.round(1.5 * r.standard_normal(2))]))
run("SLM stroke 3 %", lambda p, r: p.update(stroke=0.03 * r.standard_normal()))
