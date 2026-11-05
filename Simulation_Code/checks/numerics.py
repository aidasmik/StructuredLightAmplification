"""Are the stripes / fine structure in the corrections physics or a numerical artefact?
The angular-spectrum propagation is periodic (FFT) without absorbing boundaries: light that
leaves the 12.8 mm window re-enters from the other side. Test: evaluate the network correction,
the labels and the gradient optimum through (a) the standard periodic model, (b) the same chain
with an absorbing window before every propagation leg, (c) the real LCOS plant; and with the
dominant tilt removed / the pattern low-passed. Plus a direct wrap-around demonstration."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the simulation modules
import os, sys, numpy as np, torch, torch.nn.functional as Fn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.environ.setdefault("NN_RANGE", "wide")
import nn_dataset as nd, nn_train as nt, nn_eval as ne, correction as co, thin_disk_system as tds

G = nd.G; pad = (G.n - nd.CROP_LABEL) // 2; k0 = 2 * np.pi / tds.LAM
sysm = nd.System(); phys = nt.Physics(np.load(os.path.join(nt.DATA, "meta.npz"))); ref = nt.load_reference()
net, a = ne.load_model(sys.argv[1] if len(sys.argv) > 1 else "resnet_all")
R = torch.as_tensor(G.R, device=nt.DEV, dtype=torch.float32)
W = torch.exp(-(R / 5.5e-3) ** 16).to(torch.complex64)                    # absorbing window at the grid edge

# ---- 1. direct wrap-around demonstration: a tilted Gaussian over one 1 m leg -------------------
print("wrap-around test: Gaussian seed tilted by theta, propagated 1 m (periodic FFT model)")
fin = co.field_at_slm(G).field
for th in (2e-3, 4e-3, 6e-3, 8e-3, 12e-3):
    f = co.Prop(1.0, k0, G.dx).apply(fin * np.exp(1j * k0 * th * G.X))
    I = np.abs(f) ** 2; cx = (I * G.X).sum() / I.sum()
    inside = I[G.R < 2.5e-3].sum() / I.sum()
    print(f"  theta {th*1e3:4.1f} mrad: expected centre {th*1e3:+.1f} mm, simulated centroid {cx*1e3:+.1f} mm, power inside r<2.5 mm {inside:.2f}")

# ---- 2. the corrections under the three models -----------------------------------------------
def forward_ops(ops, f):
    for op in ops:
        f = phys.amp._apply(op, f)
    return f

def fid(corr, t_id, absorb):
    phi = phys.base[t_id] + Fn.pad(corr, (pad,) * 4)
    field = phys.slm_gain * torch.roll((phys.fin * torch.polar(torch.ones_like(phi), phi)).flip(-1), 1, dims=-1)
    ops = phys.amp.ops
    if absorb:
        ops = [x for op in ops for x in ((("mul", W), op) if op[0] == "prop" else (op,))]
    out = forward_ops(ops, field)
    t = phys.targets[t_id]
    return float((t.conj() * out).sum().abs() ** 2 / ((out.abs() ** 2).sum() * (t.abs() ** 2).sum())), out

def tilt_of(corr):
    """Dominant linear phase of the correction inside the beam: spectral peak of the weighted phasor."""
    n = corr.shape[0]
    spec = torch.fft.fftshift(torch.fft.fft2(torch.sqrt(phys.w) * torch.polar(torch.ones_like(corr), corr)))
    ky, kx = divmod(int(torch.argmax(spec.abs())), n)
    fx, fy = (kx - n // 2) / n, (ky - n // 2) / n                               # cycles per pixel
    yy, xx = torch.meshgrid(torch.arange(n, device=corr.device), torch.arange(n, device=corr.device), indexing="ij")
    return 2 * np.pi * (fx * xx + fy * yy), fx, fy

def lowpass(corr, sigma_px=3.0):
    z = torch.polar(torch.ones_like(corr), corr)[None, None]
    k = int(4 * sigma_px) | 1; ax = torch.arange(k, device=corr.device) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma_px ** 2)); g = g / g.sum(); ker = (g[:, None] * g[None, :])[None, None]
    return torch.atan2(Fn.conv2d(z.imag, ker, padding=k // 2), Fn.conv2d(z.real, ker, padding=k // 2))[0, 0]

rows, pics = [], {}
for i in range(3):
    rng = np.random.default_rng([i, 99]); p, st = sysm.draw_state(rng)
    T_pts = sysm.temperature_points(st["T_mean"], rng); P_meas = p["P"] * (1 + 0.01 * rng.standard_normal()); dop = sysm.doping_input(p)
    rec = sysm.record(st, p); stt = nt.state_tensors({k: np.asarray(v)[None] for k, v in rec.items()}, [0])
    for t_id, name in enumerate(nd.TARGETS):
        tgt_t = torch.as_tensor([t_id], device=nt.DEV)
        near, far = sysm.frames(sysm.output(sysm.base[name], st, p), p)
        x = nt.make_input(near[None], far[None], T_pts[None], [P_meas], dop[None], [t_id], ref, a["inputs"], a["formulation"])
        with torch.no_grad():
            corr = nt.full_correction(net(x), tgt_t, ref, a["formulation"])[0]
        pat_gs, _ = sysm.optimum(name, st, p)
        lab = torch.as_tensor(nd.crop(np.angle(np.exp(1j * (pat_gs - sysm.base[name]))), nd.CROP_LABEL), device=nt.DEV, dtype=torch.float32)
        pat_ref = ne.refine_optimum(phys, pat_gs, rec, t_id, sysm.base[name], 100)
        refc = torch.as_tensor(nd.crop(np.angle(np.exp(1j * (pat_ref - sysm.base[name]))), nd.CROP_LABEL), device=nt.DEV, dtype=torch.float32)
        phys.fidelity(corr[None], stt, tgt_t)                                   # arms the amplifier with this state
        plane, fx, fy = tilt_of(corr)
        theta = tds.LAM * np.hypot(fx, fy) / G.dx
        wrap = lambda c: torch.remainder(c + np.pi, 2 * np.pi) - np.pi
        variants = {"network": corr, "network, tilt removed": wrap(corr - plane), "network, low-pass 75 um": lowpass(corr),
                    "label (12 projections)": lab, "gradient optimum": refc, "gradient optimum, low-pass 75 um": lowpass(refc),
                    "reference correction only": ref["corr"][t_id]}
        print(f"\nstate {i} {name}: dominant tilt of the network correction {theta*1e3:.2f} mrad ({fx:+.3f}, {fy:+.3f} cycles/px, {theta*1e3:.1f} mm per metre)")
        print(f"  {'pattern':34s} periodic  absorbing  real LCOS")
        for k, c in variants.items():
            Fp, outp = fid(c, t_id, False); Fa, outa = fid(c, t_id, True)
            Fr = co.fidelity(sysm.output(sysm.base[name] + np.pad(c.cpu().numpy(), pad), st, p), sysm.targets[name])
            rows.append((i, name, k, Fp, Fa, Fr))
            print(f"  {k:34s} {Fp:8.3f}  {Fa:9.3f}  {Fr:9.3f}", flush=True)
            if i == 0 and name == "LG01" and k in ("network", "network, tilt removed", "gradient optimum"):
                pics[k] = (c.cpu().numpy(), outp[0].abs().cpu().numpy() ** 2 if outp.dim() == 3 else outp.abs().cpu().numpy() ** 2,
                           outa[0].abs().cpu().numpy() ** 2 if outa.dim() == 3 else outa.abs().cpu().numpy() ** 2)

fig, ax = plt.subplots(len(pics), 3, figsize=(10, 3.3 * len(pics)))
c0 = G.n // 2; w = 60
for r, (k, (c, Ip, Ia)) in enumerate(pics.items()):
    ax[r, 0].imshow(c, cmap="twilight", vmin=-np.pi, vmax=np.pi); ax[r, 0].set_title(k)
    ax[r, 1].imshow(Ip[c0 - w:c0 + w, c0 - w:c0 + w], cmap="inferno"); ax[r, 1].set_title("output, periodic model")
    ax[r, 2].imshow(Ia[c0 - w:c0 + w, c0 - w:c0 + w], cmap="inferno"); ax[r, 2].set_title("output, absorbing window")
    for a_ in ax[r]: a_.set_xticks([]); a_.set_yticks([])
fig.tight_layout(); out = os.path.join(ne.RESULTS, "numeric_check.png"); fig.savefig(out, dpi=100); print("saved", out)
