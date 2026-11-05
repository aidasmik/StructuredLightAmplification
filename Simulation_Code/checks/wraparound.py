"""Where do the raster lines in the output images come from? Hypothesis: light scattered to the
grid's Nyquist angle (41 mrad at 12.5 um, 21 mrad at 25 um) wraps around the periodic FFT window
and interferes with the beam. Test: same pattern, same grid, with and without an absorbing window
before every propagation leg; measure the fine-structure power of the output intensity."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the simulation modules
import os, numpy as np, torch
from scipy.ndimage import zoom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.environ.setdefault("NN_RANGE", "wide")
import nn_dataset as nd, nn_train as nt, nn_eval as ne, correction as co

G = nd.G; pad = (G.n - nd.CROP_LABEL) // 2
sysm = nd.System(); ref = nt.load_reference(); net, a = ne.load_model("resnet_all"); fine = ne.FineGrid()

def forward(amp, field, absorb, r0):
    W = torch.exp(-(torch.as_tensor(amp.grid.R, device=amp.device, dtype=torch.float32) / r0) ** 16).to(torch.complex64)
    ops = [x for op in amp.ops for x in ((("mul", W), op) if op[0] == "prop" else (op,))] if absorb else amp.ops
    f = torch.as_tensor(field.astype(np.complex64), device=amp.device)
    for op in ops:
        f = amp._apply(op, f)
    return f.cpu().numpy()

def fine_power(I, dx, f_min=5e3):
    """Fraction of the intensity's spectral power above f_min cycles/m (the raster lines)."""
    S = np.abs(np.fft.fftshift(np.fft.fft2(I - I.mean()))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(I.shape[0], dx)); FX, FY = np.meshgrid(f, f)
    return S[np.hypot(FX, FY) > f_min].sum() / S.sum()

rng = np.random.default_rng([1, 99]); p, st = sysm.draw_state(rng)
T_pts = sysm.temperature_points(st["T_mean"], rng); P_meas = p["P"] * (1 + 0.01 * rng.standard_normal()); dop = sysm.doping_input(p)
panels = []
for t_id, name in enumerate(nd.TARGETS):
    near, far = sysm.frames(sysm.output(sysm.base[name], st, p), p)
    x = nt.make_input(near[None], far[None], T_pts[None], [P_meas], dop[None], [t_id], ref, a["inputs"], a["formulation"])
    with torch.no_grad():
        corr = nt.full_correction(net(x), torch.as_tensor([t_id], device=nt.DEV), ref, a["formulation"])[0].cpu().numpy()
    pat = sysm.base[name] + np.pad(corr, pad)
    # 25 um grid
    sysm.amp.set_thermal_opd(st["opd"]); fin = co.ideal_slm(sysm.fin, pat)
    # 12.5 um grid, phasor interpolation of the same pattern
    fine.amp.set_thermal_opd(zoom(st["opd"], 2, order=1))
    z = np.exp(1j * pat); phi = np.angle(zoom(z.real, 2, order=1) + 1j * zoom(z.imag, 2, order=1)); fin2 = co.ideal_slm(fine.fin, phi)
    for label, amp, f0, dx, tgt in (("25 um", sysm.amp, fin, G.dx, sysm.targets[name]), ("12.5 um", fine.amp, fin2, fine.G2.dx, fine.targets[name])):
        for absorb in (False, True):
            out = forward(amp, f0, absorb, 5.5e-3)
            n = out.shape[0]; w = int(1.5e-3 / dx); I = np.abs(out[n // 2 - w:n // 2 + w, n // 2 - w:n // 2 + w]) ** 2
            F = co.fidelity(out, tgt); fp = fine_power(I, dx)
            print(f"{name:9s} {label:8s} {'absorbing' if absorb else 'periodic ':9s}: fidelity {F:.3f}, power of structure finer than 200 um in the output: {fp*100:5.2f} %", flush=True)
            panels.append((f"{name}, {label} grid, {'absorbing window' if absorb else 'periodic (as used)'}\nF = {F:.3f}, fine-structure power {fp*100:.2f} %", I))

fig, ax = plt.subplots(2, 4, figsize=(17, 8.5))
for k, (t, I) in enumerate(panels):
    a_ = ax[k // 4, k % 4]; a_.imshow(I, cmap="inferno"); a_.set_title(t, fontsize=9); a_.set_xticks([]); a_.set_yticks([])
fig.suptitle("Network correction, state 1: output intensity (+-1.5 mm) with the periodic FFT window and with an absorbing window")
fig.tight_layout(); out = os.path.join(ne.RESULTS, "lines_check.png"); fig.savefig(out, dpi=100); print("saved", out)
