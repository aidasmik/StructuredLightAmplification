"""
limits_benchmark.py - Step 13: operating limits of SLM pre-correction for the
system of the slides (10-pass -I relay, 1.5 mm seed, Hamamatsu X15213 SLM,
0.5 mm / 10 % Yb:YAG disk).

Three sweeps, each run as its own process (python limits_benchmark.py <part>):

    power_cold : pump power 0..400 W with the relay mirrors as designed for
                 the cold disk (what the lab has now)
    power_hot  : the same, with the relay mirrors re-tuned to the hot disk so
                 that one unit  disk -> A -> M -> A' -> disk  is again -I
                 (3 unknowns: radius of the even array mirrors, of the odd
                 ones and of M, solved from A = -1, B = 0, C = -1/f_thermal)
    pump_spot  : at 300 W with the re-tuned relay: pump spot radius and edge
                 sharpness vs the 1.7 mm beam (the OPD kink at the pump edge)
    residual   : at 300 W with the re-tuned relay: hot spots of growing rms,
                 static shape error of growing size, 3-D doping clusters of
                 +-2 / +-4 %, and the 50 Hz pump modulation of the slides

For every state and for the targets Gaussian and LG01 (vortex) it reports
    F_naive   : cold-optimal SLM pattern applied to the distorted system
    w_req     : radius of the input field the SLM would have to deliver
    P_in      : fraction of that field's power inside the seed envelope
    F_cplx_ub : best overlap a complex (phase + amplitude) SLM can reach
                (upper bound, any efficiency) and the same at >= 30 % efficiency
and, for the marked states, the full optimisation
    F_phase   : physics-optimal phase-only pattern on the real LCOS
    F_davis   : complex encoding (Davis) on the real LCOS, with efficiency
Fidelity F = |<E, E_target>|^2 / (<E,E> <E_t,E_t>).

python limits_benchmark.py plot   collects the CSV files into figures and LIMITS.md.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.ndimage import zoom

from seed_beam import Grid, results_dir
import thin_disk_system as tds
import slm
import correction as co
from correction_benchmark import hot_spot_field
from thermal import DiskThermal, fit_lens
from doping import DopedDiskThermal, doping_map_3d
from optics import _flip_x
from gpu import make_amplifier

warnings.filterwarnings("ignore")

G = Grid(**tds.FINE_GRID)
ITERS = 12
TARGETS = ("gaussian", "LG01")
P_LIST = (0, 50, 100, 150, 200, 250, 300, 350, 400)
P_FULL = (100, 200, 300, 400)
PUMP_RADIUS = float(os.environ.get("PUMP_RADIUS", 2.5e-3))     # pump spot radius [m]: the whole 5 mm disk; env var overrides
PUMP_ORDER = int(os.environ.get("PUMP_ORDER", 16))              # super-Gaussian order: 16 = flat top (300 W evenly over the disk)
RESULTS = results_dir(f"13_limits/pump_{PUMP_RADIUS*1e3:g}mm" if PUMP_RADIUS != 2.0e-3 else "13_limits")
COLD_ROC = dict(tds.MIRROR_ROC)


# --------------------------------------------------------------------------
# Disk states
# --------------------------------------------------------------------------
def thermal(P):
    """Steady thermal state of the uniform 10 % disk at pump power P."""
    return DiskThermal(pump_power=P, doping=0.10, pump_radius=PUMP_RADIUS, pump_order=PUMP_ORDER)


def per_pass_lens(opd, grid=None):
    """Focal length of the disk per reflection and the non-parabolic residual [waves rms]."""
    if opd is None:
        return np.inf, 0.0
    f, res = fit_lens(opd, grid or G, 1.0e-3, tds.LAM)
    return f, res


def shape_error(scale, grid=None):
    """Static surface error (30/20/15 nm astigmatism/trefoil/coma at r = 2 mm) x scale."""
    g = grid or G
    z = slm.zernike({"astig0": 30e-9 * scale, "trefoil30": 20e-9 * scale, "coma_x": 15e-9 * scale}, 2e-3)
    return 2 * z(g.X, g.Y)                               # OPD in reflection


def doping_opd(P, amplitude, seed=1):
    """OPD change caused by 3-D doping clusters (+-amplitude) on top of the
    uniform disk, computed on a 512 grid and interpolated (it is smooth)."""
    g = Grid(n=512, size=G.size)
    th = DopedDiskThermal(pump_power=P, doping=0.10, n_pump_passes=24, pump_radius=PUMP_RADIUS, pump_order=PUMP_ORDER)
    c3, _, z = doping_map_3d(g, th.thickness, nz=50, nominal=0.10, amplitude=amplitude, cluster_size=0.3e-3, seed=seed)
    clustered = th.solve_3d(g, c3, z)["opd"]
    uniform = DopedDiskThermal(pump_power=P, doping=0.10, doping_field=np.full((g.n, g.n), 0.10),
                               n_pump_passes=24, pump_radius=PUMP_RADIUS, pump_order=PUMP_ORDER).solve(g)["opd"]
    return zoom(clustered - uniform, G.n / g.n, order=1)


def modulated_states(P, frequency=50.0, duty=0.4):
    """Thermal OPD at the mean, hottest and coldest instant of the periodic
    steady state under a square-wave pump (superposition of step responses)."""
    th = thermal(P)
    T = 1 / frequency
    dt = 2e-4
    t = np.arange(0, 12 * T, dt)
    g_mean, g_diff = th.transient(t)
    g_mean, g_diff = np.asarray(g_mean), np.asarray(g_diff)

    def step(g, shift):                                   # g(t - shift), zero before the shift
        k = int(round(shift / dt))
        return np.concatenate([np.zeros(k), g[:len(g) - k]]) if k < len(g) else np.zeros_like(g)

    G_m = sum(step(g_mean, n * T) - step(g_mean, n * T + duty * T) for n in range(12))
    G_d = sum(step(g_diff, n * T) - step(g_diff, n * T + duty * T) for n in range(12))
    last = slice(len(t) - int(T / dt), len(t))             # one full period of the periodic state
    i_hot = last.start + np.argmax(G_m[last]); i_cold = last.start + np.argmin(G_m[last])
    states = {"mean": th.solve_at(G, G_m[last].mean(), G_d[last].mean())["opd"],
              "hottest": th.solve_at(G, G_m[i_hot], G_d[i_hot])["opd"],
              "coldest": th.solve_at(G, G_m[i_cold], G_d[i_cold])["opd"]}
    ripple = (G_m[i_hot] - G_m[i_cold]) / G_m[last].mean()
    return states, ripple


# --------------------------------------------------------------------------
# Relay re-tuning for the hot disk
# --------------------------------------------------------------------------
def retuned_mirrors(f_th):
    """Mirror radii so that one relay unit including the thermal lens of the
    disk is -I again. The even array mirror (first after the disk), the odd
    one and M get their own radius."""
    roc = dict(COLD_ROC)
    if not np.isfinite(f_th):
        return roc
    # N reflections on the disk but only N-1 relay units between them: the
    # units over-compensate slightly (f' = f (N-1)/N) so that the net lens
    # power seen at the last disk reflection is zero.
    f_th = f_th * (tds.N_PASSES - 1) / tds.N_PASSES
    T, L = tds._abcd_T, tds._abcd_L
    for m, arrays in tds.RELAY_GROUPS.items():
        L1 = tds._path_length("Disk", arrays[0])
        L2 = np.mean([tds._path_length(a, m) for a in arrays])
        fa0, fm0 = tds.design_focal_lengths("relay", L1, L2)

        def eqs(p):
            fa1, fa2, fm = p
            U = T(L1) @ L(fa2) @ T(L2) @ L(fm) @ T(L2) @ L(fa1) @ T(L1)
            return [U[0, 0] + 1, U[0, 1], U[1, 0] + 1 / f_th]

        sol, _, ier, msg = fsolve(eqs, (fa0, fa0, fm0), full_output=True)
        if ier != 1:
            raise RuntimeError(f"relay re-tune failed for {m}: {msg}")
        fa1, fa2, fm = sol
        for a in arrays:
            roc[a] = 2 * (fa1 if int(a[1:]) % 2 == 0 else fa2)
        roc[m] = 2 * fm
    roc["A1"] = roc[f"A{tds.N_ARRAY}"] = np.inf
    return roc


# --------------------------------------------------------------------------
# Evaluation of one disk state
# --------------------------------------------------------------------------
class Bench:
    def __init__(self):
        tds.DISK.thermal_opd = None
        tds.MIRROR_ROC.update(COLD_ROC)
        self.ref, _ = tds.trace(slm.flat(), G)              # cold Gaussian output = reference
        self.targets = {k: v for k, v in co.target_modes(G, self.ref).items() if k in TARGETS}
        self.fin_beam = co.field_at_slm(G)
        self.fin = self.fin_beam.field
        self.env = np.sqrt(tds.SLM_DEVICE.reflectivity) * tds.SLM_DEVICE.fill_factor * np.abs(self.fin)
        self.w_seed = radius(self.fin)
        amp = make_amplifier(G)
        self.naive = {}
        for name, tgt in self.targets.items():
            self.naive[name], hist = co.optimise_pattern(amp, self.fin, tgt, ITERS)
            print(f"cold pattern for {name}: fidelity {hist[-1]:.4f}", flush=True)

    def evaluate(self, label, opd, roc, full, extra=None):
        """Metrics of the amplifier with disk OPD `opd` and mirror radii `roc`."""
        tds.DISK.thermal_opd = opd
        tds.MIRROR_ROC.update(roc)
        amp = make_amplifier(G)
        out_flat = amp.forward(co.ideal_slm(self.fin, np.zeros_like(self.fin, dtype=float)))
        p_flat = np.vdot(out_flat, out_flat).real
        f_th, aberr = per_pass_lens(opd)
        rows = []
        for name, tgt in self.targets.items():
            out_naive = amp.forward(co.ideal_slm(self.fin, self.naive[name]))
            F_naive = co.fidelity(out_naive, tgt)
            align = np.exp(1j * np.angle(np.vdot(tgt, out_naive)))
            want = _flip_x(amp.adjoint(tgt * align))        # field the SLM would have to deliver
            w_req = radius(want)
            P_in = np.sum(np.abs(want[G.R < self.w_seed]) ** 2) / np.sum(np.abs(want) ** 2)
            ov, eff, fields = self.scan(want)
            F_ub = ov.max()
            F_ub30 = ov[eff >= 0.3].max() if np.any(eff >= 0.3) else 0.0
            row = dict(state=label, target=name, w_out=radius(out_flat) * 1e3, f_pass=f_th, aberration=aberr,
                       F_naive=F_naive, w_req=w_req * 1e3, P_in=P_in, F_cplx_ub=F_ub, F_cplx_30=F_ub30,
                       F_phase=np.nan, F_davis=np.nan, eff_davis=np.nan)
            if full:
                pattern, _ = co.optimise_pattern(amp, self.fin, tgt, ITERS)
                real = tds.SLM_DEVICE.modulate(self.fin_beam, slm.map_pattern(pattern, G), bias=tds.SLM_BIAS).field
                row["F_phase"] = co.fidelity(amp.forward(real), tgt)
                # complex encoding: the most faithful scale that keeps >= 20 % efficiency
                ok = eff >= 0.2
                k = np.flatnonzero(ok)[np.argmax(ov[ok])] if ok.any() else np.argmax(ov)
                ratio = np.abs(fields[k]) / (self.env + 1e-30)
                phase = np.angle(want * np.conj(self.fin))
                out_davis = amp.forward(co.davis_realise(self.fin_beam, ratio, phase))
                row["F_davis"] = co.fidelity(out_davis, tgt)
                row["eff_davis"] = np.vdot(out_davis, out_davis).real / p_flat
            if extra:
                row.update(extra)
            rows.append(row)
            print(f"{label:22s} {name:8s}: w_out {row['w_out']:.2f} mm, f/pass {f_th:8.2f} m, naive {F_naive:.3f}, "
                  f"required {w_req*1e3:.2f} mm ({P_in:.2f} inside seed), complex ub {F_ub:.3f} ({F_ub30:.3f} @30%)"
                  + (f", phase-only {row['F_phase']:.3f}, Davis {row['F_davis']:.3f} (eff {row['eff_davis']:.2f})" if full else ""),
                  flush=True)
        return rows

    def scan(self, want):
        """SLM-plane overlap and efficiency vs the scale of the wanted field."""
        ov, eff, fields = [], [], []
        for s in np.geomspace(0.01, 100, 60) * self.env.max() / np.abs(want).max():
            f = np.minimum(s * np.abs(want), self.env) * np.exp(1j * np.angle(want))
            ov.append(co.fidelity(f, want)); eff.append(np.vdot(f, f).real / np.vdot(self.env, self.env).real)
            fields.append(f)
        return np.array(ov), np.array(eff), fields


def radius(field):
    """Second-moment radius (= w for a Gaussian)."""
    I = np.abs(field) ** 2
    return np.sqrt(2 * np.sum(I * G.R ** 2) / I.sum())


def write_csv(rows, name):
    import csv
    with open(os.path.join(RESULTS, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# The three parts
# --------------------------------------------------------------------------
def run_power(retune):
    b = Bench()
    rows = []
    for P in P_LIST:
        opd = thermal(P).opd(G) if P > 0 else None
        f_th, _ = per_pass_lens(opd)
        roc = retuned_mirrors(f_th) if retune else COLD_ROC
        extra = dict(P=P, R_M1=roc["M1"], R_M2=roc["M2"], R_A_even=roc["A2"], R_A_odd=roc["A3"])
        rows += b.evaluate(f"{P:.0f} W {'hot' if retune else 'cold'} relay", opd, roc, P in P_FULL, extra)
    write_csv(rows, "power_hot_relay.csv" if retune else "power_cold_relay.csv")


def run_residual():
    b = Bench()
    P = 300
    base = thermal(P).opd(G)
    f_th, _ = per_pass_lens(base)
    roc = retuned_mirrors(f_th)
    rows = []
    for rms in (0.05, 0.1, 0.2, 0.3, 0.5):
        th = thermal(P)
        opd = th.solve(G, Q=th.heat_flux(G) * (1 + hot_spot_field(G, rms=rms)))["opd"]
        rows += b.evaluate(f"hot spots {rms:.2f}", opd, roc, rms in (0.1, 0.2, 0.5), dict(kind="hot_spots", value=rms))
    for scale in (0.5, 1, 2, 4, 8, 16):
        rows += b.evaluate(f"shape x{scale:g}", base + shape_error(scale), roc, scale in (1, 4, 16), dict(kind="shape", value=scale))
    for amp in (0.02, 0.04):
        rows += b.evaluate(f"doping +-{amp*100:.0f}%", base + doping_opd(P, amp), roc, amp == 0.02, dict(kind="doping", value=amp))
    states, ripple = modulated_states(P)
    f_mean, _ = per_pass_lens(states["mean"])
    roc_m = retuned_mirrors(f_mean)
    print(f"50 Hz modulation: temperature ripple {ripple:.1%} of the mean, mean lens {f_mean:.1f} m/pass", flush=True)
    rows += b.evaluate("50 Hz mean", states["mean"], roc_m, True, dict(kind="modulation", value=0))
    # pattern optimised at the mean state, applied at the extremes of the cycle
    tds.DISK.thermal_opd = states["mean"]; tds.MIRROR_ROC.update(roc_m)
    amp_mean = make_amplifier(G)
    patterns = {n: co.optimise_pattern(amp_mean, b.fin, t, ITERS)[0] for n, t in b.targets.items()}
    for inst in ("hottest", "coldest"):
        tds.DISK.thermal_opd = states[inst]
        amp_i = make_amplifier(G)
        for n, t in b.targets.items():
            F = co.fidelity(amp_i.forward(co.ideal_slm(b.fin, patterns[n])), t)
            rows.append(dict(state=f"50 Hz {inst} (mean pattern)", target=n, w_out=np.nan, f_pass=per_pass_lens(states[inst])[0],
                             aberration=np.nan, F_naive=np.nan, w_req=np.nan, P_in=np.nan, F_cplx_ub=np.nan, F_cplx_30=np.nan,
                             F_phase=F, F_davis=np.nan, eff_davis=np.nan, kind="modulation", value=ripple))
            print(f"50 Hz {inst:8s}, pattern of the mean state: {n:8s} fidelity {F:.3f}", flush=True)
    write_csv(rows, "residual_300W.csv")


def run_pump_spot():
    """300 W, re-tuned relay: pump spot radius (and edge sharpness) vs the
    1.7 mm beam on the disk. Outside the pumped area only the bending term
    is left, so the OPD has a kink at the pump edge that the beam wings see."""
    b = Bench()
    rows = []
    for r_pump, order in ((2.0e-3, 8), (2.25e-3, 8), (2.5e-3, 8), (3.0e-3, 8), (3.5e-3, 8), (2.5e-3, 4), (3.0e-3, 4)):
        th = DiskThermal(pump_power=300, doping=0.10, pump_radius=r_pump, pump_order=order)
        opd = th.opd(G)
        f_th, _ = per_pass_lens(opd)
        roc = retuned_mirrors(f_th)
        full = (r_pump, order) in ((2.5e-3, 8), (3.0e-3, 8), (3.5e-3, 8), (3.0e-3, 4))
        rows += b.evaluate(f"pump {r_pump*1e3:.2f} mm, order {order}", opd, roc, full,
                           dict(kind="pump_spot", value=r_pump * 1e3, order=order, T_front=th.solve(G)["T_front"].max() - 300))
    write_csv(rows, "pump_spot_300W.csv")


# --------------------------------------------------------------------------
# Figures and summary
# --------------------------------------------------------------------------
def load(name):
    import csv
    with open(os.path.join(RESULTS, name)) as fh:
        return [{k: (v if k in ("state", "target", "kind") else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def plot_all():
    cold, hot, res = load("power_cold_relay.csv"), load("power_hot_relay.csv"), load("residual_300W.csv")
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for col, (rows, title) in enumerate([(cold, "relay as designed for the cold disk"), (hot, "relay re-tuned to the hot disk")]):
        for i, name in enumerate(TARGETS):
            r = [x for x in rows if x["target"] == name]
            P = [x["P"] for x in r]
            a = ax[i, col]
            a.plot(P, [x["F_naive"] for x in r], "k--", label="no correction (cold pattern)")
            a.plot(P, [x["F_cplx_ub"] for x in r], color=f"C{i}", alpha=0.4, label="complex SLM, upper bound")
            a.plot(P, [x["F_cplx_30"] for x in r], color=f"C{i}", ls=":", label="complex SLM at >= 30 % efficiency")
            full = [x for x in r if np.isfinite(x["F_phase"])]
            a.plot([x["P"] for x in full], [x["F_phase"] for x in full], "o-", color=f"C{i}", label="phase-only optimal, real LCOS")
            a.plot([x["P"] for x in full], [x["F_davis"] for x in full], "s", color=f"C{i}", mfc="none", label="complex Davis, real LCOS (>= 20 % eff.)")
            a.set_title(f"{name}: {title}", fontsize=10); a.set_xlabel("pump power [W]"); a.set_ylabel("output fidelity")
            a.set_ylim(0, 1.02); a.grid(alpha=0.3)
            if i == 0 and col == 0:
                a.legend(fontsize=7, loc="lower left")
        a2 = ax[0, 2] if col == 0 else ax[1, 2]
        r = [x for x in rows if x["target"] == "gaussian"]
        a2.plot([x["P"] for x in r], [x["w_req"] for x in r], "r-", label="input radius the SLM must deliver")
        a2.plot([x["P"] for x in r], [x["w_out"] for x in r], "b-", label="output radius, no correction")
        a2.axhline(r[0]["w_req"], color="gray", ls="--", label="seed radius at the SLM")
        a2.set_title(f"beam sizes, {title}", fontsize=10); a2.set_xlabel("pump power [W]"); a2.set_ylabel("radius [mm]")
        a2.set_ylim(0, 12); a2.grid(alpha=0.3); a2.legend(fontsize=7)
    fig.suptitle(f"Limits of SLM pre-correction vs pump power (10-pass relay, 1.5 mm seed, Hamamatsu X15213, pump radius {PUMP_RADIUS*1e3:g} mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "limits_vs_pump_power.png"), dpi=110)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for a, kind, xlabel in zip(ax, ("hot_spots", "shape", "doping"),
                               ("hot-spot rms (fraction of the heat load)", "shape error scale (1 = 30/20/15 nm astig/trefoil/coma)", "doping cluster amplitude")):
        for i, name in enumerate(TARGETS):
            r = [x for x in res if x["kind"] == kind and x["target"] == name]
            v = [x["value"] for x in r]
            a.plot(v, [x["F_naive"] for x in r], "--", color=f"C{i}", alpha=0.6, label=f"{name}: no correction")
            a.plot(v, [x["F_cplx_30"] for x in r], ":", color=f"C{i}", label=f"{name}: complex SLM, >= 30 % eff.")
            full = [x for x in r if np.isfinite(x["F_phase"])]
            a.plot([x["value"] for x in full], [x["F_phase"] for x in full], "o", color=f"C{i}", label=f"{name}: phase-only, real LCOS")
            a.plot([x["value"] for x in full], [x["F_davis"] for x in full], "s", color=f"C{i}", mfc="none", label=f"{name}: Davis, real LCOS")
        a.set_xlabel(xlabel, fontsize=9); a.set_ylabel("output fidelity"); a.set_ylim(0, 1.02); a.grid(alpha=0.3)
    ax[0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"Residual distortions at 300 W with the relay re-tuned to the hot disk (pump radius {PUMP_RADIUS*1e3:g} mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "limits_residual_300W.png"), dpi=110)

    lines = ["# Operating limits of SLM pre-correction (Step 13)\n",
             f"System: 10-pass -I relay, 1.5 mm seed, Hamamatsu X15213, 0.5 mm / 10 % Yb:YAG, pump radius {PUMP_RADIUS*1e3:g} mm, uniform-disk thermal model.",
             "Fidelity = normalised overlap of the output with the target mode. 'complex ub' = best any complex SLM can do.\n",
             "## Pump power, relay as designed for the cold disk\n",
             "| P [W] | target | f/pass [m] | out radius [mm] | required input [mm] | no corr. | phase-only | complex ub | complex >=30 % | Davis (eff) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for rows, head in ((cold, None), (hot, "## Pump power, relay re-tuned to the hot disk (M1/M2 and array radii per power)\n")):
        if head:
            lines += ["", head, "| P [W] | target | f/pass [m] | out radius [mm] | required input [mm] | no corr. | phase-only | complex ub | complex >=30 % | Davis (eff) | R_M1 | R_A even/odd |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in rows:
            ph = f"{x['F_phase']:.2f}" if np.isfinite(x["F_phase"]) else "-"
            dv = f"{x['F_davis']:.2f} ({x['eff_davis']:.2f})" if np.isfinite(x["F_davis"]) else "-"
            line = (f"| {x['P']:.0f} | {x['target']} | {x['f_pass']:.1f} | {x['w_out']:.2f} | {x['w_req']:.2f} | {x['F_naive']:.2f} | {ph} | "
                    f"{x['F_cplx_ub']:.2f} | {x['F_cplx_30']:.2f} | {dv} |")
            if head:
                line += f" {x['R_M1']:.3f} | {x['R_A_even']:.3f}/{x['R_A_odd']:.3f} |"
            lines.append(line)
    lines += ["", "## Residual distortions at 300 W, re-tuned relay\n",
              "| state | target | no corr. | required input [mm] | phase-only | complex ub | complex >=30 % | Davis (eff) |", "|---|---|---|---|---|---|---|---|"]
    for x in res:
        ph = f"{x['F_phase']:.2f}" if np.isfinite(x["F_phase"]) else "-"
        dv = f"{x['F_davis']:.2f} ({x['eff_davis']:.2f})" if np.isfinite(x["F_davis"]) else "-"
        fn = f"{x['F_naive']:.2f}" if np.isfinite(x["F_naive"]) else "-"
        wr = f"{x['w_req']:.2f}" if np.isfinite(x["w_req"]) else "-"
        ub = f"{x['F_cplx_ub']:.2f} | {x['F_cplx_30']:.2f}" if np.isfinite(x["F_cplx_ub"]) else "- | -"
        lines.append(f"| {x['state']} | {x['target']} | {fn} | {wr} | {ph} | {ub} | {dv} |")
    if os.path.exists(os.path.join(RESULTS, "pump_spot_300W.csv")):
        spot = load("pump_spot_300W.csv")
        fig, ax = plt.subplots(figsize=(7, 4.3))
        for i, name in enumerate(TARGETS):
            for order, mk in ((8, "-"), (4, "--")):
                r = [x for x in spot if x["target"] == name and x["order"] == order]
                v = [x["value"] for x in r]
                ax.plot(v, [x["F_naive"] for x in r], mk, color="k", alpha=0.5 if i == 0 else 0.25, label=f"{name}: no correction, order {order:.0f}")
                ax.plot(v, [x["F_cplx_30"] for x in r], mk, color=f"C{i}", alpha=0.5, label=f"{name}: complex SLM >= 30 % eff., order {order:.0f}")
                full = [x for x in r if np.isfinite(x["F_phase"])]
                ax.plot([x["value"] for x in full], [x["F_phase"] for x in full], "o" if order == 8 else "^", color=f"C{i}", label=f"{name}: phase-only, order {order:.0f}")
        ax.axvline(1.7, color="gray", ls=":", label="beam radius on the disk")
        ax.set_xlabel("pump spot radius [mm] (1/e^2, super-Gaussian order 8 or 4)"); ax.set_ylabel("output fidelity"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="lower right"); ax.set_title("300 W, relay re-tuned: pump spot vs 1.7 mm beam")
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "limits_pump_spot_300W.png"), dpi=110)
        lines += ["", "## Pump spot radius at 300 W, re-tuned relay (beam radius on the disk 1.7 mm)\n",
                  "| pump radius [mm] | order | T_front max [K] | f/pass [m] | out radius [mm] | target | no corr. | required input [mm] | phase-only | complex ub | complex >=30 % | Davis (eff) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in spot:
            ph = f"{x['F_phase']:.2f}" if np.isfinite(x["F_phase"]) else "-"
            dv = f"{x['F_davis']:.2f} ({x['eff_davis']:.2f})" if np.isfinite(x["F_davis"]) else "-"
            lines.append(f"| {x['value']:.2f} | {x['order']:.0f} | {x['T_front']:.0f} | {x['f_pass']:.1f} | {x['w_out']:.2f} | {x['target']} | {x['F_naive']:.2f} | {x['w_req']:.2f} | {ph} | "
                         f"{x['F_cplx_ub']:.2f} | {x['F_cplx_30']:.2f} | {dv} |")
    with open(os.path.join(RESULTS, "LIMITS.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("saved", os.path.abspath(RESULTS))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "plot"
    {"power_cold": lambda: run_power(False), "power_hot": lambda: run_power(True),
     "residual": run_residual, "pump_spot": run_pump_spot, "plot": plot_all}[part]()
