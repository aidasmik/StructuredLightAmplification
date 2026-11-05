"""
nn_dataset.py - Steps 17/19/20: training data for the neural-network pre-correction.

One example = one random disk state. Four parameter ranges (environment NN_RANGE):

                      "narrow" (Step 17)            "wide" (Step 19)       "real" (Step 20, default)
    pump power        300 W +- 5 % (uniform)        240 .. 360 W           240 .. 360 W
    pump spot size    nominal                       +- 5 %                 +- 5 %
    pump decentre     N(0, 0.1 mm) in x and y       N(0, 0.15 mm)          N(0, 0.15 mm)
    hot spots         rms 0 .. 7 % of the heat      0 .. 10 %              0 .. 10 %
                      load inside the 1 mm beam, new random pattern each time
                      (the Step 16 state, "10 %" over r < 2 mm, is 6.7 %)
    shape error       random low-order Zernikes (7 terms), surface rms inside r = 1.5 mm
                      equal to 0 .. 1.5 x nominal   0 .. 2 x               0 .. 2 x
                      (nominal: 30/20/15 nm astig / trefoil / coma)
    doping clusters   +-0 .. 2 % (bank of N_DOPING)  0 .. 3 %               0 .. 3 %
    lens tolerance    15 %                          45 %                   45 %

"real" adds the gain of the disk and a layer of nuisance effects the lab has and the
model would otherwise never see (one random draw per example, 1-sigma values in NUIS):
    gain              quasi-three-level gain map per reflection from the state's own pump,
                      temperature and doping (gain.py; no saturation, no population lens)
    seed beam         5 low-order Zernike terms (0.2 rad each at r = 1.5 mm), pointing
                      (0.4 rad tilt at 1.5 mm = 40 urad), position (1.5 px = 40 um), power +-5 %
    relay mirrors     tilt N(0, 5 urad) per axis on each of the 23 physical mirrors (20 urad
                      would cost 0.4 in fidelity even at the optimum: a case for re-alignment)
    SLM               phase-stroke (look-up table) error N(0, 3 %)
    cameras           each: shift N(0, 1.5 px), rotation N(0, 0.7 deg), scale N(0, 1.5 %),
                      gain N(0, 4 %), dark offset N(0, 0.3 % of full scale);
                      far-field lens defocus N(0, 0.3 rad at the beam radius)

What the lab can measure (inputs):
    * near-field camera (50 um pixels) and far-field camera (focal plane of a
      f = 1 m lens, 25 um pixels), 12 bit, 1 % noise, FIXED exposure, taken with
      the uncorrected target pattern on the SLM (flat for the Gaussian, vortex
      for LG01) - as in the feedback loop of the slides,
    * a few temperature readings on the disk (pyrometer spots: centre and four
      points at r = 1.5 mm, 0.5 K noise); the full 32 x 32 map is stored too
      but is not meant as a network input,
    * pump power from the meter (1 % noise),
    * the doping map of the crystal (thickness-mean deviation from nominal,
      known once from the supplier's / a one-off measurement).

What the network should produce (labels):
    * the phase-only SLM correction from 12 alternating projections with the true
      state (optimal pattern minus the base pattern, wrapped, 256 x 256 = 6.4 mm
      around the beam at the SLM; the physics loss of nn_train.py goes beyond it),
    * also stored for the physics loss: the true disk OPD and gain (50 um crops) and
      the nuisance parameters (seed, mirror tilts, stroke); the fidelities before /
      after the optimal correction.

The relay stays tuned to the nominal 300 W disk (tuned once, as in the lab).
States whose per-pass thermal lens differs by more than LENS_TOL in power from
the tuning point are redrawn: a detuned 10-pass relay cannot be fixed from the
input plane (optimum < 0.8 beyond +-25 %). The narrow range keeps 15 % (such a
disk would be re-tuned in the lab); the others only cut hopeless outliers,
the pump range itself moves the lens by -23 .. +24 %.

"setup" (Step 21, default) is "real" with the nuisances as they are in a lab: fixed for a given
setup and recorded once in its reference state, plus a small drift between shots (DRIFT). There
are N_SETUPS setups (nuisance draw from the setup index); example i belongs to setup i % N_SETUPS,
and `nn_dataset.py reference` records one reference per setup (reference.npz, leading setup axis).
"real" instead draws all nuisances new for every example - as if the laser were rebuilt between
shots - which forces the network to hedge against errors the reference would have absorbed.

Usage:  python nn_dataset.py N_EXAMPLES [worker_index n_workers]
        python nn_dataset.py reference        (the reference state -> reference.npz)
Output: Results/<dataset>/shards/shard_<worker>_<k>.npz, meta.npz, reference.npz
        (17_nn_dataset narrow, 19_nn_wide_dataset wide, 20_nn_real_dataset real, 21_nn_setup_dataset setup)
"""
import os
import sys
import time
import warnings
from dataclasses import replace

import numpy as np
from scipy.ndimage import shift as nd_shift, zoom, map_coordinates, affine_transform

import limits_benchmark as lb
import thin_disk_system as tds
import slm
import correction as co
from correction_benchmark import hot_spot_field
from seed_beam import Grid, results_dir
from gpu import make_amplifier
from spgd import Camera
from two_camera_correction import to_far
from doping import DopedDiskThermal, doping_map_3d
from gain import gain_map, N_SITES

warnings.filterwarnings("ignore")

G = Grid(**tds.SIM_GRID)
P0 = 300.0
TARGETS = ("gaussian", "LG01")
SHARD = 100                                   # examples per file
N_DOPING = int(os.environ.get("N_DOPING", 16))                                 # pre-computed doping cluster maps
CROP_CAM = 128                                # camera pixels stored per side
CROP_LABEL = 256                              # simulation pixels stored for the label (6.4 mm at the SLM)
N_THERM = 32                                  # thermal camera pixels over the disk
R_HOT = 1.0e-3                                # the hot-spot rms is measured inside the beam radius
RANGES = {                                    # parameter ranges of the random states, see the module docstring
    "narrow": dict(P=(0.95, 1.05), pump_scale=(1.0, 1.0), decentre=0.1e-3, hot_max=0.07, shape_max=1.5, dop_max=1.0,
                   lens_tol=0.15, T_scale=5.0, P_scale=15.0, realism=False, dataset="17_nn_dataset", train="18_nn"),
    "wide": dict(P=(0.8, 1.2), pump_scale=(0.95, 1.05), decentre=0.15e-3, hot_max=0.10, shape_max=2.0, dop_max=1.5,
                 lens_tol=0.45, T_scale=20.0, P_scale=60.0, realism=False, dataset="19_nn_wide_dataset", train="19_nn_wide"),
    "real": dict(P=(0.8, 1.2), pump_scale=(0.95, 1.05), decentre=0.15e-3, hot_max=0.10, shape_max=2.0, dop_max=1.5,
                 lens_tol=0.45, T_scale=20.0, P_scale=60.0, realism=True, dataset="20_nn_real_dataset", train="20_nn_real"),
    "setup": dict(P=(0.8, 1.2), pump_scale=(0.95, 1.05), decentre=0.15e-3, hot_max=0.10, shape_max=2.0, dop_max=1.5,
                  lens_tol=0.45, T_scale=20.0, P_scale=60.0, realism=True, setups=64, dataset="21_nn_setup_dataset", train="21_nn_setup")}
RANGE_NAME = os.environ.get("NN_RANGE", "setup")
RANGE = RANGES[RANGE_NAME]
REALISM = RANGE["realism"]                    # gain + nuisance layer on
N_SETUPS = RANGE.get("setups", 0)             # > 0: nuisances fixed per setup + drift (range "setup")
HOT_MAX = RANGE["hot_max"]                    # largest hot-spot rms inside the beam
LENS_TOL = RANGE["lens_tol"]                  # accepted |1/f - 1/f0| / (1/f0) of the per-pass thermal lens
T_SCALE, P_SCALE = RANGE["T_scale"], RANGE["P_scale"]   # network input normalisation [K], [W]
RESULTS = results_dir(RANGE["dataset"])
TRAIN_RESULTS = RANGE["train"]                # where nn_train / nn_eval put their results
SHARDS = os.path.join(RESULTS, "shards")
os.makedirs(SHARDS, exist_ok=True)

NOMINAL_SHAPE = {"astig0": 30e-9, "trefoil30": 20e-9, "coma_x": 15e-9}
R_SHAPE = 1.5e-3                              # the shape rms is measured inside this radius (beam + margin)
T_POINTS = np.array([(0, 0), (1.5e-3, 0), (-1.5e-3, 0), (0, 1.5e-3), (0, -1.5e-3)])   # pyrometer spots [m]
SHAPE_TERMS = ("astig0", "astig45", "coma_x", "coma_y", "trefoil0", "trefoil30", "spherical")
# nuisance layer (1-sigma values): seed = [5 Zernike terms, tilt_x, tilt_y, shift_x px, shift_y px, power]
SEED_TERMS = ("defocus", "astig0", "astig45", "coma_x", "coma_y")
R_SEED = 1.5e-3                               # the seed aberrations and tilts are given at this radius [rad]
NUIS = dict(seed_ab=0.2, seed_tilt=0.4, seed_shift=1.5, seed_power=0.05, mirror_tilt=5e-6, stroke=0.03,
            cam_shift=1.5, cam_rot=0.7, cam_scale=0.015, cam_gain=0.04, cam_offset=0.003, far_defocus=0.3)
# drift of the nuisances between shots of one setup (1-sigma), on top of the setup's fixed values
DRIFT = dict(seed_ab=0.03, seed_tilt=0.05, seed_shift=0.0, seed_power=0.02, mirror_tilt=1e-6, stroke=0.005,
             cam_shift=0.2, cam_rot=0.05, cam_scale=0.002, cam_gain=0.01, cam_offset=0.001, far_defocus=0.03)
MIRROR_NAMES = list(dict.fromkeys(n for n in tds.PATH if n not in ("BS", "SLM", "Disk", "Out")))   # 23 physical mirrors


def crop(a, n):
    c = a.shape[0] // 2
    return a[c - n // 2:c + n // 2, c - n // 2:c + n // 2]


def doping_entry(seed):
    """One doping-cluster realisation (+-2 %): OPD difference to the uniform disk
    [m, reflection] and the thickness-mean doping deviation (c / c_nominal - 1)."""
    th = DopedDiskThermal(pump_power=P0, doping=0.10, n_pump_passes=24, pump_radius=lb.PUMP_RADIUS, pump_order=lb.PUMP_ORDER)
    c3, _, z = doping_map_3d(G, th.thickness, nz=50, nominal=0.10, amplitude=0.02, cluster_size=0.3e-3, seed=seed)
    clustered = th.solve_3d(G, c3, z)["opd"]
    uniform = DopedDiskThermal(pump_power=P0, doping=0.10, doping_field=np.full((G.n, G.n), 0.10),
                               n_pump_passes=24, pump_radius=lb.PUMP_RADIUS, pump_order=lb.PUMP_ORDER).solve(G)["opd"]
    return clustered - uniform, c3.mean(axis=0) / 0.10 - 1


# --------------------------------------------------------------------------
# The fixed part of the system
# --------------------------------------------------------------------------
class System:
    def __init__(self):
        th = lb.thermal(P0)
        f_th, _ = lb.per_pass_lens(th.opd(G), G)
        self.f0 = f_th                                                 # the lens the relay is tuned to
        tds.DISK.thermal_opd = None
        tds.MIRROR_ROC.update(lb.COLD_ROC)
        self.ref, _ = tds.trace(slm.flat(), G)                        # cold Gaussian output
        self.targets = {k: v for k, v in co.target_modes(G, self.ref).items() if k in TARGETS}
        self.w_out = float(np.mean(self.ref.widths()))                # output beam radius (far-field defocus scale)
        tds.DISK.thermal_opd = th.opd(G)
        tds.MIRROR_ROC.update(lb.retuned_mirrors(f_th))              # relay tuned once, to the nominal disk
        tds.DISK.gain = np.ones((G.n, G.n)) if REALISM else None      # slot for the per-state gain map
        self.amp = make_amplifier(G)
        tds.DISK.gain = None
        self.fin_beam = co.field_at_slm(G)
        self.fin = self.fin_beam.field
        self.base = {"gaussian": np.zeros((G.n, G.n)), "LG01": slm.vortex(1)(G.X, G.Y)}
        self.Q0 = th.heat_flux(G)
        self.th = th
        self.n_col0 = N_SITES * th.doping * th.thickness              # Yb ions per area, nominal doping
        # seed aberration basis: 5 Zernike terms + tilt x / y, unit amplitude at R_SEED
        self.seed_basis = np.stack([slm.zernike({t: 1.0}, R_SEED)(G.X, G.Y) for t in SEED_TERMS]
                                   + [G.X / R_SEED, G.Y / R_SEED]).astype(np.float32)
        # doping cluster OPD bank (slow 3-D solves, done once and cached)
        bank = os.path.join(RESULTS, f"doping_bank_{N_DOPING}.npz")
        if not os.path.exists(bank):
            entries = [doping_entry(100 + i) for i in range(N_DOPING)]
            np.savez(bank, opd=np.array([e[0] for e in entries], np.float32), c_dev=np.array([e[1] for e in entries], np.float32))
        b = np.load(bank)
        self.doping, self.doping_map = b["opd"], b["c_dev"]
        # nominal shape error: its rms inside R_SHAPE sets the scale of the random ones
        self.shape_rms0 = self._shape_rms(2 * slm.zernike(NOMINAL_SHAPE, 2e-3)(G.X, G.Y))
        # cameras with a fixed exposure: set on the nominal disk (Gaussian target); with gain the
        # brightest expected frame is the highest pump power, so the exposure is set there
        self.near_cam = Camera(binning=int(round(50e-6 / G.dx)), blur=0.5, noise=0.01, bits=12)
        self.far_cam = Camera(binning=1, blur=0.5, noise=0.01, bits=12)
        p = self.nominal_params(P0 * RANGE["P"][1] if REALISM else P0)
        out = self.output(self.base["gaussian"], self.state(p), p)
        self.near_cam.image(out); self.far_cam.image(to_far(out, self.ref))   # sets the exposures
        head = 0.6 if REALISM else 0.7                                        # headroom for brighter states
        self.near_cam._scale *= head; self.far_cam._scale *= head
        self.disk_mask = G.R < tds.DISK.diameter / 2

    # ---- the state of the system -------------------------------------------------
    def nominal_params(self, P=P0, setup=None):
        """Nominal disk at pump P, no nuisance effects - or the fixed nuisances of `setup`."""
        p = dict(P=P, pump_scale=1.0, dx=0.0, dy=0.0, hot_rms=0.0, hot_seed=0, shape_scale=0.0,
                 shape_dir=np.ones(len(SHAPE_TERMS)), dop_amp=0.0, dop_idx=0)
        if REALISM:
            p.update(seed=np.array([0.0] * 9 + [1.0]), tilts=np.zeros((len(MIRROR_NAMES), 2)), stroke=0.0, cam=np.zeros(13))
            if setup is not None:
                p.update(self.setup_nuisance(setup))
        return p

    @staticmethod
    def nuisance(rng, n):
        """One draw of the nuisance layer with the 1-sigma values n (NUIS or DRIFT)."""
        cam = lambda: [*(n["cam_shift"] * rng.standard_normal(2)), n["cam_rot"] * rng.standard_normal(), n["cam_scale"] * rng.standard_normal(),
                       n["cam_gain"] * rng.standard_normal(), n["cam_offset"] * rng.standard_normal()]
        return dict(seed=np.array([*(n["seed_ab"] * rng.standard_normal(len(SEED_TERMS))), *(n["seed_tilt"] * rng.standard_normal(2)),
                                   *np.round(n["seed_shift"] * rng.standard_normal(2)), 1 + rng.uniform(-n["seed_power"], n["seed_power"])]),
                    tilts=n["mirror_tilt"] * rng.standard_normal((len(MIRROR_NAMES), 2)),
                    stroke=n["stroke"] * rng.standard_normal(),
                    cam=np.array([*cam(), *cam(), n["far_defocus"] * rng.standard_normal()]))     # near, far, far-lens defocus

    def setup_nuisance(self, setup):
        """The fixed nuisances of setup number `setup` (reproducible from the index)."""
        return self.nuisance(np.random.default_rng([setup, 7]), NUIS)

    def random_params(self, rng, setup=None):
        r = RANGE
        p = dict(P=P0 * rng.uniform(*r["P"]), pump_scale=rng.uniform(*r["pump_scale"]),
                 dx=rng.normal(0, r["decentre"]), dy=rng.normal(0, r["decentre"]),
                 hot_rms=rng.uniform(0, HOT_MAX), hot_seed=int(rng.integers(1 << 31)),
                 shape_scale=rng.uniform(0, r["shape_max"]),
                 shape_dir=rng.standard_normal(len(SHAPE_TERMS)),
                 dop_amp=rng.uniform(0, r["dop_max"]), dop_idx=int(rng.integers(N_DOPING)))
        if REALISM and setup is None:                                      # the nuisance layer, new for this example
            p.update(self.nuisance(rng, NUIS))
        elif REALISM:                                                      # the setup's fixed nuisances + drift
            base, d = self.setup_nuisance(setup), self.nuisance(rng, DRIFT)
            seed = base["seed"].copy(); seed[:9] += d["seed"][:9]; seed[9] *= d["seed"][9]
            p.update(seed=seed, tilts=base["tilts"] + d["tilts"], stroke=base["stroke"] + d["stroke"], cam=base["cam"] + d["cam"], setup=setup)
        return p

    def state(self, p):
        """Disk state for the parameters p: OPD (reflection, m), thickness-mean temperature,
        and with REALISM the power gain map per reflection."""
        s = p.get("pump_scale", 1.0)                                               # pump spot magnified by s, same power
        Q = affine_transform(self.Q0, np.array([1 / s, 1 / s]), offset=G.n / 2 * (1 - 1 / s), order=1) / s ** 2 if s != 1.0 else self.Q0
        Q = nd_shift(Q, (p["dy"] / G.dx, p["dx"] / G.dx), order=1) * p["P"] / P0
        hot = hot_spot_field(G, rng_seed=p["hot_seed"], rms=1.0)
        Q = Q * (1 + p["hot_rms"] * hot / (hot[G.R < R_HOT].std() + 1e-30))        # rms set inside the beam
        th_state = self.th.solve(G, Q=Q)
        # static shape: random direction in the Zernike space, rms inside R_SHAPE = nominal rms x scale
        coeff = np.asarray(p["shape_dir"], float) * 1e-9
        shape = 2 * slm.zernike(dict(zip(SHAPE_TERMS, coeff)), 2e-3)(G.X, G.Y)        # OPD in reflection
        k = self.shape_rms0 * p["shape_scale"] / (self._shape_rms(shape) + 1e-30)
        shape, coeff = shape * k, coeff * k
        p["shape_coeff"] = coeff
        st = dict(opd=th_state["opd"] + shape + p["dop_amp"] * self.doping[p["dop_idx"]], T_mean=th_state["T_mean"])
        if REALISM:                                                                # gain from the same pump, T and doping
            n_col = self.n_col0 * (1 + p["dop_amp"] * self.doping_map[p["dop_idx"]])
            st["gain"] = gain_map(th_state["Q"] / self.th.eta_heat, n_col, th_state["T_mean"])[0]
        return st

    @staticmethod
    def _shape_rms(opd):
        m = G.R < R_SHAPE
        return np.sqrt(np.mean((opd[m] - opd[m].mean()) ** 2))

    def draw_state(self, rng, max_tries=30, setup=None):
        """Random parameters + state, redrawn until the thermal lens is within LENS_TOL of the tuning point."""
        for _ in range(max_tries):
            p = self.random_params(rng, setup)
            st = self.state(p)
            p["lens_change"] = self.f0 / lb.per_pass_lens(st["opd"], G)[0] - 1
            if abs(p["lens_change"]) <= LENS_TOL:
                return p, st
        return p, st

    def record(self, st, p):
        """What the shards keep of a state for the physics loss: 50 um crops (6.4 mm) and the nuisance parameters."""
        r = dict(opd=crop(zoom(st["opd"], 0.5, order=1), CROP_LABEL // 2).astype(np.float32))
        if REALISM:
            r["gain"] = crop(zoom(st["gain"], 0.5, order=1), CROP_LABEL // 2).astype(np.float16)
            r["seed"], r["tilts"], r["stroke"] = p["seed"].astype(np.float32), p["tilts"].astype(np.float32), np.float32(p["stroke"])
        return r

    # ---- the optics with this state ----------------------------------------------
    def seed_field(self, p):
        """Seed field at the SLM: nominal, or shifted, aberrated and power-scaled (p["seed"])."""
        s = p.get("seed")
        if s is None:
            return self.fin
        f = np.roll(self.fin, (int(s[8]), int(s[7])), axis=(0, 1))                 # shift y, x [px]
        return f * np.exp(1j * np.tensordot(s[:7], self.seed_basis, 1)) * np.sqrt(s[9])

    def _arm(self, st, p):
        """Put the state into the amplifier and the SLM."""
        self.amp.set_thermal_opd(st["opd"])
        if REALISM:
            self.amp.set_gain(st["gain"])
            self.amp.set_mirror_tilts(dict(zip(MIRROR_NAMES, p["tilts"])))
            tds.SLM_DEVICE.stroke = 1 + p["stroke"]

    def output(self, phase_map, st, p):
        """Field at the output for the SLM phase map with the system in state (st, p) (real LCOS model)."""
        self._arm(st, p)
        beam = replace(self.fin_beam, field=self.seed_field(p))
        field = tds.SLM_DEVICE.modulate(beam, slm.map_pattern(phase_map, G), bias=tds.SLM_BIAS).field
        return self.amp.forward(field)

    def optimum(self, name, st, p, iterations=12):
        """Phase-only pattern for target `name` from alternating projections with the true state (ideal SLM)."""
        self._arm(st, p)
        return co.optimise_pattern(self.amp, self.seed_field(p), self.targets[name], iterations)

    def frames(self, out, p):
        """Near- and far-field camera crops of the output field, with this state's camera errors."""
        c = p.get("cam")
        far_in = out if c is None else out * np.exp(1j * c[12] * (G.R / self.w_out) ** 2)   # far-field lens defocus
        near, far = self.near_cam.image(out), self.far_cam.image(to_far(far_in, self.ref))
        if c is not None:
            near, far = self._register(near, c[0:6]), self._register(far, c[6:12])
        return crop(near, CROP_CAM), crop(far, CROP_CAM)

    @staticmethod
    def _register(img, c):
        """Misregistered camera: c = [shift_y, shift_x (px), rotation (deg), scale, gain, dark offset]."""
        a, s = np.deg2rad(c[2]), 1 + c[3]
        M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) / s
        centre = np.array(img.shape) / 2
        img = affine_transform(img, M, offset=centre - M @ centre + c[:2], order=1, mode="nearest")
        return np.clip(img * (1 + c[4]) + c[5], 0, 1)

    def thermal_image(self, T_mean, rng):
        """Thermal camera: N_THERM x N_THERM over the disk, 0.5 K noise."""
        d = tds.DISK.diameter
        n_px = int(round(d / G.dx))
        T = crop(T_mean, n_px)
        T = zoom(T, N_THERM / n_px, order=1)[:N_THERM, :N_THERM]
        return T + 0.5 * rng.standard_normal(T.shape)

    def temperature_points(self, T_mean, rng=None, noise=0.5):
        """Pyrometer readings at T_POINTS, 0.5 K noise (none without rng)."""
        coords = [T_POINTS[:, 1] / G.dx + G.n / 2, T_POINTS[:, 0] / G.dx + G.n / 2]
        T = map_coordinates(T_mean, coords, order=1)
        return T if rng is None else T + noise * rng.standard_normal(len(T_POINTS))

    def doping_input(self, p):
        """Known doping map of this crystal: deviation from nominal, 50 um pixels, 6.4 mm crop."""
        return crop(zoom(p["dop_amp"] * self.doping_map[p["dop_idx"]], 0.5, order=1), CROP_LABEL // 2)

    # ---- the reference state and one complete example -----------------------------
    def reference(self, setup=None):
        """The reference state recorded once in the lab: tuned system, nominal 300 W disk,
        no deviations. Noise-free camera images (averaged) with the base pattern, pyrometer
        readings, and the optimal correction for this state. The network works with
        differences to this reference and adds its output to the reference correction."""
        p = self.nominal_params(setup=setup)
        st = self.state(p)
        ref = dict(P=P0, T_pts=self.temperature_points(st["T_mean"]).astype(np.float32), opd=st["opd"].astype(np.float32))
        noise = self.near_cam.noise, self.far_cam.noise
        self.near_cam.noise = self.far_cam.noise = 0.0
        for name in TARGETS:
            out = self.output(self.base[name], st, p)
            ref[f"near_{name}"], ref[f"far_{name}"] = (a.astype(np.float32) for a in self.frames(out, p))
            pattern, hist = self.optimum(name, st, p)
            ref[f"corr_{name}"] = crop(np.angle(np.exp(1j * (pattern - self.base[name]))), CROP_LABEL).astype(np.float32)
            ref[f"F0_{name}"], ref[f"Fopt_{name}"] = hist[0], hist[-1]
        self.near_cam.noise, self.far_cam.noise = noise
        return ref

    def example(self, rng, iterations=12, setup=None):
        p, st = self.draw_state(rng, setup=setup)
        ex = dict(P_meas=p["P"] * (1 + 0.01 * rng.standard_normal()),
                  T_pts=self.temperature_points(st["T_mean"], rng).astype(np.float32),
                  T_img=self.thermal_image(st["T_mean"], rng).astype(np.float16),
                  dop_map=self.doping_input(p).astype(np.float16),
                  params=np.array([p["P"], p["pump_scale"], p["dx"], p["dy"], p["hot_rms"], p["hot_seed"], p["shape_scale"],
                                   p["dop_amp"], p["dop_idx"], *p["shape_coeff"], p["lens_change"]], dtype=np.float64),
                  **self.record(st, p))
        if REALISM:
            ex["cam"] = p["cam"].astype(np.float32)
        if setup is not None:
            ex["setup"] = np.int32(setup)
        for name in TARGETS:
            tgt = self.targets[name]
            out = self.output(self.base[name], st, p)                                  # what the lab sees
            near, far = self.frames(out, p)
            ex[f"near_{name}"], ex[f"far_{name}"] = near.astype(np.float16), far.astype(np.float16)
            ex[f"F0_{name}"] = co.fidelity(out, tgt)
            pattern, hist = self.optimum(name, st, p, iterations)                      # label (ideal SLM)
            corr = np.angle(np.exp(1j * (pattern - self.base[name])))                  # smooth correction only
            ex[f"label_{name}"] = crop(corr, CROP_LABEL).astype(np.float16)
            ex[f"Fopt_{name}"] = hist[-1]
            ex[f"Freal_{name}"] = co.fidelity(self.output(pattern, st, p), tgt)        # optimum through the real LCOS
        return ex


def save_meta(sysm):
    np.savez_compressed(os.path.join(RESULTS, "meta.npz"),
                        grid_n=G.n, grid_size=G.size, crop_cam=CROP_CAM, crop_label=CROP_LABEL,
                        near_pixel=50e-6, far_pixel=G.dx, f_lens=1.0, thermal_pixels=N_THERM, T_points=T_POINTS,
                        targets=np.array(TARGETS), ref=sysm.ref.field.astype(np.complex64),
                        fin=sysm.fin.astype(np.complex64),
                        near_scale=sysm.near_cam._scale, far_scale=sysm.far_cam._scale,
                        param_names=np.array(["P", "pump_scale", "dx", "dy", "hot_rms", "hot_seed", "shape_scale", "dop_amp", "dop_idx", *SHAPE_TERMS, "lens_change"]),
                        range_name=RANGE_NAME, realism=REALISM, mirror_names=np.array(MIRROR_NAMES),
                        seed_terms=np.array(SEED_TERMS), r_seed=R_SEED,
                        **{f"target_{k}": v.astype(np.complex64) for k, v in sysm.targets.items()},
                        **{f"base_{k}": v.astype(np.float32) for k, v in sysm.base.items()})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reference":            # python nn_dataset.py reference
        sysm = System()
        if N_SETUPS:                                                   # one reference per setup, stacked
            refs = [sysm.reference(s) for s in range(N_SETUPS)]
            ref = {k: np.stack([r[k] for r in refs]) for k in refs[0]}
        else:
            ref = sysm.reference()
        np.savez(os.path.join(RESULTS, "reference.npz"), **ref)
        print("reference state: " + ", ".join(f"{t}: uncorrected {np.mean(ref[f'F0_{t}']):.3f}, corrected {np.mean(ref[f'Fopt_{t}']):.3f}" for t in TARGETS)
              + (f" (mean over {N_SETUPS} setups)" if N_SETUPS else ""))
        sys.exit()
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    worker, n_workers = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    t0 = time.time()
    sysm = System()
    if worker == 0:
        save_meta(sysm)
    print(f"system ready in {time.time() - t0:.0f} s; amplifier {type(sysm.amp).__name__}; range {RANGE_NAME}", flush=True)

    mine = range(worker, n_total, n_workers)
    buf, k = [], 0
    t1 = time.time()
    for j, i in enumerate(mine):
        ex = sysm.example(np.random.default_rng([i, 17]), setup=i % N_SETUPS if N_SETUPS else None)   # reproducible per example index
        ex["index"] = i
        buf.append(ex)
        if len(buf) == SHARD or j == len(mine) - 1:
            arrays = {key: np.stack([e[key] for e in buf]) for key in buf[0]}
            np.savez_compressed(os.path.join(SHARDS, f"shard_{worker}_{k:04d}.npz"), **arrays)
            F0 = arrays["F0_gaussian"].mean(), arrays["F0_LG01"].mean()
            Fo = arrays["Fopt_gaussian"].mean(), arrays["Fopt_LG01"].mean()
            print(f"worker {worker}: {j + 1}/{len(mine)} examples, {(time.time() - t1) / (j + 1):.2f} s each; "
                  f"mean fidelity uncorrected {F0[0]:.3f}/{F0[1]:.3f}, optimum {Fo[0]:.3f}/{Fo[1]:.3f}", flush=True)
            buf, k = [], k + 1
    print(f"done in {(time.time() - t0) / 60:.1f} min")
