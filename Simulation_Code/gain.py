"""
gain.py - Small-signal gain of the Yb:YAG thin disk (quasi-three-level),
as a per-reflection power gain map G(x, y) that the disk applies to the
beam like its thermal OPD.

Physics (steady state, CW, no ASE, seed far below saturation):

    upper-state ions per area     N2 = P_abs * tau / (h nu_p)
    double-pass gain coefficient  g  = 2 [ sigma_e N2 - sigma_a(T) (N_col - N2) ]
    power gain per reflection     G  = exp(g)

with P_abs(x, y) the absorbed pump power per area (from the thermal model,
heat flux / eta_heat, so the pump profile and the multi-pass absorption are
the same as in the heat load), N_col(x, y) = integral of the local Yb density
over the thickness (3-D doping field included), tau = 0.95 ms the upper-state
lifetime, sigma_e = 2.1e-24 m^2 the emission and sigma_a the re-absorption
cross-section at 1030 nm. Yb:YAG is quasi-three-level: the lower laser level
(612 cm^-1) is thermally populated, so sigma_a grows with temperature,
    sigma_a(T) = sigma_a(300 K) exp(880 K (1/300 K - 1/T)),
and the disk only amplifies where the inversion beta = N2 / N_col exceeds
beta_trans = sigma_a / (sigma_a + sigma_e) (~5 % at room temperature).

Saturation by the seed is neglected (0.1 W on 1.7 mm is ~1 W/cm^2 against a
saturation intensity of ~9 kW/cm^2). The resonant phase of the gain
(population lens) is not included.
"""
import numpy as np

H_PLANCK, C_LIGHT = 6.626e-34, 2.998e8
LAM_PUMP, LAM_LASER = 969e-9, 1030e-9
TAU = 0.95e-3                      # upper-state lifetime [s]
SIGMA_EM = 2.1e-24                 # emission cross-section at 1030 nm [m^2]
SIGMA_ABS_300 = 1.2e-25            # re-absorption cross-section at 1030 nm, 300 K [m^2]
N_SITES = 1.38e28                  # Y sites per m^3


def sigma_abs(T):
    """Re-absorption cross-section at 1030 nm vs temperature [K] (Boltzmann, 612 cm^-1)."""
    return SIGMA_ABS_300 * np.exp(880.0 * (1 / 300.0 - 1 / np.asarray(T, float)))


def gain_map(p_abs, n_col, T=300.0):
    """Power gain per reflection G(x, y) and the inversion beta(x, y).

    p_abs : absorbed pump power per area [W/m^2]
    n_col : Yb ions per area (density integrated over the thickness) [m^-2]
    T     : temperature map or scalar [K]
    """
    n2 = p_abs * TAU / (H_PLANCK * C_LIGHT / LAM_PUMP)
    n2 = np.minimum(n2, 0.5 * n_col)                       # pump bleaching limit (never reached here)
    beta = n2 / np.maximum(n_col, 1e-30)
    g = 2 * (SIGMA_EM * n2 - sigma_abs(T) * (n_col - n2))
    return np.exp(g), beta


def transparency_intensity(T=300.0, thickness=0.5e-3, doping=0.10, eta_abs=0.9):
    """Incident pump intensity [W/m^2] at which the disk becomes transparent at 1030 nm."""
    n_col = N_SITES * doping * thickness
    beta_t = sigma_abs(T) / (sigma_abs(T) + SIGMA_EM)
    return beta_t * n_col * (H_PLANCK * C_LIGHT / LAM_PUMP) / TAU / eta_abs


if __name__ == "__main__":
    from thermal import DiskThermal
    from seed_beam import Grid
    print(f"transparency pump intensity at 300 K: {transparency_intensity()/1e7:.2f} kW/cm^2, "
          f"at 400 K: {transparency_intensity(400)/1e7:.2f} kW/cm^2")
    g = Grid(n=256, size=12.8e-3)
    print(f"{'pump radius':>12s} {'I_pump':>10s} {'T centre':>9s} {'beta':>6s} {'G/reflection':>13s} {'10 passes':>10s}")
    for r in (2.0e-3, 2.5e-3, 3.0e-3, 4.0e-3, 6.0e-3):
        th = DiskThermal(pump_power=300, doping=0.10, pump_radius=r)
        st = th.solve(g)
        G, beta = gain_map(st["Q"] / th.eta_heat, N_SITES * th.doping * th.thickness, st["T_mean"])
        c = g.n // 2
        I = 300 / (np.pi * r ** 2)
        print(f"{r*1e3:10.1f} mm {I/1e7:7.2f} kW/cm2 {st['T_mean'][c, c]:7.0f} K {beta[c, c]:6.3f} {G[c, c]:13.3f} {G[c, c]**10:10.3f}")
