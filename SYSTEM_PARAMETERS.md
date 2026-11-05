# System parameters

Values taken from the project presentation (`Literature/NSYSU_FTMC_discussion-1.pdf`,
Taiwan-Lithuania Laser Project Meeting, Vilnius, 19 June 2026) and the assumptions that fill the gaps.

| parameter | value in the model | source |
|---|---|---|
| seed laser | Yb:YAG CW oscillator, 1030 nm, grating tuned | slide 9 |
| seed power | 0.1 W (`SEED_POWER`) | slides 10, 15: 0.1 W fundamental, 0.2 W vortex |
| seed / target beam | collimated Gaussian, 1.0 mm radius (`W0`), 1.1 mm on the disk | user (slide 11 says 1.5 mm radius, which would clip the 5 mm disk) |
| number of disk passes | 10 (`N_PASSES`), 20 array mirrors | slide 10: images after 1, 2, 5, 10 passes |
| amplifier topology | mirror array + end mirrors M1/M2 (PSI/ETH routing) | slide 9 layout |
| imaging design | −I relay per unit (`DESIGN = "relay"`): array mirrors R = 1.0 m, M1/M2 R = −0.52 m | assumption: the only design that carries a 1.5 mm beam with practical radii in the drawn geometry and reproduces the seed at the output |
| array arc radius / M distance | 1.0 m / 0.5 m from the disk | assumption from the drawing proportions |
| pump | 300 W at 969 nm, fibre-coupled diode, multi-pass pump module | slides 9, 10 |
| pump modulation | ~50 Hz, ~40 % duty in the amplification traces | slide 10 (not yet used in the model) |
| pump spot | the whole disk: 2.5 mm radius, flat top (300 W evenly over the disk) | user |
| disk | Yb:YAG, 0.5 mm thick, 5 mm diameter, flat, AR front / HR back | confirmed by the user |
| doping | 10 at.% nominal (confirmed by the user), ±2 % clustered non-uniformity (assumption) | user / assumption |
| heat sink | diamond, 1.5 mm | assumption |
| SLM | Hamamatsu X15213 series LCOS: 12.5 µm pitch, 1272 × 1024 px, fill factor 0.96, R = 0.97, 8 bit | slide 4 |
| SLM position | 0.3 m from the beam splitter in the SLM arm | drawing proportions |
| diagnostics | CCD in the SLM arm, power meter and photodiode at the output | slide 9 |
| control used so far | SPGD on a correlation metric with Zernike parametrisation, ~100–300 iterations | slides 7, 11–13 |
| measured amplification | ×2.5 fundamental, ×1.5 vortex after 10 passes; strong astigmatic distortion | slide 10 |

Default simulation grid: `SIM_GRID = n=512, 12.8 mm` (dx = 25 µm). Checked against the 12.5 µm grid: output
fidelity >= 0.998 for Gaussian, vortex and aberrated patterns, 6x faster. `FINE_GRID = n=1024, 12.8 mm`
(dx = 12.5 µm = one SLM pixel) is used by the correction and limits benchmarks (blazed carrier, Davis encoding).
