# SPGD as in Xie et al. 2015 (Step 22)

Disk state of Steps 14/16 (300 W, hot spots, nominal shape error), far-field camera, 14 Zernike terms,
one-sided update a += eta s (C2 - C1), two frames per iteration. C = the loop's own feedback signal
(paper: far-field intensity overlap with the ideal mode, normalised; pearson: image correlation of the
slides). Fidelity = true field fidelity of the displayed pattern (mean of the last 5 iterations).

| target | metric | delta | eta | iterations | frames | C start -> end | fidelity start -> end (best) |
|---|---|---|---|---|---|---|---|
| gaussian | paper | 0.01 | 3 | 100 | 200 | 0.444 -> 0.753 | 0.556 -> 0.885 (0.885) |
| gaussian | paper | 0.01 | 10 | 100 | 200 | 0.443 -> 0.765 | 0.556 -> 0.866 (0.869) |
| gaussian | paper | 0.01 | 30 | 100 | 200 | 0.443 -> 0.766 | 0.556 -> 0.872 (0.876) |
| gaussian | paper | 0.05 | 3 | 100 | 200 | 0.442 -> 0.764 | 0.556 -> 0.869 (0.876) |
| gaussian | paper | 0.05 | 10 | 100 | 200 | 0.444 -> 0.750 | 0.556 -> 0.837 (0.867) |
| gaussian | paper | 0.05 | 30 | 100 | 200 | 0.443 -> 0.393 | 0.556 -> 0.422 (0.618) |
| gaussian | paper | 0.15 | 3 | 100 | 200 | 0.441 -> 0.720 | 0.556 -> 0.776 (0.852) |
| gaussian | paper | 0.15 | 10 | 100 | 200 | 0.443 -> 0.403 | 0.556 -> 0.418 (0.675) |
| gaussian | paper | 0.15 | 30 | 100 | 200 | 0.442 -> 0.185 | 0.556 -> 0.290 (0.684) |
| gaussian | paper | 0.01 | 3 | 300 | 600 | 0.442 -> 0.763 | 0.556 -> 0.869 (0.883) |
| gaussian | pearson | 0.01 | 3 | 300 | 600 | 0.820 -> 0.953 | 0.556 -> 0.882 (0.882) |
| LG01 | paper | 0.01 | 3 | 100 | 200 | 0.400 -> 0.632 | 0.421 -> 0.708 (0.708) |
| LG01 | paper | 0.01 | 10 | 100 | 200 | 0.400 -> 0.641 | 0.421 -> 0.703 (0.705) |
| LG01 | paper | 0.01 | 30 | 100 | 200 | 0.400 -> 0.641 | 0.421 -> 0.703 (0.706) |
| LG01 | paper | 0.05 | 3 | 100 | 200 | 0.400 -> 0.639 | 0.421 -> 0.702 (0.706) |
| LG01 | paper | 0.05 | 10 | 100 | 200 | 0.400 -> 0.613 | 0.421 -> 0.650 (0.696) |
| LG01 | paper | 0.05 | 30 | 100 | 200 | 0.400 -> 0.391 | 0.421 -> 0.120 (0.526) |
| LG01 | paper | 0.15 | 3 | 100 | 200 | 0.399 -> 0.580 | 0.421 -> 0.563 (0.669) |
| LG01 | paper | 0.15 | 10 | 100 | 200 | 0.401 -> 0.308 | 0.421 -> 0.042 (0.523) |
| LG01 | paper | 0.15 | 30 | 100 | 200 | 0.399 -> 0.187 | 0.421 -> 0.054 (0.421) |
| LG01 | paper | 0.01 | 3 | 300 | 600 | 0.399 -> 0.641 | 0.421 -> 0.708 (0.709) |
| LG01 | pearson | 0.01 | 3 | 300 | 600 | 0.711 -> 0.860 | 0.421 -> 0.658 (0.680) |

Same state, other methods (Results/readme/iterations.csv):

| target | SPGD slides (Step 14) | response matrix (Step 16) | network, 2 frames | ceiling |
|---|---|---|---|---|
| gaussian | 0.174 (377 frames) | 0.825 (240 frames) | 0.964 | 0.943 |
| LG01 | 0.016 (380 frames) | 0.667 (238 frames) | 0.898 | 0.792 |
