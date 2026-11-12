# GPU pipelines

Shell scripts that ran the long jobs on the GPU PC (RTX 2080 SUPER, Ubuntu). They expect the
project at `~/ThinDiskLaser/Simulation` and a Python environment at `~/.venvs/thindisk`
(numpy, scipy, matplotlib, torch with CUDA); adjust `PY` and the `cd` line for another machine.
Start them detached, e.g. `setsid nohup ./nn_pipeline_setup.sh > /dev/null 2>&1 < /dev/null &`.

| script | step | does |
|---|---|---|
| `nn_pipeline.sh` | 17/18 | narrow-range dataset, residual ResNet with all inputs and cameras only, evaluation |
| `nn_pipeline_wide.sh` | 19 | wide range (240-360 W): 20 000 examples, both networks, evaluation |
| `nn_pipeline_real.sh` | 20 | wide range + gain + nuisances drawn per example |
| `nn_train_real.sh` | 20 | training part again after an out-of-memory kill (memory-mapped loader since) |
| `nn_followup.sh` | 20 | fine-tune from the wide network; re-evaluation of Steps 18/19 with the fixed propagation |
| `nn_pipeline_setup.sh` | 21 | nuisances fixed per setup (64 setups, one reference each) + drift |
| `nn_v2.sh` | 21 | wider ResNet (3.6 M) with the setup's reference images as input, no label term |
| `nn_shapes_pipeline.sh`, `nn_shapes_v2.sh`, `nn_shapes_v3.sh` | 23 | any-shape network trained on the fidelity alone (superseded) |
| `nn_shapes_relay.sh`, `nn_shapes_free.sh` | 23 | power-aware banks and two-camera networks, SLM imaged onto the disk / 2.9 m before it |
| `nn_shapes_onecam.sh` | 23 | one-camera network (far field only) |
| `nn_shapes_iter.sh` | 24 | networks against the published SPGD, both layouts |

A dataset of 20 000 examples takes ~6-8 h with 6 workers; a training run 6-15 h.
