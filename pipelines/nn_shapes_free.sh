#!/bin/bash
# Step 23, fifth run: layout as modelled (SLM 2.91 m of free space before the disk), power-aware design
# and loss, width 64, 9 shape families. Queued: starts when the imaged-layout run (nn_shapes_relay.sh) ends.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=wide PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../Results/23_nn_shapes; LOG=$R/pipeline.log
until tail -1 $LOG | grep -q "RELAY ALL DONE\|relay .*FAILED"; do sleep 60; done
# the old banks were designed for similarity alone (they clip at the disk aperture): keep them under another name
for f in bank bank_v2; do mv $R/$f.npz $R/${f}_fidelity.npz; mv $R/$f.log $R/${f}_fidelity.log; done
echo "$(date) free: bank" >> $LOG
$PY nn_shapes.py bank > $R/bank.log 2>&1 || { echo "$(date) free bank FAILED" >> $LOG; exit 1; }
$PY nn_shapes.py extend > $R/bank_v2.log 2>&1 || { echo "$(date) free extend FAILED" >> $LOG; exit 1; }
echo "$(date) free: training shapes_free" >> $LOG
$PY nn_shapes.py train shapes_free --bank bank_v2.npz --width 64 --epochs 14 --steps 3000 > $R/train_shapes_free.log 2>&1 || { echo "$(date) free train FAILED" >> $LOG; exit 1; }
echo "$(date) free: evaluating" >> $LOG
$PY nn_shapes.py eval shapes_free > $R/eval_shapes_free.log 2>&1 || { echo "$(date) free eval FAILED" >> $LOG; exit 1; }
echo "$(date) FREE ALL DONE" >> $LOG
