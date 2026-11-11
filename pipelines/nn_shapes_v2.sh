#!/bin/bash
# Step 23, second run: 9 shape families, width-128 network.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=wide PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../Results/23_nn_shapes; LOG=$R/pipeline.log
rm -rf $R/smoke2 $R/bank_smoke.npz $R/eval_smoke2.csv $R/NN_SHAPES_smoke2.md $R/shapes_smoke2.png
echo "$(date) extending the bank (v2)" >> $LOG
$PY nn_shapes.py extend > $R/bank_v2.log 2>&1 || { echo "$(date) v2 bank FAILED" >> $LOG; exit 1; }
echo "$(date) training shapes_v2" >> $LOG
$PY nn_shapes.py train shapes_v2 --bank bank_v2.npz --width 128 --epochs 16 --steps 3000 > $R/train_shapes_v2.log 2>&1 || { echo "$(date) v2 train FAILED" >> $LOG; exit 1; }
echo "$(date) evaluating shapes_v2" >> $LOG
$PY nn_shapes.py eval shapes_v2 > $R/eval_shapes_v2.log 2>&1 || { echo "$(date) v2 eval FAILED" >> $LOG; exit 1; }
echo "$(date) V2 ALL DONE" >> $LOG
