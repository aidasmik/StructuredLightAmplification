#!/bin/bash
# Step 23 pipeline on the GPU PC: hologram bank, training, evaluation of the shape network.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=wide PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../Results/23_nn_shapes; LOG=$R/pipeline.log
rm -rf $R/smoke $R/eval_smoke.csv $R/NN_SHAPES_smoke.md $R/shapes_smoke.png $R/bank.npz
echo "$(date) bank" >> $LOG
$PY nn_shapes.py bank > $R/bank.log 2>&1 || { echo "$(date) bank FAILED" >> $LOG; exit 1; }
echo "$(date) training shapes_v1" >> $LOG
$PY nn_shapes.py train shapes_v1 --epochs 14 --steps 3000 > $R/train_shapes_v1.log 2>&1 || { echo "$(date) train FAILED" >> $LOG; exit 1; }
echo "$(date) evaluating" >> $LOG
$PY nn_shapes.py eval shapes_v1 > $R/eval_shapes_v1.log 2>&1 || { echo "$(date) eval FAILED" >> $LOG; exit 1; }
echo "$(date) ALL DONE" >> $LOG
