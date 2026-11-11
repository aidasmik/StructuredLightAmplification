#!/bin/bash
# Step 23, third run: network output at the disk plane (psi), width 64, 9 shape families.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=wide PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../Results/23_nn_shapes; LOG=$R/pipeline.log
rm -rf $R/smoke3 $R/smoke3.log
echo "$(date) training shapes_v3 (disk plane)" >> $LOG
$PY nn_shapes.py train shapes_v3 --bank bank_v2.npz --plane disk --width 64 --epochs 14 --steps 3000 > $R/train_shapes_v3.log 2>&1 || { echo "$(date) v3 train FAILED" >> $LOG; exit 1; }
echo "$(date) evaluating shapes_v3" >> $LOG
$PY nn_shapes.py eval shapes_v3 > $R/eval_shapes_v3.log 2>&1 || { echo "$(date) v3 eval FAILED" >> $LOG; exit 1; }
echo "$(date) V3 ALL DONE" >> $LOG
