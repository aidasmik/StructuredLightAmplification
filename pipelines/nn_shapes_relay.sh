#!/bin/bash
# Step 23, fourth run: SLM imaged onto the disk, power-aware design and loss, width 64, 9 shape families.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=wide PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../Results/23_nn_shapes; LOG=$R/pipeline.log
echo "$(date) relay: bank" >> $LOG
$PY nn_shapes.py bank --relay > $R/bank_relay.log 2>&1 || { echo "$(date) relay bank FAILED" >> $LOG; exit 1; }
$PY nn_shapes.py extend --relay > $R/bank_v2_relay.log 2>&1 || { echo "$(date) relay extend FAILED" >> $LOG; exit 1; }
echo "$(date) relay: training shapes_relay" >> $LOG
$PY nn_shapes.py train shapes_relay --relay --bank bank_v2_relay.npz --width 64 --epochs 14 --steps 3000 > $R/train_shapes_relay.log 2>&1 || { echo "$(date) relay train FAILED" >> $LOG; exit 1; }
echo "$(date) relay: evaluating" >> $LOG
$PY nn_shapes.py eval shapes_relay > $R/eval_shapes_relay.log 2>&1 || { echo "$(date) relay eval FAILED" >> $LOG; exit 1; }
echo "$(date) RELAY ALL DONE" >> $LOG
