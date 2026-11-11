#!/bin/bash
# Step 23, sixth run: ONE camera (far field), layout as modelled, intensity score (phase not rated), width 64.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=wide PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../Results/23_nn_shapes; LOG=$R/pipeline.log
echo "$(date) onecam: smoke" >> $LOG
$PY nn_shapes.py train smoke1 --one-camera --bank bank_v2.npz --epochs 1 --steps 30 > $R/smoke1_train.log 2>&1 && $PY nn_shapes.py eval smoke1 --ceiling-steps 2 > $R/smoke1_eval.log 2>&1 || { echo "$(date) onecam smoke FAILED" >> $LOG; exit 1; }
echo "$(date) onecam: training shapes_onecam" >> $LOG
$PY nn_shapes.py train shapes_onecam --one-camera --bank bank_v2.npz --width 64 --epochs 14 --steps 3000 > $R/train_shapes_onecam.log 2>&1 || { echo "$(date) onecam train FAILED" >> $LOG; exit 1; }
$PY nn_shapes.py eval shapes_onecam > $R/eval_shapes_onecam.log 2>&1 || { echo "$(date) onecam eval FAILED" >> $LOG; exit 1; }
echo "$(date) ONECAM ALL DONE" >> $LOG
