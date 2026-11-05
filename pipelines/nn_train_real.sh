#!/bin/bash
# Training + evaluation part of the Step 20 pipeline (the dataset already exists).
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=real PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../Results/20_nn_real; LOG=$R/pipeline.log
echo "$(date) training (restart after the out-of-memory kill)" >> $LOG
$PY nn_train.py resnet_all --inputs all --epochs-sup 6 --epochs-phys 12 > $R/train_resnet_all.log 2>&1
echo "$(date) resnet_all trained" >> $LOG
$PY nn_eval.py resnet_all --n-test 100 > $R/eval_resnet_all.log 2>&1
echo "$(date) resnet_all evaluated" >> $LOG
$PY nn_train.py resnet_cams --inputs cams --epochs-sup 6 --epochs-phys 12 > $R/train_resnet_cams.log 2>&1
echo "$(date) resnet_cams trained" >> $LOG
$PY nn_eval.py resnet_all resnet_cams --n-test 100 > $R/eval_both.log 2>&1
echo "$(date) ALL DONE" >> $LOG
