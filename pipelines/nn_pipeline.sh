#!/bin/bash
# Overnight pipeline on the GPU PC: wait for the 4 dataset workers, train the residual ResNet
# (all inputs), evaluate it, train the cameras-only variant, evaluate both.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
LOG=../Results/18_nn/pipeline.log
mkdir -p ../Results/18_nn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date) waiting for the dataset" >> $LOG
until [ "$(grep -l 'done in' ../Results/17_nn_dataset/logs/worker_*.log 2>/dev/null | wc -l)" -ge 4 ]; do sleep 120; done
echo "$(date) dataset complete: $(ls ../Results/17_nn_dataset/shards | wc -l) shards" >> $LOG
$PY nn_train.py resnet_all --arch resnet --formulation residual --inputs all > ../Results/18_nn/train_resnet_all.log 2>&1
echo "$(date) resnet_all trained" >> $LOG
$PY nn_eval.py resnet_all --n-test 100 > ../Results/18_nn/eval_resnet_all.log 2>&1
echo "$(date) resnet_all evaluated" >> $LOG
$PY nn_train.py resnet_cams --arch resnet --formulation residual --inputs cams > ../Results/18_nn/train_resnet_cams.log 2>&1
echo "$(date) resnet_cams trained" >> $LOG
$PY nn_eval.py resnet_all resnet_cams --n-test 100 > ../Results/18_nn/eval_both.log 2>&1
echo "$(date) ALL DONE" >> $LOG
