#!/bin/bash
# Step 21b: wider ResNet (3.6 M) with the setup's reference images as input, no label term in the
# physics phase, longer training; then evaluate next to the first per-setup network.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=setup PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../Results/21_nn_setup; LOG=$R/pipeline.log
echo "$(date) training resnet_v2 (width 128, reference images, no label term)" >> $LOG
$PY nn_train.py resnet_v2 --inputs all --width 128 --ref-images --sup-weight 0 --epochs-sup 3 --epochs-phys 14 > $R/train_resnet_v2.log 2>&1
echo "$(date) resnet_v2 trained" >> $LOG
cp $R/NN_EVAL.md $R/NN_EVAL_resnet_ft.md
$PY nn_eval.py resnet_ft resnet_v2 --n-test 100 > $R/eval_v2.log 2>&1
echo "$(date) V2 DONE" >> $LOG
