#!/bin/bash
# Follow-up on the GPU PC: (1) fine-tune the wide-range network on the realism set (physics loss only),
# evaluate it next to the network trained from scratch; (2) re-evaluate the narrow and wide ranges with
# the corrected ceiling, the fine-grid scoring and the absorbing boundary.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=../Results/20_nn_real/pipeline.log
echo "$(date) fine-tune from the wide network" >> $LOG
NN_RANGE=real $PY nn_train.py resnet_ft --inputs all --epochs-sup 0 --epochs-phys 12 --lr 3e-4 --init ../Results/19_nn_wide/resnet_all/model.pt > ../Results/20_nn_real/train_resnet_ft.log 2>&1
echo "$(date) resnet_ft trained" >> $LOG
NN_RANGE=real $PY nn_eval.py resnet_all resnet_ft --n-test 100 > ../Results/20_nn_real/eval_ft.log 2>&1
echo "$(date) resnet_ft evaluated" >> $LOG
for r in 19_nn_wide 18_nn; do cp ../Results/$r/NN_EVAL.md ../Results/$r/NN_EVAL_v1_periodic.md; done
NN_RANGE=wide $PY nn_eval.py resnet_all resnet_cams --n-test 100 > ../Results/19_nn_wide/eval_both_v2.log 2>&1
echo "$(date) wide re-evaluated" >> $LOG
NN_RANGE=narrow $PY nn_eval.py resnet_all resnet_cams --n-test 100 > ../Results/18_nn/eval_both_v2.log 2>&1
echo "$(date) FOLLOWUP DONE" >> $LOG
