#!/bin/bash
# Wide-range pipeline on the GPU PC (Step 19): 20 000 examples with 6 workers, train the residual
# ResNet with all inputs, evaluate, train the cameras-only variant, evaluate both.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=wide PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=../Results/19_nn_wide_dataset; R=../Results/19_nn_wide; LOG=$R/pipeline.log
mkdir -p $R $D/logs
echo "$(date) generating 20000 examples" >> $LOG
for w in 0 1 2 3 4 5; do $PY nn_dataset.py 20000 $w 6 > $D/logs/worker_$w.log 2>&1 & done; wait
echo "$(date) dataset complete: $(ls $D/shards | wc -l) shards" >> $LOG
$PY nn_train.py resnet_all --inputs all --epochs-sup 6 --epochs-phys 12 > $R/train_resnet_all.log 2>&1
echo "$(date) resnet_all trained" >> $LOG
$PY nn_eval.py resnet_all --n-test 100 > $R/eval_resnet_all.log 2>&1
echo "$(date) resnet_all evaluated" >> $LOG
$PY nn_train.py resnet_cams --inputs cams --epochs-sup 6 --epochs-phys 12 > $R/train_resnet_cams.log 2>&1
echo "$(date) resnet_cams trained" >> $LOG
$PY nn_eval.py resnet_all resnet_cams --n-test 100 > $R/eval_both.log 2>&1
echo "$(date) ALL DONE" >> $LOG
