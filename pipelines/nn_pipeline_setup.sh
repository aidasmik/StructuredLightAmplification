#!/bin/bash
# Step 21 on the GPU PC: nuisances fixed per setup (64 setups, one reference each) + drift.
# Dataset now (CPU), training once the follow-up evaluations have freed the GPU.
cd ~/ThinDiskLaser/Simulation/Simulation_Code
PY=~/.venvs/thindisk/bin/python
export NN_RANGE=setup PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=../Results/21_nn_setup_dataset; R=../Results/21_nn_setup; LOG=$R/pipeline.log
mkdir -p $R $D/logs
echo "$(date) references for 64 setups" >> $LOG
$PY nn_dataset.py reference > $D/logs/reference.log 2>&1
echo "$(date) generating 20000 examples" >> $LOG
for w in 0 1 2 3 4 5; do $PY nn_dataset.py 20000 $w 6 > $D/logs/worker_$w.log 2>&1 & done; wait
echo "$(date) dataset complete: $(ls $D/shards | wc -l) shards" >> $LOG
until grep -q "FOLLOWUP DONE" ../Results/20_nn_real/pipeline.log; do sleep 300; done
$PY nn_train.py resnet_ft --inputs all --epochs-sup 0 --epochs-phys 12 --lr 3e-4 --init ../Results/19_nn_wide/resnet_all/model.pt > $R/train_resnet_ft.log 2>&1
echo "$(date) resnet_ft trained" >> $LOG
$PY nn_eval.py resnet_ft --n-test 100 > $R/eval_resnet_ft.log 2>&1
echo "$(date) ALL DONE" >> $LOG
