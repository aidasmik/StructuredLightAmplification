#!/bin/bash
cd ~/ThinDiskLaser/Simulation/Simulation_Code
R=../Results/23_nn_shapes
NN_RANGE=wide ~/.venvs/thindisk/bin/python nn_shapes_iter.py > $R/iter_free.log 2>&1 && echo "$(date) iter free done" >> $R/pipeline.log || echo "$(date) iter free FAILED" >> $R/pipeline.log
NN_RANGE=wide ~/.venvs/thindisk/bin/python nn_shapes_iter.py --relay > $R/iter_relay.log 2>&1 && echo "$(date) ITER ALL DONE" >> $R/pipeline.log || echo "$(date) iter relay FAILED (ITER END)" >> $R/pipeline.log
