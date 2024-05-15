#!/bin/bash
 source activate chard
 which python
python chard/run.py --dataset ThreeRing --flow mmd --kernel Gaussian --step_size 0.01 --bandwidth 0.3 --step_num 500000 --source_particle_num 300 --opt sgd --seed 42
