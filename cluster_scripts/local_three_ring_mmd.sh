#!/bin/bash
 source activate chard
 which python
python run.py --dataset ThreeRing --flow drmmd --kernel Gaussian --lmbda 0.01 --step_size 0.1 --bandwidth 0.3 --step_num 100000 --source_particle_num 300 --adaptive_lmbda --opt sgd --seed 42
python run.py --dataset ThreeRing --flow drmmd --kernel Gaussian --lmbda 0.1 --step_size 0.1 --bandwidth 0.3 --step_num 100000 --source_particle_num 300 --adaptive_lmbda --opt sgd --seed 43
python run.py --dataset ThreeRing --flow drmmd --kernel Gaussian --lmbda 1.0 --step_size 0.1 --bandwidth 0.3 --step_num 100000 --source_particle_num 300 --adaptive_lmbda --opt sgd --seed 44
# python run.py --dataset ThreeRing --flow drmmd --kernel Gaussian --lmbda 0.01 --step_size 0.1 --bandwidth 0.3 --step_num 100000 --source_particle_num 300 --adaptive_lmbda --opt sgd --seed 45


python run.py --dataset ThreeRing --flow drmmd --kernel Gaussian --lmbda 0.001 --step_size 0.1 --bandwidth 0.15 --step_num 100000 --source_particle_num 300 --opt sgd --seed 46

# python run.py --dataset ThreeRing --flow drmmd --kernel Gaussian --lmbda 0.01 --step_size 0.1 --bandwidth 0.15 --step_num 100000 --source_particle_num 300 --opt sgd --seed 46
