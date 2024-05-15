#!/bin/bash
source activate mmd_flow
which python

# python train_student_teacher.py --device 0 --loss mmd_noise_injection --lr 0.1 --with_noise true --seed 1
# python train_student_teacher.py --device 0 --loss mmd_noise_injection --lr 0.1 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.06 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.1 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.03 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.01 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 1.0 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.001 --seed 1

# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.06 --with_noise true --noise_decay_freq 1000 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.1 --with_noise true --noise_decay_freq 1000 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.03 --with_noise true --noise_decay_freq 1000 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.01 --with_noise true --noise_decay_freq 1000 --seed 1

# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.06 --with_noise true --noise_decay_freq 500 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.1 --with_noise true --noise_decay_freq 500 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.03 --with_noise true --noise_decay_freq 500 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.01 --with_noise true --noise_decay_freq 500 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.001 --with_noise true --noise_decay_freq 500 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 1.0 --with_noise true --noise_decay_freq 500 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 10.0 --with_noise true --noise_decay_freq 500 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 100.0 --with_noise true --noise_decay_freq 500 --seed 1


# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.06 --with_noise true --noise_decay_freq 200 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.1 --with_noise true --noise_decay_freq 200 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.03 --with_noise true --noise_decay_freq 200 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.01 --with_noise true --noise_decay_freq 200 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 0.001 --with_noise true --noise_decay_freq 200 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 1.0 --with_noise true --noise_decay_freq 200 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 10.0 --with_noise true --noise_decay_freq 200 --seed 1
# python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 100.0 --with_noise true --noise_decay_freq 200 --seed 1


python /home/zongchen/chard/student_teacher/train.py --device 0 --loss mmd_noise_injection --lr 0.1 --lmbda 10.0 --noise_decay_freq 1000 --seed 1
python train_student_teacher.py --device 0 --loss chard --lr 0.1 --lmbda 100.0 --noise_decay_freq 1000 --seed 1
