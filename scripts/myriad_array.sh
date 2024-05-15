#!/bin/bash

# bash /home/ucabzc9/Scratch/chard/scripts/myriad_array.sh /home/ucabzc9/Scratch/chard/scripts/configs/three_ring_config.txt 

jobs_in_parallel=$(wc -l < "$1")
echo $jobs_in_parallel
echo $1

qsub -t 1-${jobs_in_parallel} /home/ucabzc9/Scratch/chard/scripts/myriad_base.sh "$1"