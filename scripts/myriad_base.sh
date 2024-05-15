#$ -l mem=10G
#$ -l h_rt=12:00:00
#$ -R y
#$ -S /bin/bash
#$ -wd /home/ucabzc9/Scratch/
#$ -j y
#$ -N chard

JOB_PARAMS=$(sed "${SGE_TASK_ID}q;d" "$1")
echo "Job params: $JOB_PARAMS"

contains_param() {
    [[ $1 =~ $2 ]]
}

# Running date and nvidia-smi is useful to get some info in case the job crashes.

#module -f unload compilers mpi gcc-libs
#module load beta-modules
#module load gcc-libs/10.2.0
#module load cuda/11.2.0/gnu-10.2.0

## Load conda
module -f unload compilers
module load beta-modules
module load gcc-libs/10.2.0
module load compilers/gnu/4.9.2
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

if contains_param "$JOB_PARAMS" "--dataset ThreeRing"; then
    echo "Activating environment 'chard'"
    conda activate /lustre/home/ucabzc9/.conda/envs/chard
    ## Print out the date and nivida-smi for debugging
    date
    ## Check if the environment is correct.
    which pip
    which python3
    pwd
    python3 /home/ucabzc9/Scratch/chard/chard/run.py $JOB_PARAMS
elif contains_param "$JOB_PARAMS" "--device 0"; then
    echo "Activating environment 'mmd_flow'"
    conda activate /lustre/home/ucabzc9/.conda/envs/mmd_flow
    ## Print out the date and nivida-smi for debugging
    date
    ## Check if the environment is correct.
    which pip
    which python3
    pwd
    python3 /home/ucabzc9/Scratch/chard/student_teacher/train.py $JOB_PARAMS
else
    echo "No specific environment required for these job parameters"
    # Activate a default environment or handle this case as needed
fi
