#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1,2
#SBATCH --cpus-per-task=10
#SBATCH -t 05:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=dbg
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040
###############################################################

# load conda environment
source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.10

# check if array task id is 1
if [ $SLURM_ARRAY_TASK_ID -eq 1 ] ; then
    DATA_DIR=/home/hernandezj45/Repos/tpot2-debugging/Results/Run1/
    SPLIT_SELECT=0.95
    SCHEME=lexicase
    SEED=10063
    TASK_ID=359954
    TASK_TYPE=1

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ] ; then
    # Define the output directory
    DATA_DIR=/home/hernandezj45/Repos/tpot2-debugging/Results/Run2/
    SPLIT_SELECT=0.95
    SCHEME=lexicase
    SEED=10087
    TASK_ID=359955
    TASK_TYPE=1
fi

# let it rip
python /home/hernandezj45/Repos/tpot2-debugging/Source/experiment.py \
-split_select ${SPLIT_SELECT} \
-scheme ${SCHEME} \
-task_id ${TASK_ID} \
-n_jobs 10 \
-savepath ${DATA_DIR} \
-seed ${SEED} \
-task_type ${TASK_TYPE} \