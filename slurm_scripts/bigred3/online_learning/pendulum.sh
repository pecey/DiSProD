#!/bin/bash

#SBATCH -J pendulum-online-learning
#SBATCH -p general
#SBATCH -o pendulum-online-learning_%A_%a.out
#SBATCH -e pendulum-online-learning_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=palchatt@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=0-7:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-4

env_name=pendulum
today=$(date +'%m-%d-%Y')

module load python/3.9.8

alg=${alg}
run_name=${base_run_name}-${alg}

if [ ${noise} == 'True' ];
then
    alpha=3
else
    alpha=0
fi

seeds=(10 20 30 40 50)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

start_time=`(date +%s)`
echo Start time: ${start_time}

source ${HOME}/deeprl_py3.9/bin/activate
cd ${HOME}/awesome-sogbofa
export DISPROD_PATH=${HOME}/awesome-sogbofa

# For online learning
PYTHONPATH=. python learn_gym.py --mode=online_learning --env_name=${env_name} --run_name=${run_name}_${seed} --alg=${alg} --seed=${seed} --headless=True --render=False --compute_baseline=True --alpha=${alpha}

end_time=`(date +%s)`
echo End time: ${end_time}
