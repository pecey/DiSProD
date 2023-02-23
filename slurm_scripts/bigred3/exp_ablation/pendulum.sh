#!/bin/bash

#SBATCH -J ablation-pendulum
#SBATCH -p general
#SBATCH -o ablation-pendulum_%A_%a.out
#SBATCH -e ablation-pendulum_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=palchatt@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=0-05:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-7

env_name=pendulum
n_sample=200
alphas=(0.0 0.5 0.75 1 2 3 4 5)
alpha_val=${alphas[$SLURM_ARRAY_TASK_ID]}

today=$(date +'%m-%d-%Y')
run_name=${base_run_name}-${mode}-${alpha_val}

module load python/3.9.8

start_time=`(date +%s)`
echo Start time: ${start_time}

source ${HOME}/deeprl_py3.9/bin/activate
cd ${HOME}/awesome-sogbofa
export DISPROD_PATH=${HOME}/awesome-sogbofa

PYTHONPATH=. python run_gym.py --env_name ${env_name} --run_name ${run_name} --n_restarts=${n_sample} --n_episodes=10  --alpha=${alpha_val} --render=False --taylor_expansion_mode=${mode} --alg=sogbofa

end_time=`(date +%s)`
echo End time: ${end_time}
