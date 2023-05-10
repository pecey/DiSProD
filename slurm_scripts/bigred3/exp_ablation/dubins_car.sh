#!/bin/bash

#SBATCH -J ablation-dubins-car
#SBATCH -p general
#SBATCH -o ablation-dubins-car_%A_%a.out
#SBATCH -e ablation-dubins-car_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=0-05:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-14

env=cdc_ab
n_sample=200
alphas=(0 0.001 0.005 0.01 0.05 0.1)
alpha_val=${alphas[$SLURM_ARRAY_TASK_ID]}

today=$(date +'%m-%d-%Y')
run_name=${base_run_name}-${mode}-${alpha_val}

module load python/3.9.8

start_time=`(date +%s)`
echo Start time: ${start_time}

source ${HOME}/deeprl_py3.9/bin/activate
cd ${HOME}/disprod
export DISPROD_PATH=${HOME}/disprod

PYTHONPATH=. python run_gym.py --env ${env} --run_name ${run_name} --n_restarts=${n_sample} --n_episodes=10  --alpha=${alpha_val} --render=False --taylor_expansion_mode=${mode} --alg=disprod --map_name=cave-mini --step_size=0.2

end_time=`(date +%s)`
echo End time: ${end_time}
