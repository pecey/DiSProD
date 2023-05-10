#!/bin/bash

#SBATCH -J exp-noise-mountain-car
#SBATCH -p general
#SBATCH -o exp-noise-mountain-car_%A_%a.out
#SBATCH -e exp-noise-mountain-car_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=0-05:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-7

env=cmc
n_sample=200
depth=100
alphas=(0.0 0.0025 0.005 0.0075 0.009 0.01 0.02 0.05)
alpha_val=${alphas[$SLURM_ARRAY_TASK_ID]}

run_name=${base_run_name}-${alg}-${alpha_val}

module load python/3.9.8

if [ $alg == "disprod" ]
then
    option_name="n_restarts"
else
    option_name="n_samples"
fi

start_time=`(date +%s)`
echo Start time: ${start_time}

source ${HOME}/deeprl_py3.9/bin/activate
cd ${HOME}/disprod
export DISPROD_PATH=${HOME}/disprod

# To evaluate model
PYTHONPATH=. python run_gym.py --env=${env} --run_name=${run_name} --alg=${alg}  --${option_name}=${n_sample} --n_episodes=48 --depth=${depth} --alpha=${alpha_val} --render=False 

end_time=`(date +%s)`
echo End time: ${end_time}
