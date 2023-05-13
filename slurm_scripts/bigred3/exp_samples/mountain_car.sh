#!/bin/bash

#SBATCH -J exp-samples-mountain-car
#SBATCH -p general
#SBATCH -o exp-samples-mountain-car_%A_%a.out
#SBATCH -e exp-samples-mountain-car_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=0-05:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-10

env=cmc
depth=100
n_samples=(10 25 50 75 100 125 150 175 200 225 250)
n_sample=${n_samples[$SLURM_ARRAY_TASK_ID]}

today=$(date +'%m-%d-%Y')

if [ $noisy == "True" ]; then
    alpha_val=0.002
else
    alpha_val=0
fi

run_name=${base_run_name}-${alg}-${n_sample}

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
cd ${DISPROD_PATH}
 

PYTHONPATH=. python run_gym.py --env=${env} --run_name=${run_name} --${option_name}=${n_sample} --n_episodes=48 --alg=$alg --depth=${depth} --alpha=${alpha_val} --render=False 

end_time=`(date +%s)`
echo End time: ${end_time}
