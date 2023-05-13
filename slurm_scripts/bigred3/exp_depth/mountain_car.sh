#!/bin/bash

#SBATCH -J exp-depth-mountain-car
#SBATCH -p general
#SBATCH -o exp-depth-mountain-car_%A_%a.out
#SBATCH -e exp-depth-mountain-car_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=0-05:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-10

env=cmc
n_sample=200
depths=(20 50 75 80 90 100 110 120 130 140 150)
depth=${depths[$SLURM_ARRAY_TASK_ID]}

today=$(date +'%m-%d-%Y')

if [ $noisy == "True" ]; then
    alpha_val=0.002
else
    alpha_val=0
fi

run_name=${base_run_name}-${alg}-${depth}

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
 

# To evaluate model
PYTHONPATH=. python run_gym.py --env=${env} --run_name=${run_name} --alg=${alg}  --${option_name}=${n_sample} --n_episodes=48 --depth=${depth} --alpha=${alpha_val} --render=False 

end_time=`(date +%s)`
echo End time: ${end_time}
