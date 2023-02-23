#!/bin/bash

#SBATCH -J exp-reward-sparsity-mountain-car
#SBATCH -p general
#SBATCH -o exp-reward-sparsity-mountain-car_%A_%a.out
#SBATCH -e exp-reward-sparsity-mountain-car_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=palchatt@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-6

env_name=continuous_mountain_car
n_sample=200
depth=100
alpha_val=0

sparsity_values=(0.1 0.2 0.5 1 2 3 4)
sparsity_value=${sparsity_values[$SLURM_ARRAY_TASK_ID]}

run_name=${base_run_name}-${alg}--${depth}-${sparsity_value}

module load python/3.9.8

if [ $alg == "sogbofa" ]
then
    option_name="n_restarts"
else
    option_name="n_samples"
fi

start_time=`(date +%s)`
echo Start time: ${start_time}

source ${HOME}/deeprl_py3.9/bin/activate
cd ${HOME}/awesome-sogbofa
export DISPROD_PATH=${HOME}/awesome-sogbofa

# To evaluate model
PYTHONPATH=. python run_gym.py --env_name ${env_name} --run_name ${run_name} --${option_name}=${n_sample} --n_episodes=10 --alg=$alg --depth=${depth} --alpha=${alpha_val} --render=False --reward_sparsity=${sparsity_value}

end_time=`(date +%s)`
echo End time: ${end_time}
