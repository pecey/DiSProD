#!/bin/bash

#SBATCH -J awesome-sogbofa/half-cheetah-online-learning
#SBATCH -p gpu
#SBATCH -o half-cheetah-online-training_%A_%a.out
#SBATCH -e half-cheetah-online-training_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=palchatt@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node v100:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=16GB

env_name=half_cheetah
today=$(date +'%m-%d-%Y')

module load python/3.9.8

run_name=${today}-online-learning

start_time=`(date +%s)`
echo Start time: ${start_time}

source ${HOME}/deeprl_py3.9/bin/activate
cd ${HOME}/awesome-sogbofa
export DISPROD_PATH=${HOME}/awesome-sogbofa

# To train offline
# PYTHONPATH=. python learn_gym.py --mode=offline_training --env_name=${env_name} --run_name=${run_name} --headless=True --render=False --compute_baseline=True

# To evaluate model
# PYTHONPATH=. python learn_gym.py --mode=evaluate --env_name=${env_name} --run_name=${run_name} --headless=True --render=False --compute_baseline=True

# For online learning
# PYTHONPATH=. python learn_gym.py --mode=online_learning --env_name=${env_name} --run_name=${run_name} --headless=True --render=False --compute_baseline=True

end_time=`(date +%s)`
echo End time: ${end_time}