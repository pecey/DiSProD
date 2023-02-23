#!/bin/bash

#SBATCH -J sparse-reacher-evaluation
#SBATCH -p general
#SBATCH -o sparse-reacher-evaluation_%A_%a.out
#SBATCH -e sparse-reacher-evaluation_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=palchatt@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32GB
#SBATCH --array=0-3

env_name=sparse_reacher
today=$(date +'%m-%d-%Y')

depths=(25 50 75 100)
depth=${depths[$SLURM_ARRAY_TASK_ID]}

run_name=${base_run_name}-${alg}-${depth}

module load python/3.9.8

start_time=`(date +%s)`
echo Start time: ${start_time}

source ${HOME}/deeprl_py3.9/bin/activate
cd ${HOME}/awesome-sogbofa
export DISPROD_PATH=${HOME}/awesome-sogbofa

# To evaluate model
PYTHONPATH=. python learn_gym.py --mode=evaluate --alg=${alg} --env_name=${env_name} --run_name=${run_name} --headless=True --render=False --compute_baseline=True --depth=${depth}

end_time=`(date +%s)`
echo End time: ${end_time}
