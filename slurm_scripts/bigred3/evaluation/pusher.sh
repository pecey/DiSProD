#!/bin/bash

#SBATCH -J pusher-evaluation
#SBATCH -p general
#SBATCH -o pusher-evaluation_%A_%a.out
#SBATCH -e pusher-evaluation_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=palchatt@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=16GB

env_name=pusher
today=$(date +'%m-%d-%Y')

module load python/3.9.8

run_name=${today}-evaluation-${alg}

# depths=(15 20 25 30 35 40 50)
# depth=${depths[$SLURM_ARRAY_TASK_ID]}

start_time=`(date +%s)`
echo Start time: ${start_time}

source ${HOME}/deeprl_py3.9/bin/activate
cd ${HOME}/awesome-sogbofa
export DISPROD_PATH=${HOME}/awesome-sogbofa

# To evaluate model
PYTHONPATH=. python learn_gym.py --mode=evaluate --alg=${alg} --env_name=${env_name} --run_name=${run_name} --headless=True --render=False --compute_baseline=True

end_time=`(date +%s)`
echo End time: ${end_time}
