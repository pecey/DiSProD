#!/bin/bash

if [[ -z "${DISPROD_PATH}" ]]; then
    echo "DISPROD_PATH is not set."
    exit
fi

today=$(date +'%m-%d-%Y')
id=$(date +%s)

env_name=simple-env-v1
config_name=simple_env

# Set directory name
path=${DISPROD_PATH}/results/${env_name}
dir_name=${today}-ablation

n_episodes=1
alphas=(0.0 0.25 0.5 0.75 1 1.25 1.5 1.75 2)
taylor_expansion_modes=(complete state_var_only no_var)

for alpha in "${alphas[@]}"
do  
    for taylor_expansion_mode in "${taylor_expansion_modes[@]}"
    do
    
        echo "Running with mode: ${taylor_expansion_mode}"

        results_folder=${path}/${dir_name}/${id}/sogbofa/${alpha}/${mode}

        # Setup parent directory for exp
        mkdir -p ${results_folder}
        
       
        stdbuf -oL nohup python3 ${DISPROD_PATH}/run_gym.py --env_name ${config_name} --run_name ${dir_name}-${alpha}-${taylor_expansion_mode} --render=False --n_episodes=10 --alpha=${alpha} --taylor_expansion_mode=${taylor_expansion_mode} --alg=sogbofa --n_restarts=50 >> ${results_folder}/stdout.txt

        #python3 ${DISPROD_PATH}/scripts/summarize.py --path=${results_folder}
    done
done

git rev-parse HEAD >> hash.txt

