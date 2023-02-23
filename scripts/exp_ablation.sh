#!/bin/bash

if [[ -z "${DISPROD_PATH}" ]]; then
    echo "DISPROD_PATH is not set."
    exit
fi

today=$(date +'%m-%d-%Y')
id=$(date +%s)

env_name=pendulum-v2
config_name=pendulum

# Set directory name
path=${DISPROD_PATH}/results/${env_name}
dir_name=${today}-ablation

n_episodes=1
seeds=(10 20 30 40 50 60 70 80 90 100)
alphas=(0.0 0.5 0.75 1 2 3 4 5)

mean_propagation_configs=(True False False)
fop_only_configs=(False True False)
modes=(mean_prop_only fop_only complete)



for alpha in "${alphas[@]}"
do  
    for i in 0 1 2
    do
        mean_propagation=${mean_propagation_configs[${i}]}
        fop_only=${fop_only_configs[${i}]}
        mode=${modes[$i]}

        echo "Running with mean_propagation: ${mean_propagation} and fop_only: ${fop_only} on $env_name"

        results_folder=${path}/${dir_name}/${id}/sogbofa/${alpha}/${mode}

        # Setup parent directory for exp
        mkdir -p ${results_folder}
        
        for seed in "${seeds[@]}"
        do
        stdbuf -oL nohup python3 ${DISPROD_PATH}/run_gym.py --env_name ${config_name} --run_name ${dir_name} --n_episodes=${n_episodes} --seed=${seed} --render=False --log_file=${results_folder}/results_${seed}.txt --alpha_sim=${alpha} --alpha_planner=${alpha} --mean_propagation=${mean_propagation} --first_order_partials_only=${fop_only} --n_restarts=200 >> ${results_folder}/stdout.txt

        cat ${results_folder}/results_${seed}.txt | grep  "Score" >> ${results_folder}/summary.txt
        done

        mkdir -p ${results_folder}/raw
        mv ${path}/${dir_name}/${id}sogbofa/results*.txt ${results_folder}/raw
        mv ${path}/${dir_name}/${id}sogbofa/failures*.txt ${results_folder}/raw


        python3 ${DISPROD_PATH}/scripts/summarize.py --path=${results_folder}
    done
done

git rev-parse HEAD >> hash.txt

