#!/bin/bash

if [[ -z "${DISPROD_PATH}" ]]; then
    echo "DISPROD_PATH is not set."
    exit
fi

today=$(date +'%m-%d-%Y')
id=$(date +%s)

sparsity_values=(0.0000001 0.000001 0.00001 0.0001 0.001 0.01 0.1 1)

env_name=continuous-mountain-car-v2
config_name=continuous_mountain_car

depths=(100 150 200)
seeds=(10 20 30 40 50 60 70 80 90 100)

if [ $1 == "all" ]; then
    algorithms=("cem" "mppi" "sogbofa")
else
    algorithms=($1)
fi

# Set directory name
path=${DISPROD_PATH}/results/${env_name}
dir_name=${today}-reward-sparsity-vs-performance

n_episodes=1

for algorithm in "${algorithms[@]}"
do
    echo "Running exp with $algorithm on $env_name"
    
    results_folder=${path}/${dir_name}/${id}/${algorithm}

    # Setup parent directory for exp
    mkdir -p ${results_folder}

    if [ $algorithm == "sogbofa" ]
    then
        script_name="run_gym.py"
        option_name="n_restarts"
        option_value=1000
    else
        script_name="run_planning_baseline.py"
        option_name="n_samples"
        option_value=1000
    fi

    for sparsity_value in "${sparsity_values[@]}"
    do
        for depth in "${depths[@]}"
        do
        for seed in "${seeds[@]}"
        do
        stdbuf -oL nohup python3 ${DISPROD_PATH}/${script_name} --env_name ${config_name} --run_name ${dir_name}_${algorithm}_${depth} --${option_name}=${option_value} --n_episodes=${n_episodes} --alg=$algorithm --depth=${depth} --reward_sparsity=${sparsity_value} --seed=${seed} --render=False --log_file=${results_folder}/${sparsity_value}_${depth}_${seed}.txt >> ${results_folder}/stdout.txt
        done
        done
    done

    # By default the results are saved in results/env_name/run_name/results.txt
    mv ${path}/${dir_name}_${algorithm}_[0-9]* ${results_folder}
    cd ${results_folder}


    for filename in *
    do
        mv $filename/frames/0.png $filename.png 2>> ${results_folder}/stderr.txt
        mv $filename/results*.txt $filename.txt 2>> ${results_folder}/stderr.txt
        mv $filename/failures*.txt "${filename}_failures".txt 2>> ${results_folder}/stderr.txt
        rmdir $filename/frames 2>> ${results_folder}/stderr.txt
        rmdir $filename 2>> ${results_folder}/stderr.txt
    done
done

git rev-parse HEAD >> hash.txt