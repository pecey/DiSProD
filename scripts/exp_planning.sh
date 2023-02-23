#!/bin/bash

if [[ -z "${DISPROD_PATH}" ]]; then
    echo "DISPROD_PATH is not set."
    exit
fi

today=$(date +'%m-%d-%Y')
id=$(date +%s)

environment=$2

# Set env name and depth
if [ $environment == "cartpole" ]; then
    # Env: Cart Pole
    env_name=continuous-cartpole-v1
    config_name=continuous_cartpole
elif [ $environment == "mountain_car" ]; then
    # Env: Mountain Car
    env_name=continuous-mountain-car-v2
    config_name=continuous_mountain_car
elif [ $environment == "pendulum" ]; then
    # Env: Pendulum
    env_name=pendulum-v2
    config_name=pendulum
else
    echo "Unknown environment specified"
    exit 1
fi

if [ $1 == "all" ]; then
    algorithms=("cem" "mppi" "sogbofa")
else
    algorithms=($1)
fi

# Set directory name
path=${DISPROD_PATH}/results/${env_name}
dir_name=${today}-planning

n_episodes=1
seeds=(10 20 30 40 50 60 70 80 90 100)

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
    else
        script_name="run_planning_baseline.py"
        option_name="n_samples"
    fi

    for seed in "${seeds[@]}"
    do
    stdbuf -oL nohup python3 ${DISPROD_PATH}/${script_name} --env_name ${config_name} --run_name ${dir_name} --n_episodes=${n_episodes} --seed=${seed} --alg=$algorithm --render=False --log_file=${results_folder}/results_${seed}.txt >> ${results_folder}/stdout.txt

    cat ${results_folder}/results_${seed}.txt | grep  "Score" >> ${results_folder}/summary.txt
    done

    mkdir -p ${results_folder}/raw
    mv ${path}/${dir_name}/results*.txt ${results_folder}/raw

    python3 ${DISPROD_PATH}/scripts/summarize.py --path=${results_folder}
done

git rev-parse HEAD >> hash.txt

