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
    alphas=(0.0 0.5 0.75 1 2 5 7.5 10)
elif [ $environment == "mountain_car" ]; then
    # Env: Mountain Car
    env_name=continuous-mountain-car-v2
    config_name=continuous_mountain_car
    alphas=(0.0 0.0025 0.005 0.0075 0.009 0.01 0.02 0.05)
elif [ $environment == "pendulum" ]; then
    # Env: Pendulum
    env_name=pendulum-v2
    config_name=pendulum
    alphas=(0.0 0.5 0.75 1 2 3 4 5)
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
dir_name=${today}-noise-vs-performance

n_episodes=10

for algorithm in "${algorithms[@]}"
do
    echo "Running exp with $algorithm on $env_name"

    results_folder=${path}/${dir_name}/${id}/${algorithm}

    # Setup parent directory for exp
    mkdir -p ${results_folder}

    if [ $algorithm == "sogbofa" ]
    then
        script_name="run_gym.py"
    else
        script_name="run_planning_baseline.py"
    fi

    for alpha in "${alphas[@]}"
    do
    stdbuf -oL nohup python3 ${DISPROD_PATH}/${script_name} --env_name ${config_name} --run_name ${dir_name}_${algorithm}_${alpha} --alpha_sim=$alpha --alpha_planner=$alpha --n_episodes=$n_episodes --alg=$algorithm --render=False >> ${results_folder}/stdout.txt
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

python3 ${DISPROD_PATH}/scripts/plot_graphs.py --run_id=${id} --run_name ${dir_name} --env_name ${env_name} --filename "exp3" --xlabel "Alpha" -x ${alphas[@]}