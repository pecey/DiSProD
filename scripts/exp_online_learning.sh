#!/bin/bash

if [[ -z "${DISPROD_PATH}" ]]; then
    echo "DISPROD_PATH is not set."
    exit
fi

today=$(date +'%m-%d-%Y')
id=$(date +%s)

environment=$1

mode="online_learning"

# Set env name
if [ $environment == "cartpole" ]; then
    env_name=continuous-cartpole-v1
    config_name=continuous_cartpole
elif [ $environment == "mountain_car" ]; then
    env_name=continuous-mountain-car-v2
    config_name=continuous_mountain_car
elif [ $environment == "pendulum" ]; then
    env_name=pendulum-v2
    config_name=pendulum
else
    echo "$environment is not a valid environment."
    exit 1
fi


# # Env: Pusher
# env_name=pusher-v2
# config_name=pusher


echo "Running ${mode} on ${env_name}"

# Set directory name
path=${DISPROD_PATH}/results/${env_name}/learning
dir_name=${today}-${mode}-${id}

# Setup parent directory for exp
mkdir -p ${path}/${dir_name}


stdbuf -oL nohup python3 ${DISPROD_PATH}/learn_gym.py --mode=${mode} --env_name ${config_name}  --run_name ${dir_name} --render=False --compute_baseline=True >> ${path}/${dir_name}/stdout.txt


git rev-parse HEAD >> ${path}/${dir_name}/hash.txt

cat ${path}/${dir_name}/logs/output.log | grep "Mean" > ${path}/${dir_name}/logs/summary.txt


