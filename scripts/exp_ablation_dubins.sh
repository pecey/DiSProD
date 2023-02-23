#!/bin/bash

if [[ -z "${DISPROD_PATH}" ]]; then
    echo "DISPROD_PATH is not set."
    exit
fi

today=$(date +'%m-%d-%Y')
id=$(date +%s)

alphas=(0 0.5 0.75 1 2 3 4 5)

config_name=continuous_dubins_car_w_velocity
env_name=continuous-dubins-car-w-velocity-state-v0

algorithm=sogbofa

path=${DISPROD_PATH}/results/${env_name}/planning

maps=(cave-mini)
taylor_expansion_modes=(complete state_var_only action_var_only no_var)

for alpha_val in "${alphas[@]}"
do
    for taylor_expansion_mode in "${taylor_expansion_modes[@]}"
    do
        # Set directory name  
        dir_name=${today}-${algorithm}-complete-suite-${alpha_val}-${taylor_expansion_mode}-${id}

        mkdir -p ${path}/$dir_name
        
        for map in "${maps[@]}"
        do
        echo "Evaluating ${config_name} using ${algorithm}. Taylor expansion mode: ${taylor_expansion_mode}. Map: ${map}"

        python3 ${DISPROD_PATH}/run_gym.py --env_name ${config_name} --map_name $map --run_name dubins_suite_$map --alg=${algorithm} --taylor_expansion_mode=${taylor_expansion_mode} --log_file=${path}/${dir_name}/${map}_log.txt --n_episodes=10 --alpha_plan=${alpha_val} --alpha_sim=${alpha_val} --render=False

        mv ${path}/dubins_suite_$map ${path}/${dir_name}
        done

        cd ${path}/$dir_name

        for filename in *
        do
            mv $filename/frames/0.png $filename.png
            mv $filename/results*.txt $filename.txt
            rmdir $filename/frames
            rmdir $filename
        done

        for map in "${maps[@]}"
        do
            summary=`tail -n1 dubins_suite_${map}.txt`	
            echo "${map} ${summary}" >> summary.txt
        done 

        git rev-parse HEAD >> hash.txt
    done

    mkdir ${path}/${today}-${algorithm}-complete-suite-${alpha_val}
    mv ${path}/${today}-${algorithm}-complete-suite-${alpha_val}-* ${path}/${today}-${algorithm}-complete-suite-${alpha_val}
done
