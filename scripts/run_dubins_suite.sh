#!/bin/bash


today=$(date +'%m-%d-%Y')
id=$(date +%s)

# algorithm: sogbofa/cem/mppi
algorithm=$1

# Uncomment EITHER pair
config_name=continuous_dubins_car_w_velocity
env_name=continuous-dubins-car-w-velocity-state-v0

# config_name=continuous_dubins_car
# env_name=continuous-dubins-car-v0

echo "Evaluating ${config_name} using ${algorithm}."

script_name="run_gym.py"


# Set directory name
path=${DISPROD_PATH}/results/${env_name}/${alg}/
dir_name=${today}-${algorithm}-complete-suite-${id}

mkdir -p ${path}/$dir_name

for map in no-ob-1 no-ob-2 no-ob-3 no-ob-4 no-ob-5 ob-1 ob-2 ob-3 ob-4 ob-5 ob-6 ob-7 ob-8 ob-9 ob-10 ob-11 cave-mini
do
   python3 ${DISPROD_PATH}/${script_name} --env_name ${config_name} --map_name $map --run_name dubins_suite_$map --alg=${algorithm} --n_episodes 5 --log_file=${path}/${dir_name}/${map}_log.txt --render False
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
	summary=`tail -n1 dubins_suite_${map}_${alg}.txt`	
	echo "${map} ${summary}" >> `summary_${alg}.txt`
done 

git rev-parse HEAD >> hash.txt
