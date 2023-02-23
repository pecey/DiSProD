env_name=$1
today=$(date +'%m-%d-%Y')
identifier=$(date +%s)
system=$SERVER_NAME
exp_name=exp_depth

if [ "$2" == "noisy" ]; then
    base_run_name=$(echo ${today}-${exp_name}-noisy | tr _ -)
    noisy=True
else
    base_run_name=$(echo ${today}-${exp_name} | tr _ -)
    noisy=False
fi

echo "Triggering ${exp_name} with ${env_name} on ${system}. Noise: ${noisy}"

SLURM_LOG_PATH=${HOME}/awesome-sogbofa/slurm_logs/${env_name}/${exp_name}/${today}

# Create the directory to hold the logs
mkdir -p ${SLURM_LOG_PATH}

export OMP_NUM_THREADS=12

declare -A env_mapping
env_mapping['cartpole']="continuous-cartpole-v1"
env_mapping['pendulum']="pendulum-v2"
env_mapping['mountain_car']="continuous-mountain-car-v2"

# CEM
jid1=$(sbatch --export=ALL,base_run_name=${base_run_name},alg=cem,noisy=${noisy} --parsable -o ${SLURM_LOG_PATH}/${identifier}_cem.out  -e ${SLURM_LOG_PATH}/${identifier}_cem.err ${HOME}/awesome-sogbofa/slurm_scripts/${system}/${exp_name}/${env_name}.sh)

# MPPI
jid2=$(sbatch --export=ALL,base_run_name=${base_run_name},alg=mppi,noisy=${noisy} --parsable -o ${SLURM_LOG_PATH}/${identifier}_mppi.out  -e ${SLURM_LOG_PATH}/${identifier}_mppi.err ${HOME}/awesome-sogbofa/slurm_scripts/${system}/${exp_name}/${env_name}.sh)

# SOGBOFA
jid3=$(sbatch --export=ALL,base_run_name=${base_run_name},alg=sogbofa,noisy=${noisy} --parsable -o ${SLURM_LOG_PATH}/${identifier}_sogbofa.out  -e ${SLURM_LOG_PATH}/${identifier}_sogbofa.err ${HOME}/awesome-sogbofa/slurm_scripts/${system}/${exp_name}/${env_name}.sh)

git rev-parse HEAD >> ${SLURM_LOG_PATH}/${identifier}_hash.txt

# Cleanup
results_path=${HOME}/awesome-sogbofa/results/${env_mapping[$env_name]}/planning

sbatch --dependency=afterany:$jid1:$jid2:$jid3 -o ${SLURM_LOG_PATH}/${identifier}_cleanup.out  -e ${SLURM_LOG_PATH}/${identifier}_cleanup.err --wrap="mkdir -p ${results_path}/${base_run_name} && mv ${results_path}/${base_run_name}-* ${results_path}/${base_run_name} && mv ${SLURM_LOG_PATH}/${identifier}_hash.txt ${results_path}/${base_run_name}"

