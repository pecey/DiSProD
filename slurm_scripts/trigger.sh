exp_name=$1
env_name=$2
alg=$3
today=$(date +'%m-%d-%Y')
identifier=$(date +%s)
system=$SERVER_NAME

base_run_name=$(echo ${today}-${exp_name} | tr _ -)
echo "Triggering ${exp_name} with ${env_name} on ${system}"

SLURM_LOG_PATH=${HOME}/awesome-sogbofa/slurm_logs/${env_name}/${exp_name}/${today}

# Create the directory to hold the logs
mkdir -p ${SLURM_LOG_PATH}

export OMP_NUM_THREADS=12

declare -A env_mapping
env_mapping['cartpole']="continuous-cartpole-v1"
env_mapping['pendulum']="pendulum-v2"
env_mapping['mountain_car']="continuous-mountain-car-v2"

jid1=$(sbatch --export=ALL,alg=${alg},base_run_name=${base_run_name} --parsable -o ${SLURM_LOG_PATH}/${identifier}_main.out  -e ${SLURM_LOG_PATH}/${identifier}_main.err ${HOME}/awesome-sogbofa/slurm_scripts/${system}/${exp_name}/${env_name}.sh)

git rev-parse HEAD >> ${SLURM_LOG_PATH}/${identifier}_hash.txt

# Cleanup

results_path=${HOME}/awesome-sogbofa/results/${env_mapping[$env_name]}/planning

sbatch --dependency=afterany:$jid1 -o ${SLURM_LOG_PATH}/${identifier}_cleanup.out  -e ${SLURM_LOG_PATH}/${identifier}_cleanup.err --wrap="mkdir -p ${results_path}/${base_run_name} && mv ${results_path}/${base_run_name}-${alg} ${results_path}/${base_run_name} && mv ${SLURM_LOG_PATH}/${identifier}_hash.txt ${results_path}/${base_run_name}"

