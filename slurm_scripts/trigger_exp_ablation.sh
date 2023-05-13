env_name=$1
today=$(date +'%m-%d-%Y')
identifier=$(date +%s)
system=$SERVER_NAME

base_run_name=${today}-ablation
echo "Triggering ablation with ${env_name} on ${system}"

SLURM_LOG_PATH=${DISPROD_PATH}/slurm_logs/${env_name}/exp_ablation/${today}

# Create the directory to hold the logs
mkdir -p ${SLURM_LOG_PATH}

export OMP_NUM_THREADS=12

declare -A env_mapping
env_mapping['cartpole']="continuous-cartpole-v1"
env_mapping['pendulum']="pendulum-v2"
env_mapping['mountain_car']="continuous-mountain-car-v2"
env_mapping['sparse_mountain_car']="sparse-continuous-mountain-car-v1"
env_mapping['dubins_car']="continuous-dubins-car-ablation-v0"
env_mapping['simple_env']="simple-env-v1"

# Run experiments
jid1=$(sbatch --export=ALL,mode=complete,base_run_name=${base_run_name} --parsable -o ${SLURM_LOG_PATH}/${identifier}_complete.out  -e ${SLURM_LOG_PATH}/${identifier}_complete.err ${DISPROD_PATH}/slurm_scripts/${system}/exp_ablation/${env_name}.sh)

jid2=$(sbatch --export=ALL,mode=state_var_only,base_run_name=${base_run_name} --parsable -o ${SLURM_LOG_PATH}/${identifier}_state_var.out  -e ${SLURM_LOG_PATH}/${identifier}_state_var.err ${DISPROD_PATH}/slurm_scripts/${system}/exp_ablation/${env_name}.sh)

# jid3=$(sbatch --export=ALL,mode=action_var_only,base_run_name=${base_run_name} --parsable -o ${SLURM_LOG_PATH}/${identifier}_action_var.out  -e ${SLURM_LOG_PATH}/${identifier}_action_var.err ${DISPROD_PATH}/slurm_scripts/${system}/exp_ablation/${env_name}.sh)

jid4=$(sbatch --export=ALL,mode=no_var,base_run_name=${base_run_name} --parsable -o ${SLURM_LOG_PATH}/${identifier}_no_var.out  -e ${SLURM_LOG_PATH}/${identifier}_no_var.err ${DISPROD_PATH}/slurm_scripts/${system}/exp_ablation/${env_name}.sh)

git rev-parse HEAD >> ${SLURM_LOG_PATH}/${identifier}_hash.txt

# Cleanup
results_path=${DISPROD_PATH}/results/${env_mapping[$env_name]}/planning

sbatch --dependency=afterany:$jid1:$jid2:$jid4 -o ${SLURM_LOG_PATH}/${identifier}_cleanup.out  -e ${SLURM_LOG_PATH}/${identifier}_cleanup.err --wrap="mkdir -p ${results_path}/${base_run_name} && mv ${results_path}/${base_run_name}-* ${results_path}/${base_run_name} && mv ${SLURM_LOG_PATH}/${identifier}_hash.txt ${results_path}/${base_run_name}"

