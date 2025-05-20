export REPO_HOME=$(pwd)

final_folder="$REPO_HOME/verl/outputs_alf"  

bash_path=$REPO_HOME/examples/grpo_trainer/alf.sh


project_name=$(basename "$bash_path" .sh)
timestamp=$(date +"%Y%m%d_%H%M")
date_stamp=$(date +"%Y%m%d")
res_folder="${final_folder}/${date_stamp}/${project_name}/${timestamp}/rank_${RANK}"
echo "saving in ${res_folder}"
mkdir -p "$res_folder"

server_logging_folder="${res_folder}/server"
mkdir -p "$server_logging_folder"

output_folder="${res_folder}/outputs"
export LOG_DIR="$output_folder"

ckpt_folder="${res_folder}/ckpt"
export CKPT_DIR="$ckpt_folder"

tensorboard_folder="${res_folder}/tensorboard"
export TENSORBOARD_DIR="$tensorboard_folder"
mkdir -p "$tensorboard_folder"
echo "tb saving in ${TENSORBOARD_DIR}"

source /opt/conda/etc/profile.d/conda.sh
PORT=8000
if ss -tuln | grep -q ":$PORT "; then
    echo "端口 $PORT 已被占用"
else
    echo "$PORT 未被占用"
    conda activate /path/to/alfworld-env
    cd $REPO_HOME/verl/alfworld_server/server
    server_cmd="python start_server.py --num_servers 8"

    nohup $server_cmd > "${server_logging_folder}/run_stdout.log" 2> "${server_logging_folder}/run_stderr.log" &
    server_pid=$!
    echo "server Process ID: $server_pid Check logs in ${server_logging_folder}/"
    conda deactivate
fi

cd $REPO_HOME
conda activate /path/to/embodied-r1-env
cmd="bash ${bash_path}"
echo "Running $cmd"

$cmd 2>&1 | tee "${res_folder}/run_stdout.log" "${res_folder}/run_stderr.log"
echo "Check logs in ${res_folder}/" 