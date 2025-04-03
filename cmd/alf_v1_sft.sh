final_folder="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl/outputs_demo"  

bash_path=/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/examples/grpo_trainer/alf_v1_sft.sh


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

source /opt/conda/etc/profile.d/conda.sh
PORT=8000
if ss -tuln | grep -q ":$PORT "; then
    echo "端口 $PORT 已被占用"
else
    echo "$PORT 未被占用"
    conda activate /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/lji_env/alfworld
    cd /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl/alfworld_server/server
    server_cmd="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/lji_env/alfworld/bin/python start_server.py --num_servers 8"

    nohup $server_cmd > "${server_logging_folder}/run_stdout.log" 2> "${server_logging_folder}/run_stderr.log" &
    server_pid=$!
    echo "server Process ID: $server_pid Check logs in ${server_logging_folder}/"
    conda deactivate
fi

cd /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod
conda activate /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/lji_env/verl_spmd
cmd="bash ${bash_path}"
echo "Running $cmd"

$cmd 2>&1 | tee "${res_folder}/run_stdout.log" "${res_folder}/run_stderr.log"
echo "Check logs in ${res_folder}/" 