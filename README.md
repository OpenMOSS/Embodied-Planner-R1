<h1 style="text-align: center;">Embodied-Planner-R1: Learning to Explore and Act via Pure Reinforcement Learning</h1>

## Installation
1. Embodied-Planner-R1 is based on verl with vLLM>=0.8
```
# Create the conda environment
conda create -n Embodied-Planner-R1 python=3.10
conda activate Embodied-Planner-R1

cd Embodied-Planner-R1
pip3 install -e .

# Install the latest stable version of vLLM
pip3 install vllm==0.8.3

# Install flash-attn
pip3 install flash-attn --no-build-isolation
pip3 install tensorboard
```

2. Prepare environment for ALFWorld
```
conda create -n alfworld python=3.9
conda activate alfworld

# download task for training
pip install alfworld
pip install fastapi
pip install uvicorn
alfworld-download --data-dir ./get_data/alfworld
```

3. Prepare environment for ScienceWorld
```
conda create --name scienceworld python=3.8
conda activate scienceworld

pip install scienceworld
conda install -y -c conda-forge openjdk=11
pip install fastapi
pip install uvicorn
```

## 2. Prepare for data
```
# get task data for rl training
cd get_data
bash get_data_for_training.sh
```

## 3. Start training
```
# Remember to replace the path in the shell script with your local path
bash cmd/alf.sh

bash cmd/sci_easy.sh
```

## 4. Evaluation
```
# We follow the framework of MINT to evaluate models.
cd verl/eval_agent
conda create -n eval_agent python=3.10
conda activate eval_agent
bash setup.sh

conda create -n vllm python=3.10
conda activate vllm
pip install vllm

# deploy the model
python -m vllm.entrypoints.openai.api_server --served-model-name embodied_r1_alfworld --model /path/to/model --port 8000 --disable-frontend-multiprocessing --gpu-memory-utilization 0.99 --disable-frontend-multiprocessing --max-model-len 4096 --enforce-eager

# start evaluation
conda activate eval_agent

python -m eval_agent.main --agent_config er1_alfworld --exp_config alfworld_v2 --split dev --verbose # you can find more examples in eval.sh

```