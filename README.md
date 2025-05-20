<h1 style="text-align: center;">Embodied R1: Incentivizing Environment Interaction
Ability in LLMs via Reinforcement Learning</h1>

## Installation
1. Embodied-r1 is based on verl with vLLM>=0.8
```
# Create the conda environment
conda create -n embodied-r1 python==3.10
conda activate embodied-r1

cd embodied-r1
pip3 install -e .

# Install the latest stable version of vLLM
pip3 install vllm==0.8.3

# Install flash-attn
pip3 install flash-attn --no-build-isolation
```

2. Prepare environment for ALFWorld
```
conda create -n alfworld python=3.9
conda activate alfworld

# download task for training
pip install alfworld
alfworld-download
```

3. Prepare environment for ScienceWorld
```
conda create --name scienceworld python=3.8
conda activate scienceworld

pip install scienceworld
```

## 2. Prepare for data
```
# get task data for rl training
bash get_data/get_data_for_training.sh
```

## 3. Start training
```
# Remember to replace the path in the shell script with your local path
bash cmd/alf.sh

bash cmd/sci_easy.sh
```
