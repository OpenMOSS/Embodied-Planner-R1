<div align="center">

# Embodied-Planner-R1
<div>
   🌠Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning 🚀
</div>
</div>

<div>
<br>

<div align="center">

[![Hugging Face Model](https://img.shields.io/badge/models-%23000000?style=for-the-badge&logo=huggingface&logoColor=000&logoColor=white)]()
[![Hugging Face Data](https://img.shields.io/badge/data-%23000000?style=for-the-badge&logo=huggingface&logoColor=000&logoColor=white)]()
[![Paper](https://img.shields.io/badge/Paper-%23000000?style=for-the-badge&logo=arxiv&logoColor=000&labelColor=white)]()
</div>
</div>

We introduce <strong>Embodied Planner-R1</strong>, a novel outcome-driven reinforcement learning framework that enables LLMs to develop interactive capabilities through autonomous exploration.

Embodied Planner-R1 enables LLM agents to learn causal relationships between actions and environmental feedback through <strong>multi-turn</strong> interactions, allowing them to update their policies based on an outcome reward.

<p align="center">
<img src=figs/alf_curve.jpg width=700/>
<img src=figs/alf_performance.jpg width=700/>
</p>




## 🔥Releases
<strong>[2025/07/01]</strong>
- 🌌 Full training code and scripts are available. 
- 🤗 We open-source our model weights in [huggingface]()



## 🚀 Installation
We separate the VERL training framework from the environment and wrap the environment into a [server](verl/alfworld_server/server) for interaction.

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

2. Prepare the environment for ALFWorld
```
conda create -n alfworld python=3.9
conda activate alfworld

# download task for training
pip install alfworld
pip install fastapi
pip install uvicorn
alfworld-download --data-dir ./get_data/alfworld
```

3. Prepare the environment for ScienceWorld
```
conda create --name scienceworld python=3.8
conda activate scienceworld

pip install scienceworld
conda install -y -c conda-forge openjdk=11
pip install fastapi
pip install uvicorn
```

## 🛠️ Data preparation
We need to prepare tasks for reinforcement learning.
```
# get task data for rl training
cd get_data
bash get_data_for_training.sh
```

## 🕹️ Quick Start
In our experimental setup, we used a 1×8 A100 (80GB) for training, with detailed training parameters provided in [examples/grpo_trainer/alf.sh](examples/grpo_trainer/alf.sh).

```
# Remember to replace the path in the shell script with your local path
bash cmd/alf.sh

bash cmd/sci_easy.sh
```

## 🎮 Evaluation
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


## Acknowledgements
The training codebase is primarily based on [Verl](https://github.com/volcengine/verl), while the evaluation framework is adapted from [MINT](https://github.com/xingyaoww/mint-bench). Our model builds upon the foundation of [`Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). We deeply appreciate their excellent contributions.


## Citation
```
@article{fei2025unleashing,
  title={Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning},
  author={Fei, Zhaoye and Ji, Li and Wang, Siyin and Shi, Junhao and Gong, Jingjing and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2506.23127},
  year={2025}
}
```


