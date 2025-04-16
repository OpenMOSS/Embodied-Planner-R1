#!/bin/sh
set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS
# HOME='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl'

export system_prompt='You are a helpful assistant to do some scientific experiment in an environment.\nYou should explore the environment and find the items you need to complete the experiment.\n\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway.\nYou can teleport to any room in one step.\nThe available actions are:\nactivate OBJ\nclose OBJ\nconnect OBJ to OBJ\ndeactivate OBJ\ndisconnect OBJ\ndunk OBJ in OBJ\neat OBJ\nflush OBJ\nfocus on OBJ\ngo LOC\ninventory\nlook around\nlook at OBJ\nlook in OBJ\nmix OBJ\nmove OBJ to OBJ\nopen OBJ\npick up OBJ\npour OBJ in OBJ\nput down OBJ\nread OBJ\nuse OBKJ on OBJ\nteleport to LOC\nwait: wait 10 steps\nwait1: wait 1 step\ntask: check your task\ndone: indicate that you believe the task is complete\nWhen arrive a new location, you should use look around to check the OBj you can interact with.\nUse focus on OBJ only neccessary as incorrect use will cause environment ends.\nDo not proceed with any further exploration or actions until you receive the feedback from the environment after your action.\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>'
start_port=8000
model_path='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/data/Qwen/Qwen2.5-7B-Instruct'

python -m verl.eval.scienceworld.test \
    algorithm.adv_estimator=grpo \
    data.train_files=/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/data/ScienceWolrd/train_dataset.json \
    data.val_files=/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/data/ScienceWolrd/valid_dataset.json \
    data.train_batch_size=64 \
    +data.max_length=4096 \
    +data.max_steps=30 \
    +data.easy="easy" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    '+data.system_prompt="'"$system_prompt"'"' \
    reward_model.reward_manager=alf \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sci \
    actor_rollout_ref.rollout.max_model_len=8192 \
    +actor_rollout_ref.rollout.url="$start_port" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.balance_batch=False \
    trainer.logger=[console,tensorboard] \
    trainer.project_name=verl_grpo_demo_alf_debug \
    trainer.experiment_name=qwen2_7b_alf \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
