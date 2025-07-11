#!/bin/sh
set -x

system_prompt='You are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:Thought: your thoughts.\nAction: your next action.\n\nThe available actions are:\n1. `go to (receptacle)`\n2. `open (receptacle)`\n3. `close (receptacle)`\n4. `take (object) from (receptacle)`\n5. `move (object) to (receptacle)`\n6. `examine (something) with (object)`\n7. `use (object)`\n8. `heat (object) with (receptacle)`\n9. `clean (object) with (receptacle)`\n10. `cool (object) with (receptacle)`\n11. `slice (object) with (object)` - slice an object using a sharp object\n12. `look` - look around your current location\n13. `inventory` - check your current inventory\n14. `done` - Indicate that you believe the task is complete\nWhere `(object)` refers to manipulable objects and `(receptacle)` refers to receptacles or locations in the environment.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output: Nothing happens, that means the previous action is invalid and you should try more options.\nYou can only hold one object at a time. Before taking a new object, make sure you have placed down any object you are currently holding.\nYou should not assume or anticipate the feedback.\nEven if you have planned multiple steps ahead, you should only execute one action at a time\nDo not proceed with any further exploration or actions until you receive the feedback from the environment after your action.\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>'
start_port=8000
model_path='/path/to/Qwen2.5-7B-Instruct'

ray start --head
python -m verl.trainer.main_ppo_alf \
    algorithm.adv_estimator=grpo \
    data.train_files=get_data/rl/alf_train.json \
    data.val_files=get_data/rl/alf_valid_seen.json \
    data.train_batch_size=128 \
    +data.max_length=4096 \
    +data.max_steps=30 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    '+data.system_prompt="'"$system_prompt"'"' \
    reward_model.reward_manager=alf \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=alf \
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
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    +trainer.log_dir="$LOG_DIR" \
    trainer.default_local_dir="$CKPT_DIR"
    "$@"
